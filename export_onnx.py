#!/usr/bin/env python3
# Rustrainer-OCR A GUI Utility to train/fine tune OCR Models written in Rust.
# Copyright (C) 2026 Mohammad Najm
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: Mohammad Najm <najm.devops@gmail.com>
# https://github.com/najmdevstudio/Rustrainer_OCR
"""
Companion script for plate-ocr: loads exported Burn weights and produces an ONNX model.

Usage:
    python export_onnx.py <weights_dir> [--output plate_ocr.onnx]

The <weights_dir> must contain manifest.json and the .bin weight files
produced by `plate-ocr export`.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ── Architecture (must mirror src/model/crnn.rs exactly) ──────────────────────

class CrnnOcr(nn.Module):
    def __init__(self, num_classes: int = 37, lstm_hidden: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # After CNN: [batch, 256, 4, 32] → reshape to [batch, 32, 1024]
        self.lstm = nn.LSTM(
            input_size=256 * 4,
            hidden_size=lstm_hidden,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 32, 128]
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))

        # x: [batch, 256, 4, 32]
        batch, channels, height, width = x.size()
        # Permute to [batch, width, channels*height]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, width, channels * height)

        # BiLSTM
        x, _ = self.lstm(x)

        # Linear + log_softmax
        x = self.linear(x)
        x = torch.log_softmax(x, dim=2)

        # Transpose to [time, batch, classes] for CTC convention
        x = x.permute(1, 0, 2)
        return x


# ── Weight loading ────────────────────────────────────────────────────────────

def load_bin(path: Path, shape: list[int]) -> np.ndarray:
    """Load a raw little-endian f32 binary file into a numpy array."""
    data = path.read_bytes()
    count = 1
    for s in shape:
        count *= s
    values = struct.unpack(f"<{count}f", data)
    return np.array(values, dtype=np.float32).reshape(shape)


def load_weights(weights_dir: str, model: CrnnOcr) -> None:
    """Load exported weights (manifest.json + raw little-endian .bin files, produced by
    `plate-ocr export`) into the PyTorch model.

    Tensor names in the manifest are already PyTorch `state_dict` keys (the Rust exporter
    renames/transposes everything — including merging the BiLSTM's per-gate layers back into
    PyTorch's combined `weight_ih_l0`/`weight_hh_l0`/... tensors — before writing the manifest),
    so loading is a direct 1:1 assignment.
    """
    wdir = Path(weights_dir)
    with open(wdir / "manifest.json") as f:
        manifest = json.load(f)

    state = model.state_dict()
    manifest_names = set()

    for entry in manifest:
        name, filename, shape = entry["name"], entry["file"], entry["shape"]
        manifest_names.add(name)
        tensor = torch.from_numpy(load_bin(wdir / filename, shape))

        if name not in state:
            print(f"WARNING: '{name}' from the manifest has no matching PyTorch parameter, skipping.")
            continue
        if tuple(state[name].shape) != tuple(tensor.shape):
            raise ValueError(
                f"Shape mismatch for '{name}': model expects {tuple(state[name].shape)}, "
                f"manifest has {tuple(tensor.shape)}."
            )
        state[name] = tensor

    # `num_batches_tracked` is a PyTorch-only bookkeeping buffer (batch counter used to compute
    # the running-stats momentum) with no Burn equivalent; it isn't needed for inference/ONNX
    # export, so it's left at its PyTorch-initialized default (0) instead of being loaded.
    missing = sorted(
        name
        for name in set(state.keys()) - manifest_names
        if not name.endswith("num_batches_tracked")
    )
    if missing:
        raise ValueError(f"Missing tensor(s) in manifest '{wdir / 'manifest.json'}': {missing}")

    model.load_state_dict(state)
    print(f"Loaded {len(manifest)} parameter tensor(s) from {weights_dir}")


# ── ONNX export ──────────────────────────────────────────────────────────────

def export(weights_dir: str, output_path: str) -> None:
    model = CrnnOcr()
    load_weights(weights_dir, model)
    model.eval()

    dummy_input = torch.randn(1, 1, 32, 128)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["log_probs"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "log_probs": {1: "batch_size"},
        },
        opset_version=17,
        # Use the stable TorchScript-based exporter: it directly supports `dynamic_axes` and
        # doesn't require the optional `onnxscript` package that the newer dynamo-based exporter
        # needs.
        dynamo=False,
    )
    print(f"ONNX model exported to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export plate-ocr weights to ONNX")
    parser.add_argument("weights_dir", help="Directory with exported .bin weights and manifest.json")
    parser.add_argument("--output", "-o", default="plate_ocr.onnx", help="Output ONNX file path")
    args = parser.parse_args()

    export(args.weights_dir, args.output)
