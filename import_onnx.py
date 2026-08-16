#!/usr/bin/env python3
"""
Companion script for plate-ocr: converts a trained/exported .onnx model back into a plain
PyTorch state dict, so it can be used as a `--pretrained` fine-tuning source (Burn checkpoints
and .pt/.pth files are read directly by the Rust binary; for .onnx it shells out to this script
first, then loads the resulting .pt through the same code path — see src/interop/onnx_import.rs
and src/interop/pytorch_import.rs).

Usage:
    python import_onnx.py <input.onnx> <output.pt>

The ONNX graph is expected to follow this project's fixed CRNN architecture (see
src/model/crnn.rs / export_onnx.py): 4x (Conv2d + BatchNormalization), a bidirectional LSTM, and
a final linear layer. Weight/bias tensors are located positionally (by walking the graph's nodes
and using each operator's own fixed input order) rather than by initializer name, since exporter
-assigned initializer names aren't guaranteed to be human-readable or stable across PyTorch/ONNX
versions.

Notes on ONNX's LSTM convention (see https://onnx.ai/onnx/operators/onnx__LSTM.html):
  - Gates are packed in "i, o, f, c" order (input, output, forget, cell), whereas PyTorch's
    combined `weight_ih_l0` etc. use "i, f, g, o" order. This script reorders accordingly.
  - `W`/`R`/`B` stack both directions along their leading axis for a bidirectional LSTM.
"""

import sys

import numpy as np
import onnx
import torch
from onnx import numpy_helper


def reorder_iofc_to_ifgo(matrix: np.ndarray, hidden: int) -> np.ndarray:
    """Reorders a `[4*hidden, ...]` array from ONNX's `i, o, f, c` gate order to PyTorch's
    `i, f, g, o` order."""
    i, o, f, c = np.split(matrix, 4, axis=0)
    return np.concatenate([i, f, c, o], axis=0)


def find_bias_for(output_name: str, graph, initializers: dict) -> np.ndarray | None:
    """Finds a constant `Add` operand chained to `output_name`, e.g. the bias added after a
    `MatMul` when a `Linear` layer wasn't fused into a single `Gemm` node."""
    for node in graph.node:
        if node.op_type == "Add" and output_name in node.input:
            other = node.input[0] if node.input[1] == output_name else node.input[1]
            if other in initializers:
                return initializers[other]
    return None


def matmul_weight(node, initializers: dict) -> np.ndarray:
    for name in node.input:
        if name in initializers:
            return initializers[name]
    raise ValueError(f"Could not find a constant weight operand for MatMul node '{node.name}'.")


def convert(input_path: str, output_path: str) -> None:
    model = onnx.load(input_path)
    graph = model.graph
    initializers = {t.name: numpy_helper.to_array(t) for t in graph.initializer}

    state_dict: dict[str, torch.Tensor] = {}

    conv_index = 0
    bn_index = 0
    lstm_node = None
    gemm_node = None
    matmul_node = None

    for node in graph.node:
        if node.op_type == "Conv":
            conv_index += 1
            state_dict[f"conv{conv_index}.weight"] = torch.from_numpy(initializers[node.input[1]].copy())
            if len(node.input) > 2:
                state_dict[f"conv{conv_index}.bias"] = torch.from_numpy(initializers[node.input[2]].copy())
        elif node.op_type == "BatchNormalization":
            bn_index += 1
            state_dict[f"bn{bn_index}.weight"] = torch.from_numpy(initializers[node.input[1]].copy())
            state_dict[f"bn{bn_index}.bias"] = torch.from_numpy(initializers[node.input[2]].copy())
            state_dict[f"bn{bn_index}.running_mean"] = torch.from_numpy(initializers[node.input[3]].copy())
            state_dict[f"bn{bn_index}.running_var"] = torch.from_numpy(initializers[node.input[4]].copy())
        elif node.op_type == "LSTM":
            lstm_node = node
        elif node.op_type == "Gemm":
            gemm_node = node
        elif node.op_type == "MatMul" and matmul_node is None:
            matmul_node = node

    if conv_index != 4:
        raise ValueError(
            f"Expected 4 Conv nodes (plate-ocr's fixed CRNN architecture), found {conv_index} in "
            f"'{input_path}'. Is this an ONNX export of a compatible model?"
        )
    if bn_index not in (0, 4):
        raise ValueError(
            f"Expected either 0 (folded into the Conv weights) or 4 (separate) "
            f"BatchNormalization nodes, found {bn_index} in '{input_path}'."
        )
    if bn_index == 0:
        # Exporters commonly fold BatchNorm (an affine transform in eval mode) directly into the
        # preceding Conv's weight/bias when the model is exported in eval mode — the individual
        # gamma/beta/running_mean/running_var values are then unrecoverable (and unnecessary):
        # the already-fused `convN.weight`/`convN.bias` above reproduce the exact same math.
        # Leaving `bnN.*` out of the state dict makes the Rust side (which loads with
        # `allow_partial`) keep those layers at Burn's fresh-init identity transform
        # (gamma=1, beta=0, mean=0, var=1), so the merged model computes identically.
        print("No separate BatchNormalization nodes found; assuming they were fused into the Conv weights.")

    if lstm_node is None:
        raise ValueError(f"No LSTM node found in '{input_path}'.")

    w = initializers[lstm_node.input[1]]  # [num_directions, 4*hidden, input]
    r = initializers[lstm_node.input[2]]  # [num_directions, 4*hidden, hidden]
    b = initializers[lstm_node.input[3]] if len(lstm_node.input) > 3 and lstm_node.input[3] else None

    num_directions, four_hidden, _input_size = w.shape
    hidden = four_hidden // 4
    if num_directions != 2:
        raise ValueError(f"Expected a bidirectional LSTM (2 directions), found {num_directions}.")

    for direction_index, suffix in enumerate(["", "_reverse"]):
        weight_ih = reorder_iofc_to_ifgo(w[direction_index], hidden)
        weight_hh = reorder_iofc_to_ifgo(r[direction_index], hidden)
        state_dict[f"lstm.weight_ih_l0{suffix}"] = torch.from_numpy(weight_ih.copy())
        state_dict[f"lstm.weight_hh_l0{suffix}"] = torch.from_numpy(weight_hh.copy())

        if b is not None:
            wb, rb = np.split(b[direction_index], 2)  # each [4*hidden]
            state_dict[f"lstm.bias_ih_l0{suffix}"] = torch.from_numpy(reorder_iofc_to_ifgo(wb, hidden).copy())
            state_dict[f"lstm.bias_hh_l0{suffix}"] = torch.from_numpy(reorder_iofc_to_ifgo(rb, hidden).copy())
        else:
            state_dict[f"lstm.bias_ih_l0{suffix}"] = torch.zeros(4 * hidden)
            state_dict[f"lstm.bias_hh_l0{suffix}"] = torch.zeros(4 * hidden)

    # Final linear head: PyTorch's `nn.Linear` on a 3D input (our LSTM output, [batch, time,
    # features]) is typically exported as `MatMul` (+ `Add` for the bias) rather than `Gemm`
    # (which the ONNX spec restricts to rank-2 inputs); both are handled here.
    if matmul_node is not None:
        weight = matmul_weight(matmul_node, initializers)
        # The exporter constant-folds `weight.t()` into the initializer, i.e. it is already
        # `[in, out]`; PyTorch's `nn.Linear.weight` expects `[out, in]`.
        state_dict["linear.weight"] = torch.from_numpy(weight.T.copy())
        bias = find_bias_for(matmul_node.output[0], graph, initializers)
        if bias is not None:
            state_dict["linear.bias"] = torch.from_numpy(bias.copy())
    elif gemm_node is not None:
        weight = initializers[gemm_node.input[1]]
        trans_b = next((attr.i for attr in gemm_node.attribute if attr.name == "transB"), 0)
        if not trans_b:
            weight = weight.T
        state_dict["linear.weight"] = torch.from_numpy(weight.copy())
        if len(gemm_node.input) > 2:
            state_dict["linear.bias"] = torch.from_numpy(initializers[gemm_node.input[2]].copy())
    else:
        raise ValueError(f"No MatMul/Gemm node found for the final linear layer in '{input_path}'.")

    if "linear.bias" not in state_dict:
        state_dict["linear.bias"] = torch.zeros(state_dict["linear.weight"].shape[0])

    torch.save(state_dict, output_path)
    print(f"Converted '{input_path}' -> '{output_path}' ({len(state_dict)} tensors).")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python import_onnx.py <input.onnx> <output.pt>", file=sys.stderr)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
