# Rustrainer-OCR

A pure-Rust OCR training and inference utility for license plate recognition, built with [Burn](https://burn.dev) deep learning framework. Trains a CRNN (CNN + BiLSTM + CTC) model — or a lighter Conv-CTC model with no recurrent layer — and supports ONNX export.

## Features

- **Pure Rust** training pipeline using Burn 0.21
- **AMD ROCm** GPU acceleration (native CubeCL backend)
- **Vulkan** fallback for AMD GPUs without ROCm
- **CPU** fallback via ndarray
- **Two model architectures** — CRNN (Conv+BiLSTM+CTC) and Conv-CTC (Conv-only, no recurrent layer) — auto-detected when fine-tuning and shown in the log/GUI before training starts (see [Architectures](#architectures))
- **ONNX export** via companion Python script
- **Fine-tune from PyTorch or ONNX models**, in addition to this project's own checkpoints — the format *and* architecture are auto-detected
- **Automatic Python setup**: missing `numpy`/`onnx`/`torch` packages are installed on demand, right before they're needed for ONNX fine-tuning
- **GUI wizard** that walks you through training/fine-tuning with live progress, a loss graph and a terminal-style log
- **CLI** for training, inference, and export
- **Single-file distribution**: prebuilt binaries on GitHub Releases, an install script, and an `extract-scripts` command so even a lone downloaded binary can hand out its bundled Python helpers

## Installation

Prebuilt binaries are published as single downloadable files on the project's GitHub Releases
page — there's no separate installer package, no bundled runtime, just one executable per
platform (plus the small `import_onnx.py`/`export_onnx.py` helper scripts alongside it, needed
only for the ONNX-related features described further down).

### Quick Install (Linux/macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/najmdevstudio/Rustrainer_OCR/main/install.sh | sh
```

This detects your OS/CPU, downloads the matching `plate-ocr` binary, and installs it to
`~/.local/bin` (override with `PLATE_OCR_INSTALL_DIR`); on Linux it also adds an
application-menu entry so the GUI wizard shows up like any other installed app. See the comments
at the top of [`install.sh`](install.sh) for every option it accepts (pinning a version, the
`vulkan` backend instead of `cpu`, skipping the desktop entry, etc).


### Manual Download

Grab the archive for your platform from the Releases page, extract it, and run the `plate-ocr`
(or `plate-ocr.exe`) binary inside. Prebuilt binaries are available for Linux x86_64 (`cpu` and
`vulkan` variants), Windows x86_64 (`cpu`), and macOS x86_64/arm64 (`cpu`). The default `rocm`
backend needs the ROCm/HIP SDK and an AMD GPU, which prebuilt-binary CI runners don't have, so it
isn't distributed this way — build it from source instead (see [Build](#build) below).

If all you have is the standalone binary (e.g. it was copied over on its own), it can still hand
you the two Python helper scripts on request, since they're embedded in it at compile time:

```bash
plate-ocr extract-scripts --output-dir .
```

### Build from Source

`cargo build --release` (see [Build](#build) below) produces the exact same kind of single,
self-contained executable that the Releases page distributes, at `target/release/plate-ocr`.

## Prerequisites

### ROCm (recommended for AMD GPUs)

```bash
# Install ROCm (Ubuntu/Fedora)
sudo apt install rocm-dev hip-runtime-amd rocblas-dev  # Ubuntu
sudo dnf install rocm-dev hip-runtime-amd rocblas-devel  # Fedora

# Verify
rocminfo | head -20
hipcc --version

# For RDNA 4 GPUs (RX 9070 etc.) if not yet in ROCm's official list:
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

### Python (for ONNX export, and for fine-tuning from a PyTorch/ONNX model)

Fine-tuning from an `.onnx` file needs Python 3 plus the `torch`, `numpy` and `onnx` packages.
You don't need to install these yourself: the first time you fine-tune from an `.onnx` file,
`plate-ocr` checks for them and automatically runs `pip install` for whichever are missing,
streaming progress to the terminal/GUI log (installing `torch` can take a little while). If
automatic installation fails (no `pip`, no internet, ...), it reports exactly which package(s)
failed and how to install them manually:

```bash
pip install torch numpy onnx
```

> If these packages live under a different interpreter than the `python3` on your `PATH` (a
> virtualenv, pyenv, conda, etc.), point `plate-ocr` at it with `PLATE_OCR_PYTHON=/path/to/python3`
> — both the dependency check/install and the ONNX conversion itself use this interpreter.

> Fine-tuning from a `.pt`/`.pth` file, or from this project's own checkpoints, needs no Python
> at all (`export_onnx.py`, run manually, is the only thing that always needs Python).

> Only have the `plate-ocr` binary itself (e.g. from a GitHub release) and not the rest of the
> repo? Run `plate-ocr extract-scripts` to write `import_onnx.py`/`export_onnx.py` next to it —
> both are embedded in the binary at compile time.

## Dataset Format

```
dataset/
├── train/
│   ├── images/          # Cropped plate images (PNG/JPEG)
│   └── labels.csv       # CSV with columns: image_name,label
└── valid/
    ├── images/
    └── labels.csv
```

Example `labels.csv`:
```csv
image_name,label
plate_001.png,ABC1234
plate_002.jpg,XY 56 Z89
```

> **Note:** Spaces in labels are automatically stripped during encoding. Only alphanumeric characters (`0-9`, `A-Z`) are used.

## Build

```bash
# AMD ROCm GPU (default)
cargo build --release

# AMD GPU via Vulkan (no ROCm needed)
cargo build --release --no-default-features --features vulkan

# CPU only
cargo build --release --no-default-features --features cpu
```

Each of these produces one self-contained executable at `target/release/plate-ocr` — the release
profile (see `Cargo.toml`) is tuned with thin LTO and a stripped binary specifically so this
single file is lean enough to hand out on its own, e.g. as a GitHub release asset.

## Usage

### GUI Wizard (recommended)

Running the app with no arguments (or with the `gui` command) opens a native window — using whichever window manager/desktop your OS already provides — that walks you through the whole process:

1. **Choose the process** — New Model Training or Fine-Tuning.
2. **Choose the parameters** — dataset base directory and the rest of the training parameters, prefilled with sensible defaults for the flow you picked (editable, with native folder/file pickers). New Model Training lets you pick the architecture; Fine-Tuning auto-detects it from the file you choose.
3. **Watch it train** — the detected/selected architecture is shown up front, followed by a terminal-style output pane, a live loss graph, and a progress bar showing overall completion.
4. **See the result** — success or a clearly-formatted failure message, closed with a single "OK".

```bash
cargo run --release
# or explicitly:
cargo run --release -- gui
```

The `train`, `infer` and `export` subcommands below remain available for scripting/automation.

### Train from Scratch

```bash
cargo run --release -- train \
    --data-dir dataset \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001 \
    --output-dir checkpoints

# Train the lighter Conv-CTC architecture instead (see Architectures below)
cargo run --release -- train \
    --data-dir dataset \
    --output-dir checkpoints \
    --architecture conv-ctc
```

`--architecture` (`crnn`, the default, or `conv-ctc`) only applies when training from scratch;
it's ignored (and the architecture auto-detected instead) whenever `--pretrained` is given.

### Fine-tune a Pretrained Model

`--pretrained` accepts three formats, auto-detected from the file extension — and for each
format, plate-ocr also auto-detects *which architecture* the file holds (see
[Architectures](#architectures)), printing/showing it before training starts:

| Extension            | Source                                                              |
|----------------------|----------------------------------------------------------------------|
| *(none)*, `.mpk`     | This project's own Burn checkpoint (e.g. `checkpoints/plate_ocr_final`) |
| `.pt`, `.pth`        | A PyTorch state dict (`torch.save(model.state_dict(), ...)`)         |
| `.onnx`              | An ONNX model (converted on the fly via the bundled `import_onnx.py`; Python + torch/numpy/onnx installed automatically if missing) |

For `.pt`/`.pth`/`.onnx` sources, the convolutional backbone, batch-norm layers and FC/linear
head(s) are matched by parameter name (see the PyTorch mirrors in `export_onnx.py`); for the CRNN
architecture, the BiLSTM's combined `weight_ih_l0`/`weight_hh_l0`/... tensors are automatically
split into this project's per-gate layout. If a `.onnx` graph has batch-norm folded into the
convolutions (the common case for models exported in eval mode), those layers are left at their
identity initialization and the already-fused convolution weights take over the same computation.

This project's `--pretrained` import is designed to round-trip files *this same tool* produced
(via `train`/`export`) — either architecture, but not arbitrary third-party OCR models (different
projects almost always use a different CNN backbone/head shape). If a file doesn't match either
supported architecture, you'll get a clear error explaining what was found and what's supported,
instead of a crash.

```bash
# Fine-tune all layers from a previous checkpoint
cargo run --release -- train \
    --data-dir dataset \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.0001 \
    --output-dir checkpoints \
    --pretrained checkpoints/plate_ocr_final

# Fine-tune only the head (LSTM+linear, or FC layers for Conv-CTC); freeze the CNN backbone
cargo run --release -- train \
    --data-dir dataset \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.0001 \
    --output-dir checkpoints \
    --pretrained checkpoints/plate_ocr_final \
    --freeze-backbone

# Fine-tune from a PyTorch state dict
cargo run --release -- train \
    --data-dir dataset \
    --epochs 20 \
    --output-dir checkpoints \
    --pretrained pretrained_model.pt

# Fine-tune from an ONNX model
cargo run --release -- train \
    --data-dir dataset \
    --epochs 20 \
    --output-dir checkpoints \
    --pretrained pretrained_model.onnx
```

### Inference

```bash
cargo run --release -- infer \
    --model-path checkpoints/plate_ocr_final \
    --image test_plate.png
```

### Export to ONNX

Works the same way for a freshly-trained model or one that was fine-tuned (from a Burn
checkpoint, a `.pt`/`.pth` file, or an `.onnx` model):

```bash
# Step 1: Export weights from a Burn checkpoint (produces manifest.json + one .bin per tensor,
# already renamed/transposed/merged into PyTorch's own state_dict conventions)
cargo run --release -- export \
    --model-path checkpoints/plate_ocr_final \
    --output-dir weights

# Step 2: Convert to ONNX using the Python helper (if you only have the plate-ocr binary and
# not export_onnx.py, first run: plate-ocr extract-scripts)
python export_onnx.py weights --output plate_ocr.onnx
```

## Architectures

Both architectures share the exact same 4-layer CNN backbone (and therefore the same dataset,
image size and CLI/GUI flow) — they only differ in the head applied to the CNN's per-timestep
features. `plate-ocr` auto-detects which one a `--pretrained` file uses (from its tensors/graph
for `.pt`/`.onnx`, or from a small sidecar file next to its own checkpoints) and reports it (via
`log`/the GUI) before training starts; new training runs pick one explicitly with `--architecture`
(default `crnn`).

### CRNN (`crnn`, default)

```
Input [batch, 1, 32, 128] (grayscale)
  → Conv2d(1→64) + BN + ReLU + MaxPool(2×2)
  → Conv2d(64→128) + BN + ReLU + MaxPool(2×2)
  → Conv2d(128→256) + BN + ReLU
  → Conv2d(256→256) + BN + ReLU + MaxPool(2×1)
  → Reshape to [batch, 32, 1024]
  → BiLSTM(1024→256)
  → Linear(512→37)
  → LogSoftmax
  → CTC Loss / CTC Greedy Decode
```

Higher capacity, thanks to the bidirectional LSTM modeling context across the whole sequence —
generally the better choice when accuracy matters more than speed/size.

### Conv-CTC (`conv-ctc`)

```
Input [batch, 1, 32, 128] (grayscale)
  → (same 4x Conv2d + BN + ReLU + MaxPool backbone as CRNN)
  → Reshape to [batch, 32, 1024]
  → Linear(1024→256) + ReLU
  → Linear(256→37)
  → LogSoftmax
  → CTC Loss / CTC Greedy Decode
```

No recurrent layer: a small feed-forward head is applied independently to each timestep's CNN
features instead of a BiLSTM. Fewer parameters and faster to train/run — a good choice when
speed/size matter more than squeezing out the last bit of accuracy, or as a quick baseline.

Character vocabulary (both architectures): `0-9 A-Z` + CTC blank (37 classes).

## Environment Variables

```bash
# RDNA 4 compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Select specific GPU (multi-GPU)
export HIP_VISIBLE_DEVICES=0

# ROCm memory tuning
export HSA_ENABLE_SDMA=0
```

## Docker

```dockerfile
FROM rocm/dev-ubuntu-24.04:6.3
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app
COPY . .
RUN cargo build --release
```

## License

Rustrainer-OCR is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version — see [LICENSE](LICENSE) for the full text.

```
Rustrainer-OCR  Copyright (C) 2026  Mohammad Najm
This program comes with ABSOLUTELY NO WARRANTY; for details run `plate-ocr show w`.
This is free software, and you are welcome to redistribute it
under certain conditions; run `plate-ocr show c` for details.
```

This notice is printed automatically whenever the CLI is run from an interactive terminal; the
GUI wizard shows the same information (plus the relevant license sections) in its **About** box.

**Contact:** Mohammad Najm — <najm.devops@gmail.com> — <https://github.com/najmdevstudio/Rustrainer_OCR>
