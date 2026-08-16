# Rustrainer-OCR

A pure-Rust OCR training and inference utility for license plate recognition, built with [Burn](https://burn.dev) deep learning framework. Trains a CRNN (CNN + BiLSTM + CTC) model and supports ONNX export.

## Features

- **Pure Rust** training pipeline using Burn 0.21
- **AMD ROCm** GPU acceleration (native CubeCL backend)
- **Vulkan** fallback for AMD GPUs without ROCm
- **CPU** fallback via ndarray
- **ONNX export** via companion Python script
- **Fine-tune from PyTorch or ONNX models**, in addition to this project's own checkpoints — the format is auto-detected from the file extension
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

```bash
pip install torch numpy onnx
```

> If these packages live under a different interpreter than the `python3` on your `PATH` (a
> virtualenv, pyenv, conda, etc.), point the fine-tuning-from-`.onnx` bridge at it with
> `PLATE_OCR_PYTHON=/path/to/python3`.

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
2. **Choose the parameters** — dataset base directory and the rest of the training parameters, prefilled with sensible defaults for the flow you picked (editable, with native folder/file pickers).
3. **Watch it train** — a terminal-style output pane, a live loss graph, and a progress bar showing overall completion.
4. **See the result** — success or failure, closed with a single "OK".

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
```

### Fine-tune a Pretrained Model

`--pretrained` accepts three formats, auto-detected from the file extension:

| Extension            | Source                                                              |
|----------------------|----------------------------------------------------------------------|
| *(none)*, `.mpk`     | This project's own Burn checkpoint (e.g. `checkpoints/plate_ocr_final`) |
| `.pt`, `.pth`        | A PyTorch state dict (`torch.save(model.state_dict(), ...)`)         |
| `.onnx`              | An ONNX model (converted on the fly via the bundled `import_onnx.py`, requires Python + torch/numpy/onnx) |

For `.pt`/`.pth`/`.onnx` sources, the convolutional backbone, batch-norm layers and final linear
layer are matched by parameter name (see the `CrnnOcr` mirror in `export_onnx.py`), and the
BiLSTM's combined `weight_ih_l0`/`weight_hh_l0`/... tensors are automatically split into this
project's per-gate layout. If a `.onnx` graph has batch-norm folded into the convolutions (the
common case for models exported in eval mode), those layers are left at their identity
initialization and the already-fused convolution weights take over the same computation.

```bash
# Fine-tune all layers from a previous checkpoint
cargo run --release -- train \
    --data-dir dataset \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.0001 \
    --output-dir checkpoints \
    --pretrained checkpoints/plate_ocr_final

# Fine-tune only LSTM + linear head (freeze CNN backbone)
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

## Architecture

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

Character vocabulary: `0-9 A-Z` + CTC blank (37 classes).

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

GPLv3 — see [LICENSE](LICENSE).
