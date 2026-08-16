#!/bin/sh
# plate-ocr installer
#
# Downloads the prebuilt plate-ocr binary that matches this machine from the project's GitHub
# Releases page and installs it (together with the import_onnx.py/export_onnx.py helper
# scripts) onto the host. Works on Linux and macOS; Windows users should grab the .zip asset
# from the Releases page directly.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/OWNER/REPO/main/install.sh | sh
#
# Environment variables (all optional):
#   PLATE_OCR_REPO         "owner/repo" on GitHub. Defaults to the placeholder below -- edit it
#                           once this project has a real GitHub repository, or export the
#                           variable instead of editing the script.
#   PLATE_OCR_VERSION      Release tag to install, e.g. "v0.2.0" (default: latest release).
#   PLATE_OCR_FEATURE      Backend variant: "cpu" (default, works everywhere) or "vulkan"
#                           (Linux only, AMD/other GPU acceleration without ROCm). The default
#                           "rocm" backend isn't distributed as a prebuilt binary -- it needs the
#                           ROCm/HIP SDK and an AMD GPU, see README.md to build it from source.
#   PLATE_OCR_INSTALL_DIR  Where to place the binary (default: "$HOME/.local/bin").
#   PLATE_OCR_NO_DESKTOP   Set to "1" to skip creating a Linux application-menu entry.
#
# This script does not verify a cryptographic signature or checksum of the downloaded archive;
# review the release assets on GitHub yourself if that matters for your use case.

set -eu

REPO="${PLATE_OCR_REPO:-OWNER/REPO}"
VERSION="${PLATE_OCR_VERSION:-latest}"
FEATURE="${PLATE_OCR_FEATURE:-cpu}"
INSTALL_DIR="${PLATE_OCR_INSTALL_DIR:-$HOME/.local/bin}"

info() { printf 'plate-ocr installer: %s\n' "$1"; }
die() {
    printf 'plate-ocr installer: error: %s\n' "$1" >&2
    exit 1
}
command_exists() { command -v "$1" >/dev/null 2>&1; }

if [ "$REPO" = "OWNER/REPO" ]; then
    die "PLATE_OCR_REPO is not set and install.sh still has its placeholder. Set \
PLATE_OCR_REPO=owner/repo (or edit the default in this script) and try again."
fi

case "$(uname -s)" in
    Linux) OS=linux ;;
    Darwin) OS=darwin ;;
    *) die "Unsupported OS: $(uname -s). Download a release manually from \
https://github.com/$REPO/releases, or build from source with 'cargo build --release'." ;;
esac

case "$(uname -m)" in
    x86_64 | amd64) ARCH=x86_64 ;;
    arm64 | aarch64) ARCH=arm64 ;;
    *) die "Unsupported architecture: $(uname -m)." ;;
esac

case "$FEATURE" in
    cpu | vulkan) ;;
    *) die "Unknown PLATE_OCR_FEATURE '$FEATURE'; expected 'cpu' or 'vulkan'." ;;
esac

EXT=tar.gz
case "$OS" in
    linux)
        [ "$ARCH" = "x86_64" ] || die "No prebuilt Linux release for architecture '$ARCH' yet; \
build from source instead: cargo build --release --no-default-features --features $FEATURE"
        NAME="linux-x86_64-$FEATURE"
        ;;
    darwin)
        [ "$FEATURE" = "cpu" ] || die "The '$FEATURE' backend is Linux-only; use \
PLATE_OCR_FEATURE=cpu on macOS."
        NAME="macos-$ARCH-cpu"
        ;;
esac

ASSET="plate-ocr-$NAME.$EXT"
if [ "$VERSION" = "latest" ]; then
    URL="https://github.com/$REPO/releases/latest/download/$ASSET"
else
    URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM

info "Downloading $ASSET ($VERSION) for $NAME..."
if command_exists curl; then
    curl -fsSL "$URL" -o "$TMP_DIR/$ASSET" || die "Download failed: $URL"
elif command_exists wget; then
    wget -q "$URL" -O "$TMP_DIR/$ASSET" || die "Download failed: $URL"
else
    die "Need either curl or wget to download the release archive."
fi

tar -xzf "$TMP_DIR/$ASSET" -C "$TMP_DIR" || die "Failed to extract $ASSET"
EXTRACTED_DIR="$TMP_DIR/plate-ocr-$NAME"
[ -d "$EXTRACTED_DIR" ] || die "Unexpected archive layout in $ASSET"

mkdir -p "$INSTALL_DIR"
cp "$EXTRACTED_DIR/plate-ocr" "$INSTALL_DIR/plate-ocr"
chmod +x "$INSTALL_DIR/plate-ocr"
# Sibling helper scripts: plate-ocr's own lookup for import_onnx.py already checks next to its
# own executable first, so keeping them side by side here just works. Not fatal if missing --
# the binary can also fall back to the copy embedded in it via 'plate-ocr extract-scripts'.
cp "$EXTRACTED_DIR/import_onnx.py" "$EXTRACTED_DIR/export_onnx.py" "$INSTALL_DIR/" 2>/dev/null || true
chmod +x "$INSTALL_DIR/import_onnx.py" "$INSTALL_DIR/export_onnx.py" 2>/dev/null || true

info "Installed plate-ocr to $INSTALL_DIR/plate-ocr"

case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *)
        info "NOTE: $INSTALL_DIR is not on your PATH. Add this to your shell profile:"
        info "  export PATH=\"$INSTALL_DIR:\$PATH\""
        ;;
esac

if [ "$OS" = "linux" ] && [ "${PLATE_OCR_NO_DESKTOP:-0}" != "1" ]; then
    DESKTOP_DIR="$HOME/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"
    cat >"$DESKTOP_DIR/plate-ocr.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Plate OCR
Comment=Train/fine-tune and run the plate-ocr CRNN OCR model
Exec=$INSTALL_DIR/plate-ocr gui
Terminal=false
Categories=Development;Utility;
EOF
    command_exists update-desktop-database && update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
    info "Added a Plate OCR entry to your application menu."
fi

info "Done! Run 'plate-ocr' to launch the GUI wizard, or 'plate-ocr --help' for CLI commands."
