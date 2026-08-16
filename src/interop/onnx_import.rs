//! Fine-tuning from an `.onnx` model file: bridges to a small Python helper script
//! (`import_onnx.py`, built on the `onnx`/`numpy`/`torch` packages already required for the
//! `export` command — see README.md) that converts the ONNX graph's weights into a plain
//! PyTorch state dict. That state dict is then loaded through the exact same code path as a
//! native `.pt` file ([`pytorch_import`]).
//!
//! Hand-rolling a full ONNX protobuf/graph parser (including operator-attribute handling and
//! the ONNX `LSTM` operator's own gate-order convention) in Rust would be a large amount of
//! fragile, hard-to-validate code; delegating the graph walk to Python lets us reuse the
//! mature, well-tested `onnx` package instead.

use std::path::{Path, PathBuf};
use std::process::Command;

use burn::prelude::*;

use crate::model::crnn::{CrnnOcr, CrnnOcrConfig};

use super::pytorch_import;

/// Locates the bundled `import_onnx.py` helper, checking next to the running executable first
/// (for packaged installs), then the current working directory, then the crate root (so it
/// works with plain `cargo run`). If none of those exist — e.g. a lone `plate-ocr` binary
/// downloaded from a GitHub release with no sibling files — falls back to the copy embedded
/// into the binary at compile time (see [`crate::scripts`]), writing it out to a temp file so it
/// can still be handed to the `python3` subprocess below.
fn locate_helper_script() -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("import_onnx.py"));
    }
    candidates.push(PathBuf::from("import_onnx.py"));
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("import_onnx.py"));

    if let Some(found) = candidates.into_iter().find(|p| p.is_file()) {
        return Ok(found);
    }

    let extracted = std::env::temp_dir().join("plate-ocr-import_onnx.py");
    std::fs::write(&extracted, crate::scripts::IMPORT_ONNX_PY).map_err(|e| {
        format!(
            "Could not find import_onnx.py next to the executable or in the project directory, \
             and failed to extract the copy embedded in the binary to '{}': {e}",
            extracted.display()
        )
    })?;
    Ok(extracted)
}

/// Converts `path` (an `.onnx` file) to a temporary PyTorch state dict via the bundled Python
/// helper, then loads it exactly like a native `.pt` file.
pub fn load<B: Backend>(
    path: &Path,
    config: &CrnnOcrConfig,
    device: &B::Device,
    mut log: impl FnMut(String),
) -> Result<CrnnOcr<B>, String> {
    let script = locate_helper_script()?;
    let tmp_pt = std::env::temp_dir().join(format!(
        "plate_ocr_onnx_import_{}_{}.pt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    // Allow overriding the interpreter (e.g. a virtualenv/pyenv/conda install that has
    // 'onnx'/'numpy'/'torch') for setups where the default `python3` on PATH doesn't.
    let python = std::env::var("PLATE_OCR_PYTHON").unwrap_or_else(|_| "python3".to_string());

    log(format!(
        "Converting '{}' to a PyTorch state dict via {python} {}...",
        path.display(),
        script.display()
    ));

    let run = Command::new(&python).arg(&script).arg(path).arg(&tmp_pt).output();

    let output = match run {
        Ok(output) => output,
        Err(e) => {
            return Err(format!(
                "Failed to launch '{python} {}': {e}. Fine-tuning from an .onnx file requires Python 3 \
                 with the 'onnx', 'numpy' and 'torch' packages installed (see README.md). If they're \
                 installed under a different interpreter, point to it with the PLATE_OCR_PYTHON \
                 environment variable.",
                script.display()
            ));
        }
    };

    for line in String::from_utf8_lossy(&output.stdout).lines() {
        log(format!("[import_onnx.py] {line}"));
    }

    if !output.status.success() {
        return Err(format!(
            "import_onnx.py failed to convert '{}' (exit code {:?}):\n{}",
            path.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let result = pytorch_import::load::<B>(&tmp_pt, config, device, &mut log);
    let _ = std::fs::remove_file(&tmp_pt);
    result
}
