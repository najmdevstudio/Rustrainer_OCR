// Rustrainer-OCR A GUI Utility to train/fine tune OCR Models written in Rust.
// Copyright (C) 2026 Mohammad Najm
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// Contact: Mohammad Najm <najm.devops@gmail.com>
// https://github.com/najmdevstudio/Rustrainer_OCR

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

use crate::model::OcrModel;

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
/// helper, then loads it exactly like a native `.pt` file (auto-detecting its architecture).
pub fn load<B: Backend>(
    path: &Path,
    device: &B::Device,
    mut log: impl FnMut(String),
) -> Result<OcrModel<B>, String> {
    let script = locate_helper_script()?;
    let tmp_pt = std::env::temp_dir().join(format!(
        "plate_ocr_onnx_import_{}_{}.pt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));

    let python = crate::pydeps::python_executable();
    // Make sure numpy/onnx/torch are actually available before shelling out below — installs
    // whichever are missing instead of letting import_onnx.py die on its first `import` line.
    crate::pydeps::ensure_dependencies(&python, &mut log)?;

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
        let stderr = String::from_utf8_lossy(&output.stderr);
        for line in stderr.lines() {
            log(format!("[import_onnx.py] {line}"));
        }
        return Err(format!(
            "import_onnx.py could not convert '{}': {}",
            path.display(),
            summarize_python_error(&stderr)
        ));
    }

    let result = pytorch_import::load::<B>(&tmp_pt, device, &mut log);
    let _ = std::fs::remove_file(&tmp_pt);
    result
}

/// Reduces a Python traceback down to its final exception line (e.g. `ValueError: ...` or
/// `ModuleNotFoundError: No module named 'numpy'`) for a concise, front-and-center error message
/// — the full traceback is still available above it since every stderr line is also sent to
/// `log` before this is called.
fn summarize_python_error(stderr: &str) -> String {
    stderr
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .next_back()
        .unwrap_or("(no error output captured; run with the terminal log open for details)")
        .to_string()
}
