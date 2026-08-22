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

//! Ensures the Python interpreter used for the ONNX interop bridge (see
//! `crate::interop::onnx_import`) has the third-party packages it needs, installing whichever
//! are missing automatically instead of letting the bundled `import_onnx.py` helper fail
//! part-way through with a raw `ModuleNotFoundError` traceback.
//!
//! This only runs right before an `.onnx` file is about to be converted — training/fine-tuning
//! from this project's own checkpoints or a `.pt`/`.pth` file never touches Python at all, so
//! there's no reason to pay this cost (or require internet access) unless it's actually needed.

use std::process::{Command, Stdio};

/// (Python import name, PyPI package name) for every package `import_onnx.py`/`export_onnx.py`
/// need. They happen to be identical for all three, but are kept separate since that's not
/// guaranteed for packages in general.
const REQUIRED_PACKAGES: &[(&str, &str)] = &[("numpy", "numpy"), ("onnx", "onnx"), ("torch", "torch")];

/// Resolves which Python interpreter to use: the `PLATE_OCR_PYTHON` environment variable if
/// set (for a virtualenv/pyenv/conda install), otherwise plain `python3` on `PATH`.
pub fn python_executable() -> String {
    std::env::var("PLATE_OCR_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

/// Checks for [`REQUIRED_PACKAGES`] and installs (via `pip`) whichever `packages` are missing,
/// streaming human-readable progress through `log`. A no-op (besides the check itself) if
/// everything is already importable by `python`.
pub fn ensure_dependencies(python: &str, mut log: impl FnMut(String)) -> Result<(), String> {
    ensure_packages(python, REQUIRED_PACKAGES, &mut log)
}

fn ensure_packages(
    python: &str,
    packages: &[(&'static str, &'static str)],
    log: &mut impl FnMut(String),
) -> Result<(), String> {
    log(format!("Checking Python environment ('{python}') for required packages..."));
    let missing = missing_packages(python, packages)?;
    if missing.is_empty() {
        log("All required Python packages are already installed.".to_string());
        return Ok(());
    }

    let pip_names: Vec<&str> = missing.iter().map(|(_, pip_name)| *pip_name).collect();
    log(format!(
        "Missing Python package(s): {}. Attempting automatic installation via pip (this can take \
         a while, especially for torch)...",
        pip_names.join(", ")
    ));

    let output = Command::new(python)
        .args(["-m", "pip", "install", "--disable-pip-version-check"])
        .args(&pip_names)
        .output()
        .map_err(|e| {
            format!(
                "Could not automatically install {} — failed to launch '{python} -m pip install \
                 {}': {e}. Install manually with `pip install {}`, or point the PLATE_OCR_PYTHON \
                 environment variable at an interpreter that already has them.",
                pip_names.join(", "),
                pip_names.join(" "),
                pip_names.join(" ")
            )
        })?;

    for line in String::from_utf8_lossy(&output.stdout).lines() {
        log(format!("[pip] {line}"));
    }
    for line in String::from_utf8_lossy(&output.stderr).lines() {
        log(format!("[pip] {line}"));
    }

    if !output.status.success() {
        return Err(format!(
            "Automatic installation of {} failed (pip exit code {:?}). Install manually with \
             `{python} -m pip install {}`, or point the PLATE_OCR_PYTHON environment variable at \
             an interpreter that already has them.",
            pip_names.join(", "),
            output.status.code(),
            pip_names.join(" ")
        ));
    }

    // Re-check: pip can report success while still leaving a package unimportable by this exact
    // interpreter (e.g. it installed into a different environment than the one being checked).
    let still_missing = missing_packages(python, packages)?;
    if !still_missing.is_empty() {
        let still_missing_pip: Vec<&str> = still_missing.iter().map(|(_, pip_name)| *pip_name).collect();
        return Err(format!(
            "pip reported success, but {} still can't be imported by '{python}'. This can happen \
             when pip installs into a different environment than the interpreter being used. Try \
             `{python} -m pip install {}` explicitly, or point PLATE_OCR_PYTHON elsewhere.",
            still_missing_pip.join(", "),
            still_missing_pip.join(" ")
        ));
    }

    log(format!("Successfully installed: {}", pip_names.join(", ")));
    Ok(())
}

/// Returns the subset of `packages` not currently importable by `python`.
///
/// The probe's own stdout/stderr (e.g. `ModuleNotFoundError` tracebacks for whichever packages
/// turn out to be missing — an expected, routine outcome here, not a real error) are
/// deliberately discarded rather than inherited from this process: they'd otherwise dump
/// confusing raw Python tracebacks straight to the terminal/GUI log for something this function
/// already reports cleanly via its return value.
fn missing_packages(
    python: &str,
    packages: &[(&'static str, &'static str)],
) -> Result<Vec<(&'static str, &'static str)>, String> {
    let mut missing = Vec::new();
    for &(import_name, pip_name) in packages {
        let status = Command::new(python)
            .args(["-c", &format!("import {import_name}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        match status {
            Ok(status) if status.success() => {}
            Ok(_) => missing.push((import_name, pip_name)),
            Err(e) => {
                return Err(format!(
                    "Could not run '{python}' ({e}). Fine-tuning from an .onnx file requires \
                     Python 3 on PATH (or point the PLATE_OCR_PYTHON environment variable at one)."
                ));
            }
        }
    }
    Ok(missing)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uses the standard library's `sys` module (always importable, no network/pip involved) to
    /// verify the "already satisfied" path without touching real third-party packages.
    #[test]
    fn missing_packages_is_empty_for_stdlib_module() {
        let missing = missing_packages("python3", &[("sys", "sys")]).expect("failed to run python3");
        assert!(missing.is_empty(), "expected 'sys' to be importable, got missing: {missing:?}");
    }

    /// A package name that will never exist should always be reported missing — this exercises
    /// the detection logic offline/deterministically, without ever invoking pip.
    #[test]
    fn missing_packages_detects_a_nonexistent_package() {
        let fake = ("definitely_not_a_real_package_xyz123", "definitely-not-a-real-package-xyz123");
        let missing = missing_packages("python3", &[fake]).expect("failed to run python3");
        assert_eq!(missing, vec![fake]);
    }

    #[test]
    fn python_executable_respects_env_override() {
        // SAFETY: this test is not run concurrently with other code that reads this specific
        // environment variable (it's exclusive to plate-ocr, not a general one like PATH).
        unsafe {
            std::env::set_var("PLATE_OCR_PYTHON", "/custom/python3");
        }
        assert_eq!(python_executable(), "/custom/python3");
        unsafe {
            std::env::remove_var("PLATE_OCR_PYTHON");
        }
    }
}
