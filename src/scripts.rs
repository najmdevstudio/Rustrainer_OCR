//! Embeds the companion Python helper scripts (`import_onnx.py`, `export_onnx.py`) directly
//! into the compiled binary at build time, so a single downloaded executable is enough to
//! bootstrap the whole utility rather than requiring a separate download/clone of the repo.
//!
//! Used two ways:
//! - The `extract-scripts` CLI command writes both files to disk on request.
//! - [`crate::interop::onnx_import`] falls back to the embedded copy of `import_onnx.py`
//!   whenever it can't find one already sitting next to the executable/in the working directory
//!   (see that module for the full lookup order).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Source of `import_onnx.py` (fine-tuning from an `.onnx` model), embedded at compile time.
pub const IMPORT_ONNX_PY: &str = include_str!("../import_onnx.py");

/// Source of `export_onnx.py` (converting a `plate-ocr export`ed checkpoint into `.onnx`),
/// embedded at compile time. Not shelled out to automatically — Python isn't otherwise a
/// dependency of the `export` subcommand — so `extract-scripts` is the only way to retrieve it
/// from a standalone binary.
pub const EXPORT_ONNX_PY: &str = include_str!("../export_onnx.py");

/// Writes both companion scripts into `dir` (creating it if needed) and marks them executable
/// on Unix, since both start with a `#!/usr/bin/env python3` shebang. Returns the paths written.
pub fn write_all(dir: &Path) -> io::Result<[PathBuf; 2]> {
    fs::create_dir_all(dir)?;
    let import_path = dir.join("import_onnx.py");
    let export_path = dir.join("export_onnx.py");
    write_script(&import_path, IMPORT_ONNX_PY)?;
    write_script(&export_path, EXPORT_ONNX_PY)?;
    Ok([import_path, export_path])
}

fn write_script(path: &Path, contents: &str) -> io::Result<()> {
    fs::write(path, contents)?;
    set_executable(path)
}

#[cfg(unix)]
fn set_executable(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> io::Result<()> {
    Ok(())
}
