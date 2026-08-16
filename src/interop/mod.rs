//! Bridges pretrained models supplied in formats other than Burn's own checkpoint format
//! (PyTorch `.pt`/`.pth` state dicts, or `.onnx` models) into a fresh [`CrnnOcr`], so they can
//! be used as a fine-tuning starting point. Burn's native checkpoints (produced by earlier runs
//! of this project) keep working exactly as before, unaffected by this module.

pub mod lstm_gates;
mod onnx_import;
mod pytorch_import;

use std::path::Path;

use burn::module::Module;
use burn::prelude::*;
use burn::record::CompactRecorder;

use crate::model::crnn::{CrnnOcr, CrnnOcrConfig};

/// Loads `path` as a pretrained/fine-tuning starting point, auto-detecting the format from its
/// file extension:
/// - `.pt` / `.pth` -> PyTorch state dict (see [`pytorch_import`])
/// - `.onnx` -> ONNX model, converted via a bundled Python helper (see [`onnx_import`])
/// - anything else -> Burn's own checkpoint format (the historical, pre-existing behavior)
///
/// `log` receives human-readable progress lines, useful for surfacing in the CLI/GUI output.
pub fn load_pretrained<B: Backend>(
    path: &str,
    config: &CrnnOcrConfig,
    device: &B::Device,
    log: impl FnMut(String),
) -> Result<CrnnOcr<B>, String> {
    let path_ref = Path::new(path);
    let extension = path_ref
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());

    match extension.as_deref() {
        Some("pt") | Some("pth") => pytorch_import::load::<B>(path_ref, config, device, log),
        Some("onnx") => onnx_import::load::<B>(path_ref, config, device, log),
        _ => {
            let model = config.init::<B>(device);
            model
                .load_file(path, &CompactRecorder::new(), device)
                .map_err(|e| format!("Failed to load Burn checkpoint '{path}': {e}"))
        }
    }
}
