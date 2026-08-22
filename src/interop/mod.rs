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

//! Bridges pretrained models supplied in formats other than Burn's own checkpoint format
//! (PyTorch `.pt`/`.pth` state dicts, or `.onnx` models) into a fresh [`OcrModel`], so they can
//! be used as a fine-tuning starting point. Burn's native checkpoints (produced by earlier runs
//! of this project) keep working exactly as before, unaffected by this module.
//!
//! Every format is auto-detected down to which of plate-ocr's supported
//! [`Architecture`](crate::model::Architecture)s it holds — `.pt`/`.onnx` by inspecting the
//! tensors/graph themselves, this project's own checkpoints via a small sidecar file (see
//! `crate::model::architecture`) — so callers never need to know which one they're getting
//! ahead of time; they can call [`OcrModel::architecture`] on the result instead.

pub mod lstm_gates;
mod onnx_import;
mod pytorch_import;

use std::path::Path;

use burn::module::Module;
use burn::prelude::*;
use burn::record::CompactRecorder;

use crate::model::conv_ctc::ConvCtcOcrConfig;
use crate::model::crnn::CrnnOcrConfig;
use crate::model::{Architecture, OcrModel};

/// Loads `path` as a pretrained/fine-tuning starting point, auto-detecting the format from its
/// file extension:
/// - `.pt` / `.pth` -> PyTorch state dict (see [`pytorch_import`])
/// - `.onnx` -> ONNX model, converted via a bundled Python helper (see [`onnx_import`])
/// - anything else -> Burn's own checkpoint format (the historical, pre-existing behavior)
///
/// `log` receives human-readable progress lines, useful for surfacing in the CLI/GUI output.
pub fn load_pretrained<B: Backend>(
    path: &str,
    device: &B::Device,
    log: impl FnMut(String),
) -> Result<OcrModel<B>, String> {
    let path_ref = Path::new(path);
    let extension = path_ref
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());

    match extension.as_deref() {
        Some("pt") | Some("pth") => pytorch_import::load::<B>(path_ref, device, log),
        Some("onnx") => onnx_import::load::<B>(path_ref, device, log),
        _ => load_burn_checkpoint::<B>(path, device),
    }
}

/// Loads one of this project's own Burn checkpoints, picking the right model config based on
/// the architecture sidecar file written alongside it by `crate::training::train` (defaulting to
/// [`Architecture::CrnnBiLstm`] for checkpoints saved before that sidecar existed).
fn load_burn_checkpoint<B: Backend>(path: &str, device: &B::Device) -> Result<OcrModel<B>, String> {
    match Architecture::read_sidecar(path) {
        Architecture::CrnnBiLstm => {
            let model = CrnnOcrConfig::new()
                .init::<B>(device)
                .load_file(path, &CompactRecorder::new(), device)
                .map_err(|e| format!("Failed to load Burn checkpoint '{path}': {e}"))?;
            Ok(OcrModel::CrnnBiLstm(model))
        }
        Architecture::ConvCtc => {
            let model = ConvCtcOcrConfig::new()
                .init::<B>(device)
                .load_file(path, &CompactRecorder::new(), device)
                .map_err(|e| format!("Failed to load Burn checkpoint '{path}': {e}"))?;
            Ok(OcrModel::ConvCtc(model))
        }
    }
}
