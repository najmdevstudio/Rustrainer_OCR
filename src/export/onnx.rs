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

//! Exports trained model weights so the companion `export_onnx.py` script can reconstruct the
//! architecture in PyTorch and call `torch.onnx.export()`.
//!
//! Every tensor is written under its PyTorch `state_dict` name (see the PyTorch mirrors in
//! `export_onnx.py`): the CNN backbone, batch-norm layers and FC/linear head(s) are simple
//! renames/transposes, while `CrnnOcr`'s BiLSTM per-gate layers are merged back into PyTorch's
//! combined `weight_ih_l0`/`weight_hh_l0`/... convention via [`crate::interop::lstm_gates`] (the
//! exact inverse of how those files are read back in for fine-tuning). Which architecture
//! `model_path` is gets auto-detected via its checkpoint sidecar file (see
//! [`crate::model::architecture`]) and recorded in the written manifest, so `export_onnx.py`
//! knows which PyTorch class to reconstruct without guessing.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use burn::module::Module;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::DType;
use burn_store::{ModuleSnapshot, TensorSnapshot};
use serde::Serialize;

use crate::interop::lstm_gates::{self, GateWeights, LstmGates};
use crate::model::conv_ctc::{ConvCtcOcr, ConvCtcOcrConfig};
use crate::model::crnn::{CrnnOcr, CrnnOcrConfig};
use crate::model::Architecture;

#[derive(Serialize)]
struct ManifestEntry {
    name: String,
    file: String,
    shape: Vec<usize>,
}

/// Top-level `manifest.json` shape: which architecture these tensors belong to (so
/// `export_onnx.py` knows which PyTorch class to instantiate) plus the tensors themselves.
#[derive(Serialize)]
struct Manifest {
    architecture: &'static str,
    tensors: Vec<ManifestEntry>,
}

/// A tensor plus its shape, keyed by Burn's own internal dotted path (e.g. `"bn1.gamma"` or
/// `"lstm.forward.input_gate.input_transform.weight"`).
type TensorMap = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Export trained model weights for ONNX conversion, auto-detecting `model_path`'s architecture
/// from its checkpoint sidecar file (defaulting to [`Architecture::CrnnBiLstm`] for checkpoints
/// saved before that sidecar existed).
///
/// Writes `manifest.json` (`{"architecture": ..., "tensors": [{name, file, shape}, ...]}`) plus
/// one raw little-endian `.bin` file per tensor into `output_dir`. A companion Python script
/// (`export_onnx.py`) then loads these directly into a PyTorch state dict and calls
/// `torch.onnx.export()`.
pub fn export_weights<B: Backend>(model_path: &str, output_dir: &str, device: &B::Device) -> Result<(), String> {
    let architecture = Architecture::read_sidecar(model_path);
    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path)
        .map_err(|e| format!("Failed to create output directory '{output_dir}': {e}"))?;

    let tensors = match architecture {
        Architecture::CrnnBiLstm => export_crnn::<B>(model_path, output_path, device)?,
        Architecture::ConvCtc => export_conv_ctc::<B>(model_path, output_path, device)?,
    };
    let tensor_count = tensors.len();

    let manifest = Manifest {
        architecture: architecture.id(),
        tensors,
    };
    let manifest_json =
        serde_json::to_string_pretty(&manifest).map_err(|e| format!("Failed to serialize manifest: {e}"))?;
    fs::write(output_path.join("manifest.json"), manifest_json)
        .map_err(|e| format!("Failed to write manifest.json: {e}"))?;

    log::info!(
        "Exported {tensor_count} tensor(s) ({architecture}) to {}",
        output_path.display()
    );
    log::info!("To convert to ONNX, run: python export_onnx.py {output_dir}");
    Ok(())
}

/// Loads a [`CrnnOcr`] checkpoint and writes its backbone, final linear layer and (gate-merged)
/// BiLSTM tensors.
fn export_crnn<B: Backend>(
    model_path: &str,
    output_path: &Path,
    device: &B::Device,
) -> Result<Vec<ManifestEntry>, String> {
    let config = CrnnOcrConfig::new();
    let model: CrnnOcr<B> = config.init(device);
    let model = model
        .load_file(model_path, &CompactRecorder::new(), device)
        .map_err(|e| format!("Failed to load model checkpoint '{model_path}': {e}"))?;

    let hidden = config.lstm_hidden;
    let lstm_input = model.lstm_input_dim();
    let tensors = tensor_map_from_snapshots(model.collect(None, None, false))?;

    let mut manifest = Vec::new();
    write_backbone_tensors(output_path, &mut manifest, &tensors)?;

    // Final classifier: Burn's `Linear` stores `[d_input, d_output]`, PyTorch expects
    // `[d_output, d_input]`.
    let (shape, data) = tensors["linear.weight"].clone();
    write_tensor(
        output_path,
        &mut manifest,
        "linear.weight",
        vec![shape[1], shape[0]],
        lstm_gates::transpose(&data, shape[0], shape[1]),
    )?;
    let (shape, data) = tensors["linear.bias"].clone();
    write_tensor(output_path, &mut manifest, "linear.bias", shape, data)?;

    // BiLSTM: merge Burn's independent per-gate layers back into PyTorch's combined tensors.
    for (direction, suffix) in [("forward", ""), ("reverse", "_reverse")] {
        let gates = collect_lstm_gates(&tensors, direction)?;
        let (weight_ih, weight_hh, bias_ih, bias_hh) = lstm_gates::merge_ifgo(&gates, lstm_input, hidden);
        write_tensor(
            output_path,
            &mut manifest,
            &format!("lstm.weight_ih_l0{suffix}"),
            vec![4 * hidden, lstm_input],
            weight_ih,
        )?;
        write_tensor(
            output_path,
            &mut manifest,
            &format!("lstm.weight_hh_l0{suffix}"),
            vec![4 * hidden, hidden],
            weight_hh,
        )?;
        write_tensor(
            output_path,
            &mut manifest,
            &format!("lstm.bias_ih_l0{suffix}"),
            vec![4 * hidden],
            bias_ih,
        )?;
        write_tensor(
            output_path,
            &mut manifest,
            &format!("lstm.bias_hh_l0{suffix}"),
            vec![4 * hidden],
            bias_hh,
        )?;
    }

    Ok(manifest)
}

/// Loads a [`ConvCtcOcr`] checkpoint and writes its backbone and 2-layer FC head. Unlike the
/// BiLSTM above, both FC layers are plain `Linear`s matched/transposed directly, with no gate
/// merging required.
fn export_conv_ctc<B: Backend>(
    model_path: &str,
    output_path: &Path,
    device: &B::Device,
) -> Result<Vec<ManifestEntry>, String> {
    let model: ConvCtcOcr<B> = ConvCtcOcrConfig::new().init(device);
    let model = model
        .load_file(model_path, &CompactRecorder::new(), device)
        .map_err(|e| format!("Failed to load model checkpoint '{model_path}': {e}"))?;

    let tensors = tensor_map_from_snapshots(model.collect(None, None, false))?;

    let mut manifest = Vec::new();
    write_backbone_tensors(output_path, &mut manifest, &tensors)?;

    // Both FC layers: Burn's `Linear` stores `[d_input, d_output]`, PyTorch expects
    // `[d_output, d_input]`.
    for fc in ["fc1", "fc2"] {
        let (shape, data) = tensors[&format!("{fc}.weight")].clone();
        write_tensor(
            output_path,
            &mut manifest,
            &format!("{fc}.weight"),
            vec![shape[1], shape[0]],
            lstm_gates::transpose(&data, shape[0], shape[1]),
        )?;
        let (shape, data) = tensors[&format!("{fc}.bias")].clone();
        write_tensor(output_path, &mut manifest, &format!("{fc}.bias"), shape, data)?;
    }

    Ok(manifest)
}

/// Writes the 4x (Conv2d + BatchNorm) backbone tensors shared by both architectures. PyTorch's
/// `BatchNorm2d` uses `weight`/`bias`, Burn's uses `gamma`/`beta`; everything else is a direct
/// name/shape match.
fn write_backbone_tensors(
    output_path: &Path,
    manifest: &mut Vec<ManifestEntry>,
    tensors: &TensorMap,
) -> Result<(), String> {
    for i in 1..=4 {
        let (shape, data) = tensors[&format!("conv{i}.weight")].clone();
        write_tensor(output_path, manifest, &format!("conv{i}.weight"), shape, data)?;
        let (shape, data) = tensors[&format!("conv{i}.bias")].clone();
        write_tensor(output_path, manifest, &format!("conv{i}.bias"), shape, data)?;

        let (shape, data) = tensors[&format!("bn{i}.gamma")].clone();
        write_tensor(output_path, manifest, &format!("bn{i}.weight"), shape, data)?;
        let (shape, data) = tensors[&format!("bn{i}.beta")].clone();
        write_tensor(output_path, manifest, &format!("bn{i}.bias"), shape, data)?;
        let (shape, data) = tensors[&format!("bn{i}.running_mean")].clone();
        write_tensor(output_path, manifest, &format!("bn{i}.running_mean"), shape, data)?;
        let (shape, data) = tensors[&format!("bn{i}.running_var")].clone();
        write_tensor(output_path, manifest, &format!("bn{i}.running_var"), shape, data)?;
    }
    Ok(())
}

/// Writes one tensor's raw little-endian `.bin` file and appends its [`ManifestEntry`].
fn write_tensor(
    output_path: &Path,
    manifest: &mut Vec<ManifestEntry>,
    name: &str,
    shape: Vec<usize>,
    data: Vec<f32>,
) -> Result<(), String> {
    let file_name = format!("{}.bin", name.replace('.', "_"));
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(output_path.join(&file_name), bytes).map_err(|e| format!("Failed to write '{file_name}': {e}"))?;
    manifest.push(ManifestEntry {
        name: name.to_string(),
        file: file_name,
        shape,
    });
    Ok(())
}

/// Converts a model's collected leaf-tensor snapshots into a [`TensorMap`] keyed by Burn's own
/// dotted path.
fn tensor_map_from_snapshots(snapshots: Vec<TensorSnapshot>) -> Result<TensorMap, String> {
    snapshots
        .into_iter()
        .map(|snapshot| {
            let path = snapshot.full_path();
            let shape: Vec<usize> = snapshot.shape.iter().copied().collect();
            let data = snapshot
                .to_data()
                .map_err(|e| format!("Failed to read tensor '{path}': {e}"))?
                .convert_dtype(DType::F32)
                .to_vec::<f32>()
                .map_err(|e| format!("Failed to convert tensor '{path}' to f32: {e:?}"))?;
            Ok((path, (shape, data)))
        })
        .collect()
}

/// Rebuilds an [`LstmGates`] for one direction from Burn's own dotted-path tensor map (the
/// inverse of how `crate::interop::pytorch_import` injects merged weights back in).
fn collect_lstm_gates(tensors: &TensorMap, direction: &str) -> Result<LstmGates, String> {
    let get = |gate: &str, transform: &str, field: &str| -> Result<Vec<f32>, String> {
        let key = format!("lstm.{direction}.{gate}.{transform}.{field}");
        tensors
            .get(&key)
            .map(|(_, data)| data.clone())
            .ok_or_else(|| format!("Missing expected tensor '{key}' while exporting"))
    };
    let gate = |name: &str| -> Result<GateWeights, String> {
        Ok(GateWeights {
            input_weight: get(name, "input_transform", "weight")?,
            input_bias: get(name, "input_transform", "bias")?,
            hidden_weight: get(name, "hidden_transform", "weight")?,
            hidden_bias: get(name, "hidden_transform", "bias")?,
        })
    };

    Ok(LstmGates {
        input_gate: gate("input_gate")?,
        forget_gate: gate("forget_gate")?,
        cell_gate: gate("cell_gate")?,
        output_gate: gate("output_gate")?,
    })
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use std::collections::HashSet;

    type TestBackend = burn::backend::NdArray;

    #[test]
    fn export_writes_manifest_with_all_expected_tensors() {
        let device = Default::default();
        let base = std::env::temp_dir().join("plate_ocr_export_test");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).expect("failed to create temp dir");

        let config = CrnnOcrConfig::new();
        let model: CrnnOcr<TestBackend> = config.init(&device);
        let ckpt_path = base.join("ckpt");
        model
            .save_file(&ckpt_path, &CompactRecorder::new())
            .expect("failed to save checkpoint");

        let weights_dir = base.join("weights");
        export_weights::<TestBackend>(ckpt_path.to_str().unwrap(), weights_dir.to_str().unwrap(), &device)
            .expect("export_weights failed");

        let manifest_str =
            fs::read_to_string(weights_dir.join("manifest.json")).expect("manifest.json missing");
        let manifest: serde_json::Value = serde_json::from_str(&manifest_str).expect("invalid manifest.json");
        assert_eq!(manifest["architecture"], "crnn_bilstm");
        let tensors = manifest["tensors"].as_array().expect("tensors should be an array");
        assert_eq!(tensors.len(), 34, "expected 34 exported tensors, got {}", tensors.len());

        let names: HashSet<String> = tensors.iter().map(|e| e["name"].as_str().unwrap().to_string()).collect();
        for expected in [
            "conv1.weight",
            "conv1.bias",
            "bn1.weight",
            "bn1.bias",
            "bn1.running_mean",
            "bn1.running_var",
            "linear.weight",
            "linear.bias",
            "lstm.weight_ih_l0",
            "lstm.weight_hh_l0",
            "lstm.bias_ih_l0",
            "lstm.bias_hh_l0",
            "lstm.weight_ih_l0_reverse",
            "lstm.weight_hh_l0_reverse",
            "lstm.bias_ih_l0_reverse",
            "lstm.bias_hh_l0_reverse",
        ] {
            assert!(names.contains(expected), "manifest missing '{expected}'");
        }

        // linear.weight must come out transposed to PyTorch's [out, in] convention.
        let linear_entry = tensors.iter().find(|e| e["name"] == "linear.weight").unwrap();
        assert_eq!(linear_entry["shape"], serde_json::json!([37, 512]));

        // Combined LSTM tensors must have PyTorch's [4*hidden, ...] convention.
        let weight_ih = tensors.iter().find(|e| e["name"] == "lstm.weight_ih_l0").unwrap();
        assert_eq!(weight_ih["shape"], serde_json::json!([1024, 1024]));
        let bias_ih = tensors.iter().find(|e| e["name"] == "lstm.bias_ih_l0").unwrap();
        assert_eq!(bias_ih["shape"], serde_json::json!([1024]));
    }

    #[test]
    fn export_conv_ctc_writes_manifest_with_all_expected_tensors() {
        let device = Default::default();
        let base = std::env::temp_dir().join("plate_ocr_export_conv_ctc_test");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).expect("failed to create temp dir");

        let config = ConvCtcOcrConfig::new();
        let model: ConvCtcOcr<TestBackend> = config.init(&device);
        let ckpt_path = base.join("ckpt");
        model
            .save_file(&ckpt_path, &CompactRecorder::new())
            .expect("failed to save checkpoint");
        Architecture::ConvCtc
            .write_sidecar(ckpt_path.to_str().unwrap())
            .expect("failed to write architecture sidecar");

        let weights_dir = base.join("weights");
        export_weights::<TestBackend>(ckpt_path.to_str().unwrap(), weights_dir.to_str().unwrap(), &device)
            .expect("export_weights failed");

        let manifest_str =
            fs::read_to_string(weights_dir.join("manifest.json")).expect("manifest.json missing");
        let manifest: serde_json::Value = serde_json::from_str(&manifest_str).expect("invalid manifest.json");
        assert_eq!(manifest["architecture"], "conv_ctc");
        let tensors = manifest["tensors"].as_array().expect("tensors should be an array");
        assert_eq!(tensors.len(), 28, "expected 28 exported tensors, got {}", tensors.len());

        let names: HashSet<String> = tensors.iter().map(|e| e["name"].as_str().unwrap().to_string()).collect();
        for expected in ["conv1.weight", "bn1.running_var", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"] {
            assert!(names.contains(expected), "manifest missing '{expected}'");
        }
        assert!(!names.iter().any(|n| n.starts_with("lstm.")), "Conv-CTC export must not contain LSTM tensors");

        // fc1.weight must come out transposed to PyTorch's [out, in] convention.
        let fc1_entry = tensors.iter().find(|e| e["name"] == "fc1.weight").unwrap();
        assert_eq!(fc1_entry["shape"], serde_json::json!([256, 1024]));
        let fc2_entry = tensors.iter().find(|e| e["name"] == "fc2.weight").unwrap();
        assert_eq!(fc2_entry["shape"], serde_json::json!([37, 256]));
    }
}
