//! Exports trained model weights so the companion `export_onnx.py` script can reconstruct the
//! architecture in PyTorch and call `torch.onnx.export()`.
//!
//! Every tensor is written under its PyTorch `state_dict` name (see the `CrnnOcr` mirror in
//! `export_onnx.py`): the CNN backbone, batch-norm layers and final linear head are simple
//! renames/transposes, while the BiLSTM's per-gate layers are merged back into PyTorch's
//! combined `weight_ih_l0`/`weight_hh_l0`/... convention via [`crate::interop::lstm_gates`] (the
//! exact inverse of how those files are read back in for fine-tuning).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use burn::module::Module;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::DType;
use burn_store::ModuleSnapshot;
use serde::Serialize;

use crate::interop::lstm_gates::{self, GateWeights, LstmGates};
use crate::model::crnn::{CrnnOcr, CrnnOcrConfig};

#[derive(Serialize)]
struct ManifestEntry {
    name: String,
    file: String,
    shape: Vec<usize>,
}

/// A tensor plus its shape, keyed by Burn's own internal dotted path (e.g. `"bn1.gamma"` or
/// `"lstm.forward.input_gate.input_transform.weight"`).
type TensorMap = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Export trained model weights for ONNX conversion.
///
/// Writes `manifest.json` (tensor name -> file -> shape) plus one raw little-endian `.bin` file
/// per tensor into `output_dir`. A companion Python script (`export_onnx.py`) then loads these
/// directly into a PyTorch state dict and calls `torch.onnx.export()`.
pub fn export_weights<B: Backend>(model_path: &str, output_dir: &str, device: &B::Device) {
    let config = CrnnOcrConfig::new();
    let model: CrnnOcr<B> = config.init(device);
    let model = model
        .load_file(model_path, &CompactRecorder::new(), device)
        .expect("Failed to load model checkpoint");

    let hidden = config.lstm_hidden;
    let lstm_input = model.lstm_input_dim();

    let tensors = collect_tensors(&model);

    let output_path = Path::new(output_dir);
    fs::create_dir_all(output_path).expect("Failed to create output directory");

    let mut manifest = Vec::new();
    let mut write = |name: &str, shape: Vec<usize>, data: Vec<f32>| {
        let file_name = format!("{}.bin", name.replace('.', "_"));
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(output_path.join(&file_name), bytes)
            .unwrap_or_else(|e| panic!("Failed to write {file_name}: {e}"));
        manifest.push(ManifestEntry {
            name: name.to_string(),
            file: file_name,
            shape,
        });
    };

    // CNN backbone + batch-norm layers: direct name/shape match with PyTorch, aside from the
    // gamma/beta -> weight/bias rename that PyTorch's BatchNorm2d uses.
    for i in 1..=4 {
        let (shape, data) = tensors[&format!("conv{i}.weight")].clone();
        write(&format!("conv{i}.weight"), shape, data);
        let (shape, data) = tensors[&format!("conv{i}.bias")].clone();
        write(&format!("conv{i}.bias"), shape, data);

        let (shape, data) = tensors[&format!("bn{i}.gamma")].clone();
        write(&format!("bn{i}.weight"), shape, data);
        let (shape, data) = tensors[&format!("bn{i}.beta")].clone();
        write(&format!("bn{i}.bias"), shape, data);
        let (shape, data) = tensors[&format!("bn{i}.running_mean")].clone();
        write(&format!("bn{i}.running_mean"), shape, data);
        let (shape, data) = tensors[&format!("bn{i}.running_var")].clone();
        write(&format!("bn{i}.running_var"), shape, data);
    }

    // Final classifier: Burn's `Linear` stores `[d_input, d_output]`, PyTorch expects
    // `[d_output, d_input]`.
    let (shape, data) = tensors["linear.weight"].clone();
    write(
        "linear.weight",
        vec![shape[1], shape[0]],
        lstm_gates::transpose(&data, shape[0], shape[1]),
    );
    let (shape, data) = tensors["linear.bias"].clone();
    write("linear.bias", shape, data);

    // BiLSTM: merge Burn's independent per-gate layers back into PyTorch's combined tensors.
    for (direction, suffix) in [("forward", ""), ("reverse", "_reverse")] {
        let gates = collect_lstm_gates(&tensors, direction);
        let (weight_ih, weight_hh, bias_ih, bias_hh) = lstm_gates::merge_ifgo(&gates, lstm_input, hidden);
        write(&format!("lstm.weight_ih_l0{suffix}"), vec![4 * hidden, lstm_input], weight_ih);
        write(&format!("lstm.weight_hh_l0{suffix}"), vec![4 * hidden, hidden], weight_hh);
        write(&format!("lstm.bias_ih_l0{suffix}"), vec![4 * hidden], bias_ih);
        write(&format!("lstm.bias_hh_l0{suffix}"), vec![4 * hidden], bias_hh);
    }

    let manifest_json = serde_json::to_string_pretty(&manifest).expect("Failed to serialize manifest");
    fs::write(output_path.join("manifest.json"), manifest_json).expect("Failed to write manifest.json");

    log::info!("Exported {} tensor(s) to {}", manifest.len(), output_path.display());
    log::info!("To convert to ONNX, run: python export_onnx.py {}", output_dir);
}

/// Collects every leaf parameter tensor from `model`, keyed by Burn's own dotted path.
fn collect_tensors<B: Backend>(model: &CrnnOcr<B>) -> TensorMap {
    model
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| {
            let path = snapshot.full_path();
            let shape: Vec<usize> = snapshot.shape.iter().copied().collect();
            let data = snapshot
                .to_data()
                .unwrap_or_else(|e| panic!("Failed to read tensor '{path}': {e}"))
                .convert_dtype(DType::F32)
                .to_vec::<f32>()
                .unwrap_or_else(|e| panic!("Failed to convert tensor '{path}' to f32: {e:?}"));
            (path, (shape, data))
        })
        .collect()
}

/// Rebuilds an [`LstmGates`] for one direction from Burn's own dotted-path tensor map (the
/// inverse of how `crate::interop::pytorch_import` injects merged weights back in).
fn collect_lstm_gates(tensors: &TensorMap, direction: &str) -> LstmGates {
    let get = |gate: &str, transform: &str, field: &str| -> Vec<f32> {
        let key = format!("lstm.{direction}.{gate}.{transform}.{field}");
        tensors
            .get(&key)
            .unwrap_or_else(|| panic!("Missing expected tensor '{key}' while exporting"))
            .1
            .clone()
    };
    let gate = |name: &str| GateWeights {
        input_weight: get(name, "input_transform", "weight"),
        input_bias: get(name, "input_transform", "bias"),
        hidden_weight: get(name, "hidden_transform", "weight"),
        hidden_bias: get(name, "hidden_transform", "bias"),
    };

    LstmGates {
        input_gate: gate("input_gate"),
        forget_gate: gate("forget_gate"),
        cell_gate: gate("cell_gate"),
        output_gate: gate("output_gate"),
    }
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
        export_weights::<TestBackend>(
            ckpt_path.to_str().unwrap(),
            weights_dir.to_str().unwrap(),
            &device,
        );

        let manifest_str =
            fs::read_to_string(weights_dir.join("manifest.json")).expect("manifest.json missing");
        let manifest: Vec<serde_json::Value> =
            serde_json::from_str(&manifest_str).expect("invalid manifest.json");
        assert_eq!(manifest.len(), 34, "expected 34 exported tensors, got {}", manifest.len());

        let names: HashSet<String> = manifest
            .iter()
            .map(|e| e["name"].as_str().unwrap().to_string())
            .collect();
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
        let linear_entry = manifest.iter().find(|e| e["name"] == "linear.weight").unwrap();
        assert_eq!(linear_entry["shape"], serde_json::json!([37, 512]));

        // Combined LSTM tensors must have PyTorch's [4*hidden, ...] convention.
        let weight_ih = manifest.iter().find(|e| e["name"] == "lstm.weight_ih_l0").unwrap();
        assert_eq!(weight_ih["shape"], serde_json::json!([1024, 1024]));
        let bias_ih = manifest.iter().find(|e| e["name"] == "lstm.bias_ih_l0").unwrap();
        assert_eq!(bias_ih["shape"], serde_json::json!([1024]));
    }
}
