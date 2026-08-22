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

//! Loads a PyTorch (`.pt` / `.pth`) state dict as a pretrained/fine-tuning starting point,
//! auto-detecting which of plate-ocr's two architectures it holds (see
//! [`crate::model::architecture`]) from the tensor names it contains — a `lstm.weight_ih_l0`
//! tensor means [`CrnnOcr`], its absence means [`ConvCtcOcr`].
//!
//! The convolutional backbone, batch-norm layers and FC/linear head(s) are matched by name
//! automatically via [`burn_store::PytorchStore`] (which transposes linear weights and renames
//! `weight`/`bias` to `gamma`/`beta` for norm layers). `CrnnOcr`'s BiLSTM is handled separately:
//! PyTorch stores it as one combined matrix per gate-group (`lstm.weight_ih_l0`, ...) while Burn
//! keeps every gate as an independent layer, so those tensors are split with
//! [`crate::interop::lstm_gates`] and injected directly as [`TensorSnapshot`]s.
//!
//! Expected PyTorch key names (matching the PyTorch mirrors in `export_onnx.py`):
//! - `CrnnOcr`: `conv{1..4}.{weight,bias}`, `bn{1..4}.{weight,bias,running_mean,running_var}`,
//!   `lstm.{weight_ih_l0,weight_hh_l0,bias_ih_l0,bias_hh_l0}` (+ `_reverse` variants),
//!   `linear.{weight,bias}`.
//! - `ConvCtcOcr`: the same `conv{1..4}`/`bn{1..4}` keys, plus `fc1.{weight,bias}` and
//!   `fc2.{weight,bias}` (no `lstm.*` keys).

use std::path::Path;

use burn::module::ParamId;
use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use burn_store::{ModuleSnapshot, ModuleStore, PytorchStore, TensorSnapshot};

use crate::model::conv_ctc::{ConvCtcOcr, ConvCtcOcrConfig};
use crate::model::crnn::{CrnnOcr, CrnnOcrConfig};
use crate::model::OcrModel;

use super::lstm_gates::{self, GateWeights};

/// Loads a PyTorch state dict file for use as a fine-tuning starting point, auto-detecting
/// whether it is a [`CrnnOcr`] or a [`ConvCtcOcr`]. `log` receives human-readable progress lines.
pub fn load<B: Backend>(
    path: &Path,
    device: &B::Device,
    mut log: impl FnMut(String),
) -> Result<OcrModel<B>, String> {
    let mut store = PytorchStore::from_file(path).allow_partial(true);

    let has_lstm = store
        .get_snapshot("lstm.weight_ih_l0")
        .ok()
        .flatten()
        .is_some();

    if has_lstm {
        log("PyTorch import: found 'lstm.weight_ih_l0' -> detected architecture: CRNN (Conv + BiLSTM + CTC).".to_string());
        load_crnn::<B>(path, device, &mut store, &mut log).map(OcrModel::CrnnBiLstm)
    } else {
        log("PyTorch import: no 'lstm.weight_ih_l0' tensor -> detected architecture: Conv-CTC (Conv-only, no recurrent layer).".to_string());
        load_conv_ctc::<B>(path, device, &mut store, &mut log).map(OcrModel::ConvCtc)
    }
}

/// Loads the CNN backbone, BiLSTM (via manual gate-merging) and final linear layer into a fresh
/// [`CrnnOcr`].
fn load_crnn<B: Backend>(
    path: &Path,
    device: &B::Device,
    store: &mut PytorchStore,
    log: &mut impl FnMut(String),
) -> Result<CrnnOcr<B>, String> {
    let config = CrnnOcrConfig::new();
    let mut model = config.init::<B>(device);

    // Expected LSTM shapes, taken from the freshly-initialized model (ground truth for this
    // architecture) rather than hard-coded, so this keeps working if `CrnnOcrConfig` changes.
    let hidden = config.lstm_hidden;
    let lstm_input = model.lstm_input_dim();

    let result = model
        .load_from(store)
        .map_err(|e| format!("Failed to read PyTorch checkpoint '{}': {e}", path.display()))?;
    log(format!(
        "PyTorch import: matched {} tensor(s) by name (CNN backbone + classifier); {} skipped, {} left for the LSTM merge below.",
        result.applied.len(),
        result.skipped.len(),
        result.missing.len(),
    ));
    if !result.errors.is_empty() {
        return Err(format!(
            "Errors while applying PyTorch weights to '{}': {:?}",
            path.display(),
            result.errors
        ));
    }

    let mut lstm_snapshots = Vec::new();
    for (direction, suffix) in [("forward", ""), ("reverse", "_reverse")] {
        let weight_ih = read_f32(store, &format!("lstm.weight_ih_l0{suffix}"), path)?;
        let weight_hh = read_f32(store, &format!("lstm.weight_hh_l0{suffix}"), path)?;
        let bias_ih = read_f32(store, &format!("lstm.bias_ih_l0{suffix}"), path)?;
        let bias_hh = read_f32(store, &format!("lstm.bias_hh_l0{suffix}"), path)?;

        let expected_ih_len = 4 * hidden * lstm_input;
        let expected_hh_len = 4 * hidden * hidden;
        let expected_bias_len = 4 * hidden;
        if weight_ih.len() != expected_ih_len
            || weight_hh.len() != expected_hh_len
            || bias_ih.len() != expected_bias_len
            || bias_hh.len() != expected_bias_len
        {
            return Err(format!(
                "LSTM ({direction} direction) shape mismatch in '{}': expected weight_ih/weight_hh/bias sizes \
                 {expected_ih_len}/{expected_hh_len}/{expected_bias_len}, got {}/{}/{} (bias_hh {}). \
                 The pretrained model's architecture doesn't match this project's CRNN (hidden size {hidden}, \
                 LSTM input size {lstm_input}).",
                path.display(),
                weight_ih.len(),
                weight_hh.len(),
                bias_ih.len(),
                bias_hh.len()
            ));
        }

        let gates = lstm_gates::split_ifgo(&weight_ih, &weight_hh, &bias_ih, &bias_hh, lstm_input, hidden);
        for (gate_name, gate) in gates.in_ifgo_order() {
            push_gate_snapshots(&mut lstm_snapshots, direction, gate_name, gate);
        }
        log(format!("PyTorch import: merged LSTM {direction}-direction gates."));
    }

    // Only the 32 LSTM leaf tensors are provided here (everything else was already applied
    // above via `load_from`), so `apply_result.missing` will naturally list every *other*
    // model parameter — that's expected for this partial apply, not an error.
    let expected_applied = lstm_snapshots.len();
    let apply_result = model.apply(lstm_snapshots, None, None, false);
    if !apply_result.errors.is_empty() {
        return Err(format!(
            "Errors while injecting merged LSTM weights from '{}': {:?}",
            path.display(),
            apply_result.errors
        ));
    }
    if apply_result.applied.len() != expected_applied {
        return Err(format!(
            "Internal error: expected to inject {expected_applied} merged LSTM tensors into the \
             model while importing '{}', but only {} were applied: {:?}",
            path.display(),
            apply_result.applied.len(),
            apply_result.applied
        ));
    }

    Ok(model)
}

/// Loads the CNN backbone and 2-layer FC head into a fresh [`ConvCtcOcr`]. Since it has no
/// recurrent layer to hand-merge, every tensor is matched by name directly.
fn load_conv_ctc<B: Backend>(
    path: &Path,
    device: &B::Device,
    store: &mut PytorchStore,
    log: &mut impl FnMut(String),
) -> Result<ConvCtcOcr<B>, String> {
    let mut model = ConvCtcOcrConfig::new().init::<B>(device);

    let result = model
        .load_from(store)
        .map_err(|e| format!("Failed to read PyTorch checkpoint '{}': {e}", path.display()))?;
    log(format!(
        "PyTorch import: matched {} tensor(s) by name (CNN backbone + FC head); {} skipped, {} missing.",
        result.applied.len(),
        result.skipped.len(),
        result.missing.len(),
    ));
    if !result.errors.is_empty() {
        return Err(format!(
            "Errors while applying PyTorch weights to '{}': {:?}",
            path.display(),
            result.errors
        ));
    }
    if !result.missing.is_empty() {
        return Err(format!(
            "'{}' is missing {} tensor(s) expected by plate-ocr's Conv-CTC architecture: {:?}. Make \
             sure it is a plain state dict (e.g. saved via `torch.save(model.state_dict(), ...)`) \
             using the same layer names as this project's Conv-CTC model (see `export_onnx.py`).",
            path.display(),
            result.missing.len(),
            result.missing
        ));
    }

    // `fc_input_dim` (unlike `CrnnOcr::lstm_input_dim`) doesn't need to be read here for manual
    // shape validation: `load_from`'s by-name matching already validates every tensor's shape
    // (including `fc1`'s) against the freshly-initialized model, since no hand-merged tensors
    // are injected separately for this architecture.
    let _ = model.fc_input_dim();

    Ok(model)
}

fn read_f32(store: &mut PytorchStore, name: &str, path: &Path) -> Result<Vec<f32>, String> {
    let snapshot = store
        .get_snapshot(name)
        .map_err(|e| format!("Failed to read '{name}' from '{}': {e}", path.display()))?
        .ok_or_else(|| {
            format!(
                "'{}' does not contain a '{name}' tensor. Make sure it is a plain state dict (e.g. saved via \
                 `torch.save(model.state_dict(), ...)`) using the same layer names as this project's CRNN \
                 (see `export_onnx.py`).",
                path.display()
            )
        })?;
    let data = snapshot
        .to_data()
        .map_err(|e| format!("Failed to materialize '{name}' from '{}': {e}", path.display()))?
        .convert_dtype(DType::F32);
    data.to_vec::<f32>()
        .map_err(|e| format!("Failed to convert '{name}' from '{}' to f32: {e:?}", path.display()))
}

#[cfg(all(test, feature = "cpu"))]
mod cross_validation_tests {
    //! Manual cross-validation against a real PyTorch `.pt` file produced (and independently
    //! forward-passed) by actual PyTorch, to numerically verify the LSTM gate-merge math and
    //! the `burn-store` integration end-to-end. Ignored by default since it depends on fixture
    //! files produced out-of-band by a Python/PyTorch script (see the project's dev notes); run
    //! explicitly with `cargo test -- --ignored` after generating the fixtures.
    use super::*;
    use burn::tensor::TensorData;

    type TestBackend = burn::backend::NdArray;

    #[test]
    #[ignore]
    fn matches_real_pytorch_forward_pass() {
        let fixture_dir = std::path::Path::new("/tmp/plate_ocr_export_test");
        let device = Default::default();

        let mut logs = Vec::new();
        let model = load::<TestBackend>(&fixture_dir.join("direct.pt"), &device, |line| logs.push(line))
            .expect("failed to load direct.pt");
        for line in &logs {
            println!("{line}");
        }
        let OcrModel::CrnnBiLstm(model) = model else {
            panic!("expected direct.pt to be auto-detected as CrnnBiLstm");
        };

        let input_bytes = std::fs::read(fixture_dir.join("fixture_input.bin")).expect("missing fixture input");
        let input_f32: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let input = Tensor::<TestBackend, 4>::from_data(TensorData::new(input_f32, [1, 1, 32, 128]), &device);

        let output = model.forward(input);
        let output = output.into_data().to_vec::<f32>().unwrap();

        let expected_bytes = std::fs::read(fixture_dir.join("fixture_output.bin")).expect("missing fixture output");
        let expected: Vec<f32> = expected_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(output.len(), expected.len());
        let max_diff = output
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("max abs diff vs real PyTorch forward pass: {max_diff}");
        assert!(max_diff < 1e-3, "max abs diff too large: {max_diff}");
    }
}

/// Appends the four leaf-tensor [`TensorSnapshot`]s (`input_transform.{weight,bias}`,
/// `hidden_transform.{weight,bias}`) for one LSTM gate, using Burn's own internal path naming
/// (`lstm.<direction>.<gate>.<transform>.<field>`, as produced by `CrnnOcr`'s own `collect()`).
fn push_gate_snapshots(out: &mut Vec<TensorSnapshot>, direction: &str, gate_name: &str, gate: &GateWeights) {
    let base = ["lstm", direction, gate_name];
    let leaf = |transform: &str, field: &str, data: TensorData| {
        let mut path_stack: Vec<String> = base.iter().map(|s| s.to_string()).collect();
        path_stack.push(transform.to_string());
        path_stack.push(field.to_string());
        TensorSnapshot::from_data(data, path_stack, Vec::new(), ParamId::new())
    };

    let hidden = gate.input_bias.len();
    let input = gate.input_weight.len() / hidden;

    out.push(leaf(
        "input_transform",
        "weight",
        TensorData::new(gate.input_weight.clone(), [input, hidden]),
    ));
    out.push(leaf(
        "input_transform",
        "bias",
        TensorData::new(gate.input_bias.clone(), [hidden]),
    ));
    out.push(leaf(
        "hidden_transform",
        "weight",
        TensorData::new(gate.hidden_weight.clone(), [hidden, hidden]),
    ));
    out.push(leaf(
        "hidden_transform",
        "bias",
        TensorData::new(gate.hidden_bias.clone(), [hidden]),
    ));
}
