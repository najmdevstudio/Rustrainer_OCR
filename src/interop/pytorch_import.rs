//! Loads a PyTorch (`.pt` / `.pth`) state dict as a pretrained/fine-tuning starting point for
//! [`CrnnOcr`].
//!
//! The convolutional backbone, batch-norm layers and final linear head are matched by name
//! automatically via [`burn_store::PytorchStore`] (which transposes linear weights and renames
//! `weight`/`bias` to `gamma`/`beta` for norm layers). The BiLSTM is handled separately: PyTorch
//! stores it as one combined matrix per gate-group (`lstm.weight_ih_l0`, ...) while Burn keeps
//! every gate as an independent layer, so those tensors are split with
//! [`crate::interop::lstm_gates`] and injected directly as [`TensorSnapshot`]s.
//!
//! Expected PyTorch key names (matching the `CrnnOcr` PyTorch mirror in `export_onnx.py`):
//! `conv{1..4}.{weight,bias}`, `bn{1..4}.{weight,bias,running_mean,running_var}`,
//! `lstm.{weight_ih_l0,weight_hh_l0,bias_ih_l0,bias_hh_l0}` (+ `_reverse` variants),
//! `linear.{weight,bias}`.

use std::path::Path;

use burn::module::ParamId;
use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use burn_store::{ModuleSnapshot, ModuleStore, PytorchStore, TensorSnapshot};

use crate::model::crnn::{CrnnOcr, CrnnOcrConfig};

use super::lstm_gates::{self, GateWeights};

/// Loads a PyTorch state dict file into a freshly-initialized [`CrnnOcr`], for use as a
/// fine-tuning starting point. `log` receives human-readable progress lines.
pub fn load<B: Backend>(
    path: &Path,
    config: &CrnnOcrConfig,
    device: &B::Device,
    mut log: impl FnMut(String),
) -> Result<CrnnOcr<B>, String> {
    let mut model = config.init::<B>(device);

    // Expected LSTM shapes, taken from the freshly-initialized model (ground truth for this
    // architecture) rather than hard-coded, so this keeps working if `CrnnOcrConfig` changes.
    let hidden = config.lstm_hidden;
    let lstm_input = model.lstm_input_dim();

    let mut store = PytorchStore::from_file(path).allow_partial(true);

    let result = model
        .load_from(&mut store)
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
        let weight_ih = read_f32(&mut store, &format!("lstm.weight_ih_l0{suffix}"), path)?;
        let weight_hh = read_f32(&mut store, &format!("lstm.weight_hh_l0{suffix}"), path)?;
        let bias_ih = read_f32(&mut store, &format!("lstm.bias_ih_l0{suffix}"), path)?;
        let bias_hh = read_f32(&mut store, &format!("lstm.bias_hh_l0{suffix}"), path)?;

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
        let config = CrnnOcrConfig::new();

        let mut logs = Vec::new();
        let model = load::<TestBackend>(&fixture_dir.join("direct.pt"), &config, &device, |line| {
            logs.push(line)
        })
        .expect("failed to load direct.pt");
        for line in &logs {
            println!("{line}");
        }

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
