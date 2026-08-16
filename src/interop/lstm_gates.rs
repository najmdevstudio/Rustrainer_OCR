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

//! Pure tensor-shape math for converting between the combined-matrix LSTM gate convention
//! used by PyTorch/ONNX and Burn's per-gate [`GateController`](burn::nn::lstm::GateController)
//! decomposition.
//!
//! PyTorch (`torch.nn.LSTM`) packs the 4 gates for one direction into a single matrix, stacked
//! in `i, f, g(cell), o` order: `weight_ih_l0` has shape `[4*hidden, input]`, `weight_hh_l0` has
//! shape `[4*hidden, hidden]`, `bias_ih_l0` / `bias_hh_l0` have shape `[4*hidden]`. Burn instead
//! keeps every gate — and every one of its two affine transforms — as an independent `Linear`
//! layer with logical weight shape `[d_input, d_output]` (see `burn::nn::lstm::{Lstm, BiLstm,
//! GateController}`). Converting between the two therefore requires (1) slicing out each gate's
//! `hidden`-sized chunk and (2) transposing the weight, since PyTorch stores `[out, in]` while
//! Burn stores `[in, out]`.
//!
//! ONNX's `LSTM` operator uses the same combined-matrix idea but a different gate order
//! (`i, o, f, c`) and packs both directions into one leading axis; that reordering is handled by
//! the Python `import_onnx.py` helper before the resulting tensors ever reach this module, so
//! everything here only has to deal with PyTorch's `i, f, g, o` order.

const NUM_GATES: usize = 4;

/// One LSTM gate's two affine transforms, laid out the way
/// [`GateController`](burn::nn::lstm::GateController) expects: `weight` has shape
/// `[d_in, d_out]`, `bias` has shape `[d_out]` (Burn's `Linear` convention).
#[derive(Debug, Clone)]
pub struct GateWeights {
    /// `input_transform.weight`, shape `[input, hidden]`.
    pub input_weight: Vec<f32>,
    /// `input_transform.bias`, shape `[hidden]`.
    pub input_bias: Vec<f32>,
    /// `hidden_transform.weight`, shape `[hidden, hidden]`.
    pub hidden_weight: Vec<f32>,
    /// `hidden_transform.bias`, shape `[hidden]`.
    pub hidden_bias: Vec<f32>,
}

/// The four gates of a single-direction LSTM, named after Burn's own `Lstm` fields.
#[derive(Debug, Clone)]
pub struct LstmGates {
    pub input_gate: GateWeights,
    pub forget_gate: GateWeights,
    pub cell_gate: GateWeights,
    pub output_gate: GateWeights,
}

impl LstmGates {
    /// Iterates over the four gates paired with their Burn field name, in PyTorch's `i, f, g, o`
    /// order.
    pub fn in_ifgo_order(&self) -> [(&'static str, &GateWeights); NUM_GATES] {
        [
            ("input_gate", &self.input_gate),
            ("forget_gate", &self.forget_gate),
            ("cell_gate", &self.cell_gate),
            ("output_gate", &self.output_gate),
        ]
    }
}

/// Transposes a row-major `[rows, cols]` matrix into `[cols, rows]`.
pub(crate) fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Splits PyTorch-style combined `weight_ih` / `weight_hh` / `bias_ih` / `bias_hh` tensors
/// (`torch.nn.LSTM`'s `i, f, g, o` gate order) into Burn's four independent gates.
///
/// - `weight_ih`: `[4*hidden, input]`
/// - `weight_hh`: `[4*hidden, hidden]`
/// - `bias_ih`, `bias_hh`: `[4*hidden]` each (pass all-zero vectors for bias-less LSTMs)
pub fn split_ifgo(
    weight_ih: &[f32],
    weight_hh: &[f32],
    bias_ih: &[f32],
    bias_hh: &[f32],
    input: usize,
    hidden: usize,
) -> LstmGates {
    let chunk =
        |data: &[f32], rows: usize, cols: usize, index: usize| -> Vec<f32> {
            data[index * rows * cols..(index + 1) * rows * cols].to_vec()
        };
    let chunk_bias =
        |data: &[f32], index: usize| -> Vec<f32> { data[index * hidden..(index + 1) * hidden].to_vec() };

    let gate = |index: usize| GateWeights {
        input_weight: transpose(&chunk(weight_ih, hidden, input, index), hidden, input),
        input_bias: chunk_bias(bias_ih, index),
        hidden_weight: transpose(&chunk(weight_hh, hidden, hidden, index), hidden, hidden),
        hidden_bias: chunk_bias(bias_hh, index),
    };

    LstmGates {
        input_gate: gate(0),
        forget_gate: gate(1),
        cell_gate: gate(2),
        output_gate: gate(3),
    }
}

/// The inverse of [`split_ifgo`]: merges Burn's four independent gates back into PyTorch's
/// combined `i, f, g, o` layout, ready to be saved as a PyTorch-compatible tensor.
///
/// Returns `(weight_ih, weight_hh, bias_ih, bias_hh)`.
pub fn merge_ifgo(gates: &LstmGates, input: usize, hidden: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut weight_ih = Vec::with_capacity(NUM_GATES * hidden * input);
    let mut weight_hh = Vec::with_capacity(NUM_GATES * hidden * hidden);
    let mut bias_ih = Vec::with_capacity(NUM_GATES * hidden);
    let mut bias_hh = Vec::with_capacity(NUM_GATES * hidden);

    for (_name, gate) in gates.in_ifgo_order() {
        weight_ih.extend(transpose(&gate.input_weight, input, hidden));
        weight_hh.extend(transpose(&gate.hidden_weight, hidden, hidden));
        bias_ih.extend(&gate.input_bias);
        bias_hh.extend(&gate.hidden_bias);
    }

    (weight_ih, weight_hh, bias_ih, bias_hh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_then_merge_round_trips() {
        let input = 3;
        let hidden = 2;
        let weight_ih: Vec<f32> = (0..NUM_GATES * hidden * input).map(|v| v as f32).collect();
        let weight_hh: Vec<f32> = (0..NUM_GATES * hidden * hidden).map(|v| v as f32 * 0.5).collect();
        let bias_ih: Vec<f32> = (0..NUM_GATES * hidden).map(|v| v as f32 + 0.1).collect();
        let bias_hh: Vec<f32> = (0..NUM_GATES * hidden).map(|v| v as f32 - 0.1).collect();

        let gates = split_ifgo(&weight_ih, &weight_hh, &bias_ih, &bias_hh, input, hidden);
        let (out_ih, out_hh, out_bias_ih, out_bias_hh) = merge_ifgo(&gates, input, hidden);

        assert_eq!(out_ih, weight_ih);
        assert_eq!(out_hh, weight_hh);
        assert_eq!(out_bias_ih, bias_ih);
        assert_eq!(out_bias_hh, bias_hh);
    }

    #[test]
    fn gate_chunks_are_assigned_in_ifgo_order() {
        // Each gate's chunk is filled with a distinct constant so we can check that
        // `split_ifgo` assigns "i, f, g, o" to the correspondingly-named Burn gate.
        let (input, hidden) = (1, 1);
        let weight_ih = [10.0, 20.0, 30.0, 40.0]; // i, f, g, o
        let weight_hh = [1.0, 2.0, 3.0, 4.0];
        let bias_ih = [100.0, 200.0, 300.0, 400.0];
        let bias_hh = [0.0; 4];

        let gates = split_ifgo(&weight_ih, &weight_hh, &bias_ih, &bias_hh, input, hidden);

        assert_eq!(gates.input_gate.input_weight, vec![10.0]);
        assert_eq!(gates.forget_gate.input_weight, vec![20.0]);
        assert_eq!(gates.cell_gate.input_weight, vec![30.0]);
        assert_eq!(gates.output_gate.input_weight, vec![40.0]);
        assert_eq!(gates.input_gate.input_bias, vec![100.0]);
        assert_eq!(gates.output_gate.input_bias, vec![400.0]);
    }

    #[test]
    fn transpose_matches_matrix_semantics() {
        // weight_ih for a single gate, input=3, hidden=2: PyTorch layout [hidden, input].
        let weight_ih = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let weight_hh = [0.0; 4]; // hidden=2 -> [2,2]
        let bias_ih = [0.0; 2];
        let bias_hh = [0.0; 2];

        // Treat this as if NUM_GATES chunks were all identical for simplicity by only checking
        // the first gate via a direct call to the private `transpose` semantics through
        // `split_ifgo` with hidden=2 covering 1 "gate" worth of rows (we fabricate 4 identical
        // gates to keep chunk sizes valid).
        let mut ih4 = Vec::new();
        let mut hh4 = Vec::new();
        let mut bih4 = Vec::new();
        let mut bhh4 = Vec::new();
        for _ in 0..4 {
            ih4.extend_from_slice(&weight_ih);
            hh4.extend_from_slice(&weight_hh);
            bih4.extend_from_slice(&bias_ih);
            bhh4.extend_from_slice(&bias_hh);
        }

        let gates = split_ifgo(&ih4, &hh4, &bih4, &bhh4, 3, 2);
        // [[1,2,3],[4,5,6]] transposed -> [[1,4],[2,5],[3,6]] flattened = [1,4,2,5,3,6]
        assert_eq!(gates.input_gate.input_weight, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
