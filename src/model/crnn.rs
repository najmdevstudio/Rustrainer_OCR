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

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
use burn::nn::lstm::{BiLstm, BiLstmConfig};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;


#[derive(Config, Debug)]
pub struct CrnnOcrConfig {
    #[config(default = 37)]
    pub num_classes: usize,
    #[config(default = 256)]
    pub lstm_hidden: usize,
}

/// CRNN model for OCR: CNN feature extractor → BiLSTM → Linear classifier.
///
/// Input:  [batch, 1, 32, 128] grayscale images
/// Output: [time=32, batch, num_classes] log-probabilities
#[derive(Module, Debug)]
pub struct CrnnOcr<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    pool1: MaxPool2d,

    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    pool2: MaxPool2d,

    conv3: Conv2d<B>,
    bn3: BatchNorm<B>,

    conv4: Conv2d<B>,
    bn4: BatchNorm<B>,
    pool4: MaxPool2d,

    lstm: BiLstm<B>,
    linear: Linear<B>,

    num_classes: usize,
}

impl CrnnOcrConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CrnnOcr<B> {
        let conv1 = Conv2dConfig::new([1, 64], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let bn1 = BatchNormConfig::new(64).init(device);
        let pool1 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let conv2 = Conv2dConfig::new([64, 128], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let bn2 = BatchNormConfig::new(128).init(device);
        let pool2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let conv3 = Conv2dConfig::new([128, 256], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let bn3 = BatchNormConfig::new(256).init(device);

        let conv4 = Conv2dConfig::new([256, 256], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Same)
            .init(device);
        let bn4 = BatchNormConfig::new(256).init(device);
        let pool4 = MaxPool2dConfig::new([2, 1]).with_strides([2, 1]).init();

        // After CNN: [batch, 256, 4, 32] → reshape to [batch, 32, 1024]
        let lstm_input = 256 * 4;
        let lstm = BiLstmConfig::new(lstm_input, self.lstm_hidden, true)
            .init(device);

        let linear = LinearConfig::new(self.lstm_hidden * 2, self.num_classes)
            .init(device);

        CrnnOcr {
            conv1, bn1, pool1,
            conv2, bn2, pool2,
            conv3, bn3,
            conv4, bn4, pool4,
            lstm, linear,
            num_classes: self.num_classes,
        }
    }
}

impl<B: Backend> CrnnOcr<B> {
    /// The BiLSTM's expected input feature size (`d_input`), i.e. the flattened CNN output
    /// width. Used by `crate::interop` to validate/shape external (PyTorch/ONNX) weights before
    /// merging them into the BiLSTM's per-gate layers.
    pub(crate) fn lstm_input_dim(&self) -> usize {
        self.lstm.forward.input_gate.input_transform.weight.shape().dims::<2>()[0]
    }

    /// Freeze CNN backbone layers (conv + batchnorm) so only LSTM and linear head are trained.
    /// Returns a new model with the backbone parameters marked as non-trainable.
    pub fn freeze_backbone(self) -> Self {
        Self {
            conv1: self.conv1.no_grad(),
            bn1: self.bn1.no_grad(),
            pool1: self.pool1,
            conv2: self.conv2.no_grad(),
            bn2: self.bn2.no_grad(),
            pool2: self.pool2,
            conv3: self.conv3.no_grad(),
            bn3: self.bn3.no_grad(),
            conv4: self.conv4.no_grad(),
            bn4: self.bn4.no_grad(),
            pool4: self.pool4,
            lstm: self.lstm,
            linear: self.linear,
            num_classes: self.num_classes,
        }
    }

    /// Forward pass.
    /// Input:  `images` [batch, 1, 32, 128]
    /// Output: log-probabilities [time=32, batch, num_classes]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = self.conv1.forward(images);
        let x = self.bn1.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.pool1.forward(x);

        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.pool2.forward(x);

        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = burn::tensor::activation::relu(x);

        let x = self.conv4.forward(x);
        let x = self.bn4.forward(x);
        let x = burn::tensor::activation::relu(x);
        let x = self.pool4.forward(x);

        // x: [batch, 256, 4, 32]
        let [batch, channels, height, width] = x.dims();

        // Reshape to [batch, 32, 1024]
        let x = x.swap_dims(2, 3); // [batch, 256, 32, 4]
        let x = x.swap_dims(1, 2); // [batch, 32, 256, 4]
        let x = x.reshape([batch, width, channels * height]);

        // BiLSTM
        let (lstm_out, _) = self.lstm.forward(x, None);

        // Linear projection
        let output = self.linear.forward(lstm_out);

        // log_softmax over class dimension
        let output = log_softmax(output, 2);

        // CTC expects [time, batch, classes]
        output.swap_dims(0, 1)
    }
}

#[cfg(all(test, feature = "cpu"))]
mod path_diagnostic {
    use super::*;
    use burn_store::ModuleSnapshot;

    type TestBackend = burn::backend::NdArray;

    /// `crate::interop` builds parameter paths like `lstm.forward.input_gate.input_transform.weight`
    /// by hand (to inject merged PyTorch/ONNX LSTM weights via `Module::apply`), so this locks in
    /// Burn's own internal path naming/shapes as a regression check.
    #[test]
    fn param_paths_match_what_interop_expects() {
        let device = Default::default();
        let model = CrnnOcrConfig::new().init::<TestBackend>(&device);
        let paths: std::collections::HashMap<String, Vec<usize>> = model
            .collect(None, None, false)
            .into_iter()
            .map(|s| (s.full_path(), s.shape.iter().copied().collect()))
            .collect();

        assert_eq!(paths["conv1.weight"], vec![64, 1, 3, 3]);
        assert_eq!(paths["bn1.gamma"], vec![64]);
        assert_eq!(paths["bn1.running_mean"], vec![64]);
        assert_eq!(paths["linear.weight"], vec![512, 37]);
        assert_eq!(
            paths["lstm.forward.input_gate.input_transform.weight"],
            vec![1024, 256]
        );
        assert_eq!(
            paths["lstm.forward.input_gate.hidden_transform.weight"],
            vec![256, 256]
        );
        assert_eq!(paths["lstm.reverse.output_gate.hidden_transform.bias"], vec![256]);
        assert_eq!(model.lstm_input_dim(), 1024);
    }
}
