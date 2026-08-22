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

pub mod architecture;
pub mod conv_ctc;
pub mod crnn;

use burn::prelude::*;

pub use architecture::Architecture;
use conv_ctc::ConvCtcOcr;
use crnn::CrnnOcr;

/// Either of plate-ocr's supported OCR models, after loading/initializing — used wherever code
/// (inference, export, the fine-tuning entry point) needs to hold "some model, of whichever
/// architecture was detected/selected" without caring which one beyond dispatching `forward`.
///
/// This is a plain tagged union rather than a Burn [`burn::module::Module`] itself: the two
/// variants are trained through their own separate, concretely-typed pipelines (see
/// `crate::training::train`), so nothing needs `OcrModel` itself to be a `Module`.
#[derive(Debug)]
pub enum OcrModel<B: Backend> {
    CrnnBiLstm(CrnnOcr<B>),
    ConvCtc(ConvCtcOcr<B>),
}

impl<B: Backend> OcrModel<B> {
    /// Initializes a fresh model of `architecture`, with default hyperparameters.
    pub fn init_default(architecture: Architecture, device: &B::Device) -> Self {
        match architecture {
            Architecture::CrnnBiLstm => OcrModel::CrnnBiLstm(crnn::CrnnOcrConfig::new().init(device)),
            Architecture::ConvCtc => OcrModel::ConvCtc(conv_ctc::ConvCtcOcrConfig::new().init(device)),
        }
    }

    pub fn architecture(&self) -> Architecture {
        match self {
            OcrModel::CrnnBiLstm(_) => Architecture::CrnnBiLstm,
            OcrModel::ConvCtc(_) => Architecture::ConvCtc,
        }
    }

    /// Forward pass, common to every architecture: [batch, 1, 32, 128] -> [time, batch, classes]
    /// log-probabilities.
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        match self {
            OcrModel::CrnnBiLstm(model) => model.forward(images),
            OcrModel::ConvCtc(model) => model.forward(images),
        }
    }

    /// Freeze the CNN backbone (conv/bn layers) of whichever architecture this is, leaving only
    /// its head (LSTM+linear, or the FC head) trainable.
    pub fn freeze_backbone(self) -> Self {
        match self {
            OcrModel::CrnnBiLstm(model) => OcrModel::CrnnBiLstm(model.freeze_backbone()),
            OcrModel::ConvCtc(model) => OcrModel::ConvCtc(model.freeze_backbone()),
        }
    }
}
