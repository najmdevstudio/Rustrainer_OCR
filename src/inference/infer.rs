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

use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::module::Module;

use crate::data::dataset::{IMG_HEIGHT, IMG_WIDTH};
use crate::data::vocab;
use crate::model::crnn::CrnnOcrConfig;
use crate::model::conv_ctc::ConvCtcOcrConfig;
use crate::model::{Architecture, OcrModel};

/// Loads a Burn checkpoint saved by `plate-ocr train`/`train --pretrained ...`, auto-detecting
/// which architecture it is via the small sidecar file `crate::model::architecture` writes next
/// to every checkpoint (defaulting to [`Architecture::CrnnBiLstm`] for checkpoints saved before
/// that sidecar existed).
pub fn load_model<B: Backend>(model_path: &str, device: &B::Device) -> Result<OcrModel<B>, String> {
    let architecture = Architecture::read_sidecar(model_path);
    let model = match architecture {
        Architecture::CrnnBiLstm => {
            let model = CrnnOcrConfig::new()
                .init::<B>(device)
                .load_file(model_path, &CompactRecorder::new(), device)
                .map_err(|e| format!("Failed to load checkpoint '{model_path}': {e}"))?;
            OcrModel::CrnnBiLstm(model)
        }
        Architecture::ConvCtc => {
            let model = ConvCtcOcrConfig::new()
                .init::<B>(device)
                .load_file(model_path, &CompactRecorder::new(), device)
                .map_err(|e| format!("Failed to load checkpoint '{model_path}': {e}"))?;
            OcrModel::ConvCtc(model)
        }
    };
    Ok(model)
}

pub fn preprocess_image<B: Backend>(image_path: &str, device: &B::Device) -> Tensor<B, 4> {
    let img = image::open(image_path)
        .unwrap_or_else(|e| panic!("Failed to open image {}: {}", image_path, e));

    let img = img
        .resize_exact(
            IMG_WIDTH as u32,
            IMG_HEIGHT as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_luma8();

    let pixels: Vec<f32> = img.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    let data = TensorData::new(pixels, [1, 1, IMG_HEIGHT, IMG_WIDTH]);
    Tensor::from_data(data, device)
}

pub fn recognize<B: Backend>(model: &OcrModel<B>, image_path: &str, device: &B::Device) -> String {
    let image = preprocess_image::<B>(image_path, device);
    let log_probs = model.forward(image);
    // log_probs: [time=32, batch=1, classes=37]

    let predictions = log_probs.argmax(2); // [time, batch]
    let predictions = predictions.squeeze::<1>(); // [time]

    let data = predictions.into_data();
    let indices: Vec<i64> = data.to_vec().expect("Failed to extract prediction indices");
    let indices: Vec<usize> = indices.iter().map(|&v| v as usize).collect();

    vocab::decode(&indices)
}
