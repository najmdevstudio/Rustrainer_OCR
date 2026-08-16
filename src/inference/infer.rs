use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::module::Module;

use crate::data::dataset::{IMG_HEIGHT, IMG_WIDTH};
use crate::data::vocab;
use crate::model::crnn::{CrnnOcr, CrnnOcrConfig};

pub fn load_model<B: Backend>(model_path: &str, device: &B::Device) -> CrnnOcr<B> {
    let config = CrnnOcrConfig::new();
    let model = config.init::<B>(device);
    model
        .load_file(model_path, &CompactRecorder::new(), device)
        .expect("Failed to load model checkpoint")
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

pub fn recognize<B: Backend>(model: &CrnnOcr<B>, image_path: &str, device: &B::Device) -> String {
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
