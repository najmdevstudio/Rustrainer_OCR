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

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::vocab;

pub const IMG_HEIGHT: usize = 32;
pub const IMG_WIDTH: usize = 128;
pub const MAX_LABEL_LEN: usize = 15;
/// Number of time steps output by the CNN (width / 4 due to two 2x2 pools on width).
pub const TIME_STEPS: usize = IMG_WIDTH / 4;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlateRecord {
    pub image_name: String,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct PlateSample {
    pub image: Vec<f32>,
    pub label: Vec<usize>,
    #[allow(dead_code)]
    pub label_len: usize,
}

pub struct PlateDataset {
    samples: Vec<PlateRecord>,
    images_dir: PathBuf,
}

impl PlateDataset {
    pub fn new(split_dir: &str) -> Self {
        let split_path = PathBuf::from(split_dir);
        let csv_path = split_path.join("labels.csv");
        let images_dir = split_path.join("images");

        let mut reader = csv::Reader::from_path(&csv_path)
            .unwrap_or_else(|e| panic!("Failed to open {}: {}", csv_path.display(), e));

        let samples: Vec<PlateRecord> = reader
            .deserialize()
            .filter_map(|r: Result<PlateRecord, _>| match r {
                Ok(record) => Some(record),
                Err(e) => {
                    log::warn!("Skipping CSV row: {}", e);
                    None
                }
            })
            .collect();

        if samples.is_empty() {
            log::error!("No samples loaded from {}! Check that CSV has columns: image_name,label", csv_path.display());
        }

        log::info!("Loaded {} samples from {}", samples.len(), csv_path.display());

        Self { samples, images_dir }
    }

    fn load_image(&self, filename: &str) -> Vec<f32> {
        let path = self.images_dir.join(filename);
        let img = image::open(&path)
            .unwrap_or_else(|e| panic!("Failed to open image {}: {}", path.display(), e));

        let img = img
            .resize_exact(
                IMG_WIDTH as u32,
                IMG_HEIGHT as u32,
                image::imageops::FilterType::Triangle,
            )
            .to_luma8();

        img.pixels().map(|p| p.0[0] as f32 / 255.0).collect()
    }
}

impl Dataset<PlateSample> for PlateDataset {
    fn get(&self, index: usize) -> Option<PlateSample> {
        let record = self.samples.get(index)?;
        let image = self.load_image(&record.image_name);
        let label = vocab::encode(&record.label);
        let label_len = label.len();
        Some(PlateSample { image, label, label_len })
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

/// Batched tensors ready for the model.
#[derive(Debug, Clone)]
pub struct PlateBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 2, Int>,
    pub input_lengths: Tensor<B, 1, Int>,
    pub target_lengths: Tensor<B, 1, Int>,
}

#[derive(Clone, Default)]
pub struct PlateBatcher;

impl<B: Backend> Batcher<B, PlateSample, PlateBatch<B>> for PlateBatcher {
    fn batch(&self, items: Vec<PlateSample>, device: &B::Device) -> PlateBatch<B> {
        let batch_size = items.len();

        let images: Vec<Tensor<B, 4>> = items
            .iter()
            .map(|s| {
                let data = TensorData::new(s.image.clone(), [1, IMG_HEIGHT, IMG_WIDTH]);
                Tensor::<B, 3>::from_data(data, device).unsqueeze_dim(0)
            })
            .collect();
        let images = Tensor::cat(images, 0);

        let mut target_data = vec![0i64; batch_size * MAX_LABEL_LEN];
        let mut target_len_data = Vec::with_capacity(batch_size);

        for (i, sample) in items.iter().enumerate() {
            let len = sample.label.len().min(MAX_LABEL_LEN);
            for j in 0..len {
                target_data[i * MAX_LABEL_LEN + j] = sample.label[j] as i64;
            }
            target_len_data.push(len as i64);
        }

        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_data, [batch_size, MAX_LABEL_LEN]),
            device,
        );

        let target_lengths = Tensor::<B, 1, Int>::from_data(
            TensorData::new(target_len_data, [batch_size]),
            device,
        );

        let input_len_data = vec![TIME_STEPS as i64; batch_size];
        let input_lengths = Tensor::<B, 1, Int>::from_data(
            TensorData::new(input_len_data, [batch_size]),
            device,
        );

        PlateBatch {
            images,
            targets,
            input_lengths,
            target_lengths,
        }
    }
}
