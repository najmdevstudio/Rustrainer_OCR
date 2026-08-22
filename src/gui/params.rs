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

//! Editable training parameters shown on the "Parameters" screen.

use crate::model::Architecture;
use crate::training::train::TrainConfig;

/// Which high-level flow the user picked on the first screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    NewTraining,
    FineTuning,
}

impl Mode {
    pub fn label(&self) -> &'static str {
        match self {
            Mode::NewTraining => "New Model Training",
            Mode::FineTuning => "Fine-Tuning",
        }
    }
}

/// Training parameters, prefilled with sensible defaults for each [`Mode`] (mirroring the
/// examples documented in the README) but fully editable before starting.
#[derive(Debug, Clone)]
pub struct Params {
    pub data_dir: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub output_dir: String,
    /// Path to a pretrained model to fine-tune from. Only used in [`Mode::FineTuning`].
    /// Accepts a Burn checkpoint, a PyTorch `.pt`/`.pth` state dict, or an `.onnx` model — the
    /// format is auto-detected from the file extension (see `crate::interop`).
    pub pretrained: String,
    pub freeze_backbone: bool,
    /// Which architecture to train from scratch. Only user-editable in [`Mode::NewTraining`];
    /// [`Mode::FineTuning`] always auto-detects the architecture from `pretrained` instead (this
    /// field is simply ignored in that case).
    pub architecture: Architecture,
}

impl Params {
    pub fn defaults_for(mode: Mode) -> Self {
        match mode {
            Mode::NewTraining => Self {
                data_dir: "dataset".to_string(),
                epochs: 50,
                batch_size: 64,
                learning_rate: 1e-3,
                output_dir: "checkpoints".to_string(),
                pretrained: String::new(),
                freeze_backbone: false,
                architecture: Architecture::CrnnBiLstm,
            },
            Mode::FineTuning => Self {
                data_dir: "dataset".to_string(),
                epochs: 20,
                batch_size: 64,
                learning_rate: 1e-4,
                output_dir: "checkpoints".to_string(),
                pretrained: "checkpoints/plate_ocr_final".to_string(),
                freeze_backbone: false,
                architecture: Architecture::CrnnBiLstm,
            },
        }
    }

    /// Validates the fields, returning a human readable error message on failure.
    pub fn validate(&self, mode: Mode) -> Result<(), String> {
        if self.data_dir.trim().is_empty() {
            return Err("Please choose a dataset base directory.".to_string());
        }
        if self.epochs == 0 {
            return Err("Epochs must be at least 1.".to_string());
        }
        if self.batch_size == 0 {
            return Err("Batch size must be at least 1.".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("Learning rate must be greater than 0.".to_string());
        }
        if self.output_dir.trim().is_empty() {
            return Err("Please choose an output directory for checkpoints.".to_string());
        }
        if mode == Mode::FineTuning && self.pretrained.trim().is_empty() {
            return Err(
                "Please choose a pretrained model to fine-tune (checkpoint, .pt/.pth, or .onnx)."
                    .to_string(),
            );
        }
        Ok(())
    }

    pub fn to_train_config(&self) -> TrainConfig {
        let pretrained = if self.pretrained.trim().is_empty() {
            None
        } else {
            Some(self.pretrained.clone())
        };

        TrainConfig::new(self.data_dir.clone())
            .with_num_epochs(self.epochs)
            .with_batch_size(self.batch_size)
            .with_learning_rate(self.learning_rate)
            .with_output_dir(self.output_dir.clone())
            .with_pretrained(pretrained)
            .with_freeze_backbone(self.freeze_backbone)
            .with_architecture(self.architecture)
    }
}
