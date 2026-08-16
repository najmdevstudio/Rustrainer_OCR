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

//! Runs the training job on a background thread so the GUI event loop stays responsive.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::Sender;
use std::thread;

use crate::training::train::TrainConfig;

use super::progress::GuiEvent;

/// Spawns the training job in the background and reports the final outcome through `sender`
/// (as a [`GuiEvent::Finished`]) once it completes or fails.
pub fn spawn_training(config: TrainConfig, sender: Sender<GuiEvent>) {
    thread::spawn(move || {
        let output_dir = config.output_dir.clone();
        let renderer_sender = sender.clone();

        let result = catch_unwind(AssertUnwindSafe(|| {
            let device = crate::backend::device();
            crate::training::train::run_with_progress::<crate::backend::TrainBackend>(
                config,
                device,
                Some(renderer_sender),
            );
        }));

        let outcome = match result {
            Ok(()) => Ok(format!(
                "Training complete. Model saved to {output_dir}/plate_ocr_final"
            )),
            Err(payload) => Err(panic_message(payload)),
        };

        let _ = sender.send(GuiEvent::Finished(outcome));
    });
}

/// Extracts a readable message out of a `catch_unwind` panic payload.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        message.to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "Training failed due to an unknown error.".to_string()
    }
}
