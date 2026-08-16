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
