//! Graphical wizard that guides the user through the whole training process:
//! choose a process -> configure parameters -> watch live progress -> see the result.
//!
//! Built with `eframe`/`egui`, which opens a regular, native OS window managed by whichever
//! window manager the host platform already provides (no custom chrome, no web view).

mod app;
mod params;
pub(crate) mod progress;
mod worker;

use eframe::egui;

use app::WizardApp;

/// Launches the GUI wizard. Blocks until the window is closed.
pub fn run_gui() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([880.0, 680.0])
            .with_min_inner_size([680.0, 520.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Plate OCR — Training Wizard",
        options,
        Box::new(|_cc| Ok(Box::new(WizardApp::default()))),
    )
}
