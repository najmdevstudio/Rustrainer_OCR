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
