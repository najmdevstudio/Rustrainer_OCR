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

//! The four-screen training wizard: choose mode -> parameters -> live progress -> result.

use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use eframe::egui::{self, Color32, RichText};
use egui_plot::{Line, Plot, PlotPoints};

use crate::model::Architecture;

use super::params::{Mode, Params};
use super::progress::{GuiEvent, Phase};
use super::worker;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    ChooseMode,
    Parameters,
    Progress,
    Result,
}

/// Top level eframe application implementing the whole wizard flow.
pub struct WizardApp {
    screen: Screen,
    mode: Mode,
    params: Params,
    error: Option<String>,
    show_about: bool,

    events: Option<Receiver<GuiEvent>>,
    logs: Vec<String>,
    train_series: Vec<[f64; 2]>,
    valid_series: Vec<[f64; 2]>,
    train_x: f64,
    valid_x: f64,
    epoch: usize,
    num_epochs: usize,
    fraction: f32,
    /// Which architecture was detected (fine-tuning) or selected (new training), once known —
    /// shown at the top of the progress screen, before training actually starts changing it.
    architecture: Option<String>,
    outcome: Option<Result<String, String>>,
}

impl Default for WizardApp {
    fn default() -> Self {
        Self {
            screen: Screen::ChooseMode,
            mode: Mode::NewTraining,
            params: Params::defaults_for(Mode::NewTraining),
            error: None,
            show_about: false,
            events: None,
            logs: Vec::new(),
            train_series: Vec::new(),
            valid_series: Vec::new(),
            train_x: 0.0,
            valid_x: 0.0,
            epoch: 0,
            num_epochs: 0,
            fraction: 0.0,
            architecture: None,
            outcome: None,
        }
    }
}

const MAX_LOG_LINES: usize = 2000;

impl WizardApp {
    fn drain_events(&mut self) {
        let Some(rx) = &self.events else {
            return;
        };

        while let Ok(event) = rx.try_recv() {
            match event {
                GuiEvent::Log(line) => {
                    self.logs.push(line);
                    if self.logs.len() > MAX_LOG_LINES {
                        let overflow = self.logs.len() - MAX_LOG_LINES;
                        self.logs.drain(0..overflow);
                    }
                }
                GuiEvent::Architecture(label) => {
                    self.architecture = Some(label);
                }
                GuiEvent::Progress {
                    epoch,
                    num_epochs,
                    fraction,
                } => {
                    self.epoch = epoch;
                    self.num_epochs = num_epochs;
                    self.fraction = fraction;
                }
                GuiEvent::Metric { phase, value, .. } => match phase {
                    Phase::Train => {
                        self.train_x += 1.0;
                        self.train_series.push([self.train_x, value]);
                    }
                    Phase::Valid => {
                        self.valid_x += 1.0;
                        self.valid_series.push([self.valid_x, value]);
                    }
                },
                GuiEvent::Finished(outcome) => {
                    self.outcome = Some(outcome);
                    self.screen = Screen::Result;
                }
            }
        }
    }

    fn start_training(&mut self) {
        match self.params.validate(self.mode) {
            Ok(()) => {
                self.error = None;
                let (tx, rx) = mpsc::channel();
                self.events = Some(rx);
                self.logs.clear();
                self.train_series.clear();
                self.valid_series.clear();
                self.train_x = 0.0;
                self.valid_x = 0.0;
                self.epoch = 0;
                self.num_epochs = self.params.epochs;
                self.fraction = 0.0;
                self.architecture = None;
                self.outcome = None;

                let config = self.params.to_train_config();
                worker::spawn_training(config, tx);
                self.screen = Screen::Progress;
            }
            Err(message) => self.error = Some(message),
        }
    }

    fn ui_choose_mode(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(40.0);
            ui.vertical_centered(|ui| {
                ui.heading("Plate OCR — Training Wizard");
                ui.add_space(8.0);
                ui.label("What would you like to do?");
                ui.add_space(30.0);

                let button_size = egui::vec2(300.0, 60.0);
                if ui
                    .add_sized(
                        button_size,
                        egui::Button::selectable(
                            self.mode == Mode::NewTraining,
                            "🆕  New Model Training",
                        ),
                    )
                    .clicked()
                {
                    self.mode = Mode::NewTraining;
                }
                ui.add_space(10.0);
                if ui
                    .add_sized(
                        button_size,
                        egui::Button::selectable(self.mode == Mode::FineTuning, "🔁  Fine-Tuning"),
                    )
                    .clicked()
                {
                    self.mode = Mode::FineTuning;
                }

                ui.add_space(30.0);
                if ui
                    .add_sized([140.0, 36.0], egui::Button::new("Next ▶"))
                    .clicked()
                {
                    self.params = Params::defaults_for(self.mode);
                    self.error = None;
                    self.screen = Screen::Parameters;
                }
            });
        });
    }

    fn ui_parameters(&mut self, ctx: &egui::Context) {
        let mode = self.mode;
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(format!("Parameters — {}", mode.label()));
            ui.label("Defaults are prefilled below; adjust anything before starting.");
            ui.add_space(10.0);

            egui::Grid::new("params_grid")
                .num_columns(3)
                .spacing([8.0, 12.0])
                .show(ui, |ui| {
                    ui.label("Dataset base directory:");
                    ui.text_edit_singleline(&mut self.params.data_dir);
                    if ui.button("Browse…").clicked()
                        && let Some(path) = rfd::FileDialog::new().pick_folder()
                    {
                        self.params.data_dir = path.display().to_string();
                    }
                    ui.end_row();

                    ui.label("Epochs:");
                    ui.add(egui::DragValue::new(&mut self.params.epochs).range(1..=100_000));
                    ui.end_row();

                    ui.label("Batch size:");
                    ui.add(egui::DragValue::new(&mut self.params.batch_size).range(1..=4096));
                    ui.end_row();

                    ui.label("Learning rate:");
                    ui.add(
                        egui::DragValue::new(&mut self.params.learning_rate)
                            .speed(0.0001)
                            .range(0.0..=1.0),
                    );
                    ui.end_row();

                    ui.label("Output (checkpoints) directory:");
                    ui.text_edit_singleline(&mut self.params.output_dir);
                    if ui.button("Browse…").clicked()
                        && let Some(path) = rfd::FileDialog::new().pick_folder()
                    {
                        self.params.output_dir = path.display().to_string();
                    }
                    ui.end_row();

                    if mode == Mode::NewTraining {
                        ui.label("Architecture:");
                        egui::ComboBox::new("architecture_combo", "")
                            .selected_text(self.params.architecture.label())
                            .show_ui(ui, |ui| {
                                for architecture in Architecture::ALL {
                                    ui.selectable_value(
                                        &mut self.params.architecture,
                                        architecture,
                                        architecture.label(),
                                    );
                                }
                            });
                        ui.end_row();

                        ui.label("Freeze CNN backbone:");
                        ui.checkbox(&mut self.params.freeze_backbone, "only train the head");
                        ui.end_row();
                    }

                    if mode == Mode::FineTuning {
                        ui.label("Pretrained model (checkpoint / .pt / .onnx):");
                        ui.text_edit_singleline(&mut self.params.pretrained);
                        if ui.button("Browse…").clicked()
                            && let Some(path) = rfd::FileDialog::new()
                                .add_filter("Model files", &["mpk", "pt", "pth", "onnx"])
                                .add_filter("All files", &["*"])
                                .pick_file()
                        {
                            self.params.pretrained = path.display().to_string();
                        }
                        ui.end_row();

                        ui.label("Architecture:");
                        ui.label("auto-detected from the file above once training starts");
                        ui.end_row();

                        ui.label("Freeze CNN backbone:");
                        ui.checkbox(&mut self.params.freeze_backbone, "only train the head");
                        ui.end_row();
                    }
                });

            ui.add_space(16.0);
            if let Some(error) = &self.error {
                ui.colored_label(Color32::from_rgb(220, 60, 60), error);
                ui.add_space(8.0);
            }

            ui.horizontal(|ui| {
                if ui.button("◀ Back").clicked() {
                    self.screen = Screen::ChooseMode;
                }
                if ui.button("Start Training ▶").clicked() {
                    self.start_training();
                }
            });
        });
    }

    fn ui_progress(&mut self, ctx: &egui::Context) {
        ctx.request_repaint_after(Duration::from_millis(100));

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(format!("Training in progress — {}", self.mode.label()));
            ui.add_space(4.0);
            match &self.architecture {
                Some(architecture) => {
                    ui.label(RichText::new(format!("Architecture: {architecture}")).strong());
                }
                None => {
                    ui.label(RichText::new("Architecture: detecting…").weak());
                }
            }
            ui.add_space(8.0);

            let text = if self.num_epochs > 0 {
                format!(
                    "Epoch {}/{} — {:.0}%",
                    self.epoch,
                    self.num_epochs,
                    self.fraction * 100.0
                )
            } else {
                "Starting…".to_string()
            };
            ui.add(egui::ProgressBar::new(self.fraction).text(text));
            ui.add_space(12.0);

            ui.label("Loss");
            Plot::new("loss_plot")
                .height(220.0)
                .allow_scroll(false)
                .legend(egui_plot::Legend::default())
                .show(ui, |plot_ui| {
                    if !self.train_series.is_empty() {
                        plot_ui.line(
                            Line::new("train loss", PlotPoints::from(self.train_series.clone()))
                                .color(Color32::from_rgb(90, 170, 255)),
                        );
                    }
                    if !self.valid_series.is_empty() {
                        plot_ui.line(
                            Line::new(
                                "validation loss",
                                PlotPoints::from(self.valid_series.clone()),
                            )
                            .color(Color32::from_rgb(255, 170, 60)),
                        );
                    }
                });

            ui.add_space(12.0);
            ui.label("Output");
            egui::Frame::new()
                .fill(Color32::from_rgb(18, 18, 18))
                .inner_margin(8)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .max_height(220.0)
                        .show(ui, |ui| {
                            for line in &self.logs {
                                ui.label(
                                    RichText::new(line)
                                        .monospace()
                                        .color(Color32::from_rgb(80, 240, 120)),
                                );
                            }
                        });
                });
        });
    }

    fn ui_result(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(60.0);
            ui.vertical_centered(|ui| {
                match &self.outcome {
                    Some(Ok(message)) => {
                        ui.label(RichText::new("✅").size(64.0));
                        ui.add_space(10.0);
                        ui.heading("Training completed successfully");
                        ui.add_space(8.0);
                        ui.label(message);
                    }
                    Some(Err(message)) => {
                        ui.label(RichText::new("❌").size(64.0));
                        ui.add_space(10.0);
                        ui.heading("Training failed");
                        ui.add_space(8.0);
                        egui::Frame::new()
                            .fill(Color32::from_rgb(40, 20, 20))
                            .stroke(egui::Stroke::new(1.0_f32, Color32::from_rgb(200, 70, 70)))
                            .inner_margin(10)
                            .show(ui, |ui| {
                                ui.set_width(560.0);
                                egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                                    ui.label(
                                        RichText::new(message)
                                            .monospace()
                                            .color(Color32::from_rgb(255, 150, 150)),
                                    );
                                });
                            });
                    }
                    None => {
                        ui.label("Unknown result.");
                    }
                }

                ui.add_space(30.0);
                if ui
                    .add_sized([140.0, 36.0], egui::Button::new("OK"))
                    .clicked()
                {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
        });
    }

    /// GPLv3's suggested "about box" for GUI programs: the short no-warranty/free-to-redistribute
    /// notice, contact info, and the same warranty/conditions excerpts as the CLI's `show w`/`show c`.
    fn ui_about_window(&mut self, ctx: &egui::Context) {
        if !self.show_about {
            return;
        }

        let mut open = true;
        egui::Window::new(format!("About {}", crate::license::PROGRAM_NAME))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(520.0)
            .show(ctx, |ui| {
                ui.heading(crate::license::PROGRAM_NAME);
                ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                ui.add_space(6.0);
                ui.label(crate::license::COPYRIGHT_LINE);
                ui.add_space(10.0);
                ui.label("This program comes with ABSOLUTELY NO WARRANTY; see \"Warranty\" below.");
                ui.label(
                    "This is free software, and you are welcome to redistribute it under certain conditions; see \"Conditions\" below.",
                );
                ui.add_space(10.0);
                ui.label(format!("Contact: {}", crate::license::CONTACT));
                ui.add_space(12.0);

                egui::CollapsingHeader::new("Warranty (GPLv3 sections 15-17)").show(ui, |ui| {
                    egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                        ui.label(RichText::new(crate::license::warranty_section()).monospace());
                    });
                });
                egui::CollapsingHeader::new("Redistribution Conditions (GPLv3 sections 4-6)").show(
                    ui,
                    |ui| {
                        egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                            ui.label(
                                RichText::new(crate::license::conditions_section()).monospace(),
                            );
                        });
                    },
                );

                ui.add_space(12.0);
                ui.hyperlink_to("Full license text", "https://www.gnu.org/licenses/gpl-3.0.html");
            });
        self.show_about = open;
    }
}

impl eframe::App for WizardApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_events();

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("About").clicked() {
                    self.show_about = true;
                }
            });
        });

        match self.screen {
            Screen::ChooseMode => self.ui_choose_mode(ctx),
            Screen::Parameters => self.ui_parameters(ctx),
            Screen::Progress => self.ui_progress(ctx),
            Screen::Result => self.ui_result(ctx),
        }

        self.ui_about_window(ctx);
    }
}
