//! The four-screen training wizard: choose mode -> parameters -> live progress -> result.

use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use eframe::egui::{self, Color32, RichText};
use egui_plot::{Line, Plot, PlotPoints};

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

    events: Option<Receiver<GuiEvent>>,
    logs: Vec<String>,
    train_series: Vec<[f64; 2]>,
    valid_series: Vec<[f64; 2]>,
    train_x: f64,
    valid_x: f64,
    epoch: usize,
    num_epochs: usize,
    fraction: f32,
    outcome: Option<Result<String, String>>,
}

impl Default for WizardApp {
    fn default() -> Self {
        Self {
            screen: Screen::ChooseMode,
            mode: Mode::NewTraining,
            params: Params::defaults_for(Mode::NewTraining),
            error: None,
            events: None,
            logs: Vec::new(),
            train_series: Vec::new(),
            valid_series: Vec::new(),
            train_x: 0.0,
            valid_x: 0.0,
            epoch: 0,
            num_epochs: 0,
            fraction: 0.0,
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

                        ui.label("Freeze CNN backbone:");
                        ui.checkbox(
                            &mut self.params.freeze_backbone,
                            "only train LSTM + linear head",
                        );
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
                        egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                            ui.label(message);
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
}

impl eframe::App for WizardApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_events();

        match self.screen {
            Screen::ChooseMode => self.ui_choose_mode(ctx),
            Screen::Parameters => self.ui_parameters(ctx),
            Screen::Progress => self.ui_progress(ctx),
            Screen::Result => self.ui_result(ctx),
        }
    }
}
