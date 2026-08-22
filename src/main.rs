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

mod backend;
mod data;
mod export;
mod gui;
mod inference;
mod interop;
mod license;
mod model;
mod pydeps;
mod scripts;
mod training;

use std::io::IsTerminal;
use std::path::Path;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "plate-ocr")]
#[command(about = "Train and run OCR on license plate images")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the graphical training wizard (also the default when no command is given).
    Gui,
    /// Train the CRNN OCR model on a plate dataset.
    Train {
        #[arg(long, default_value = "dataset")]
        data_dir: String,
        #[arg(long, default_value_t = 50)]
        epochs: usize,
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value = "checkpoints")]
        output_dir: String,
        /// Path to a pretrained model to fine-tune: a Burn checkpoint (default), a PyTorch
        /// state dict (.pt/.pth), or an ONNX model (.onnx, requires Python + onnx/numpy/torch;
        /// missing packages are installed automatically). If omitted, trains from scratch with
        /// random initialization.
        #[arg(long)]
        pretrained: Option<String>,
        /// Freeze CNN backbone layers during fine-tuning (only train the head: LSTM+linear, or
        /// the FC layers for --architecture conv-ctc).
        #[arg(long, default_value_t = false)]
        freeze_backbone: bool,
        /// Which architecture to train from scratch. Ignored when --pretrained is set: the
        /// architecture is auto-detected from that file instead (and shown in the log/GUI
        /// before training starts).
        #[arg(long, value_enum, default_value = "crnn")]
        architecture: CliArchitecture,
    },
    /// Run inference on a single plate image.
    Infer {
        #[arg(long)]
        model_path: String,
        #[arg(long)]
        image: String,
    },
    /// Export model weights for ONNX conversion.
    Export {
        #[arg(long)]
        model_path: String,
        #[arg(long, default_value = "weights")]
        output_dir: String,
    },
    /// Write the bundled import_onnx.py/export_onnx.py helper scripts to disk.
    /// Useful when all you have is this single compiled binary (e.g. downloaded from a GitHub
    /// release) and you need the standalone Python helpers for `--pretrained *.onnx`
    /// fine-tuning or for `export`'s ONNX conversion step.
    ExtractScripts {
        #[arg(long, default_value = ".")]
        output_dir: String,
    },
    /// Show parts of the GNU GPL: `show w` for the warranty disclaimer, `show c` for the
    /// redistribution conditions (the same text printed by the startup notice's instructions).
    Show {
        part: ShowPart,
    },
}

/// Which part of the license `show` should print.
#[derive(Clone, Copy, clap::ValueEnum)]
enum ShowPart {
    #[value(name = "w")]
    W,
    #[value(name = "c")]
    C,
}

/// CLI spelling of `model::Architecture` (see the `Train` subcommand's `--architecture` flag).
#[derive(Clone, Copy, clap::ValueEnum)]
enum CliArchitecture {
    #[value(name = "crnn")]
    Crnn,
    #[value(name = "conv-ctc")]
    ConvCtc,
}

impl From<CliArchitecture> for model::Architecture {
    fn from(value: CliArchitecture) -> Self {
        match value {
            CliArchitecture::Crnn => model::Architecture::CrnnBiLstm,
            CliArchitecture::ConvCtc => model::Architecture::ConvCtc,
        }
    }
}

fn main() {
    // Default to showing informational progress (dataset loading, architecture
    // detection/selection, Python dependency checks, ...) even when the user hasn't set
    // RUST_LOG themselves — silence-until-a-cryptic-final-error is exactly what makes failures
    // hard to diagnose. RUST_LOG, if set, still takes priority as usual.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // GPLv3's "How to Apply These Terms to Your New Programs": print a short notice when the
    // program does terminal interaction and is being run interactively.
    if std::io::stdout().is_terminal() {
        println!("{}", license::short_notice());
        println!();
    }

    let cli = Cli::parse();

    match cli.command {
        None | Some(Commands::Gui) => {
            if let Err(e) = gui::run_gui() {
                log::error!("Failed to launch GUI: {e}");
                std::process::exit(1);
            }
        }
        Some(Commands::Train {
            data_dir,
            epochs,
            batch_size,
            lr,
            output_dir,
            pretrained,
            freeze_backbone,
            architecture,
        }) => {
            let config = training::train::TrainConfig::new(data_dir)
                .with_num_epochs(epochs)
                .with_batch_size(batch_size)
                .with_learning_rate(lr)
                .with_output_dir(output_dir)
                .with_pretrained(pretrained)
                .with_freeze_backbone(freeze_backbone)
                .with_architecture(architecture.into());

            run_training(config);
        }
        Some(Commands::Infer { model_path, image }) => {
            run_inference(&model_path, &image);
        }
        Some(Commands::Export {
            model_path,
            output_dir,
        }) => {
            run_export(&model_path, &output_dir);
        }
        Some(Commands::ExtractScripts { output_dir }) => {
            run_extract_scripts(&output_dir);
        }
        Some(Commands::Show { part }) => match part {
            ShowPart::W => println!("{}", license::warranty_section()),
            ShowPart::C => println!("{}", license::conditions_section()),
        },
    }
}

fn run_training(config: training::train::TrainConfig) {
    let device = backend::device();
    match training::train::run::<backend::TrainBackend>(config, device) {
        Ok(message) => println!("{message}"),
        Err(message) => {
            log::error!("{message}");
            std::process::exit(1);
        }
    }
}

fn run_inference(model_path: &str, image_path: &str) {
    let device = backend::device();
    match inference::infer::load_model::<backend::InferBackend>(model_path, &device) {
        Ok(model) => {
            println!("Loaded {} model from {model_path}", model.architecture());
            let result = inference::infer::recognize(&model, image_path, &device);
            println!("Recognized plate: {}", result);
        }
        Err(message) => {
            log::error!("Failed to load model '{model_path}': {message}");
            std::process::exit(1);
        }
    }
}

fn run_export(model_path: &str, output_dir: &str) {
    let device = backend::device();
    if let Err(message) = export::onnx::export_weights::<backend::InferBackend>(model_path, output_dir, &device) {
        log::error!("Failed to export '{model_path}': {message}");
        std::process::exit(1);
    }
}

fn run_extract_scripts(output_dir: &str) {
    match scripts::write_all(Path::new(output_dir)) {
        Ok(paths) => {
            for path in paths {
                println!("Wrote {}", path.display());
            }
        }
        Err(e) => {
            log::error!("Failed to extract helper scripts to '{output_dir}': {e}");
            std::process::exit(1);
        }
    }
}
