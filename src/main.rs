mod backend;
mod data;
mod export;
mod gui;
mod inference;
mod interop;
mod model;
mod scripts;
mod training;

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
        /// state dict (.pt/.pth), or an ONNX model (.onnx, requires Python + onnx/numpy/torch).
        /// If omitted, trains from scratch with random initialization.
        #[arg(long)]
        pretrained: Option<String>,
        /// Freeze CNN backbone layers during fine-tuning (only train LSTM + linear head).
        #[arg(long, default_value_t = false)]
        freeze_backbone: bool,
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
}

fn main() {
    env_logger::init();
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
        }) => {
            let config = training::train::TrainConfig::new(data_dir)
                .with_num_epochs(epochs)
                .with_batch_size(batch_size)
                .with_learning_rate(lr)
                .with_output_dir(output_dir)
                .with_pretrained(pretrained)
                .with_freeze_backbone(freeze_backbone);

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
    }
}

fn run_training(config: training::train::TrainConfig) {
    let device = backend::device();
    training::train::run::<backend::TrainBackend>(config, device);
}

fn run_inference(model_path: &str, image_path: &str) {
    let device = backend::device();
    let model = inference::infer::load_model::<backend::InferBackend>(model_path, &device);
    let result = inference::infer::recognize(&model, image_path, &device);
    println!("Recognized plate: {}", result);
}

fn run_export(model_path: &str, output_dir: &str) {
    let device = backend::device();
    export::onnx::export_weights::<backend::InferBackend>(model_path, output_dir, &device);
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
