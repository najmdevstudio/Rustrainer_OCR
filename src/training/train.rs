use std::sync::mpsc::Sender;

use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::module::Module;
use burn::nn::loss::CTCLossConfig;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::{
    InferenceStep, Learner, SequenceOutput, SupervisedTraining, TrainOutput, TrainStep,
};

use crate::data::dataset::{PlateBatch, PlateBatcher, PlateDataset};
use crate::gui::progress::{GuiEvent, GuiRenderer};
use crate::model::crnn::{CrnnOcr, CrnnOcrConfig};


#[derive(Config, Debug)]
pub struct TrainConfig {
    pub data_dir: String,
    #[config(default = 50)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = "\"checkpoints\".to_string()")]
    pub output_dir: String,
    /// Path to a pretrained model to fine-tune from: a Burn checkpoint, a PyTorch `.pt`/`.pth`
    /// state dict, or an `.onnx` model (format auto-detected by extension, see
    /// `crate::interop::load_pretrained`). If `None`, trains from scratch.
    pub pretrained: Option<String>,
    /// If true, freeze CNN backbone (conv/bn layers) and only train LSTM + linear head.
    #[config(default = false)]
    pub freeze_backbone: bool,
}

#[derive(Module, Debug)]
pub struct OcrTrainModule<B: Backend> {
    pub model: CrnnOcr<B>,
}

impl<B: Backend> OcrTrainModule<B> {
    pub fn new(model: CrnnOcr<B>) -> Self {
        Self { model }
    }

    fn forward_step(&self, batch: PlateBatch<B>) -> SequenceOutput<B> {
        let log_probs = self.model.forward(batch.images);
        // log_probs: [time, batch, classes]

        let ctc = CTCLossConfig::new()
            .with_blank(0)
            .with_zero_infinity(true)
            .init();

        let per_sample_loss = ctc.forward(
            log_probs.clone(),
            batch.targets.clone(),
            batch.input_lengths,
            batch.target_lengths,
        );
        // per_sample_loss: [batch]
        let loss = per_sample_loss.mean().unsqueeze();

        // Transpose log_probs to [batch, time, classes] for SequenceOutput
        let logits = log_probs.swap_dims(0, 1);

        SequenceOutput::new(loss, logits, None, batch.targets)
    }
}


impl<B: AutodiffBackend> TrainStep for OcrTrainModule<B> {
    type Input = PlateBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let output = self.forward_step(batch);
        let loss = output.loss.clone();
        TrainOutput::new(self, loss.backward(), output)
    }
}

impl<B: Backend> InferenceStep for OcrTrainModule<B> {
    type Input = PlateBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, batch: Self::Input) -> Self::Output {
        self.forward_step(batch)
    }
}

/// Trains (or fine-tunes) the model. Equivalent to [`run_with_progress`] with no progress
/// channel, which is what the CLI uses.
pub fn run<B: AutodiffBackend>(config: TrainConfig, device: B::Device) {
    run_with_progress::<B>(config, device, None);
}

/// Same as [`run`], but when `events` is `Some`, progress, metrics and log lines are also
/// streamed through the channel (used by the GUI wizard to display live feedback) in addition
/// to the normal `log` output.
pub fn run_with_progress<B: AutodiffBackend>(
    config: TrainConfig,
    device: B::Device,
    events: Option<Sender<GuiEvent>>,
) {
    let emit_log = |message: String| {
        log::info!("{message}");
        if let Some(sender) = &events {
            let _ = sender.send(GuiEvent::Log(message));
        }
    };

    let train_dir = format!("{}/train", config.data_dir);
    let valid_dir = format!("{}/valid", config.data_dir);

    emit_log(format!("Loading dataset from {}", config.data_dir));
    let train_dataset = PlateDataset::new(&train_dir);
    let valid_dataset = PlateDataset::new(&valid_dir);
    emit_log(format!(
        "Loaded {} training samples and {} validation samples",
        train_dataset.len(),
        valid_dataset.len()
    ));

    let batcher = PlateBatcher;

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(42)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    let model_config = CrnnOcrConfig::new();
    let model = match &config.pretrained {
        Some(pretrained_path) => {
            emit_log(format!("Loading pretrained model from: {}", pretrained_path));
            let model = crate::interop::load_pretrained::<B>(
                pretrained_path,
                &model_config,
                &device,
                |line| emit_log(line),
            )
            .expect("Failed to load pretrained model");
            if config.freeze_backbone {
                emit_log(
                    "Freezing CNN backbone — only LSTM and linear head will be trained"
                        .to_string(),
                );
                model.freeze_backbone()
            } else {
                emit_log("Fine-tuning all layers".to_string());
                model
            }
        }
        None => {
            emit_log("Training from scratch with random initialization".to_string());
            model_config.init::<B>(&device)
        }
    };
    let train_module = OcrTrainModule::new(model);

    let mut training =
        SupervisedTraining::new(&config.output_dir, dataloader_train, dataloader_valid)
            .metrics((LossMetric::new(),))
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(config.num_epochs);

    if let Some(sender) = &events {
        training = training.renderer(GuiRenderer::new(sender.clone()));
    }

    let training = training.summary();

    emit_log(format!("Starting training for {} epoch(s)...", config.num_epochs));

    let result = training.launch(Learner::new(
        train_module,
        AdamConfig::new().init(),
        config.learning_rate,
    ));

    let final_path = format!("{}/plate_ocr_final", config.output_dir);
    // Save the inner `CrnnOcr` directly (not the `OcrTrainModule` wrapper): `infer`, `export`
    // and the pretrained/fine-tuning loaders (`crate::interop`) all expect a checkpoint whose
    // record is `CrnnOcr`'s own, not one nested under an extra `model` field.
    result
        .model
        .model
        .save_file(&final_path, &CompactRecorder::new())
        .expect("Failed to save trained model");

    emit_log(format!("Training complete. Model saved to {}", final_path));
}

// These tests run a tiny real training loop end-to-end on the CPU (`ndarray`) backend, so they
// only compile/run when the `cpu` feature is enabled (`cargo test --features cpu`).
#[cfg(all(test, feature = "cpu"))]
mod progress_tests {
    use super::*;
    use image::{GrayImage, Luma};
    use std::fs;
    use std::sync::mpsc;

    type TestBackend = burn::backend::Autodiff<burn::backend::NdArray>;

    /// Writes a tiny (train + valid) dataset under `base`, just large enough to run one real
    /// training step.
    fn write_tiny_dataset(base: &std::path::Path) {
        for split in ["train", "valid"] {
            let images_dir = base.join(split).join("images");
            fs::create_dir_all(&images_dir).expect("failed to create images dir");

            let mut csv = String::from("image_name,label\n");
            for i in 0..4 {
                let name = format!("plate_{i}.png");
                let image = GrayImage::from_pixel(
                    crate::data::dataset::IMG_WIDTH as u32,
                    crate::data::dataset::IMG_HEIGHT as u32,
                    Luma([128]),
                );
                image
                    .save(images_dir.join(&name))
                    .expect("failed to write dummy image");
                csv.push_str(&format!("{name},AB{i}12\n"));
            }
            fs::write(base.join(split).join("labels.csv"), csv).expect("failed to write labels.csv");
        }
    }

    #[test]
    fn run_with_progress_streams_logs_progress_and_metrics() {
        let data_dir = std::env::temp_dir().join(format!(
            "plate_ocr_gui_progress_test_{}",
            std::process::id()
        ));
        write_tiny_dataset(&data_dir);
        let output_dir = data_dir.join("out");

        let config = TrainConfig::new(data_dir.to_string_lossy().to_string())
            .with_num_epochs(1)
            .with_batch_size(2)
            .with_num_workers(1)
            .with_output_dir(output_dir.to_string_lossy().to_string());

        let (tx, rx) = mpsc::channel();
        let device = Default::default();

        run_with_progress::<TestBackend>(config, device, Some(tx));

        let (mut saw_log, mut saw_progress, mut saw_metric) = (false, false, false);
        while let Ok(event) = rx.try_recv() {
            match event {
                GuiEvent::Log(_) => saw_log = true,
                GuiEvent::Progress { fraction, .. } => {
                    saw_progress = true;
                    assert!((0.0..=1.0).contains(&fraction));
                }
                GuiEvent::Metric { value, .. } => {
                    saw_metric = true;
                    assert!(value.is_finite());
                }
                GuiEvent::Finished(_) => {
                    panic!("run_with_progress must not send Finished itself (that's the GUI worker's job)");
                }
            }
        }

        assert!(saw_log, "expected at least one Log event");
        assert!(saw_progress, "expected at least one Progress event");
        assert!(saw_metric, "expected at least one Metric (loss) event");

        let _ = fs::remove_dir_all(&data_dir);
    }

    /// End-to-end regression check for `crate::interop::load_pretrained`'s dispatch: fine-tuning
    /// from a Burn checkpoint (the pre-existing, non-PyTorch/ONNX behavior) must keep working
    /// once routed through it.
    #[test]
    fn fine_tuning_from_a_burn_checkpoint_still_works() {
        let data_dir = std::env::temp_dir().join(format!(
            "plate_ocr_finetune_regression_test_{}",
            std::process::id()
        ));
        write_tiny_dataset(&data_dir);
        let output_dir = data_dir.join("out");
        let device = Default::default();

        let base_config = TrainConfig::new(data_dir.to_string_lossy().to_string())
            .with_num_epochs(1)
            .with_batch_size(2)
            .with_num_workers(1)
            .with_output_dir(output_dir.to_string_lossy().to_string());

        run::<TestBackend>(base_config, device);
        let checkpoint = output_dir.join("plate_ocr_final");
        assert!(
            std::path::Path::new(&format!("{}.mpk", checkpoint.display())).exists()
                || checkpoint.exists(),
            "expected a checkpoint at {}",
            checkpoint.display()
        );

        let finetune_config = TrainConfig::new(data_dir.to_string_lossy().to_string())
            .with_num_epochs(1)
            .with_batch_size(2)
            .with_num_workers(1)
            .with_output_dir(output_dir.to_string_lossy().to_string())
            .with_pretrained(Some(checkpoint.to_string_lossy().to_string()));

        // Must not panic: this exercises `interop::load_pretrained`'s Burn-checkpoint fallback
        // branch exactly as the CLI/GUI do.
        run::<TestBackend>(finetune_config, Default::default());

        let _ = fs::remove_dir_all(&data_dir);
    }
}
