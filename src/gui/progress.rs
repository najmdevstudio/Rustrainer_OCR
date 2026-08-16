//! Bridges Burn's training loop (progress + metrics) to the GUI through a plain channel,
//! instead of Burn's own CLI/TUI dashboard.

use std::collections::HashMap;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use burn::train::LearnerSummary;
use burn::train::metric::{MetricDefinition, MetricId};
use burn::train::renderer::{
    EvaluationName, EvaluationProgress, MetricState, MetricsRenderer, MetricsRendererEvaluation,
    MetricsRendererTraining, ProgressType, TrainingProgress,
};

/// Which phase (training or validation) a metric value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Train,
    Valid,
}

/// Events streamed from the (background-threaded) training run to the GUI event loop.
#[derive(Debug, Clone)]
pub enum GuiEvent {
    /// A human readable line to append to the "terminal" output pane.
    Log(String),
    /// Overall progress across every epoch.
    Progress {
        epoch: usize,
        num_epochs: usize,
        /// Overall completion, in the `0.0..=1.0` range.
        fraction: f32,
    },
    /// A numeric metric value (e.g. loss) reported by the training loop, used to feed the graph.
    Metric {
        phase: Phase,
        #[allow(dead_code)]
        name: String,
        value: f64,
    },
    /// Sent exactly once, when the background training thread terminates.
    Finished(Result<String, String>),
}

/// Adapts Burn's [`MetricsRenderer`] to forward progress/metrics through a channel instead of
/// printing to a terminal dashboard, so the GUI wizard can display them live.
pub struct GuiRenderer {
    sender: Arc<Mutex<Sender<GuiEvent>>>,
    metric_names: HashMap<MetricId, String>,
    last_epoch_logged: usize,
}

impl GuiRenderer {
    pub fn new(sender: Sender<GuiEvent>) -> Self {
        Self {
            sender: Arc::new(Mutex::new(sender)),
            metric_names: HashMap::new(),
            last_epoch_logged: 0,
        }
    }

    fn send(&self, event: GuiEvent) {
        if let Ok(sender) = self.sender.lock() {
            let _ = sender.send(event);
        }
    }

    fn metric_name(&self, id: &MetricId) -> String {
        self.metric_names
            .get(id)
            .cloned()
            .unwrap_or_else(|| "metric".to_string())
    }

    /// Reports overall progress (based on the training phase only, epochs + current epoch
    /// iteration) and logs a line whenever a new epoch starts.
    fn report_progress(&mut self, item: &TrainingProgress) {
        let epoch = item.global_progress.items_processed;
        let num_epochs = item.global_progress.items_total.max(1);

        let epoch_fraction = match &item.progress {
            Some(local) if local.items_total > 0 => {
                local.items_processed as f32 / local.items_total as f32
            }
            _ => 0.0,
        };
        let fraction = (epoch.saturating_sub(1) as f32 + epoch_fraction) / num_epochs as f32;

        self.send(GuiEvent::Progress {
            epoch,
            num_epochs,
            fraction: fraction.clamp(0.0, 1.0),
        });

        if epoch != self.last_epoch_logged && epoch > 0 {
            self.last_epoch_logged = epoch;
            self.send(GuiEvent::Log(format!("Epoch {epoch}/{num_epochs} started")));
        }
    }
}

impl MetricsRendererTraining for GuiRenderer {
    fn update_train(&mut self, state: MetricState) {
        if let MetricState::Numeric(entry, numeric) = state {
            let name = self.metric_name(&entry.metric_id);
            self.send(GuiEvent::Metric {
                phase: Phase::Train,
                name,
                value: numeric.current(),
            });
        }
    }

    fn update_valid(&mut self, state: MetricState) {
        if let MetricState::Numeric(entry, numeric) = state {
            let name = self.metric_name(&entry.metric_id);
            self.send(GuiEvent::Metric {
                phase: Phase::Valid,
                name,
                value: numeric.current(),
            });
        }
    }

    fn render_train(&mut self, item: TrainingProgress, _progress_indicators: Vec<ProgressType>) {
        self.report_progress(&item);
    }

    fn render_valid(&mut self, item: TrainingProgress, _progress_indicators: Vec<ProgressType>) {
        // Validation is comparatively quick; we only surface its loss values (via
        // `update_valid`) and keep the overall progress bar driven by the training phase.
        let _ = item;
    }

    fn on_train_end(
        &mut self,
        _summary: Option<LearnerSummary>,
    ) -> Result<(), Box<dyn core::error::Error>> {
        Ok(())
    }
}

impl MetricsRendererEvaluation for GuiRenderer {
    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {}

    fn render_test(&mut self, _item: EvaluationProgress, _progress_indicators: Vec<ProgressType>) {}
}

impl MetricsRenderer for GuiRenderer {
    fn manual_close(&mut self) {}

    fn register_metric(&mut self, definition: MetricDefinition) {
        self.metric_names.insert(definition.metric_id, definition.name);
    }
}
