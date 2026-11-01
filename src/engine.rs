//! transcribe-cpp engine: load a GGUF model and run batch transcription.
//! Mirrors the run path Handy's worker process uses.

use anyhow::Result;
use std::path::Path;
use transcribe_cpp::{
    Backend, Device, Feature, Model, ModelOptions, RunOptions, Session, Transcript,
};

pub struct Engine {
    session: Session,
    pub streaming: bool,
    pub long_form: bool,
    #[allow(dead_code)]
    pub translate: bool,
    #[allow(dead_code)]
    pub languages: Vec<String>,
    pub backend_name: String,
    pub device_label: String,
}

impl Engine {
    /// Registers the backends compiled into transcribe-cpp. Must run once,
    /// before any model (including the VAD model) is loaded.
    pub fn init_backends() -> Result<()> {
        transcribe_cpp::init_logging();
        transcribe_cpp::init_backends_default()
            .map_err(|e| anyhow::anyhow!("Failed to initialize transcribe.cpp backends: {e}"))
    }

    /// Loads `path` (a GGUF model). `device_index` hard-selects a compute
    /// device by its `--list-devices` index; `None` uses automatic selection
    /// (GPU when available, CPU otherwise).
    pub fn load(path: &Path, device_index: Option<usize>) -> Result<Self> {
        let (backend, device) = match device_index {
            Some(index) => {
                let device = transcribe_cpp::devices()
                    .into_iter()
                    .find(|d| d.index == Some(index))
                    .ok_or_else(|| {
                        anyhow::anyhow!("No compute device with index {index} (see --list-devices)")
                    })?;
                (Backend::Auto, Some(device))
            }
            None => (Backend::Auto, None),
        };

        let model = Model::load_with(path, &ModelOptions { backend, device })
            .map_err(|e| anyhow::anyhow!("Failed to load model: {e}"))?;
        let session = model
            .session()
            .map_err(|e| anyhow::anyhow!("Failed to create session: {e}"))?;
        let capabilities = model
            .capabilities()
            .map_err(|e| anyhow::anyhow!("Failed to read model capabilities: {e}"))?;

        let backend_name = model.backend();
        let device_label = model
            .device()
            .ok()
            .map(|d| device_label(&d))
            .unwrap_or_else(|| "unknown".to_string());

        Ok(Self {
            session,
            streaming: capabilities.supports_streaming,
            translate: capabilities.supports_translate,
            languages: capabilities.languages.clone(),
            long_form: model.supports(Feature::LongForm),
            backend_name,
            device_label,
        })
    }

    pub fn run(&mut self, pcm: &[f32], options: &RunOptions) -> Result<Transcript> {
        self.session
            .run(pcm, options)
            .map_err(|e| anyhow::anyhow!("transcribe-cpp run failed: {e}"))
    }

    pub fn devices() -> Vec<Device> {
        transcribe_cpp::devices()
    }
}

pub fn device_label(d: &Device) -> String {
    if d.description.is_empty() {
        d.name.clone()
    } else {
        d.description.clone()
    }
}
