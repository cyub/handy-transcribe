//! Silero VAD, the same detector Handy uses to score audio for long-form
//! chunking. Ported from Handy's `audio_toolkit/vad/silero.rs` and
//! `audio_toolkit/vad/mod.rs`.

use anyhow::Result;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use transcribe_cpp::{Backend, Model, ModelOptions, VadSession, VadSessionOptions};

use crate::audio::WHISPER_SAMPLE_RATE;

/// One speech probability per frame; what long-form chunking cuts at.
pub trait SpeechScorer: Send {
    /// One probability per frame of `frames`, which must be whole frames.
    fn score(&mut self, frames: &[f32]) -> Result<Vec<f32>>;
    fn frame_samples(&self) -> usize;
}

pub struct SileroVad {
    // `VadSession` is Send but not Sync; only ever reached through `get_mut`,
    // so it is never locked and never poisoned.
    session: Mutex<VadSession>,
    frame_samples: usize,
}

impl SileroVad {
    /// Backends must already be initialized (the caller runs
    /// `transcribe_cpp::init_backends_default()` once at startup). The VAD
    /// model itself always runs on the CPU backend.
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model = Model::load_with(
            model_path.as_ref(),
            &ModelOptions {
                backend: Backend::Cpu,
                device: None,
            },
        )
        .map_err(|e| anyhow::anyhow!("Failed to load VAD model: {e}"))?;

        let info = model
            .vad_info()
            .map_err(|e| anyhow::anyhow!("Model does not support VAD: {e}"))?;
        if info.sample_rate != WHISPER_SAMPLE_RATE as i32 || info.frame_samples <= 0 {
            anyhow::bail!(
                "unsupported VAD model: {} Hz, {} samples per frame",
                info.sample_rate,
                info.frame_samples
            );
        }

        // The session keeps the native model alive on its own.
        let session = model
            .vad_session_with(&VadSessionOptions { n_threads: 1 })
            .map_err(|e| anyhow::anyhow!("Failed to create VAD session: {e}"))?;

        Ok(Self {
            session: Mutex::new(session),
            frame_samples: info.frame_samples as usize,
        })
    }
}

impl SpeechScorer for SileroVad {
    fn score(&mut self, frames: &[f32]) -> Result<Vec<f32>> {
        // `usize::is_multiple_of` is stable since Rust 1.87; keep the MSRV at 1.85.
        if frames.len() % self.frame_samples != 0 {
            anyhow::bail!(
                "expected whole {}-sample frames, got {} samples",
                self.frame_samples,
                frames.len()
            );
        }
        let result = self
            .session
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .stream_feed(frames)
            .map_err(|e| anyhow::anyhow!("Silero VAD error: {e}"))?;

        let expected = frames.len() / self.frame_samples;
        if result.probs.len() != expected {
            anyhow::bail!(
                "Silero VAD returned {} probabilities for {expected} frames",
                result.probs.len()
            );
        }
        Ok(result.probs)
    }

    fn frame_samples(&self) -> usize {
        self.frame_samples
    }
}
