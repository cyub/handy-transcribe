//! Orchestration: decode policy (whole vs chunked), run chunks, join text.

use anyhow::Result;
use std::path::Path;
use transcribe_cpp::{RunOptions, Task, Transcript};

use crate::chunking::{join_transcripts, Chunker};
use crate::engine::Engine;
use crate::vad::{SileroVad, SpeechScorer};

pub struct TranscribeOutcome {
    /// Joined raw text from every chunk, before post-processing.
    pub raw_text: String,
    /// The model's detected language (Whisper's LID in auto mode), if all
    /// chunks agreed on one.
    pub detected_language: Option<String>,
}

pub fn build_run_options(
    language: Option<&str>,
    translate: bool,
    vocabulary: Vec<String>,
) -> RunOptions {
    RunOptions {
        task: if translate {
            Task::Translate
        } else {
            Task::Transcribe
        },
        language: language.map(str::to_string),
        target_language: if translate {
            Some("en".to_string())
        } else {
            None
        },
        vocabulary,
        ..Default::default()
    }
}

/// Transcribes `audio` (16 kHz mono f32) with `engine`. Long audio is cut at
/// pauses with the Silero VAD unless the model keeps its accuracy on it or
/// `no_chunk` is set — the same policy as Handy's `batch_decode`.
pub fn transcribe(
    audio: Vec<f32>,
    engine: &mut Engine,
    run_options: &RunOptions,
    no_chunk: bool,
    vad_path: Option<&Path>,
) -> Result<TranscribeOutcome> {
    let chunked = !no_chunk && !engine.streaming && !engine.long_form;

    let transcripts = if chunked {
        let vad_path = vad_path
            .expect("VAD path is required when chunking")
            .to_path_buf();
        let mut chunker = Chunker::new(Box::new(move || {
            let vad = SileroVad::new(vad_path.clone())?;
            Ok(Box::new(vad) as Box<dyn SpeechScorer>)
        }));
        let mut out = Vec::new();
        for chunk in chunker.push(audio) {
            out.push(engine.run(&chunk, run_options)?);
        }
        for chunk in chunker.finish() {
            out.push(engine.run(&chunk, run_options)?);
        }
        out
    } else {
        vec![engine.run(&audio, run_options)?]
    };

    let (raw_text, detected_language) = join_chunks(transcripts);
    Ok(TranscribeOutcome {
        raw_text,
        detected_language,
    })
}

/// One transcription's text from its chunks, in order, and the model's
/// detected language. Chunks that disagree leave it unknown. A single
/// transcript passes through untouched. (Ported from Handy's `join_chunks`.)
fn join_chunks(transcripts: Vec<Transcript>) -> (String, Option<String>) {
    let mut detected = transcripts.iter().map(|t| t.language.clone());
    let language = detected
        .next()
        .flatten()
        .filter(|first| detected.all(|language| language.as_ref() == Some(first)));
    let text = match <[Transcript; 1]>::try_from(transcripts) {
        Ok([one]) => one.text,
        Err(many) => join_transcripts(&many.iter().map(|t| t.text.as_str()).collect::<Vec<_>>()),
    };
    (text, language)
}
