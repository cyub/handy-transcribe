//! Long-form chunking: long audio is decoded in chunks cut at pauses, because
//! most models' accuracy degrades past 30-40 s of audio in one run.
//!
//! Ported from Handy's `audio_toolkit/chunking.rs`. Cuts come from per-frame
//! voice activity probabilities. Every duration here is in samples of 16 kHz
//! audio, never in frames. The thresholds are Silero's.

use std::ops::Range;

use anyhow::Result;
use log::{debug, warn};
use once_cell::sync::Lazy;
use regex::Regex;

use crate::audio::WHISPER_SAMPLE_RATE;
use crate::vad::SpeechScorer;

const RATE: usize = WHISPER_SAMPLE_RATE as usize;

const TARGET_SECONDS: f64 = 30.0;
/// Audio up to this many samples is decoded in one piece; longer audio is cut.
const MAX_CHUNK_SAMPLES: usize = 40 * RATE;
const MIN_CHUNK_SAMPLES: usize = 5 * RATE;
/// Live, a cut is only final once this much audio past its window is scored.
const LOOKAHEAD_SAMPLES: usize = MIN_CHUNK_SAMPLES;

// Silero's `get_speech_timestamps` defaults.
const THRESHOLD: f32 = 0.5;
const NEG_THRESHOLD: f32 = THRESHOLD - 0.15;
const MIN_SPEECH_SAMPLES: usize = 250 * RATE / 1000;
const MIN_SILENCE_SAMPLES: usize = 100 * RATE / 1000;
const SPEECH_PAD_SAMPLES: usize = 30 * RATE / 1000;

/// Speech segments, in samples, from per-frame speech probabilities: Silero's
/// `get_speech_timestamps` without a maximum speech length. `samples` is the
/// length of the audio the probabilities score.
fn speech_segments(probs: &[f32], frame_samples: usize, samples: usize) -> Vec<Range<usize>> {
    let mut segments: Vec<Range<usize>> = Vec::new();
    let mut triggered = false;
    let mut start = 0;
    let mut temp_end: Option<usize> = None;
    for (i, &prob) in probs.iter().enumerate() {
        let at = i * frame_samples;
        if prob >= THRESHOLD {
            temp_end = None;
            if !triggered {
                triggered = true;
                start = at;
            }
            continue;
        }
        if prob < NEG_THRESHOLD && triggered {
            let end = *temp_end.get_or_insert(at);
            if at - end < MIN_SILENCE_SAMPLES {
                continue;
            }
            if end - start > MIN_SPEECH_SAMPLES {
                segments.push(start..end);
            }
            triggered = false;
            temp_end = None;
        }
    }
    if triggered && samples.saturating_sub(start) > MIN_SPEECH_SAMPLES {
        segments.push(start..samples);
    }

    for i in 0..segments.len() {
        if i == 0 {
            segments[0].start = segments[0].start.saturating_sub(SPEECH_PAD_SAMPLES);
        }
        if i + 1 == segments.len() {
            segments[i].end = (segments[i].end + SPEECH_PAD_SAMPLES).min(samples);
            continue;
        }
        let silence = segments[i + 1].start - segments[i].end;
        let pad = if silence < 2 * SPEECH_PAD_SAMPLES {
            silence / 2
        } else {
            SPEECH_PAD_SAMPLES
        };
        segments[i].end = (segments[i].end + pad).min(samples);
        segments[i + 1].start = segments[i + 1].start.saturating_sub(pad);
    }
    segments
}

/// A place a chunk may end: the middle of a pause.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CutCandidate {
    cut: usize,
    gap_seconds: f64,
}

/// One candidate in the middle of each pause, lead-in and tail included.
fn cut_candidates(segments: &[Range<usize>], samples: usize) -> Vec<CutCandidate> {
    let mut candidates = Vec::new();
    let mut silence_start = 0;
    for segment in segments
        .iter()
        .cloned()
        .chain(std::iter::once(samples..samples))
    {
        if segment.start > silence_start {
            candidates.push(CutCandidate {
                cut: (silence_start + segment.start) / 2,
                gap_seconds: (segment.start - silence_start) as f64 / RATE as f64,
            });
        }
        silence_start = silence_start.max(segment.end);
    }
    candidates
}

/// Greedy, left to right from `from`: each cut is the best-scoring pause
/// 5-40 s after the previous one, preferring long pauses near 30 s and ones
/// that leave at least 5 s after them. With no pause, it is a hard cut at 40 s,
/// or halfway when that would leave under 5 s. Audio of 40 s or less is never
/// cut. `candidates` must be in order.
fn plan_cuts(candidates: &[CutCandidate], samples: usize, from: usize) -> Vec<usize> {
    let mut cuts = Vec::new();
    let mut start = from;
    let mut first = 0;
    while samples - start > MAX_CHUNK_SAMPLES {
        while first < candidates.len() && candidates[first].cut < start + MIN_CHUNK_SAMPLES {
            first += 1;
        }
        let mut best: Option<(usize, f64)> = None;
        let mut fallback: Option<(usize, f64)> = None;
        for candidate in candidates[first..]
            .iter()
            .take_while(|candidate| candidate.cut <= start + MAX_CHUNK_SAMPLES)
        {
            let position = (candidate.cut - start) as f64 / RATE as f64;
            let off_target = if position > TARGET_SECONDS {
                0.1 * (position - TARGET_SECONDS)
            } else {
                0.05 * (TARGET_SECONDS - position)
            };
            let score = candidate.gap_seconds.min(1.0) - off_target;
            if samples - candidate.cut >= MIN_CHUNK_SAMPLES
                && best.is_none_or(|(_, best)| score > best)
            {
                best = Some((candidate.cut, score));
            }
            if fallback.is_none_or(|(_, fallback)| score > fallback) {
                fallback = Some((candidate.cut, score));
            }
        }
        let hard_cut = if samples - start < MAX_CHUNK_SAMPLES + MIN_CHUNK_SAMPLES {
            start + (samples - start) / 2
        } else {
            start + MAX_CHUNK_SAMPLES
        };
        start = best.or(fallback).map_or(hard_cut, |(cut, _)| cut);
        cuts.push(start);
    }
    cuts
}

/// Makes a fresh scorer for one clip.
pub type MakeScorer = Box<dyn Fn() -> Result<Box<dyn SpeechScorer>> + Send>;

/// Cuts audio into chunks at pauses. A whole clip pushed at once is cut
/// exactly as it is in Handy: the same scores and the same plan.
///
/// If the audio can't be scored (the VAD fails to load or to run), nothing
/// more is cut and the rest is one chunk: chunking only places cuts, so the
/// audio is still decoded, as it would be without chunking.
pub struct Chunker {
    make_scorer: MakeScorer,
    scorer: Option<Box<dyn SpeechScorer>>,
    frame_samples: usize,
    uncut: bool,
    pending: Vec<f32>,
    probs: Vec<f32>,
    start: usize,
    len: usize,
}

impl Chunker {
    pub fn new(make_scorer: MakeScorer) -> Self {
        Self {
            make_scorer,
            scorer: None,
            frame_samples: 0,
            uncut: false,
            pending: Vec::new(),
            probs: Vec::new(),
            start: 0,
            len: 0,
        }
    }

    /// Samples pushed so far, chunked or not.
    #[allow(dead_code)]
    pub fn samples(&self) -> usize {
        self.len
    }

    /// Chunks decoded so far.
    #[allow(dead_code)]
    pub fn decoded(&self) -> usize {
        self.len - self.pending.len()
    }

    /// Adds audio. Returns the chunks whose cuts this made certain, in order.
    /// A whole clip pushed at once is moved in, not copied.
    pub fn push(&mut self, pcm: impl Into<Vec<f32>>) -> Vec<Vec<f32>> {
        let pcm = pcm.into();
        self.len += pcm.len();
        if self.pending.is_empty() {
            self.pending = pcm;
        } else {
            self.pending.extend_from_slice(&pcm);
        }
        if self.uncut || self.len - self.start <= MAX_CHUNK_SAMPLES + LOOKAHEAD_SAMPLES {
            return Vec::new();
        }
        let scored = match self.score(false) {
            Ok(scored) => scored,
            Err(e) => {
                self.stop_cutting(e);
                return Vec::new();
            }
        };
        let segments = speech_segments(&self.probs, self.frame_samples, scored);
        if segments.last().map(|segment| segment.end) != Some(scored) {
            return Vec::new();
        }
        let cuts = plan_cuts(&cut_candidates(&segments, scored), scored, self.start);
        let mut chunks = Vec::new();
        for cut in cuts {
            if scored - self.start <= MAX_CHUNK_SAMPLES + LOOKAHEAD_SAMPLES {
                break;
            }
            chunks.push(self.take_until(cut));
        }
        chunks
    }

    /// Cuts what is left and returns the remaining chunks, in order. Nothing
    /// more is pushed after.
    pub fn finish(&mut self) -> Vec<Vec<f32>> {
        let mut chunks = Vec::new();
        if !self.uncut && self.len - self.start > MAX_CHUNK_SAMPLES {
            match self.score(true) {
                Ok(_) => {
                    let segments = speech_segments(&self.probs, self.frame_samples, self.len);
                    let cuts =
                        plan_cuts(&cut_candidates(&segments, self.len), self.len, self.start);
                    for cut in cuts {
                        chunks.push(self.take_until(cut));
                    }
                }
                Err(e) => self.stop_cutting(e),
            }
        }
        if self.len > self.start {
            chunks.push(self.take_until(self.len));
        }
        chunks
    }

    fn score(&mut self, pad_last: bool) -> Result<usize> {
        if self.scorer.is_none() {
            debug_assert!(self.start == 0 && self.probs.is_empty());
            let scorer = (self.make_scorer)()?;
            self.frame_samples = scorer.frame_samples();
            assert!(self.frame_samples > 0, "VAD frame size must be non-zero");
            self.scorer = Some(scorer);
        }
        let scorer = self.scorer.as_mut().expect("the scorer was made above");
        let frame = self.frame_samples;
        let scored = self.probs.len() * frame;
        let from = scored - self.start;
        let whole = (self.len - scored) / frame * frame;
        if whole > 0 {
            self.probs
                .extend(scorer.score(&self.pending[from..from + whole])?);
        }
        if pad_last && self.len > scored + whole {
            let mut tail = self.pending[from + whole..].to_vec();
            tail.resize(frame, 0.0);
            self.probs.extend(scorer.score(&tail)?);
            return Ok(self.len);
        }
        Ok(scored + whole)
    }

    fn stop_cutting(&mut self, e: anyhow::Error) {
        warn!(
            "Voice activity detection failed at {:.1}s; decoding the rest in one piece: {:#}",
            seconds(self.len),
            e
        );
        self.uncut = true;
    }

    fn take_until(&mut self, cut: usize) -> Vec<f32> {
        debug!(
            "Long-form chunk: {:.1}-{:.1}s",
            seconds(self.start),
            seconds(cut)
        );
        let chunk = self.pending.drain(..cut - self.start).collect();
        self.start = cut;
        chunk
    }
}

/// Scripts written without spaces between words, and CJK punctuation.
static UNSPACED: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Bopomofo}\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}\x{3000}-\x{303f}\x{ff00}-\x{ffef}]",
    )
    .unwrap()
});

fn is_unspaced(c: char) -> bool {
    UNSPACED.is_match(c.encode_utf8(&mut [0; 4]))
}

/// Chunk texts in order: a space between two chunks unless either side is
/// unspaced script.
pub fn join_transcripts<S: AsRef<str>>(texts: &[S]) -> String {
    let mut joined = String::new();
    let mut last: Option<char> = None;
    for text in texts {
        let text = text.as_ref().trim();
        let Some(first) = text.chars().next() else {
            continue;
        };
        if last.is_some_and(|last| !is_unspaced(last) && !is_unspaced(first)) {
            joined.push(' ');
        }
        joined.push_str(text);
        last = text.chars().last();
    }
    joined
}

fn seconds(samples: usize) -> f64 {
    samples as f64 / RATE as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(seconds: f64) -> usize {
        (seconds * RATE as f64).round() as usize
    }

    #[test]
    fn join_transcripts_spaces_latin_scripts() {
        let joined = join_transcripts(&["Hello world", "next chunk", "last"]);
        assert_eq!(joined, "Hello world next chunk last");
    }

    #[test]
    fn join_transcripts_never_spaces_cjk() {
        let joined = join_transcripts(&["你好世界", "这是测试", "结束"]);
        assert_eq!(joined, "你好世界这是测试结束");
    }

    #[test]
    fn join_transcripts_mixed_cjk_and_latin() {
        let joined = join_transcripts(&["Hello 你好", "world 世界"]);
        // "你好" ends unspaced -> no space; "world" ends latin -> space before 世界? no:
        // last="好" (unspaced) so no space after it; then "world" is latin and next "世" is unspaced.
        assert_eq!(joined, "Hello 你好world 世界");
    }

    #[test]
    fn short_audio_is_one_chunk() {
        // A whole clip of 40s or less is never cut.
        let probs = vec![1.0; s(30.0) / 512];
        let segments = speech_segments(&probs, 512, s(30.0));
        let cuts = plan_cuts(&cut_candidates(&segments, s(30.0)), s(30.0), 0);
        assert!(cuts.is_empty());
    }

    #[test]
    fn long_audio_with_pauses_is_cut() {
        // 120s of speech with a 2s pause at the 30s mark.
        let frame = 512;
        let samples = s(120.0);
        let mut probs = vec![1.0; samples / frame];
        let pause_start = s(30.0) / frame;
        let pause_end = s(32.0) / frame;
        for p in &mut probs[pause_start..pause_end] {
            *p = 0.0;
        }
        let segments = speech_segments(&probs, frame, samples);
        let cuts = plan_cuts(&cut_candidates(&segments, samples), samples, 0);
        assert!(!cuts.is_empty(), "long audio must be cut");
        // Cuts land inside the pause, in 5..40s range from the start.
        assert!(cuts[0] > s(5.0) && cuts[0] < s(40.0));
    }
}
