//! handy-transcribe: transcribe a local audio file or a URL, using the same
//! decode → VAD chunking → transcribe-cpp run → join → post-process pipeline
//! as Handy (without the recording/UI parts).

mod audio;
mod chunking;
mod download;
mod engine;
mod model;
mod post;
mod transcribe;
mod vad;

use anyhow::{anyhow, Result};
use clap::Parser;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use audio::{decode_audio_file, WHISPER_SAMPLE_RATE};
use download::{download_to_temp, is_url};
use engine::{device_label, Engine};
use model::{
    default_cache_dir, models, resolve_model, resolve_vad_model, DEFAULT_MODEL, DEFAULT_QUANT,
};
use post::{normalize_transcription_output, remove_filler_words};
use transcribe::{build_run_options, transcribe};

#[derive(Parser, Debug)]
#[command(
    name = "handy-transcribe",
    version,
    about = "Transcribe an audio file or URL, using Handy's offline transcription pipeline"
)]
struct Cli {
    /// Audio file path or http(s):// URL
    input: Option<String>,

    /// Model: a local .gguf path or a Handy catalog model id
    #[arg(short, long, default_value = DEFAULT_MODEL)]
    model: String,

    /// Quantization for catalog models (F32/F16/BF16/Q8_0/Q6_K/Q5_K_M/Q4_K_M)
    #[arg(short, long, default_value = DEFAULT_QUANT)]
    quant: String,

    /// Language hint (e.g. zh/en); default auto-detect
    #[arg(short, long)]
    language: Option<String>,

    /// Translate to English (task=translate)
    #[arg(long)]
    translate: bool,

    /// Compute device index (see --list-devices)
    #[arg(long)]
    device: Option<usize>,

    /// Disable VAD chunking of long audio
    #[arg(long)]
    no_chunk: bool,

    /// Keep filler words (uh/um); by default they are removed
    #[arg(long)]
    keep_fillers: bool,

    /// Custom words file (one per line), injected as model vocabulary
    #[arg(long)]
    custom_words: Option<PathBuf>,

    /// Write output to a file instead of stdout
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Emit machine-readable JSON
    #[arg(long)]
    json: bool,

    /// Model cache directory (default: the platform cache dir, e.g.
    /// ~/Library/Caches/handy-transcribe on macOS, ~/.cache/handy-transcribe on Linux)
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// List Handy's catalog models (with cache status and supported languages) and exit.
    /// Combine with --language to filter, and with --json for machine-readable output
    #[arg(long)]
    list_models: bool,

    /// List transcribe-cpp compute devices and exit
    #[arg(long)]
    list_devices: bool,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    let cache_dir = cli.cache_dir.clone().unwrap_or_else(default_cache_dir);

    if cli.list_models {
        return list_models(&cache_dir, cli.language.as_deref(), cli.json);
    }

    // Must happen once, before any model (including the VAD model) loads.
    Engine::init_backends()?;

    if cli.list_devices {
        list_devices();
        return Ok(());
    }

    let input = cli
        .input
        .as_deref()
        .ok_or_else(|| anyhow!("missing <INPUT> (audio file path or http(s):// URL)"))?;

    // ── Resolve input (URL → temp file, or local path) ─────────────────
    let temp = if is_url(input) {
        Some(download_to_temp(input)?)
    } else {
        None
    };
    let input_path = match &temp {
        Some(t) => t.path().to_path_buf(),
        None => PathBuf::from(input),
    };
    if !input_path.is_file() {
        return Err(anyhow!("Input file not found: {}", input));
    }

    let audio = decode_audio_file(&input_path)?;
    drop(temp);
    let audio_secs = audio.len() as f64 / WHISPER_SAMPLE_RATE as f64;
    if audio.is_empty() {
        return Err(anyhow!("The file has no audio"));
    }
    eprintln!(
        "Decoded {:.1}s of audio ({} samples @ 16 kHz)",
        audio_secs,
        audio.len()
    );

    // ── Resolve + load the model ───────────────────────────────────────
    let model_path = resolve_model(&cli.model, &cli.quant, &cache_dir)?;
    let load_start = Instant::now();
    let mut engine = Engine::load(&model_path, cli.device)?;
    let load_ms = load_start.elapsed().as_millis() as u64;
    eprintln!(
        "Loaded model {} on {} ({}) in {}ms",
        cli.model, engine.backend_name, engine.device_label, load_ms
    );

    // The model is chosen before we know what is in the audio, so a language the model does not
    // list is a likely wrong-model situation: say so instead of silently emitting garbage.
    if let Some(requested) = cli.language.as_deref() {
        if !engine.languages.is_empty() && !language_matches(requested, &engine.languages) {
            eprintln!(
                "warning: {} does not list '{}' among its languages ({}); the transcript may be \
                 wrong — try --list-models --language {}",
                cli.model,
                requested,
                engine.languages.join(", "),
                requested
            );
        }
    }

    // The VAD is only fetched when the model actually needs chunking.
    let chunked = !cli.no_chunk && !engine.streaming && !engine.long_form;
    let vad_path = if chunked {
        Some(resolve_vad_model(&cache_dir)?)
    } else {
        None
    };

    // ── Run transcription ──────────────────────────────────────────────
    let vocabulary = read_custom_words(cli.custom_words.as_deref())?;
    let run_options = build_run_options(cli.language.as_deref(), cli.translate, vocabulary);

    let t_start = Instant::now();
    let outcome = transcribe(
        audio,
        &mut engine,
        &run_options,
        cli.no_chunk,
        vad_path.as_deref(),
    )?;
    let transcribe_ms = t_start.elapsed().as_millis() as u64;

    let language_for_fillers = cli
        .language
        .as_deref()
        .or(outcome.detected_language.as_deref());
    let mut text = remove_filler_words(&outcome.raw_text, language_for_fillers, !cli.keep_fillers);
    text = normalize_transcription_output(&text);

    // ── Emit ───────────────────────────────────────────────────────────
    if cli.json {
        let rtf = if transcribe_ms > 0 {
            audio_secs / (transcribe_ms as f64 / 1000.0)
        } else {
            0.0
        };
        emit(
            &serde_json::json!({
                "model": cli.model,
                "backend": engine.backend_name,
                "device": engine.device_label,
                "audio_secs": audio_secs,
                "load_ms": load_ms,
                "transcribe_ms": transcribe_ms,
                "rtf": rtf,
                "detected_language": outcome.detected_language,
                "text": text,
            })
            .to_string(),
        );
    } else if let Some(path) = cli.output {
        std::fs::write(&path, &text)?;
        eprintln!("Wrote {} to {}", text.len(), path.display());
    } else {
        emit(&text);
    }

    Ok(())
}

/// Writes one line to stdout, treating a closed pipe as a normal end of output.
/// Without this, `handy-transcribe --list-models | head` would panic with
/// "failed printing to stdout: Broken pipe".
fn emit(line: &str) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{line}");
}

/// Primary subtag of a language code, lowercased: "zh-CN" -> "zh", "en_US" -> "en".
fn primary_tag(code: &str) -> String {
    code.split(['-', '_'])
        .next()
        .unwrap_or(code)
        .to_ascii_lowercase()
}

fn language_matches(requested: &str, supported: &[String]) -> bool {
    let want = primary_tag(requested);
    supported.iter().any(|code| primary_tag(code) == want)
}

/// Compact language column: "zh,en,ja+18".
fn languages_label(languages: &[String]) -> String {
    const MAX: usize = 18;
    if languages.is_empty() {
        return "-".to_string();
    }
    let mut label = String::new();
    let mut shown = 0usize;
    for language in languages {
        let candidate = if label.is_empty() {
            language.clone()
        } else {
            format!("{label},{language}")
        };
        if shown > 0 && candidate.len() + 4 > MAX {
            break;
        }
        label = candidate;
        shown += 1;
    }
    let rest = languages.len() - shown;
    if rest > 0 {
        label.push_str(&format!("+{rest}"));
    }
    label
}

fn model_size_bytes(model: &crate::model::ModelEntry) -> u64 {
    model
        .files
        .get(&model.quant)
        .or_else(|| model.files.values().next())
        .map(|file| file.size)
        .unwrap_or(0)
}

fn list_models(cache_dir: &Path, language: Option<&str>, as_json: bool) -> Result<()> {
    let mut entries: Vec<_> = models()
        .iter()
        .filter(|model| match language {
            // Filtering by language is a "which model can transcribe this" question, so skip the
            // helper models (language ID, VAD) that list every language but transcribe nothing.
            Some(requested) => {
                !matches!(model.role.as_str(), "langid" | "vad")
                    && language_matches(requested, &model.languages)
            }
            None => true,
        })
        .collect();

    if let Some(requested) = language {
        if entries.is_empty() {
            eprintln!("warning: no catalog model lists '{requested}'");
        }
        // Smallest first: for a given language that is usually the fastest useful option.
        entries.sort_by_key(|model| model_size_bytes(model));
    }

    if as_json {
        let payload: Vec<_> = entries
            .iter()
            .map(|model| {
                let size_bytes = model_size_bytes(model);
                let filename = model
                    .files
                    .get(&model.quant)
                    .or_else(|| model.files.values().next())
                    .map(|file| file.filename.clone())
                    .unwrap_or_default();
                serde_json::json!({
                    "id": model.id,
                    "name": model.name,
                    "family": model.family,
                    "role": model.role,
                    "quant": model.quant,
                    "description": model.description,
                    "languages": model.languages,
                    "capabilities": {
                        "streaming": model.capabilities.streaming,
                        "translate": model.capabilities.translate,
                        "language_detection": model.capabilities.language_detection,
                        "timestamps": model.capabilities.timestamps,
                    },
                    "license": model.license,
                    "parameters": model.parameters,
                    "size_bytes": size_bytes,
                    "size_mb": (size_bytes as f64 / 1_000_000.0).round() as u64,
                    "cached": cache_dir.join("models").join(filename).is_file(),
                })
            })
            .collect();
        emit(&serde_json::to_string(&payload)?);
        return Ok(());
    }

    emit(&match language {
        Some(requested) => {
            format!("Handy catalog models supporting '{requested}' (✓ = cached), smallest first:")
        }
        None => "Handy catalog models (✓ = cached):".to_string(),
    });
    emit(&format!(
        "  {:<40} {:>8}  {:<9} {:<14} {:<18} {}",
        "id", "size", "role", "family", "languages", "name"
    ));
    for model in &entries {
        let size_mb = model_size_bytes(model) as f64 / 1_000_000.0;
        let filename = model
            .files
            .get(&model.quant)
            .or_else(|| model.files.values().next())
            .map(|file| file.filename.clone())
            .unwrap_or_default();
        let mark = if cache_dir.join("models").join(filename).is_file() {
            "✓"
        } else {
            " "
        };
        emit(&format!(
            "{mark} {:<40} {:>6.0}MB  {:<9} {:<14} {:<18} {}",
            model.id,
            size_mb,
            model.role,
            model.family,
            languages_label(&model.languages),
            model.name
        ));
    }
    Ok(())
}

fn list_devices() {
    let devices = Engine::devices();
    if devices.is_empty() {
        emit("No transcribe-cpp compute devices registered.");
        return;
    }
    emit("transcribe-cpp compute devices:");
    for d in &devices {
        let index = d.index.map(|i| i.to_string()).unwrap_or_else(|| "?".into());
        emit(&format!(
            "  [{}] {} ({:?}, {} MB)",
            index,
            device_label(d),
            d.device_type,
            d.memory_total / 1_000_000
        ));
    }
}

fn read_custom_words(path: Option<&Path>) -> Result<Vec<String>> {
    let Some(path) = path else {
        return Ok(Vec::new());
    };
    let content = std::fs::read_to_string(path)?;
    Ok(content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}
