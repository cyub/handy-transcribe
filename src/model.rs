//! Model resolution: Handy's model catalog (embedded) + Hugging Face download.

use anyhow::{anyhow, Context, Result};
use once_cell::sync::Lazy;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_MODEL: &str = "Qwen3-ASR-0.6B";
pub const DEFAULT_QUANT: &str = "Q8_0";
pub const VAD_MODEL_ID: &str = "silero-vad-v6.2";

const CATALOG_JSON: &str = include_str!("../catalog.json");

#[derive(Deserialize)]
struct Catalog {
    models: Vec<ModelEntry>,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
pub struct ModelEntry {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub repository: String,
    pub revision: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub quant: String,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub parameters: Option<String>,
    #[serde(default)]
    pub languages: Vec<String>,
    pub capabilities: Capabilities,
    #[serde(default)]
    pub curated: bool,
    #[serde(default)]
    pub files: HashMap<String, ModelFile>,
}

#[derive(Deserialize, Clone)]
#[allow(dead_code)]
pub struct Capabilities {
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub translate: bool,
    #[serde(default, rename = "languageDetection")]
    pub language_detection: bool,
    #[serde(default)]
    pub timestamps: bool,
}

#[derive(Deserialize, Clone)]
pub struct ModelFile {
    pub filename: String,
    pub size: u64,
    pub sha256: String,
}

fn catalog() -> &'static Catalog {
    static CATALOG: Lazy<Catalog> =
        Lazy::new(|| serde_json::from_str(CATALOG_JSON).expect("embedded catalog.json must parse"));
    &CATALOG
}

pub fn models() -> &'static [ModelEntry] {
    &catalog().models
}

pub fn find_model(id: &str) -> Option<&'static ModelEntry> {
    catalog().models.iter().find(|m| m.id == id)
}

pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("handy-transcribe")
}

/// Resolves a `--model` argument to a local GGUF path, downloading it from
/// Hugging Face when it is a catalog model id and not already cached.
pub fn resolve_model(model_arg: &str, quant: &str, cache_dir: &Path) -> Result<PathBuf> {
    let arg_path = Path::new(model_arg);
    if arg_path.is_file() {
        return Ok(arg_path.to_path_buf());
    }
    if arg_path.exists() {
        return Err(anyhow!("Model path is not a file: {}", model_arg));
    }

    let entry = find_model(model_arg)
        .ok_or_else(|| anyhow!("Unknown model id '{}' (see --list-models)", model_arg))?;
    let file = pick_file(entry, quant)?;
    let dest = cache_dir.join("models").join(&file.filename);
    if dest.is_file() && dest_ok(&dest, file.size) {
        return Ok(dest);
    }
    download_file(entry, file, &dest)
}

/// Resolves the VAD model to a local path (downloads on demand).
pub fn resolve_vad_model(cache_dir: &Path) -> Result<PathBuf> {
    let entry = find_model(VAD_MODEL_ID)
        .ok_or_else(|| anyhow!("VAD model '{}' missing from catalog", VAD_MODEL_ID))?;
    let file = pick_file(entry, "F32")?;
    let dest = cache_dir.join("models").join(&file.filename);
    if dest.is_file() && dest_ok(&dest, file.size) {
        return Ok(dest);
    }
    download_file(entry, file, &dest)
}

fn pick_file<'a>(entry: &'a ModelEntry, quant: &str) -> Result<&'a ModelFile> {
    if let Some(file) = entry.files.get(quant) {
        return Ok(file);
    }
    if let Some(file) = entry.files.get(&entry.quant) {
        return Ok(file);
    }
    entry
        .files
        .values()
        .next()
        .ok_or_else(|| anyhow!("Model '{}' has no downloadable files", entry.id))
}

fn dest_ok(dest: &Path, expected_size: u64) -> bool {
    fs::metadata(dest)
        .map(|m| m.len() == expected_size)
        .unwrap_or(false)
}

/// Base URL for model downloads.
///
/// Defaults to the public Hugging Face Hub. Override it with
/// `HANDY_HF_ENDPOINT` (takes precedence) or the conventional `HF_ENDPOINT`
/// when you are behind a proxy or prefer a mirror, e.g.
/// `HF_ENDPOINT=https://hf-mirror.com`.
fn hf_endpoint() -> String {
    std::env::var("HANDY_HF_ENDPOINT")
        .or_else(|_| std::env::var("HF_ENDPOINT"))
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

fn download_file(entry: &ModelEntry, file: &ModelFile, dest: &Path) -> Result<PathBuf> {
    let url = format!(
        "{}/{}/resolve/{}/{}",
        hf_endpoint(),
        entry.repository,
        entry.revision,
        file.filename
    );
    fs::create_dir_all(dest.parent().expect("dest has a parent"))?;
    let partial = dest.with_extension("gguf.partial");

    eprintln!(
        "Downloading {} ({:.0} MB)...",
        file.filename,
        file.size as f64 / 1_000_000.0
    );
    let mut response = reqwest::blocking::get(&url)
        .with_context(|| format!("Failed to download {}", url))?
        .error_for_status()
        .with_context(|| format!("Download returned an error for {}", url))?;

    let mut out = fs::File::create(&partial)?;
    std::io::copy(&mut response, &mut out)?;
    drop(out);

    let bytes = fs::read(&partial)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if !digest.eq_ignore_ascii_case(&file.sha256) {
        let _ = fs::remove_file(&partial);
        return Err(anyhow!(
            "SHA-256 mismatch for {}: expected {}, got {}",
            file.filename,
            file.sha256,
            digest
        ));
    }
    fs::rename(&partial, dest)?;
    eprintln!("Saved {}", dest.display());
    Ok(dest.to_path_buf())
}
