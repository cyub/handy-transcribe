//! Audio input from a URL: download to a temporary file that is removed on
//! drop, keeping the original extension so symphonia's format probe can use it.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

pub fn is_url(input: &str) -> bool {
    input.starts_with("http://") || input.starts_with("https://")
}

pub struct TempFile(PathBuf);

impl TempFile {
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Downloads `url` to a fresh temp file, returning a guard that deletes it.
pub fn download_to_temp(url: &str) -> Result<TempFile> {
    eprintln!("Downloading audio from {}", url);
    let mut response = reqwest::blocking::get(url)
        .with_context(|| format!("Failed to download {}", url))?
        .error_for_status()
        .with_context(|| format!("Download returned an error for {}", url))?;

    let filename = filename_from_url(url);
    let dest = std::env::temp_dir().join(format!(
        "handy-transcribe-{}-{}",
        std::process::id(),
        filename
    ));

    let mut out = fs::File::create(&dest)?;
    std::io::copy(&mut response, &mut out)?;
    drop(out);

    Ok(TempFile(dest))
}

fn filename_from_url(url: &str) -> String {
    let path = url
        .split(['?', '#'])
        .next()
        .unwrap_or(url)
        .rsplit('/')
        .next()
        .unwrap_or("");
    if path.is_empty() {
        "input.bin".to_string()
    } else {
        path.to_string()
    }
}
