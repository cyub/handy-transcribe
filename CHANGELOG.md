# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation

- READMEs now invite community translations and explain that a quick-start level translation is
  enough, since the reference tables change every release.

- README now leads with a project logo, a language switcher (English / 简体中文) and a badge row fed by
  live crates.io version, download count and license, plus CI status and the latest release.

## [0.1.1] - 2026-11-01

### Fixed

- Output no longer panics with `failed printing to stdout: Broken pipe` when the reader closes early
  (for example `handy-transcribe --list-models | head`).

## [0.1.0] - 2026-10-31

### Added

- First public release: `handy-transcribe <INPUT>` transcribes a local audio/video file or a direct
  `http(s)` media URL using Handy's pipeline (symphonia decode → mono/16 kHz resample → Silero VAD
  chunking → `transcribe-cpp` inference → CJK-aware join → post-process).
- Embedded model catalog (76 models, 21 families) with on-demand Hugging Face download and SHA-256
  verification; `--list-models` shows cache status.
- Metal backend on macOS, CPU backend elsewhere; `--list-devices` reports what `transcribe-cpp`
  registered.
- Flags: `--model`, `--quant`, `--language`, `--translate`, `--device`, `--no-chunk`,
  `--keep-fillers`, `--custom-words`, `--output`, `--json`, `--cache-dir`.
- `HF_ENDPOINT` / `HANDY_HF_ENDPOINT` support for model mirrors and corporate proxies.
- `cargo binstall` metadata, prebuilt release tarballs, and a Homebrew tap.
