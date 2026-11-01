<p align="center">
  <img src="assets/logo.svg" width="72" height="72" alt="handy-transcribe logo">
</p>

<h1 align="center">handy-transcribe</h1>

<p align="center">
  <strong>English</strong> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://crates.io/crates/handy-transcribe">
    <img src="https://img.shields.io/crates/v/handy-transcribe?style=flat&logo=rust" alt="crates.io">
  </a>
  <a href="https://crates.io/crates/handy-transcribe">
    <img src="https://img.shields.io/crates/d/handy-transcribe?style=flat" alt="downloads">
  </a>
  <a href="https://github.com/cyub/handy-transcribe/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/cyub/handy-transcribe/ci.yml?style=flat&logo=githubactions&logoColor=white" alt="CI">
  </a>
  <a href="https://github.com/cyub/handy-transcribe/releases">
    <img src="https://img.shields.io/github/v/release/cyub/handy-transcribe?style=flat&logo=github" alt="release">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/handy-transcribe?style=flat" alt="license">
  </a>
  <img src="https://img.shields.io/badge/rust-1.85%2B-dea584?style=flat&logo=rust" alt="MSRV">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux-555?style=flat" alt="Platform">
</p>

<p align="center">Offline speech-to-text for audio files and URLs — one Rust binary, no API keys, no audio leaving your machine.</p>

`handy-transcribe` is a small command-line front end for the same pipeline that powers
[Handy](https://github.com/cjpais/Handy): `decode → VAD chunking → transcribe.cpp (ggml) inference →
join → post-process`. Recording, UI, and clipboard plumbing are stripped away, leaving a headless CLI
you can script, batch, or wire into an agent.

> **Unofficial.** This is a community CLI built on top of Handy's pipeline and the
> [`transcribe-cpp`](https://github.com/handy-computer/transcribe.cpp) engine. It is not affiliated
> with, or endorsed by, the Handy project or handy-computer.

## Why another transcription CLI?

- **Fully local.** Models run on your hardware (Metal on macOS, CPU elsewhere). Nothing is uploaded,
  there is no account, no per-minute billing, no rate limit.
- **Real language coverage.** 76 models from 21 families ship in the embedded catalog — Whisper,
  Parakeet, Canary, SenseVoice, Qwen3-ASR, Voxtral, Moonshine, GigaAM, Granite Speech, MOSS,
  Fun-ASR, MedASR… Use `--list-models` to see them all.
- **Fast on consumer hardware.** Measured: a 972-second (16 min) Mandarin tech talk transcribed in
  **49 s** with `Qwen3-ASR-0.6B` Q8_0 on an Apple M1 Pro — roughly **20× realtime**.
- **One binary.** No Python, no `torch`, no CUDA toolkit required for the default CPU/Metal builds.

## Install

```bash
# From crates.io (builds from source, needs a C++17 toolchain + CMake)
cargo install handy-transcribe

# Or grab a prebuilt binary (no compiler needed)
cargo binstall handy-transcribe

# Or download a tarball from the releases page:
#   macOS arm64 / macOS x86_64 / Linux x86_64 / Linux arm64
#   https://github.com/cyub/handy-transcribe/releases/latest
```

Build from source:

```bash
git clone https://github.com/cyub/handy-transcribe
cd handy-transcribe
cargo build --release        # binary at target/release/handy-transcribe
```

Requirements: Rust ≥ 1.85, CMake, and a C++17 compiler (`transcribe-cpp-sys` builds ggml through CMake).
On Debian/Ubuntu: `sudo apt install cmake build-essential pkg-config`; on macOS: `brew install cmake`.

## Quick start

```bash
# Transcribe a local file (model chosen by default, language auto-detected)
handy-transcribe ./meeting.m4a

# Transcribe a direct media URL
handy-transcribe "https://example.com/lecture.mp3"

# Write to a file instead of stdout
handy-transcribe ./interview.wav -o interview.txt

# Machine-readable output (backend, device, timings, RTF, detected language)
handy-transcribe ./podcast.mp3 --json

# Highest-accuracy multilingual model (needs GPU / Apple Silicon for comfort)
handy-transcribe ./talk.wav -m Qwen3-ASR-1.7B
```

### Working with web videos

`handy-transcribe` fetches **direct media URLs**. To transcribe something that only exists as a web
page (YouTube, Vimeo, Bilibili, X/Twitter…), extract the audio first with
[yt-dlp](https://github.com/yt-dlp/yt-dlp) and pipe the file in:

```bash
yt-dlp -x --audio-format m4a -o audio.%(ext)s "https://www.youtube.com/watch?v=..."
handy-transcribe ./audio.m4a -o transcript.txt
```

That keeps this tool dependency-free and keeps yt-dlp's extractor work in the project that does it
best.

## Options

| Option | Description |
|---|---|
| `-m, --model <id\|path>` | Catalog model id (e.g. `Qwen3-ASR-0.6B`, `whisper-large-v3-turbo`) or a local `.gguf` path. Default: `Qwen3-ASR-0.6B` |
| `-q, --quant <quant>` | Quantization for catalog models: `F32`, `F16`, `BF16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`. Default: `Q8_0` |
| `-l, --language <code>` | Language hint (`en`, `zh`, `ja`, `de`, …); default auto-detect |
| `--translate` | Translate the result to English (`task=translate`) |
| `--device <N>` | Compute device index (see `--list-devices`) |
| `--no-chunk` | Disable VAD chunking and run the whole file in one pass |
| `--keep-fillers` | Keep `uh`/`um`-style filler words (removed by default) |
| `--custom-words <file>` | Vocabulary file (one term per line) injected to bias spelling |
| `-o, --output <file>` | Write the transcript to a file instead of stdout |
| `--json` | Emit machine-readable JSON |
| `--cache-dir <dir>` | Model cache directory. Default: the platform cache dir — `~/Library/Caches/handy-transcribe` on macOS, `~/.cache/handy-transcribe` on Linux |
| `--list-models` | List catalog models with cache status and exit |
| `--list-devices` | List `transcribe-cpp` compute devices and exit |
| `-h, --help`, `-V, --version` | Usual suspects |

Accepted inputs: `mp3`, `wav`, `flac`, `ogg`/`vorbis`, `m4a`/`aac`, `aiff`, plus the audio track of
MP4-family video containers. Any sample rate is resampled to 16 kHz mono internally.

## Picking a model

The catalog is inherited from Handy and embedded in the binary, so no network call is needed to list
it:

```bash
handy-transcribe --list-models
```

A few good starting points:

| Goal | Model | Notes |
|---|---|---|
| Default, multilingual, laptop-friendly | `Qwen3-ASR-0.6B` | Apache-2.0, strong on CJK + English |
| Maximum accuracy | `Qwen3-ASR-1.7B` | Apache-2.0, GPU recommended |
| Fastest on CPU, CJK | `SenseVoiceSmall` | Very small, very fast |
| English, high accuracy | `parakeet-tdt-0.6b-v3`, `parakeet-unified-en-0.6b` | CC-BY-4.0 |
| General multilingual fallback | `whisper-large-v3-turbo`, `whisper-large-v3` | Apache-2.0 |
| Russian | `gigaam-v3-e2e-rnnt`, `gigaam-v3-ctc` | MIT |
| Arabic / Japanese / Korean / Vietnamese / Ukrainian / Chinese, tiny | `moonshine-base-*` | MIT |
| Long-form audio | streaming / long-form models (`--no-chunk` is implied for them) | see `--list-models` |

Models are downloaded on first use from the Hugging Face Hub and verified against the SHA-256 in the
catalog. To use a mirror or a corporate proxy endpoint:

```bash
export HF_ENDPOINT=https://hf-mirror.com     # or HANDY_HF_ENDPOINT, which takes precedence
handy-transcribe ./audio.mp3
```

## Output

Plain text goes to stdout; progress, model loading, and timings go to stderr, so the tool is safe to
pipe. With `--json`:

```json
{
  "model": "Qwen3-ASR-0.6B",
  "backend": "metal",
  "device": "MTL0",
  "audio_secs": 972.4,
  "load_ms": 1101,
  "transcribe_ms": 48862,
  "rtf": 19.9,
  "detected_language": "zh",
  "text": "…"
}
```

`rtf` is *audio seconds per transcription second* — higher is faster.

## How it works

1. **Decode** — `symphonia` reads the first audio track, downmixes to mono, `rubato` resamples to
   16 kHz `f32`.
2. **Chunk** — audio longer than ~40 s is cut at pauses by Silero VAD (~30 s chunks), unless the
   selected model is long-form/streaming or `--no-chunk` is given.
3. **Infer** — `transcribe-cpp` (GGUF/ggml) runs chunk by chunk, on Metal or CPU.
4. **Join** — chunks are concatenated with CJK-aware spacing (no spaces injected between
   non-spaced scripts).
5. **Post-process** — duplicate-word cleanup, filler-word removal, whitespace normalization.

## Environment variables

| Variable | Effect |
|---|---|
| `HF_ENDPOINT` / `HANDY_HF_ENDPOINT` | Model download base URL (mirror / proxy). The second wins. |
| `RUST_LOG` | Log filter for the embedded `env_logger`, e.g. `RUST_LOG=debug` |

## Platform support

| Platform | Status |
|---|---|
| macOS arm64 (Metal) | ✅ primary target, tested locally |
| macOS x86_64 | ✅ built in CI (`macos-15-intel` runner) |
| Linux x86_64 | ✅ built in CI (CPU backend) |
| Linux arm64 | ✅ built in CI on a native ARM runner (CPU backend) |
| Windows | ⚠️ untested — patches welcome |

GPU acceleration beyond Metal (Vulkan/CUDA for Linux and Windows) requires enabling the
corresponding `transcribe-cpp-sys` features and rebuilding; see the crate's docs.

## License

Code: MIT — see [LICENSE](LICENSE).

**Model licenses differ.** The catalog mixes Apache-2.0, MIT, CC-BY-4.0 and non-commercial
CC-BY-NC variants (for example `canary-1b` is CC-BY-NC-4.0, `granite-speech-5.0-470m-turboctc-nc` is
CC-BY-NC-SA-4.0). Check the license of the model you pick before commercial use; each entry's
license is embedded in `catalog.json` and printed in the catalog's source repository.

## Acknowledgements

- [Handy](https://github.com/cjpais/Handy) — the original desktop app this pipeline comes from (MIT).
- [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) — ggml-based speech-to-text
  inference for 16+ model families (MIT).
- [symphonia](https://github.com/pdeljanov/Symphonia), [rubato](https://github.com/HEnquist/rubato),
  [Silero VAD](https://github.com/snakers4/silero-vad) — decoding, resampling, voice activity.

---

中文文档：[README.zh-CN.md](README.zh-CN.md)
