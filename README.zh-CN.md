<p align="center">
  <img src="assets/logo.svg" width="72" height="72" alt="handy-transcribe logo">
</p>

<h1 align="center">handy-transcribe</h1>

<p align="center">
  <a href="README.md">English</a> ·
  <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/handy-transcribe">
    <img src="https://img.shields.io/crates/v/handy-transcribe?style=flat&logo=rust" alt="crates.io">
  </a>
  <a href="https://crates.io/crates/handy-transcribe">
    <img src="https://img.shields.io/crates/d/handy-transcribe?style=flat" alt="下载量">
  </a>
  <a href="https://github.com/cyub/handy-transcribe/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/cyub/handy-transcribe/ci.yml?style=flat&logo=githubactions&logoColor=white" alt="CI">
  </a>
  <a href="https://github.com/cyub/handy-transcribe/releases">
    <img src="https://img.shields.io/github/v/release/cyub/handy-transcribe?style=flat&logo=github" alt="Release">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/handy-transcribe?style=flat" alt="许可证">
  </a>
  <img src="https://img.shields.io/badge/rust-1.85%2B-dea584?style=flat&logo=rust" alt="MSRV">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux-555?style=flat" alt="平台">
</p>

<p align="center">本地离线语音转写：给一个音频文件或 URL，输出文本。单个 Rust 二进制，不要 API key，音频不出本机。</p>

`handy-transcribe` 是 [Handy](https://github.com/cjpais/Handy) 那条流水线的命令行外壳：
`解码 → VAD 分块 → transcribe.cpp (ggml) 推理 → 拼接 → 后处理`。去掉了录音 / UI / 剪贴板部分，
只留下可以脚本化、批量跑、给 agent 调用的无头 CLI。

> **非官方项目。** 基于 Handy 的流水线与
> [`transcribe-cpp`](https://github.com/handy-computer/transcribe.cpp) 引擎构建的社区 CLI，
> 与 Handy 项目及 handy-computer 无隶属关系。

## 特点

- **完全本地**：模型跑在本机（macOS 用 Metal，其他平台 CPU）。不上传、不需要账号、不按分钟计费。
- **语言覆盖广**：内置 76 个模型、21 个家族（Whisper、Parakeet、Canary、SenseVoice、Qwen3-ASR、
  Voxtral、Moonshine、GigaAM、Granite Speech、MOSS、Fun-ASR、MedASR……），`--list-models` 可查。
- **消费级硬件上够快**：实测 972 秒（16 分钟）中文技术演讲，`Qwen3-ASR-0.6B` Q8_0 在 Apple M1 Pro 上
  **49 秒**转完，约 **20 倍实时**。
- **单个二进制**：不需要 Python、`torch`，默认构建也不需要 CUDA 工具链。

## 安装

```bash
cargo install handy-transcribe      # 从源码构建，需要 C++17 工具链 + CMake
cargo binstall handy-transcribe     # 直接拉预编译二进制
```

也可以从 [Releases](https://github.com/cyub/handy-transcribe/releases/latest) 下载
macOS arm64 / macOS x86_64 / Linux x86_64 / Linux arm64 的压缩包。

要求：Rust ≥ 1.85、CMake、C++17 编译器（`transcribe-cpp-sys` 会用 CMake 构建 ggml）。
Debian/Ubuntu：`sudo apt install cmake build-essential pkg-config`；macOS：`brew install cmake`。

## 快速开始

```bash
# 本地文件（默认模型，自动识别语言）
handy-transcribe ./会议.m4a

# 直接媒体 URL
handy-transcribe "https://example.com/lecture.mp3"

# 写到文件
handy-transcribe ./interview.wav -o interview.txt

# JSON 输出（backend、设备、耗时、RTF、识别语言）
handy-transcribe ./podcast.mp3 --json

# 最高准确率模型（建议有 GPU / Apple Silicon）
handy-transcribe ./talk.wav -m Qwen3-ASR-1.7B
```

### 处理网页视频（YouTube 等）

本工具只接受**直接媒体 URL**。YouTube / Bilibili / Vimeo 这类网页请先用
[yt-dlp](https://github.com/yt-dlp/yt-dlp) 抽音频，再喂给本工具：

```bash
yt-dlp -x --audio-format m4a -o audio.%(ext)s "https://www.youtube.com/watch?v=..."
handy-transcribe ./audio.m4a -o transcript.txt
```

## 常用参数

| 参数 | 说明 |
|---|---|
| `-m, --model <id\|路径>` | 内置目录模型 id 或本地 `.gguf` 路径，默认 `Qwen3-ASR-0.6B` |
| `-q, --quant <档位>` | 量化档：`F32/F16/BF16/Q8_0/Q6_K/Q5_K_M/Q4_K_M`，默认 `Q8_0` |
| `-l, --language <code>` | 语言提示（`zh`/`en`/`ja`/`de`…），默认自动检测 |
| `--translate` | 翻译成英文 |
| `--device <N>` | 计算设备索引（见 `--list-devices`） |
| `--no-chunk` | 关闭 VAD 分块，整段一次推理 |
| `--keep-fillers` | 保留 uh/um 等填充词（默认移除） |
| `--custom-words <文件>` | 自定义词表（每行一词），注入模型词表以改善专有名词拼写 |
| `-o, --output <文件>` | 写文件（默认 stdout） |
| `--json` | JSON 输出 |
| `--cache-dir <目录>` | 模型缓存目录，默认跟随平台：macOS `~/Library/Caches/handy-transcribe`，Linux `~/.cache/handy-transcribe` |
| `--list-models` / `--list-devices` | 列出目录模型 / 计算设备 |

支持 mp3、wav、flac、ogg/vorbis、m4a/aac、aiff，以及 MP4 家族视频容器的音轨；任意采样率内部重采样到
16 kHz mono。

## 模型下载

首次使用自动从 Hugging Face 下载并按 `catalog.json` 里的 SHA-256 校验。国内或企业网络可用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 或用 HANDY_HF_ENDPOINT，优先级更高
handy-transcribe ./audio.mp3
```

模型许可证各不相同（Apache-2.0 / MIT / CC-BY-4.0 / 非商用 CC-BY-NC 等），商用前请确认所选模型的许可。

## 平台支持

| 平台 | 状态 |
|---|---|
| macOS arm64（Metal） | ✅ 主要目标，本机已测 |
| macOS x86_64 | ✅ CI 构建（`macos-15-intel` runner） |
| Linux x86_64 | ✅ CI 构建（CPU 后端） |
| Linux arm64 | ✅ CI 在原生 ARM runner 构建（CPU 后端） |
| Windows | ⚠️ 未测试，欢迎 PR |

## 其他语言

本 README 目前维护 English 与简体中文两版，其他语言欢迎提 issue 或 PR。

写个"快速上手"级别的短版就够了：上面那几张表格（参数、模型、平台支持）每轮发版都会变，
不需要逐字翻译，指向本文件即可。

## 许可证

代码 MIT，见 [LICENSE](LICENSE)。模型许可证单独计算。

致谢：[Handy](https://github.com/cjpais/Handy)、
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp)、
[symphonia](https://github.com/pdeljanov/Symphonia)、[rubato](https://github.com/HEnquist/rubato)、
[Silero VAD](https://github.com/snakers4/silero-vad)。

