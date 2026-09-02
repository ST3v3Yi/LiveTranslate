# LiveTranslate

**English** | [中文](README_zh.md)

Real-time audio translation for Windows. Captures system audio (WASAPI loopback) and optional microphone input, runs ASR, translates via LLM API, and displays results in a transparent overlay.

Works with any system audio — videos, livestreams, voice chat. No player modifications needed.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Windows](https://img.shields.io/badge/Platform-Windows-0078d4)
![License](https://img.shields.io/badge/License-MIT-green)

## Screenshot

![LiveTranslate](screenshot/en.png)

## Video

[![Install & Demo](https://img.shields.io/badge/Bilibili-Install%20%26%20Demo-00A1D6?logo=bilibili)](https://www.bilibili.com/video/BV1K2Awz6Euw)

## Features

- **Real-time pipeline**: System audio → VAD → ASR → LLM translation → overlay
- **Multiple ASR engines**: faster-whisper, SenseVoice, FunASR Nano, Anime-Whisper
- **Qwen3-ASR integration**: optional local Qwen3-ASR worker with recognition context, keywords, and second-pass refinement
- **Remote ASR**: offload speech recognition to a GPU machine over HTTP — see [REMOTE_ASR.md](REMOTE_ASR.md)
- **Any OpenAI-compatible API**: DeepSeek, Grok, Qwen, GPT, Ollama, vLLM, etc.
- **Streaming translation display**: Real-time character-by-character translation output
- **Per-model settings**: Streaming, structured output (JSON), context history, disable thinking
- **Terminology glossaries**: Load CSV or Markdown glossaries, normalize ASR names, and inject only matched terms into translation prompts
- **Screenshot translation**: Select a screen region, run OCR in a separate worker environment, preview the translated image, then paste, save, or copy it
- **Microphone mix-in**: Optionally mix microphone input with system audio for ASR
- **Low-latency VAD**: 32ms chunks + Silero VAD, with optional FireRedVAD streaming inference
- **Transparent overlay**: Always-on-top, click-through, draggable, 14 color themes
- **CUDA acceleration**: GPU-accelerated ASR inference
- **Auto model management**: Setup wizard, ModelScope / HuggingFace dual sources
- **Built-in benchmark**: Compare translation model speed and quality

## Fork Additions

This fork adds features for Japanese game, video, and live-stream translation:

- Qwen3-ASR can use recent accepted transcripts and user-provided keywords to improve names and short context-dependent speech.
- The terminology module supports custom CSV / Markdown glossaries. `endfield_terms.csv` is included as an example glossary for Arknights: Endfield.
- Screenshot translation uses PaddleOCR or PaddleOCR-VL in a separate Python environment so OCR failures do not stop live audio translation.
- ASR and OCR settings are available from the settings panel and are persisted locally in `user_settings.json`.

## Changelog

See [English Changelog](i18n/CHANGELOG_en.md) | [中文更新日志](i18n/CHANGELOG_zh.md)

## Requirements

- **OS**: Windows 10/11
- **Python**: 3.10–3.12 (or use the portable build)
- **GPU** (recommended): NVIDIA + CUDA 12.6 (Blackwell GPUs like RTX 50xx require CUDA 12.8)
- **Network**: Access to a translation API

## Quick Start

### Portable build (no Python required, recommended for non-developers)

Download `LiveTranslate-portable-*.zip` from this fork's [Releases](https://github.com/ST3v3Yi/LiveTranslate/releases), unzip, and double-click **`start.bat`**. The first run auto-downloads a portable Python 3.12 and installs GPU-aware dependencies — no Python installation needed. OCR and Qwen3-ASR use separate environments and are optional.

### From source

```bash
git clone https://github.com/ST3v3Yi/LiveTranslate.git
cd LiveTranslate
```

Double-click **`install.bat`** — the installer will:
1. Detect Python 3.10–3.12 (auto-install via winget if missing)
2. Create a virtual environment
3. Auto-detect NVIDIA GPU and let you choose CUDA / CPU PyTorch
4. Install all dependencies

Then double-click **`start.bat`** to launch. The setup script keeps temporary downloads inside the project directory and removes them after a successful install.

To update, double-click **`update.bat`** — it will pull the latest code and update dependencies (auto-installs Git via winget if missing).

<details>
<summary>Manual install</summary>

```bash
python -m venv .venv
.venv\Scripts\activate

# PyTorch (choose one)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126  # CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # CUDA (RTX 50xx)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu    # CPU only

# Dependencies
pip install -r requirements.txt

# Launch
.venv\Scripts\python.exe main.py
```

</details>

## First Launch

1. Setup wizard appears — choose download source (ModelScope / HuggingFace) and cache path
2. Silero VAD + the selected ASR model download automatically
3. Configure a translation API in the model dialog
4. Main UI appears when ready

### Optional Qwen3-ASR

Qwen3-ASR runs through a separate Python environment. Select **Qwen3-ASR** in Settings → VAD / ASR, then provide the model folder, Qwen3-ASR Python executable, and source directory if they are not detected automatically. You can also configure ASR context turns, maximum output tokens, keyword hints, and second-pass refinement.

### Optional screenshot translation

Enable screenshot translation in Settings → Translation. Configure the PaddleOCR / PaddleOCR-VL Python executable, model path, and device. Click **Screenshot Translation** on the overlay, select a region, review the translated image, and then choose paste, save, or copy. OCR runs in a separate worker and does not share the live ASR process.

## Translation API

Settings → Translation tab:

| Parameter | Example |
|-----------|---------|
| API Base | `https://api.deepseek.com/v1` |
| API Key | Your key, entered in the settings dialog |
| Model | `deepseek-chat` |
| Proxy | `none` / `system` / custom URL |

The repository does not contain a usable API key. Replace the placeholder with your own key in the settings dialog or local configuration.

## Architecture

```
Audio (WASAPI 32ms) → VAD (Silero) → ASR → LLM Translation → Overlay
         ↑ optional mic mix-in
```

```
main.py                 Entry point & pipeline
├── audio_capture.py    WASAPI loopback + mic mix-in
├── vad_processor.py    Silero VAD
├── asr_engine.py       faster-whisper backend
├── asr_funasr.py       Unified FunASR model selector backend
├── asr_sensevoice.py   SenseVoice backend
├── asr_funasr_nano.py  FunASR Nano backend
├── asr_anime_whisper.py Anime-Whisper backend (ja anime/galgame)
├── asr_qwen3.py        Qwen3-ASR process client with context and keyword hints
├── asr_qwen3_worker.py Qwen3-ASR worker process
├── asr_remote.py        Remote Whisper client (→ asr_server.py, see REMOTE_ASR.md)
├── translator.py       OpenAI-compatible client (streaming, JSON schema, context)
├── term_glossary.py    CSV / Markdown glossary loader and term matcher
├── screenshot_translation.py Screenshot selection and translated-image workflow
├── ocr_paddle.py       PaddleOCR worker client
├── ocr_paddle_worker.py OCR/PaddleOCR-VL worker process
├── model_manager.py    Model download & cache
├── subtitle_overlay.py PyQt6 overlay
├── control_panel.py    Settings UI (7 tabs)
├── dialogs.py          Wizard, download & model config dialogs
└── benchmark.py        Translation benchmark
```

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper inference via CTranslate2
- [FunASR](https://github.com/modelscope/FunASR) — SenseVoice / Fun-ASR-Nano
- [Anime-Whisper](https://huggingface.co/litagin/anime-whisper) — Japanese anime/galgame ASR
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice activity detection

## Star History

<a href="https://www.star-history.com/?repos=ST3v3Yi%2FLiveTranslate&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=ST3v3Yi/LiveTranslate&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=ST3v3Yi/LiveTranslate&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=ST3v3Yi/LiveTranslate&type=date&legend=top-left" />
 </picture>
</a>

## License

[MIT License](LICENSE)
