# Ollama Vox

A local macOS menubar voice assistant that records speech, transcribes with MLX Whisper, gets responses from Ollama, and speaks back using Kokoro TTS.

## Features

- **Menubar app** for easy access
- **Local speech-to-text** with `mlx-whisper`
- **Local text generation** with Ollama
- **Local text-to-speech** with `mlx-audio` + Kokoro voices
- **Status panel** showing model info, rolling latency stats, and response history

## Requirements

- macOS (Apple Silicon recommended)
- Python `3.12+`
- [Ollama](https://ollama.com/) installed and running locally

## Installation

The easiest way to install is via `pip` or `uv`:

```bash
pip install ollama-vox
# or if using uv
uv tool install ollama-vox
```

If installing from source for development:

```bash
git clone https://github.com/maadhav-codes/ollama-vox.git
cd ollama-vox
uv sync
```

## First-Time Setup

Download recommended STT + TTS models:

```bash
uv run ollama-vox --setup
```

## Run

Start the menubar application:

```bash
uv run ollama-vox
```

## Usage

1. Click the microphone icon in your macOS menubar.
2. Select **Start Listening** to speak.
3. Select **Stop Listening** when you are done. The app will process your speech and respond with audio.
4. Click **Show Status** to view latency, recent responses, and active models.

## Configuration

Settings are managed in `config.yaml`.

- **`audio`**: Adjust Voice Activity Detection (VAD) and recording limits.
- **`stt.model`**: Path to the local Whisper model.
- **`ollama`**: Set the endpoint, model name, and temperature.
- **`tts`**: Configure the voice, speaking rate, and Kokoro model path.
