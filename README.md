# Native Ollama Voiceover

A local macOS menubar voice assistant that records speech, transcribes with MLX Whisper, gets responses from Ollama, and speaks back using Kokoro TTS.

## What You Get

- Menubar app with global push-to-talk hotkey
- Local speech-to-text with `mlx-whisper`
- Local text generation with Ollama (`http://localhost:11434`)
- Local text-to-speech with `mlx-audio` + Kokoro voices
- Queue-based STT -> LLM -> TTS pipeline
- Retry + graceful fallback when Ollama fails mid-response
- Status popover (`Show Status`) with model info, latency stats, and last response

## Requirements

- macOS (Apple Silicon recommended)
- Python `3.12+`
- Ollama installed and running locally
- Ollama model available locally (default in config: `llama3.2:1b-instruct-q4_K_M`)

## Install

```bash
uv sync
```

## First-Time Setup (Model Downloads)

Download recommended STT + TTS assets:

```bash
uv run native-ollama-voiceover --setup
```

This setup downloads:

- `mlx-community/whisper-small.en-mlx-q4` -> `whisper/whisper-small.en-mlx-q4`
- `mlx-community/Kokoro-82M-4bit` -> `kokoro/Kokoro-82M-4bit`

It uses paths from `config.yaml` (`stt.model`, `tts.model`) and skips downloads if paths already exist.

## Run

```bash
uv run native-ollama-voiceover
```

## Configuration

Edit `config.yaml`.

Important fields:

- `audio`: sample rate, VAD threshold/silence, max duration
- `stt.model`: Whisper model path
- `ollama.model`, `ollama.temperature`: generation model and temperature
- `tts.model`, `tts.voice`, `tts.rate`, `tts.sample_rate`
- `hotkey.key`: global trigger combo (default `cmd+shift`)
- `queue.maxsize`, `queue.drop_policy`
- `response_style` + `styles`

## Startup Health Checks

On boot, the app checks:

- STT model path exists
- TTS model path exists
- `mlx_whisper` and `mlx_audio` imports
- Ollama reachability via `/api/tags`

## Menubar Status Popover

Open `Show Status` from the menu to view:

- Current status (`idle`, `listening`, `busy`, `speaking`, `error`)
- Active model identifiers (LLM, STT path, TTS model)
- Last + rolling average latency for STT, LLM, and TTS
- Response count and latest response text

## Build a Double-Clickable macOS App (.app)

```bash
uv sync
uv run python setup.py py2app
open dist/Native\ Ollama\ Voiceover.app
```

Notes:

- Bundle includes `config.yaml`, `whisper/`, and `kokoro/`
- macOS will request microphone permission on first run
- For unsigned local builds, use right-click -> Open if Gatekeeper warns

## Project Structure

- `main.py`: entrypoint, setup flag, health checks, dependency wiring
- `core/`: audio capture, STT, LLM client, TTS, worker pipeline
- `ui/menubar.py`: menubar app and status UI
- `config.yaml`: runtime configuration
- `setup.py`: `py2app` packaging config
