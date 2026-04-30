# Native Ollama Voiceover

A local macOS menubar voice assistant that records speech, transcribes it with MLX Whisper, sends the text to Ollama, and plays the response with Kokoro TTS.

## Features

- Push-to-talk recording from a configurable global hotkey
- Voice activity detection (VAD) and max recording limits
- Local STT with `mlx-whisper`
- Local LLM responses through an Ollama model
- Local TTS with Kokoro voices via `mlx-audio`
- Menubar status updates and queued processing pipeline

## Requirements

- macOS (Apple Silicon recommended for MLX)
- Python 3.12+
- Ollama running locally on `http://localhost:11434`
- A pulled Ollama model matching `config.yaml` (default: `llama3.2:1b-instruct-q4_K_M`)

## Installation

```bash
uv sync
```

If you do not use `uv`, install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
uv run native-ollama-voiceover
```

or:

```bash
python main.py
```

## Configuration

Main config is in `config.yaml`.

Key sections:

- `audio`: sample rate, VAD controls, max recording duration
- `stt.model`: local path to Whisper MLX model
- `ollama.model` and `ollama.temperature`: model + generation behavior
- `tts.model`, `tts.voice`, `tts.rate`, `tts.sample_rate`: Kokoro voice output
- `hotkey.key`: trigger key combo (default `cmd+shift`)
- `queue`: worker queue size and drop policy
- `response_style` + `styles`: speaking style controls

## Health Checks at Startup

On startup, the app validates:

- STT model path exists
- TTS model path exists
- `mlx_whisper` and `mlx_audio` imports work
- Ollama is reachable (`/api/tags`)

Any failures are logged with `event=startup_health_failed`.

## Project Layout

- `main.py`: app startup, config loading, dependency wiring
- `core/`: audio, STT, LLM, TTS, and worker pipeline
- `ui/`: menubar app integration
- `whisper/`: bundled Whisper model assets
- `kokoro/`: bundled Kokoro model assets and voices

## Notes

- Model assets in `kokoro/` and `whisper/` can be large.
- If your hotkey conflicts with system shortcuts, update `hotkey.key` in `config.yaml`.
