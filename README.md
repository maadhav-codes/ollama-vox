# Ollama Vox

A local macOS menubar voice assistant that records speech, transcribes with MLX Whisper, gets responses from Ollama, and speaks back using Kokoro TTS.

## Features

- **Menubar app** for easy access
- **Local speech-to-text** with `mlx-whisper`
- **Local text generation** with Ollama
- **Local text-to-speech** with `mlx-audio` + Kokoro voices
- **Status panel** showing model info, rolling latency stats, and response history
- **Developer-Friendly Codebase**: Comprehensive Google-style docstrings, full type hints, and robust pytest suite

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
uv sync --group dev
```

## First-Time Setup

On first launch, Ollama Vox prompts to download missing STT + TTS models
before starting the menubar app. Setup must complete successfully to proceed.

To run setup separately and exit afterward:

```bash
uv run ollama-vox --setup
```

## Run

Start the menubar application:

```bash
uv run ollama-vox
```

If the local Ollama server is stopped, the app asks permission to start
`ollama serve` in the background and waits up to 15 seconds for it to be ready.
Ollama must be installed and its command available on `PATH`. When quitting,
you can stop the local server (even if it was already running before this
session), leave it running, or cancel quit. Stopping it also disconnects other
apps using that server.
Remote servers must be started on their host.

## Usage

1. Click the microphone icon in your macOS menubar.
2. Select **Start Listening** to speak.
3. Select **Stop Listening** when you are done. The app will process your speech and respond with audio.
4. Click **Show Status** to view latency, recent responses, and active models.

## Development & Testing

Ollama Vox is built with a strong emphasis on reliability and developer experience. The core pipeline is fully unit-tested with mocked hardware dependencies, and the entire codebase features comprehensive Google-style docstrings and Python type hints.

To run the test suite locally:

```bash
uv run pytest
```

## Configuration

Settings are managed in `config.yaml`.

- **`audio`**: Adjust Voice Activity Detection (VAD) and recording limits.
- **`stt.model`**: Path to the local Whisper model.
- **`ollama`**: Set the endpoint, model name, and temperature.
- **`tts`**: Configure the voice, speaking rate, and Kokoro model path.

On first download, a folder picker lets you choose where to keep models
(including a folder on Desktop). Skipping the picker uses
`~/Documents/ollama-vox/models`. The app remembers your choice across launches.
Relative STT and TTS paths resolve inside that folder regardless of the launch
directory. Absolute paths and paths starting with `~` in config remain supported.

Before offering a download, setup checks existing model files in configured
and remembered locations, the project directory, common Documents/Desktop/
Downloads folders, and the Hugging Face cache. Complete models are reused in
place and their locations remembered. A selected folder is checked again before
any download prompt. Only missing or incomplete models are downloaded.

## Community

We welcome contributions and feedback from the community!

- **Contributing**: Please see our [Contributing Guide](CONTRIBUTING.md) for local setup instructions and our PR review workflow.
- **Reporting Issues**: Encountered a bug or have a feature idea? Please use our [Issue Forms](https://github.com/maadhav-codes/ollama-vox/issues/new/choose).
- **Discussions**: Have questions or want to share how you're using Ollama Vox? Join the conversation in [GitHub Discussions](https://github.com/maadhav-codes/ollama-vox/discussions).
- **Security**: If you find a security vulnerability, please follow the steps in our [Security Policy](SECURITY.md) to report it privately.
