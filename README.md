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
- Python `3.12.10+`
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

On first launch, Ollama Vox checks for existing Whisper and Kokoro model files
and prompts to download only missing models. It also checks for the English
spaCy model needed by Kokoro. Setup must complete before recording is available.

To run setup separately and exit afterward:

```bash
uv run ollama-vox --setup
```

### English Speech Support (spaCy)

spaCy is installed automatically with Ollama Vox. Kokoro's English text processor
also needs **`en_core_web_sm`**. If it is already installed, setup reuses it.
Otherwise, setup asks permission and runs spaCy's documented downloader in the
same Python environment as the app:

```bash
python -m spacy download en_core_web_sm
```

spaCy selects a model version compatible with the installed spaCy version. See
[spaCy's installation documentation](https://spacy.io/usage/models#download) and
the official [English model 3.8.0 release](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0).
The app does not hard-code a release version or download URL.

For a source checkout, you can install the model manually:

```bash
uv run python -m spacy download en_core_web_sm
```

For a `pip` installation, activate the same virtual environment used to install
and run Ollama Vox, then use the `python -m spacy download` command above. For a
`uv tool` installation, let the app's setup install it in the tool environment.
The spaCy model is a Python package installed in that environment; the model
folder picker applies to Whisper and Kokoro files.

After manually installing the spaCy model, use `uv sync --inexact --group dev`
when updating development dependencies to preserve it. A regular `uv sync`
removes packages outside the lockfile; setup will offer to reinstall it if needed.

spaCy's downloader still uses official GitHub releases. If installation times
out, check that GitHub downloads are reachable and retry setup or the manual
command. Installation failures stop startup and show an error. Model downloads
happen during setup, never while answering a spoken request.

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

1. Hold **Control + Option + Space** from your terminal, editor, or browser.
2. After the short rising tone, speak. Release the shortcut to submit your audio.
3. A falling tone confirms recording stopped; a low tone signals processing.
4. Open **Show Panel** from the tray to view the full response, including code.

Global push-to-talk registers only the configured shortcut with macOS. It does
not monitor other typing and needs no **Accessibility** or **Input Monitoring**
permission for Python, Terminal, or the app. Enable **Push-to-Talk** from the tray
menu. Microphone permission is still required for recording. If registration
fails, choose another `interaction.push_to_talk_hotkey` in `config.yaml`;
shortcuts must include Control or Command and one key. Letter shortcuts refer
to US physical key positions. Manual **Start Listening / Stop Listening** remains
available, including automatic silence detection. Holding the shortcut bypasses
silence detection while retaining the configured maximum recording duration.

Startup shows **Preparing Ollama Vox** while Kokoro warms up silently before
recording becomes available. Downloaded voice files are loaded directly.
Set `ollama.think` to `false` (default) for faster voice replies, `true` to enable
reasoning, a supported level such as `low`, or `null` to use the model default.
Some reasoning models cannot disable thinking; use their supported level instead.

Responses use concise spoken-language instructions. Spoken output removes
Markdown markers and URLs and skips fenced code, while the panel retains the
original response. Transcription runs once on the complete recording; there
are no redundant post-recording preview passes.

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
- **`ollama.system_prompt`**: Override the default instructions for concise
  spoken developer assistance.
- **`interaction`**: Set `push_to_talk_enabled`, `push_to_talk_hotkey`
  (default `<ctrl>+<alt>+<space>`), `audio_cues_enabled`, and
  `audio_cues_volume` (0 to 1).

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
