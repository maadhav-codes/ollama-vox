# Contributing to Ollama Vox

First off, thank you for considering contributing to Ollama Vox! It's people like you that make open source such a great community.

## Local Setup

To set up your development environment, follow these steps:

1. **Install Dependencies:** We use `uv` for dependency management. Sync the project and development dependencies (pytest, ruff) by running:

   ```bash
   uv sync --group dev
   ```

   _Note: Ollama Vox requires Python 3.12+. `uv` will automatically use the correct version based on the project configuration._

2. **Speech Dependencies:** Ollama Vox runs locally on **macOS** (Apple Silicon recommended).
   - Kokoro's English text normalizer requires the spaCy `en_core_web_sm` model:

     ```bash
     uv run python -m spacy download en_core_web_sm
     ```

   _Note:_ English voice synthesis is handled natively by `misaki` and spaCy; external binary tools like `espeak-ng` are not required.

## Architecture Overview

The Ollama Vox application is designed with a clear separation of concerns. UI components, including the main application window and tray icon, are located in the `ollama_vox/ui/` directory. The core engine, handling the Text-to-Speech (TTS) and Speech-to-Text (STT) pipelines, resides in the `ollama_vox/core/` directory.

## PR Process

When you are ready to submit a Pull Request, please ensure you follow these rules to maintain code quality. These commands exactly mirror our CI pipeline:

1. **Run Tests:** Ensure all unit tests pass before submitting. To test the core logic and UI components, run:

   ```bash
   uv run pytest tests/
   ```

   _Tip: For faster iteration, you can run tests for specific modules, e.g., `uv run pytest tests/core/`._

2. **Run Linter and Formatter:** We use `ruff` to keep our code clean and correctly formatted. Run it before committing:

   ```bash
   uv run ruff check .
   uv run ruff format .
   ```

3. **Descriptive PRs:** Provide a clear and concise description of what your PR does and any issues it resolves.

## Issues and Feedback

- **Bug Reports:** Please use the provided [Bug Report form](https://github.com/maadhav-codes/ollama-vox/issues/new/choose) when opening an issue.
- **Feature Requests:** Please use the provided [Feature Request form](https://github.com/maadhav-codes/ollama-vox/issues/new/choose).
