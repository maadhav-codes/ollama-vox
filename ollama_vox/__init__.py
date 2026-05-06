"""Ollama Vox — a voice assistant package for macOS.

This is the top-level Python package for Ollama Vox. Its primary job is to expose the current version string so tools like ``pip show ollama-vox`` and ``importlib.metadata`` can report it correctly.

Architecture overview
---------------------
The package is split into two sub-packages:

* ``ollama_vox.core`` — pure-Python business logic (audio, STT, LLM, TTS, pipeline workers, and configuration). These modules are independent of any GUI framework and can be tested without a display.

* ``ollama_vox.ui`` — PySide6 graphical components (system-tray icon, status panel, model-selection wizard, and the first-run setup wizard).

The application entry-point lives in ``ollama_vox.main`` and is registered as the ``ollama-vox`` console script in ``pyproject.toml``.
"""

# The canonical version of the package.
# This single string is the source-of-truth; ``pyproject.toml`` reads it via
# ``importlib.metadata`` so both always stay in sync.
__version__ = "1.0.2"
