"""Shared pytest fixtures for Ollama Vox unit tests.

This ``conftest.py`` file is automatically loaded by pytest before any test
in the ``tests/`` directory. It defines ``autouse=True`` fixtures, meaning
they apply to **every** test without needing to be explicitly requested.

Why mock these dependencies?
------------------------------
The core modules import hardware-dependent libraries at the top level:

* ``sounddevice`` — needs a real audio device (unavailable in CI).
* ``soundfile``   — would write real files to disk during tests.
* ``mlx_whisper`` — requires Apple Silicon and a downloaded model.
* ``mlx_audio``   — same as above.
* ``misaki``      — G2P library with system-level dependencies.

By patching them in ``conftest.py``, all unit tests run without any
hardware, models, or audio devices — making the test suite fast, portable,
and CI-friendly.
"""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_audio_dependencies(mocker):
    """Patch ``sounddevice`` and ``soundfile`` for every test.

    Replaces ``sd.InputStream`` with a MagicMock so tests that call
    ``AudioRecorder.start()`` do not open a real microphone stream.

    ``sf.SoundFile`` is replaced to prevent actual file writes; ``sf.read``
    is replaced to return a predictable small float32 array so
    ``AudioRecorder.stop()`` always has data to return.

    Args:
        mocker: pytest-mock's ``mocker`` fixture.

    Yields:
        tuple: ``(mock_sf, mock_read)`` — the mock SoundFile class and the
               mock ``sf.read`` function (rarely needed directly by tests).
    """
    mocker.patch("sounddevice.InputStream")

    # We might need to keep soundfile working for temp files in tests,
    # but for pure unit testing without writing to disk, we can mock sf.SoundFile and sf.read
    mock_sf = mocker.patch("soundfile.SoundFile")
    mock_read = mocker.patch("soundfile.read")
    import numpy as np

    # Return a tiny (10, 1) float32 array — matches the always_2d=True shape.
    mock_read.return_value = (np.zeros((10, 1), dtype=np.float32), 16000)

    return mock_sf, mock_read


@pytest.fixture(autouse=True)
def mock_mlx_whisper(monkeypatch):
    """Inject a fake ``mlx_whisper`` module for every test.

    Replaces the real ``mlx_whisper`` in ``sys.modules`` so that
    ``from mlx_whisper import ...`` in ``core/stt.py`` gets the mock instead.
    The mock's ``transcribe`` returns a fixed ``{"text": "mocked transcription"}``
    dict, which lets STT tests verify behaviour without a real model.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        MagicMock: The fake ``mlx_whisper`` module.
    """
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = {"text": "mocked transcription"}
    monkeypatch.setitem(sys.modules, "mlx_whisper", mock_whisper)
    return mock_whisper


@pytest.fixture(autouse=True)
def mock_misaki(monkeypatch):
    """Inject fake ``misaki`` and ``misaki.espeak`` modules for every test.

    ``misaki`` is the G2P (grapheme-to-phoneme) library used by Kokoro for
    text normalisation. It has system-level dependencies (espeak-ng) that
    are not available in CI. Replacing it with a MagicMock prevents
    ``ImportError`` when ``main.py`` is imported during tests.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        MagicMock: The fake ``misaki`` top-level module.
    """
    mock_misaki = MagicMock()
    mock_espeak = MagicMock()
    monkeypatch.setitem(sys.modules, "misaki", mock_misaki)
    monkeypatch.setitem(sys.modules, "misaki.en", MagicMock())
    monkeypatch.setitem(sys.modules, "misaki.espeak", mock_espeak)
    return mock_misaki


@pytest.fixture(autouse=True)
def mock_mlx_audio(monkeypatch):
    """Inject a fake ``mlx_audio`` module for every test.

    Prevents the real ``mlx_audio`` (which requires Apple Silicon and a
    downloaded Kokoro model) from being imported. TTS tests that need
    specific behaviour from the model object provide their own
    ``tts._model`` mock on top of this fixture.

    Args:
        monkeypatch: pytest's built-in monkeypatch fixture.

    Returns:
        MagicMock: The fake ``mlx_audio`` module.
    """
    mock_audio = MagicMock()
    monkeypatch.setitem(sys.modules, "mlx_audio", mock_audio)
    return mock_audio
