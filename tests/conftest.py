import pytest
from unittest.mock import MagicMock
import sys


@pytest.fixture(autouse=True)
def mock_audio_dependencies(mocker):
    mocker.patch("sounddevice.InputStream")

    # We might need to keep soundfile working for temp files in tests,
    # but for pure unit testing without writing to disk, we can mock sf.SoundFile and sf.read
    mock_sf = mocker.patch("soundfile.SoundFile")
    mock_read = mocker.patch("soundfile.read")
    import numpy as np

    mock_read.return_value = (np.zeros((10, 1), dtype=np.float32), 16000)

    return mock_sf, mock_read


@pytest.fixture(autouse=True)
def mock_mlx_whisper(monkeypatch):
    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = {"text": "mocked transcription"}
    monkeypatch.setitem(sys.modules, "mlx_whisper", mock_whisper)
    return mock_whisper


@pytest.fixture(autouse=True)
def mock_misaki(monkeypatch):
    mock_misaki = MagicMock()
    mock_espeak = MagicMock()
    monkeypatch.setitem(sys.modules, "misaki", mock_misaki)
    monkeypatch.setitem(sys.modules, "misaki.en", MagicMock())
    monkeypatch.setitem(sys.modules, "misaki.espeak", mock_espeak)
    return mock_misaki


@pytest.fixture(autouse=True)
def mock_mlx_audio(monkeypatch):
    mock_audio = MagicMock()
    monkeypatch.setitem(sys.modules, "mlx_audio", mock_audio)
    return mock_audio
