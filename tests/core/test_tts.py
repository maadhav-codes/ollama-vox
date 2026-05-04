import pytest
from unittest.mock import MagicMock
from ollama_vox.core.tts import TTS


@pytest.fixture
def mock_sd_play(mocker):
    return mocker.patch("sounddevice.play")


@pytest.fixture
def mock_sd_stop(mocker):
    return mocker.patch("sounddevice.stop")


def test_tts_init(mocker):
    # Mock _load_model_id_from_config to avoid config file dependency
    mocker.patch(
        "ollama_vox.core.tts.TTS._load_model_id_from_config", return_value="mock_model"
    )
    tts = TTS(voice="test_voice", rate=1.2, split_chars=100)
    assert tts.voice == "test_voice"
    assert tts.default_speed == 1.2
    assert tts.split_chars == 100


def test_tts_split_text(mocker):
    mocker.patch(
        "ollama_vox.core.tts.TTS._load_model_id_from_config", return_value="mock_model"
    )
    tts = TTS(split_chars=10)
    chunks = tts._split_text("This is a long sentence. It has parts.")
    assert chunks == ["This is a ", "long sente", "nce.", "It has par", "ts."]


def test_tts_speak(mocker, mock_sd_play):
    mocker.patch(
        "ollama_vox.core.tts.TTS._load_model_id_from_config", return_value="mock_model"
    )

    tts = TTS(voice="test_voice", rate=1.0)

    mock_model = MagicMock()
    mock_stream_result = MagicMock()
    mock_stream_result.audio = [0.1, 0.2]
    mock_model.generate.return_value = [mock_stream_result]

    tts._model = mock_model

    tts.speak("Hello")

    assert mock_model.generate.call_count == 1
    assert mock_sd_play.call_count == 1


def test_tts_stop(mocker, mock_sd_stop):
    mocker.patch(
        "ollama_vox.core.tts.TTS._load_model_id_from_config", return_value="mock_model"
    )
    tts = TTS()
    tts.stop()
    assert tts._interrupt is True
    assert mock_sd_stop.call_count == 1
