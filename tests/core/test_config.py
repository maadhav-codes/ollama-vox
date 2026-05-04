import pytest
from ollama_vox.core.config import AppConfig, ConfigValidationError


def test_app_config_defaults():
    config = AppConfig.from_dict({})
    assert config.audio.sample_rate == 16000
    assert config.audio.vad_enabled is True
    assert config.stt.model == "./whisper/whisper-small.en-mlx-q4"
    assert config.ollama.endpoint == "http://localhost:11434"
    assert config.tts.model == "./kokoro/Kokoro-82M-4bit"
    assert config.queue.maxsize == 4
    assert config.queue.drop_policy == "drop_oldest"


def test_app_config_coercion():
    data = {
        "audio": {"sample_rate": "24000", "vad_enabled": "False"},
        "queue": {"maxsize": "10"},
    }
    config = AppConfig.from_dict(data)
    assert config.audio.sample_rate == 24000
    assert config.audio.vad_enabled is False
    assert config.queue.maxsize == 10


def test_app_config_validation_error():
    with pytest.raises(
        ConfigValidationError, match="Invalid type for 'audio.sample_rate'"
    ):
        AppConfig.from_dict({"audio": {"sample_rate": "not_an_int"}})


def test_app_config_invalid_drop_policy():
    with pytest.raises(ConfigValidationError, match="Invalid queue.drop_policy"):
        AppConfig.from_dict({"queue": {"drop_policy": "invalid_policy"}})


def test_app_config_styles():
    data = {
        "styles": {"fast": {"speed": 1.5, "voice": "af_bella"}, "high": {"pitch": 1.2}}
    }
    config = AppConfig.from_dict(data)
    assert config.styles["fast"].speed == 1.5
    assert config.styles["fast"].voice == "af_bella"
    assert config.styles["high"].pitch == 1.2
    assert config.styles["fast"].pitch is None
