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


def test_app_config_type_coercion_and_bool_strings():
    data = {
        "audio": {
            "sample_rate": "24000",
            "vad_enabled": "False",
            "vad_threshold": "0.02",
        },
        "queue": {"maxsize": "10"},
    }
    config = AppConfig.from_dict(data)
    assert config.audio.sample_rate == 24000
    assert config.audio.vad_enabled is False
    assert config.audio.vad_threshold == 0.02
    assert config.queue.maxsize == 10


def test_app_config_rejects_invalid_types():
    with pytest.raises(ConfigValidationError, match="audio.sample_rate"):
        AppConfig.from_dict({"audio": {"sample_rate": "not_an_int"}})


def test_app_config_invalid_drop_policy():
    with pytest.raises(ConfigValidationError, match="Invalid queue.drop_policy"):
        AppConfig.from_dict({"queue": {"drop_policy": "invalid"}})


def test_app_config_unknown_nested_key_raises():
    with pytest.raises(ConfigValidationError, match="Unknown configuration key"):
        AppConfig.from_dict({"audio": {"unknown": 1}})


def test_app_config_styles_map_and_style_validation():
    cfg = AppConfig.from_dict(
        {
            "styles": {
                "fast": {"speed": "1.5", "voice": "af_bella"},
                "high": {"pitch": 1.2},
            }
        }
    )
    assert cfg.styles["fast"].speed == 1.5
    assert cfg.styles["fast"].voice == "af_bella"
    assert cfg.styles["fast"].pitch is None
    assert cfg.styles["high"].pitch == 1.2


def test_app_config_styles_must_be_dict():
    with pytest.raises(
        ConfigValidationError, match="Styles config must be a dictionary"
    ):
        AppConfig.from_dict({"styles": []})
