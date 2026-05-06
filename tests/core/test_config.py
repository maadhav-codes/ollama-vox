"""Unit tests for ``ollama_vox.core.config``.

Tests verify that :class:`AppConfig` (and its sub-configs) correctly:
* Apply default values when no config keys are provided.
* Coerce string values from YAML to the right Python types.
* Reject invalid values with :class:`ConfigValidationError`.
* Parse nested ``styles`` dictionaries.
"""

import pytest

from ollama_vox.core.config import AppConfig, ConfigValidationError


def test_app_config_defaults():
    """AppConfig.from_dict({}) should populate every field with its default.

    Passing an empty dict simulates a config.yaml with no content.
    All sub-config objects must fall back to their declared defaults so
    the application can start without any configuration file.
    """
    config = AppConfig.from_dict({})
    assert config.audio.sample_rate == 16000
    assert config.audio.vad_enabled is True
    assert config.stt.model == "./whisper/whisper-small.en-mlx-q4"
    assert config.ollama.endpoint == "http://localhost:11434"
    assert config.tts.model == "./kokoro/Kokoro-82M-4bit"
    assert config.queue.maxsize == 4
    assert config.queue.drop_policy == "drop_oldest"


def test_app_config_type_coercion_and_bool_strings():
    """Numeric and boolean values provided as strings must be coerced correctly.

    YAML parsers may load values as strings (e.g. ``vad_enabled: "False"``).
    ``_coerce_type`` must convert them to the proper Python types so that
    comparisons like ``if config.audio.vad_enabled:`` work as expected.
    """
    data = {
        "audio": {
            "sample_rate": "24000",  # string int → int
            "vad_enabled": "False",  # string bool → bool False
            "vad_threshold": "0.02",  # string float → float
        },
        "queue": {"maxsize": "10"},  # string int → int
    }
    config = AppConfig.from_dict(data)
    assert config.audio.sample_rate == 24000
    assert config.audio.vad_enabled is False
    assert config.audio.vad_threshold == 0.02
    assert config.queue.maxsize == 10


def test_app_config_rejects_invalid_types():
    """A non-numeric string for an int field must raise ConfigValidationError.

    ``_coerce_type`` cannot convert ``"not_an_int"`` to ``int``, so it must
    raise with a message that includes the offending field name so the user
    knows which config key to fix.
    """
    with pytest.raises(ConfigValidationError, match="audio.sample_rate"):
        AppConfig.from_dict({"audio": {"sample_rate": "not_an_int"}})


def test_app_config_invalid_drop_policy():
    """An unrecognised drop_policy value must raise ConfigValidationError.

    Only ``"drop_oldest"``, ``"drop_new"``, and ``"block"`` are valid.
    Any other string must be rejected with a clear error message.
    """
    with pytest.raises(ConfigValidationError, match="Invalid queue.drop_policy"):
        AppConfig.from_dict({"queue": {"drop_policy": "invalid"}})


def test_app_config_unknown_nested_key_raises():
    """An unrecognised key inside a known section must raise ConfigValidationError.

    Dataclasses raise ``TypeError`` for unknown keyword arguments. The config
    module must catch this and re-raise it as ``ConfigValidationError`` with a
    helpful message, rather than exposing the raw TypeError to the user.
    """
    with pytest.raises(ConfigValidationError, match="Unknown configuration key"):
        AppConfig.from_dict({"audio": {"unknown": 1}})


def test_app_config_styles_map_and_style_validation():
    """The styles section must be parsed into a dict of StyleConfig objects.

    Each style name maps to a ``StyleConfig`` with optional ``speed``,
    ``pitch``, and ``voice`` fields. Unspecified fields must remain ``None``.
    String values must be coerced to their correct numeric types.
    """
    cfg = AppConfig.from_dict(
        {
            "styles": {
                "fast": {"speed": "1.5", "voice": "af_bella"},  # speed as string
                "high": {"pitch": 1.2},  # pitch as float
            }
        }
    )
    assert cfg.styles["fast"].speed == 1.5
    assert cfg.styles["fast"].voice == "af_bella"
    assert cfg.styles["fast"].pitch is None  # not specified → None
    assert cfg.styles["high"].pitch == 1.2


def test_app_config_styles_must_be_dict():
    """Providing a list for the styles section must raise ConfigValidationError.

    The ``styles`` value must be a dict (mapping name → overrides). Providing
    any other type (e.g. a list) should be caught and reported clearly.
    """
    with pytest.raises(
        ConfigValidationError, match="Styles config must be a dictionary"
    ):
        AppConfig.from_dict({"styles": []})
