"""Unit tests for ``ollama_vox.core.tts.TTS``.

Verifies text splitting, audio playback, style application, pitch-fallback
retry logic, and the stop/interrupt mechanism — all without loading a real
Kokoro model. The ``_load_model_id_from_config`` class method is patched in
the shared ``_tts()`` factory so each test starts with a clean TTS instance.
"""

from unittest.mock import MagicMock

import numpy as np

from ollama_vox.core.tts import TTS


def _tts(mocker, **kwargs) -> TTS:
    """Factory that creates a :class:`TTS` instance with config loading mocked.

    Patches ``TTS._load_model_id_from_config`` so tests don't need a real
    ``config.yaml`` on disk. Any additional keyword arguments are forwarded
    to the ``TTS`` constructor.

    Args:
        mocker: pytest-mock's ``mocker`` fixture.
        **kwargs: Extra keyword arguments for the ``TTS`` constructor
            (e.g. ``voice``, ``rate``, ``split_chars``, ``style_map``).

    Returns:
        TTS: A freshly constructed TTS instance ready for testing.
    """
    mocker.patch(
        "ollama_vox.core.tts.TTS._load_model_id_from_config", return_value="mock_model"
    )
    return TTS(**kwargs)


def test_split_text_respects_sentence_boundaries_before_chunking(mocker):
    """_split_text() splits at sentence boundaries, then hard-splits at split_chars.

    With ``split_chars=10``, neither ``"This is one."`` nor ``"This is two."``
    fits in 10 chars, so each sentence is further hard-split every 10 chars.
    """
    tts = _tts(mocker, split_chars=10)
    chunks = list(tts._split_text("This is one. This is two."))
    assert chunks == ["This is on", "e.", "This is tw", "o."]


def test_split_text_empty_returns_empty_list(mocker):
    """_split_text() on whitespace-only text must return an empty list."""
    tts = _tts(mocker)
    assert list(tts._split_text("   ")) == []


def test_play_audio_noop_on_interrupt_or_empty(mocker):
    """_play_audio() must not call sd.play() when interrupted or given empty audio.

    Two scenarios are tested:
    * ``_interrupt=True`` → abort before playing.
    * Empty array after conversion → skip silently.
    """
    tts = _tts(mocker)
    sd_play = mocker.patch("ollama_vox.core.tts.sd.play")

    # Scenario 1: interrupted flag set.
    tts._interrupt = True
    tts._play_audio(np.array([0.1], dtype=np.float32))

    # Scenario 2: empty array (size 0 after conversion).
    tts._interrupt = False
    tts._play_audio(np.array([], dtype=np.float32))

    # Neither scenario should have triggered playback.
    sd_play.assert_not_called()


def test_speak_applies_style_and_plays_generated_audio(mocker):
    """speak() with a named style overrides voice/speed/pitch passed to model.generate().

    The ``"friendly"`` style sets voice=``"v2"``, speed=``1.3``, pitch=``1.1``.
    These must override the TTS instance defaults and be forwarded to the
    model's ``generate()`` call.
    """
    tts = _tts(
        mocker,
        voice="default_voice",
        rate=1.0,
        style_map={"friendly": {"voice": "v2", "speed": 1.3, "pitch": 1.1}},
    )

    sd_play = mocker.patch("ollama_vox.core.tts.sd.play")
    result = MagicMock()
    result.audio = [0.1, 0.2]
    model = MagicMock()
    model.generate.return_value = [result]
    # Inject the mock model to avoid loading a real Kokoro checkpoint.
    tts._model = model

    tts.speak("Hello world.", style="friendly")

    # Assert the style overrides were forwarded to model.generate().
    kwargs = model.generate.call_args.kwargs
    assert kwargs["voice"] == "v2"
    assert kwargs["speed"] == 1.3
    assert kwargs["pitch"] == 1.1
    sd_play.assert_called_once()


def test_speak_retries_after_pitch_typeerror_without_pitch(mocker):
    """speak() retries model.generate() without 'pitch' after a TypeError.

    Some Kokoro versions don't accept a ``pitch`` parameter. When
    ``model.generate()`` raises ``TypeError``, speak() must retry the
    call without ``pitch`` so synthesis still succeeds.
    """
    tts = _tts(mocker, voice="v")
    result = MagicMock()
    result.audio = [0.1]
    model = MagicMock()
    # First call raises TypeError (pitch not supported); second call succeeds.
    model.generate.side_effect = [TypeError("no pitch"), [result]]
    tts._model = model
    mocker.patch("ollama_vox.core.tts.sd.play")

    tts.speak("hi", pitch=1.2)

    first_kwargs = model.generate.call_args_list[0].kwargs
    second_kwargs = model.generate.call_args_list[1].kwargs
    # First attempt should have included pitch.
    assert "pitch" in first_kwargs
    # Retry must not include pitch.
    assert "pitch" not in second_kwargs


def test_stop_sets_interrupt_and_stops_device(mocker):
    """stop() sets _interrupt=True and calls sd.stop() to halt playback.

    This verifies the interrupt mechanism used when the user starts speaking
    while the assistant is still talking.
    """
    tts = _tts(mocker)
    sd_stop = mocker.patch("ollama_vox.core.tts.sd.stop")

    tts.stop()

    assert tts._interrupt is True
    sd_stop.assert_called_once()
