from unittest.mock import MagicMock

import numpy as np

from ollama_vox.core.tts import TTS


def _tts(mocker, **kwargs):
    mocker.patch(
        "ollama_vox.core.tts.TTS._load_model_id_from_config", return_value="mock_model"
    )
    return TTS(**kwargs)


def test_split_text_respects_sentence_boundaries_before_chunking(mocker):
    tts = _tts(mocker, split_chars=10)
    chunks = list(tts._split_text("This is one. This is two."))
    assert chunks == ["This is on", "e.", "This is tw", "o."]


def test_split_text_empty_returns_empty_list(mocker):
    tts = _tts(mocker)
    assert list(tts._split_text("   ")) == []


def test_play_audio_noop_on_interrupt_or_empty(mocker):
    tts = _tts(mocker)
    sd_play = mocker.patch("ollama_vox.core.tts.sd.play")

    tts._interrupt = True
    tts._play_audio(np.array([0.1], dtype=np.float32))
    tts._interrupt = False
    tts._play_audio(np.array([], dtype=np.float32))

    sd_play.assert_not_called()


def test_speak_applies_style_and_plays_generated_audio(mocker):
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
    tts._model = model

    tts.speak("Hello world.", style="friendly")

    kwargs = model.generate.call_args.kwargs
    assert kwargs["voice"] == "v2"
    assert kwargs["speed"] == 1.3
    assert kwargs["pitch"] == 1.1
    sd_play.assert_called_once()


def test_speak_retries_after_pitch_typeerror_without_pitch(mocker):
    tts = _tts(mocker, voice="v")
    result = MagicMock()
    result.audio = [0.1]
    model = MagicMock()
    model.generate.side_effect = [TypeError("no pitch"), [result]]
    tts._model = model
    mocker.patch("ollama_vox.core.tts.sd.play")

    tts.speak("hi", pitch=1.2)

    first_kwargs = model.generate.call_args_list[0].kwargs
    second_kwargs = model.generate.call_args_list[1].kwargs
    assert "pitch" in first_kwargs
    assert "pitch" not in second_kwargs


def test_stop_sets_interrupt_and_stops_device(mocker):
    tts = _tts(mocker)
    sd_stop = mocker.patch("ollama_vox.core.tts.sd.stop")

    tts.stop()

    assert tts._interrupt is True
    sd_stop.assert_called_once()
