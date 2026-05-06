"""Unit tests for ``ollama_vox.core.stt.STT``.

Verifies the transcription interface without a real Whisper model by
patching ``mlx_whisper.transcribe`` via the autouse fixture in conftest.py
and with per-test mocker patches where specific return values are needed.
"""

import numpy as np

from ollama_vox.core.stt import STT


def test_transcribe_empty_audio_returns_empty_and_prints_final(mocker):
    """transcribe() with empty audio must return '' and call printer.final('').

    An empty audio array (e.g. user clicked Stop immediately) must be handled
    gracefully without calling Whisper, since passing an empty array to the
    model would likely raise an error or produce meaningless output.
    """
    stt = STT(model="test-model")
    # Spy on the printer so we can assert it received the correct call.
    final = mocker.patch.object(stt.printer, "final")

    result = stt.transcribe([], sr=16000)

    assert result == ""
    # printer.final("") must be called exactly once to signal end of transcript.
    final.assert_called_once_with("")


def test_transcribe_chunked_live_and_final(mocker):
    """transcribe() calls printer.live() once per 3-second chunk and final() once.

    A 7-second audio clip at 16000 Hz produces three 3-second chunks
    (0–3 s, 3–6 s, 6–7 s), so ``printer.live`` should be called 3 times.
    The final transcription of the full audio should result in exactly one
    ``printer.final`` call.
    """
    mocker.patch(
        "ollama_vox.core.stt.mlx_whisper.transcribe", return_value={"text": "hello"}
    )
    stt = STT(model="test-model")
    live = mocker.patch.object(stt.printer, "live")
    final = mocker.patch.object(stt.printer, "final")

    # 7 seconds of silent audio — content doesn't matter, only length.
    audio = np.zeros(16000 * 7, dtype=np.float32)
    result = stt.transcribe(audio, sr=16000)

    assert result == "hello"
    # 3 chunks (0-3s, 3-6s, 6-7s) → 3 live updates.
    assert live.call_count == 3
    final.assert_called_once_with("hello")


def test__transcribe_audio_retries_and_returns_empty(mocker):
    """_transcribe_audio() returns '' after exhausting all retries on failure.

    With ``retries=1`` (2 total attempts) and a Whisper mock that always
    raises, the method must catch all exceptions, log the error, and return
    an empty string instead of propagating the exception to the caller.
    """
    mocker.patch(
        "ollama_vox.core.stt.mlx_whisper.transcribe", side_effect=Exception("fail")
    )
    # backoff_seconds=0 avoids sleep() delays in tests.
    stt = STT(model="test-model", retries=1, backoff_seconds=0)

    assert stt._transcribe_audio(np.zeros(4, dtype=np.float32)) == ""
