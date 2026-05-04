import numpy as np

from ollama_vox.core.stt import STT


def test_transcribe_empty_audio_returns_empty_and_prints_final(mocker):
    stt = STT(model="test-model")
    final = mocker.patch.object(stt.printer, "final")

    result = stt.transcribe([], sr=16000)

    assert result == ""
    final.assert_called_once_with("")


def test_transcribe_chunked_live_and_final(mocker):
    mocker.patch(
        "ollama_vox.core.stt.mlx_whisper.transcribe", return_value={"text": "hello"}
    )
    stt = STT(model="test-model")
    live = mocker.patch.object(stt.printer, "live")
    final = mocker.patch.object(stt.printer, "final")

    audio = np.zeros(16000 * 7, dtype=np.float32)
    result = stt.transcribe(audio, sr=16000)

    assert result == "hello"
    assert live.call_count == 3
    final.assert_called_once_with("hello")


def test__transcribe_audio_retries_and_returns_empty(mocker):
    mocker.patch(
        "ollama_vox.core.stt.mlx_whisper.transcribe", side_effect=Exception("fail")
    )
    stt = STT(model="test-model", retries=1, backoff_seconds=0)

    assert stt._transcribe_audio(np.zeros(4, dtype=np.float32)) == ""
