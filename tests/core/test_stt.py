import numpy as np
from ollama_vox.core.stt import STT


def test_stt_init():
    stt = STT(model="test-model")
    assert stt.model == "test-model"


def test_stt_transcribe_empty():
    stt = STT(model="test-model")
    result = stt.transcribe([], sr=16000)
    assert result == ""


def test_stt_transcribe_calls_whisper(mocker):
    mock_mlx_whisper = mocker.patch("ollama_vox.core.stt.mlx_whisper")
    mock_mlx_whisper.transcribe.return_value = {"text": "mocked transcription"}
    stt = STT(model="test-model")

    # 3 seconds of audio at 16000 sr to simulate one chunk
    audio = np.zeros(16000 * 3, dtype=np.float32)

    result = stt.transcribe(audio, sr=16000)

    assert result == "mocked transcription"
    # Transcribe should be called once for chunk, once for final
    assert mock_mlx_whisper.transcribe.call_count == 2
