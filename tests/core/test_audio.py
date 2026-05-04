import numpy as np
from ollama_vox.core.audio import AudioRecorder


def test_audio_recorder_init():
    recorder = AudioRecorder(sample_rate=24000, vad_enabled=False)
    assert recorder.sample_rate == 24000
    assert recorder.vad_enabled is False
    assert recorder.recording is False


def test_audio_recorder_start_stop(_mocker):
    recorder = AudioRecorder()
    recorder.start()
    assert recorder.recording is True
    assert recorder._tmp_path is not None

    audio = recorder.stop()
    assert recorder.recording is False
    assert recorder.stream is None
    assert recorder._tmp_path is None
    assert isinstance(audio, np.ndarray)


def test_audio_recorder_auto_stop_max_duration(mocker):
    recorder = AudioRecorder(max_duration_seconds=1.0)
    recorder.start()

    # Mock time.monotonic to simulate elapsed time
    mocker.patch("time.monotonic", return_value=recorder._started_at + 1.1)

    assert recorder.should_auto_stop() is True
    assert recorder._auto_stop_reason == "max_duration"


def test_audio_recorder_auto_stop_vad(_mocker):
    recorder = AudioRecorder(vad_silence_seconds=1.0)
    recorder.start()

    # Simulate silence callback accumulation
    recorder._silence_run_seconds = 1.1

    assert recorder.should_auto_stop() is True
    assert recorder._auto_stop_reason == "vad_silence"
