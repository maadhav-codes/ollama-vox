import numpy as np

from ollama_vox.core.audio import AudioRecorder


class _FakeStream:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _FakeWriter:
    def __init__(self):
        self.closed = False

    def write(self, _):
        return None

    def close(self):
        self.closed = True


def test_start_and_stop_lifecycle(mocker):
    fake_stream = _FakeStream()
    fake_writer = _FakeWriter()

    mocker.patch(
        "ollama_vox.core.audio.tempfile.NamedTemporaryFile",
        return_value=type(
            "F", (), {"name": "/tmp/fake.wav", "close": lambda self: None}
        )(),
    )
    mocker.patch("ollama_vox.core.audio.sf.SoundFile", return_value=fake_writer)
    mocker.patch("ollama_vox.core.audio.sd.InputStream", return_value=fake_stream)
    mocker.patch(
        "ollama_vox.core.audio.sf.read",
        return_value=(np.zeros((3, 1), dtype=np.float32), 16000),
    )
    unlink = mocker.patch("ollama_vox.core.audio.os.unlink")

    rec = AudioRecorder()
    rec.start()
    assert rec.recording is True
    assert rec.stream is fake_stream

    audio = rec.stop()
    assert rec.recording is False
    assert rec.stream is None
    assert isinstance(audio, np.ndarray)
    unlink.assert_called_once_with("/tmp/fake.wav")


def test_stop_when_not_recording_returns_empty_array():
    rec = AudioRecorder()
    out = rec.stop()
    assert out.shape == (0, 1)


def test_should_auto_stop_for_max_duration(mocker):
    rec = AudioRecorder(max_duration_seconds=1.0)
    rec.recording = True
    rec._started_at = 10.0
    mocker.patch("ollama_vox.core.audio.time.monotonic", return_value=11.1)

    assert rec.should_auto_stop() is True
    assert rec._auto_stop_reason == "max_duration"


def test_should_auto_stop_for_vad_silence(mocker):
    mocker.patch("ollama_vox.core.audio.time.monotonic", return_value=0.0)
    rec = AudioRecorder(vad_enabled=True, vad_silence_seconds=0.5)
    rec.recording = True
    rec._started_at = 0.0
    rec._silence_run_seconds = 0.6

    assert rec.should_auto_stop() is True
    assert rec._auto_stop_reason == "vad_silence"
