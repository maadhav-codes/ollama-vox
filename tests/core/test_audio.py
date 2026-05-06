"""Unit tests for ``ollama_vox.core.audio.AudioRecorder``.

These tests verify the lifecycle (start → stop), auto-stop logic, and
edge-case behaviour of the microphone recorder without touching any real
audio hardware. Hardware dependencies (sounddevice, soundfile) are replaced
by lightweight fakes defined locally or patched via ``mocker``.
"""

import numpy as np

from ollama_vox.core.audio import AudioRecorder


class _FakeStream:
    """Minimal stand-in for ``sounddevice.InputStream``.

    Records which lifecycle methods were called so tests can assert that the
    recorder correctly opens and closes the audio stream.
    """

    def __init__(self):
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        """Simulate starting the PortAudio stream."""
        self.started = True

    def stop(self):
        """Simulate stopping the PortAudio stream."""
        self.stopped = True

    def close(self):
        """Simulate closing the PortAudio stream."""
        self.closed = True


class _FakeWriter:
    """Minimal stand-in for ``soundfile.SoundFile`` in write mode.

    Tracks whether ``close()`` was called so tests can verify the recorder
    flushes data before reading the temp file back.
    """

    def __init__(self):
        self.closed = False

    def write(self, _):
        """Accept audio data without doing anything (no-op in tests)."""
        return None

    def close(self):
        """Mark the writer as closed."""
        self.closed = True


def test_start_and_stop_lifecycle(mocker):
    """start() opens a stream and stop() returns a NumPy array and cleans up.

    After start():
        * ``recording`` must be ``True``.
        * ``stream`` must be the fake stream object.

    After stop():
        * ``recording`` must be ``False``.
        * ``stream`` must be ``None`` (released).
        * The returned value must be a NumPy ndarray.
        * The temp WAV file must be deleted (``os.unlink`` called).
    """
    fake_stream = _FakeStream()
    fake_writer = _FakeWriter()

    # Patch all hardware/filesystem interactions.
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
    # Temp file must be deleted to avoid disk leaks.
    unlink.assert_called_once_with("/tmp/fake.wav")


def test_stop_when_not_recording_returns_empty_array():
    """Calling stop() before start() must return an empty (0,1) float32 array.

    This guards against the UI calling stop() multiple times or before
    any recording has begun.
    """
    rec = AudioRecorder()
    out = rec.stop()
    # Shape (0, 1) matches the always_2d=True convention used in stop().
    assert out.shape == (0, 1)


def test_should_auto_stop_for_max_duration(mocker):
    """should_auto_stop() returns True when elapsed time exceeds max_duration.

    We freeze ``time.monotonic`` to return a value that places the elapsed
    time 0.1 s beyond the 1.0 s limit, ensuring a deterministic result.
    """
    rec = AudioRecorder(max_duration_seconds=1.0)
    rec.recording = True
    rec._started_at = 10.0
    # Pretend the clock now reads 11.1 → elapsed = 1.1 s > 1.0 s limit.
    mocker.patch("ollama_vox.core.audio.time.monotonic", return_value=11.1)

    assert rec.should_auto_stop() is True
    assert rec._auto_stop_reason == "max_duration"


def test_should_auto_stop_for_vad_silence(mocker):
    """should_auto_stop() returns True when silence run exceeds the threshold.

    We manually set ``_silence_run_seconds`` above the configured threshold
    to simulate a sustained period of silence detected by the VAD callback.
    """
    mocker.patch("ollama_vox.core.audio.time.monotonic", return_value=0.0)
    rec = AudioRecorder(vad_enabled=True, vad_silence_seconds=0.5)
    rec.recording = True
    rec._started_at = 0.0
    # Simulate 0.6 s of consecutive silence (above the 0.5 s threshold).
    rec._silence_run_seconds = 0.6

    assert rec.should_auto_stop() is True
    assert rec._auto_stop_reason == "vad_silence"
