import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
import os
import time
import logging

logger = logging.getLogger(__name__)


class AudioRecorder:
    def __init__(
        self,
        sample_rate=16000,
        vad_enabled=True,
        vad_threshold=0.015,
        vad_silence_seconds=1.2,
        max_duration_seconds=20.0,
    ):
        self.sample_rate = sample_rate
        self.vad_enabled = vad_enabled
        self.vad_threshold = float(vad_threshold)
        self.vad_silence_seconds = float(vad_silence_seconds)
        self.max_duration_seconds = float(max_duration_seconds)
        self.recording = False
        self.stream = None
        self._tmp_file = None
        self._tmp_path = None
        self._writer = None
        self._started_at = None
        self._silence_run_seconds = 0.0
        self._auto_stop_reason = None

    def start(self):
        self.recording = True
        self._started_at = time.monotonic()
        self._silence_run_seconds = 0.0
        self._auto_stop_reason = None
        self._tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._tmp_path = self._tmp_file.name
        self._tmp_file.close()
        self._writer = sf.SoundFile(
            self._tmp_path, mode="w", samplerate=self.sample_rate, channels=1
        )

        def callback(indata, frames, time, status):
            if self.recording:
                self._writer.write(indata)
                if self.vad_enabled:
                    rms = float(np.sqrt(np.mean(np.square(indata.astype(np.float32)))))
                    chunk_seconds = frames / float(self.sample_rate)
                    if rms < self.vad_threshold:
                        self._silence_run_seconds += chunk_seconds
                    else:
                        self._silence_run_seconds = 0.0

        self.stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, callback=callback
        )
        self.stream.start()

    def should_auto_stop(self):
        if not self.recording or self._started_at is None:
            return False

        elapsed = time.monotonic() - self._started_at
        if elapsed >= self.max_duration_seconds:
            self._auto_stop_reason = "max_duration"
            return True

        if self.vad_enabled and self._silence_run_seconds >= self.vad_silence_seconds:
            self._auto_stop_reason = "vad_silence"
            return True

        return False

    def stop(self):
        if not self.recording:
            return np.empty((0, 1), dtype=np.float32)
        self.recording = False
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None

        audio, _ = sf.read(self._tmp_path, dtype="float32", always_2d=True)
        os.unlink(self._tmp_path)
        self._tmp_path = None
        self._tmp_file = None
        logger.info(
            "event=recording_stopped reason=%s duration_seconds=%.3f samples=%s",
            self._auto_stop_reason or "manual",
            (time.monotonic() - self._started_at) if self._started_at else 0.0,
            len(audio),
        )
        self._started_at = None
        self._silence_run_seconds = 0.0
        self._auto_stop_reason = None
        return audio
