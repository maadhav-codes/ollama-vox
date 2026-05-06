"""Microphone recording module with optional Voice Activity Detection (VAD).

This module provides the :class:`AudioRecorder` class, which captures audio from the default system microphone using the ``sounddevice`` library and writes it to a temporary WAV file while recording is in progress.

How it fits into the bigger picture
------------------------------------
``AudioRecorder`` is the *first* stage of the Ollama Vox pipeline::

    Microphone → AudioRecorder → STT → LLM → TTS → Speakers

The UI (system-tray app) calls :meth:`AudioRecorder.start` when the user begins speaking and :meth:`AudioRecorder.stop` when the user finishes. The raw audio array returned by ``stop()`` is passed directly to the STT worker thread for transcription.

Voice Activity Detection (VAD)
-------------------------------
When ``vad_enabled=True``, the recorder continuously measures the RMS energy of each incoming audio chunk. If the energy falls below ``vad_threshold`` for at least ``vad_silence_seconds`` consecutive seconds, the tray app's polling timer detects this via :meth:`AudioRecorder.should_auto_stop` and calls ``stop()`` automatically — so the user doesn't need to click a button.

Dependencies:
    * ``sounddevice`` — wraps PortAudio for cross-platform audio I/O.
    * ``soundfile``   — reads/writes WAV files.
    * ``numpy``       — array maths for RMS energy calculation.
"""

import logging
import os
import tempfile
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

# Module-level logger. Uses the Python module path as its name so log
# messages show up as "ollama_vox.core.audio" in log output, making it
# easy to filter by module.
logger = logging.getLogger(__name__)


class AudioRecorder:
    """Records audio from the default microphone with optional auto-stop.

    The recorder uses a streaming ``sounddevice.InputStream`` that writes audio chunks to a temporary WAV file in real time. When recording stops, the file is read back into a NumPy array, the temp file is deleted, and the array is returned to the caller.

    This design (write-to-file, then read-back) avoids accumulating a large in-memory buffer while recording, and ensures the data is safely on disk even if something goes wrong mid-recording.

    Auto-stop behaviour:
        The recorder itself does **not** start a background thread to call ``stop()``; instead it exposes :meth:`should_auto_stop` so the UI timer can poll it and decide when to stop. This keeps the recorder simple and free of Qt/threading dependencies.

    Thread safety:
        The ``sounddevice`` callback runs in a real-time audio thread. The only shared mutable state it touches is ``self.recording``, ``self._writer``, and ``self._silence_run_seconds``. These are safe because the GIL protects simple attribute reads/writes in CPython.

    Args:
        sample_rate (int): Audio samples per second. Must match the expected
            input rate of the STT model (Whisper expects 16000 Hz).
            Default: 16000.
        vad_enabled (bool): Enable automatic silence detection. When
            ``True``, :meth:`should_auto_stop` may return ``True`` after a
            long silence. Default: True.
        vad_threshold (float): RMS energy threshold below which audio is
            considered silent. In the range [0.0, 1.0] for float32 audio.
            Default: 0.015.
        vad_silence_seconds (float): Duration of consecutive silence (in
            seconds) required before auto-stop triggers. Default: 1.2.
        max_duration_seconds (float): Hard recording limit in seconds.
            After this many seconds, :meth:`should_auto_stop` returns
            ``True`` regardless of VAD state. Default: 20.0.

    Attributes:
        recording (bool): ``True`` while recording is active.
        stream (sounddevice.InputStream or None): The active audio stream,
            or ``None`` when not recording.
    """

    def __init__(
        self,
        sample_rate=16000,
        vad_enabled=True,
        vad_threshold=0.015,
        vad_silence_seconds=1.2,
        max_duration_seconds=20.0,
    ):
        # Store config params as instance attributes for use in start() and
        # should_auto_stop().
        self.sample_rate = sample_rate
        self.vad_enabled = vad_enabled

        # Ensure threshold and timing values are always floats, even if the
        # caller accidentally passes strings or ints.
        self.vad_threshold = float(vad_threshold)
        self.vad_silence_seconds = float(vad_silence_seconds)
        self.max_duration_seconds = float(max_duration_seconds)

        # --- State flags (reset each time start() is called) ---
        self.recording = False  # True only while mic stream is active
        self.stream = None  # sounddevice InputStream object

        # --- Temporary file handles ---
        # We write audio to a temp WAV file while recording so we don't have
        # to hold the entire recording in RAM.
        self._tmp_file = None  # NamedTemporaryFile object
        self._tmp_path = None  # String path to the temp WAV file
        self._writer = None  # sf.SoundFile writer object

        # --- Timing and VAD state ---
        self._started_at = None  # monotonic timestamp of start()
        self._silence_run_seconds = 0.0  # cumulative silence duration
        self._auto_stop_reason = None  # "max_duration" | "vad_silence" | None

    def start(self):
        """Begin capturing audio from the default microphone.

        Opens a temporary WAV file, creates a ``sounddevice.InputStream``, and starts the real-time audio callback. Each chunk of incoming audio is written to the temp file. If VAD is enabled, the chunk's RMS energy is also used to track cumulative silence duration.

        Call :meth:`stop` to end recording and retrieve the audio data.

        Side effects:
            * Creates a temporary ``.wav`` file on disk.
            * Opens a ``sounddevice.InputStream`` (claims the microphone).
            * Resets VAD counters and the auto-stop reason.

        Note:
            Calling ``start()`` when already recording will open a *second* stream and temp file, leaking the first. Always call ``stop()`` before calling ``start()`` again.
        """
        self.recording = True

        # Record the wall-clock start time so we can check elapsed duration
        # in should_auto_stop(). We use monotonic() to avoid problems if
        # the system clock changes (e.g. NTP sync).
        self._started_at = time.monotonic()
        self._silence_run_seconds = 0.0
        self._auto_stop_reason = None

        # Create a temporary WAV file. delete=False means the OS won't
        # automatically remove it when the file object is closed — we need
        # it to persist until stop() reads it back.
        self._tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._tmp_path = self._tmp_file.name
        # Close the file handle immediately; the SoundFile writer below will
        # open the same path by name.
        self._tmp_file.close()

        # Open the temp file as a SoundFile writer. We write mono (1 channel)
        # audio at the configured sample rate.
        self._writer = sf.SoundFile(
            self._tmp_path, mode="w", samplerate=self.sample_rate, channels=1
        )

        def callback(indata, frames, _time_info, _status):
            """Real-time audio callback invoked by sounddevice for each chunk.

            This function runs in a dedicated high-priority thread managed by PortAudio. It should be fast and avoid blocking I/O or heavy computation.

            Args:
                indata (numpy.ndarray): Audio samples for this chunk, shape
                    ``(frames, channels)``, dtype ``float32``.
                frames (int): Number of audio frames (samples) in ``indata``.
                _time_info: Timing info from PortAudio (not used here).
                _status: Status flags from PortAudio (not used here).
            """
            if self.recording:
                # Persist audio samples to disk immediately. SoundFile
                # handles buffering internally.
                self._writer.write(indata)

                if self.vad_enabled:
                    # Compute Root Mean Square (RMS) energy of this chunk.
                    # RMS gives a single number representing how "loud" the
                    # chunk is. Low RMS → silence, high RMS → speech.
                    # We cast to float32 first to avoid integer overflow in
                    # the squaring step.
                    rms = float(np.sqrt(np.mean(np.square(indata.astype(np.float32)))))

                    # Convert the chunk size from samples to seconds so we
                    # can accumulate silence duration independently of
                    # chunk size.
                    chunk_seconds = frames / float(self.sample_rate)

                    if rms < self.vad_threshold:
                        # This chunk is silent — accumulate silence time.
                        self._silence_run_seconds += chunk_seconds
                    else:
                        # Speech detected — reset the silence counter so
                        # that only *consecutive* silence triggers auto-stop.
                        self._silence_run_seconds = 0.0

        # Create and start the input stream. sounddevice will call `callback`
        # for each chunk of audio received from the microphone.
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, callback=callback
        )
        self.stream.start()

    def should_auto_stop(self) -> bool:
        """Check whether the recording should stop automatically.

        This method is polled by the UI timer (every ~200 ms) to decide
        whether to call :meth:`stop` without user interaction. Two conditions
        can trigger an auto-stop:

        1. **Max duration exceeded**: the recording has been running for
           longer than ``self.max_duration_seconds``.
        2. **VAD silence**: ``vad_enabled`` is ``True`` and the mic has been
           silent for at least ``self.vad_silence_seconds`` seconds in a row.

        Returns:
            bool: ``True`` if recording should stop, ``False`` otherwise.
                  Also returns ``False`` if not currently recording.

        Side effects:
            Sets ``self._auto_stop_reason`` to ``"max_duration"`` or
            ``"vad_silence"`` when returning ``True``. This string is later
            included in the log message produced by :meth:`stop`.
        """
        # Can't auto-stop if we're not recording.
        if not self.recording or self._started_at is None:
            return False

        # Check how many seconds have passed since start().
        elapsed = time.monotonic() - self._started_at

        if elapsed >= self.max_duration_seconds:
            self._auto_stop_reason = "max_duration"
            return True

        # Check if VAD has accumulated enough consecutive silence.
        if self.vad_enabled and self._silence_run_seconds >= self.vad_silence_seconds:
            self._auto_stop_reason = "vad_silence"
            return True

        return False

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a NumPy array.

        Stops the ``sounddevice`` stream, flushes and closes the WAV writer,
        reads the temp file back into a NumPy array, then deletes the temp
        file to free disk space.

        Returns:
            numpy.ndarray: Float32 audio array with shape ``(N, 1)`` where
                ``N`` is the total number of captured samples.
                Returns an empty array of shape ``(0, 1)`` if called when
                not recording (safe no-op).

        Side effects:
            * Stops and closes the ``sounddevice.InputStream``.
            * Deletes the temporary WAV file from disk.
            * Resets all recording state (``recording``, ``stream``,
              ``_writer``, ``_tmp_path``, ``_started_at``, etc.).
            * Emits an INFO log line with the stop reason and duration.

        Example:
            >>> rec = AudioRecorder()
            >>> rec.start()
            >>> audio = rec.stop()
            >>> audio.shape
            (48000, 1)  # 3 seconds at 16 kHz, for example
        """
        # If called when not recording, return a valid but empty array.
        # Shape (0, 1) matches the always_2d=True shape of sf.read().
        if not self.recording:
            return np.empty((0, 1), dtype=np.float32)

        # Signal the callback to stop writing (it checks self.recording).
        self.recording = False

        # Stop the PortAudio stream — no more callbacks after this point.
        self.stream.stop()
        self.stream.close()
        self.stream = None

        # Flush and close the SoundFile writer to ensure all buffered data
        # is written to disk before we try to read it back.
        if self._writer is not None:
            self._writer.close()
            self._writer = None

        # Read the complete audio from disk. always_2d=True ensures the
        # array always has shape (N, channels) even for mono audio.
        audio, _ = sf.read(self._tmp_path, dtype="float32", always_2d=True)

        # Clean up: delete the temp file now that we have the data in memory.
        os.unlink(self._tmp_path)
        self._tmp_path = None
        self._tmp_file = None

        # Log a summary of this recording session for debugging purposes.
        logger.info(
            "event=recording_stopped reason=%s duration_seconds=%.3f samples=%s",
            self._auto_stop_reason or "manual",
            (time.monotonic() - self._started_at) if self._started_at else 0.0,
            len(audio),
        )

        # Reset timing state so the object is ready for the next start().
        self._started_at = None
        self._silence_run_seconds = 0.0
        self._auto_stop_reason = None

        return audio
