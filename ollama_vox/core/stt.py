"""Speech-to-Text (STT) transcription module using MLX Whisper.

This module wraps the ``mlx_whisper`` library (an Apple-Silicon-optimised
port of OpenAI's Whisper) to convert a NumPy audio array into a string of
transcribed text.

How it fits into the pipeline::

    AudioRecorder → [audio array] → STT → [text string] → OllamaClient

Role in the architecture
-------------------------
:class:`STT` is the second stage of the pipeline. It receives raw float32
audio from the :class:`~ollama_vox.core.audio.AudioRecorder`, splits it into
3-second chunks for live progress display, then passes the full audio to
Whisper for the final high-quality transcription.

Dependencies:
    * ``mlx_whisper`` — runs Whisper on Apple Silicon using the MLX framework.
    * ``numpy``        — array manipulation.
    * ``rich`` (optional) — coloured terminal output. Falls back to ANSI
      escape codes if ``rich`` is not installed.
"""

import logging
import time

import mlx_whisper
import numpy as np

logger = logging.getLogger(__name__)


class _TerminalTranscriptPrinter:
    """Pretty-prints live and final transcription results to the terminal.

    This private helper class abstracts the difference between having the
    ``rich`` library installed vs. falling back to plain ANSI escape codes.

    Attributes:
        _use_rich (bool): ``True`` if ``rich`` was successfully imported.
        _console (rich.console.Console or None): Rich console, or ``None``.
    """

    def __init__(self):
        """Initialise the printer, attempting to import ``rich``."""
        self._use_rich = False
        self._console = None
        try:
            from rich.console import Console

            self._console = Console()
            self._use_rich = True
        except ImportError:
            # rich is not installed; fall back to ANSI escape codes.
            pass

    def live(self, text: str) -> None:
        """Print an intermediate (live) transcription chunk in cyan.

        Args:
            text (str): Interim transcription text. Empty strings are ignored.
        """
        text = text.strip()
        if not text:
            return
        if self._use_rich:
            self._console.print(f"[cyan][LIVE][/cyan] {text}")
        else:
            # \033[96m = bright cyan, \033[0m = colour reset
            print(f"\033[96m[LIVE]\033[0m {text}")

    def final(self, text: str) -> None:
        """Print the final, complete transcription in bold green.

        Args:
            text (str): Final transcription text (may be empty).
        """
        text = text.strip()
        if self._use_rich:
            self._console.print(f"[bold green][FINAL][/bold green] {text}")
        else:
            # \033[1;32m = bold + green, \033[0m = colour reset
            print(f"\033[1;32m[FINAL]\033[0m {text}")


class STT:
    """Speech-to-Text engine backed by MLX Whisper.

    Transcribes a NumPy float32 audio array into a text string. Implements
    exponential-backoff retry logic for transient model errors, and provides
    a chunked "live preview" pass so the user sees interim results quickly.

    Typical usage::

        stt = STT(model="./whisper/whisper-small.en-mlx-q4")
        text = stt.transcribe(audio_array, sr=16000)

    Args:
        model (str): File-system path (or HuggingFace repo ID) of the Whisper
            model directory.
        retries (int): Number of additional attempts after the first failure.
            Total attempts = ``retries + 1``. Default: 2.
        backoff_seconds (float): Base sleep time between retries. Grows
            exponentially: attempt i waits ``backoff_seconds * 2^i`` seconds.
            Default: 0.35.
    """

    def __init__(self, model, retries=2, backoff_seconds=0.35):
        self.model = model
        self.printer = _TerminalTranscriptPrinter()
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Run Whisper on a single 1-D audio array with retry logic.

        This private method wraps ``mlx_whisper.transcribe`` with
        exponential-backoff retries. Called for both 3-second live chunks
        and the final full-audio pass.

        Args:
            audio_data (numpy.ndarray): 1-D float32 audio at 16000 Hz.

        Returns:
            str: Transcribed text, stripped of whitespace.
                 Returns ``""`` if all attempts fail.
        """
        last_error = None

        # Try up to (retries + 1) times total.
        for attempt in range(self.retries + 1):
            try:
                result = mlx_whisper.transcribe(
                    audio_data,
                    path_or_hf_repo=self.model,
                )
                # mlx_whisper returns {"text": "...", "segments": [...]}
                return result.get("text", "").strip()
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    # Exponential backoff: 0.35 s, 0.70 s, 1.40 s, …
                    time.sleep(self.backoff_seconds * (2**attempt))

        # All attempts failed — log and return empty string.
        logger.exception(
            "event=stt_transcribe_failed model=%s retries=%s",
            self.model,
            self.retries,
            exc_info=last_error,
        )
        return ""

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        """Transcribe an audio array to text with a live chunked preview.

        Performs two passes:

        1. **Chunked live pass** — splits audio into 3-second chunks,
           transcribes each, and prints growing interim text to the terminal.
        2. **Final full pass** — transcribes the entire audio for the best
           accuracy, then prints and returns the result.

        Args:
            audio (numpy.ndarray): Float32 audio data (1-D or 2-D).
                Will be flattened to 1-D internally.
            sr (int): Sample rate in Hz (should be 16000 for Whisper).

        Returns:
            str: Final transcribed text, or ``""`` if audio is empty or
                 transcription fails.

        Example:
            >>> audio = np.zeros(16000, dtype=np.float32)
            >>> STT("./model").transcribe(audio, sr=16000)
            ''
        """
        # Guard: empty or None audio → nothing to transcribe.
        if audio is None or len(audio) == 0:
            self.printer.final("")
            return ""

        # Flatten to 1-D — Whisper expects a flat array.
        audio_1d = np.asarray(audio).reshape(-1)

        # --- Pass 1: chunked live preview ---
        chunk_seconds = 3
        chunk_size = int(sr * chunk_seconds)
        num_samples = len(audio_1d)
        live_segments = []

        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            chunk_audio = audio_1d[start:end]
            if len(chunk_audio) == 0:
                continue

            partial = self._transcribe_audio(chunk_audio)

            if partial:
                live_segments.append(partial)
                # Show all segments seen so far joined as a running transcript.
                self.printer.live(" ".join(live_segments).strip())

        # --- Pass 2: final full-audio transcription ---
        # Full context gives Whisper better accuracy than individual chunks.
        final_text = self._transcribe_audio(audio_1d)

        self.printer.final(final_text)
        return final_text
