"""Text-to-Speech (TTS) synthesis module using the Kokoro MLX model.

This module provides :class:`TTS`, which converts text strings into spoken
audio played through the system speakers. It uses the ``mlx-audio`` library
to load and run the Kokoro-82M model on Apple Silicon hardware.

How it fits into the pipeline::

    OllamaClient → [sentence] → TTS → [audio played to speakers]

Key features
-------------
* **Lazy model loading**: The Kokoro model is not loaded until the first call
  to :meth:`TTS.speak`, so application startup is fast even if TTS is never
  used in a session.
* **Text splitting**: Long texts are split at sentence boundaries first, then
  hard-chunked at ``split_chars`` characters. This keeps synthesis latency
  low — the first chunk starts playing before the rest is synthesised.
* **Interruptible playback**: Calling :meth:`TTS.stop` sets an interrupt flag
  and calls ``sounddevice.stop()``. The ``speak`` loop checks this flag
  between chunks and aborts immediately.
* **Named styles**: A ``style_map`` dict (from ``config.yaml``) maps style
  names (e.g. ``"friendly"``) to voice/speed/pitch overrides.
* **Pitch fallback**: Some Kokoro model versions don't support the ``pitch``
  parameter. If ``model.generate()`` raises ``TypeError``, we retry without
  ``pitch`` automatically.

Dependencies:
    * ``mlx_audio`` — runs Kokoro on Apple Silicon via the MLX framework.
    * ``sounddevice`` — plays the synthesised audio array through speakers.
    * ``numpy``       — array manipulation.
    * ``yaml``        — reads ``config.yaml`` as a fallback for ``model_id``.
"""

import logging
import re
import time
from collections.abc import Iterable
from pathlib import Path
from threading import Lock

import numpy as np
import sounddevice as sd
import yaml

logger = logging.getLogger(__name__)


class TTS:
    """Text-to-Speech engine backed by Kokoro running on MLX.

    Converts a text string to speech and plays it through the default audio
    output device. Designed to be interruptible: a call to :meth:`stop` will
    abort playback between (or during) chunks.

    Thread safety:
        :meth:`stop` may be called from any thread (e.g. the UI thread) while
        :meth:`speak` runs in the TTS worker thread. The ``_interrupt`` flag
        and ``_lock`` mutex ensure safe cross-thread communication.

    Args:
        voice (Optional[str]): Default voice identifier, e.g. ``"af_bella"``.
            Must correspond to a file in the model's ``voices/`` directory.
        rate (float): Default speech rate multiplier. 1.0 = normal speed.
            Default: 1.0.
        model_id (Optional[str]): Path to the Kokoro model directory. If
            ``None``, :meth:`_load_model_id_from_config` reads it from
            ``config.yaml``.
        sample_rate (int): Output audio sample rate in Hz. Should match the
            Kokoro model's native output rate (24000 Hz). Default: 24000.
        split_chars (int): Maximum characters per synthesis chunk.
            Default: 180.
        style_map (Optional[dict]): Mapping of style name → overrides dict.
            Each overrides dict may contain ``"voice"``, ``"speed"``, and/or
            ``"pitch"`` keys. Default: ``None`` (empty map).

    Attributes:
        _model: The loaded Kokoro model object, or ``None`` before first use.
        _interrupt (bool): When ``True``, ongoing :meth:`speak` calls abort.
        _lock (threading.Lock): Mutex protecting ``_interrupt`` flag.
    """

    def __init__(
        self,
        voice: str | None = None,
        rate: float = 1.0,
        model_id: str | None = None,
        sample_rate: int = 24000,
        split_chars: int = 180,
        style_map: dict | None = None,
    ):
        self.voice = voice
        self.default_speed = float(rate)

        # If no model_id is supplied, fall back to reading config.yaml.
        self.model_id = model_id or self._load_model_id_from_config()
        self.sample_rate = sample_rate
        self.split_chars = split_chars
        self.style_map = style_map or {}

        # --- Lazy-loaded model ---
        # None until the first speak() call; avoids loading a large model
        # at import time or when TTS is never needed.
        self._model = None

        # --- Interrupt mechanism ---
        # _interrupt is set to True by stop() to signal speak() to abort.
        # _lock protects writes to _interrupt from concurrent threads.
        self._interrupt = False
        self._lock = Lock()

    @staticmethod
    def _load_model_id_from_config() -> str:
        """Read the TTS model path from ``config.yaml`` as a fallback.

        This static method is called when no ``model_id`` is passed to the
        constructor. It parses ``config.yaml`` using
        :class:`~ollama_vox.core.config.AppConfig` and extracts the TTS
        model path.

        Returns:
            str: The ``tts.model`` value from ``config.yaml``, or the
                 hard-coded default ``"./kokoro/Kokoro-82M-4bit"`` if the
                 file cannot be read or parsed for any reason.

        Note:
            Failures are silently swallowed so that the TTS object can still
            be constructed even if the config file is missing or corrupted.
        """
        try:
            from pathlib import Path

            from ollama_vox.core.config import AppConfig

            # The config file lives two directories up from this file:
            # ollama_vox/core/tts.py → ollama_vox/ → config.yaml
            config_path = Path(__file__).parent.parent / "config.yaml"
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            config = AppConfig.from_dict(data)
            return config.tts.model
        except (
            OSError,  # File not found or permission denied
            yaml.YAMLError,  # Malformed YAML
            ImportError,  # AppConfig not importable (rare)
            AttributeError,  # Missing attribute in config object
            KeyError,  # Missing key in YAML dict
            TypeError,  # Unexpected type in config data
            ValueError,  # Validation error from AppConfig
        ):
            # Fall back to the well-known default path.
            return "./kokoro/Kokoro-82M-4bit"

    def _load_model(self):
        """Load the Kokoro model from disk on first use (lazy loading).

        Uses a simple existence check (``self._model is not None``) as a
        cache to avoid reloading the model on every call to :meth:`speak`.
        This pattern is often called "lazy initialisation" or "memoisation".

        Returns:
            The Kokoro model object returned by ``mlx_audio.tts.utils.load_model``.

        Raises:
            RuntimeError: If ``mlx-audio`` is not installed.
        """
        # Return cached model if already loaded.
        if self._model is not None:
            return self._model

        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise RuntimeError(
                "mlx-audio is not installed. Install with: pip install mlx-audio"
            ) from exc

        # Load model from the local path. Path() ensures cross-platform
        # compatibility and clean string-to-path conversion.
        self._model = load_model(Path(self.model_id))
        return self._model

    def _split_text(self, text: str) -> Iterable[str]:
        """Split text into synthesisable chunks respecting sentence boundaries.

        Long texts are first split at sentence boundaries (``"."``, ``"!"``,
        ``"?"`` followed by whitespace) to produce natural-sounding pauses.
        Pieces that are still longer than ``self.split_chars`` are then
        hard-split at exactly ``split_chars`` characters to keep each TTS
        call fast.

        Args:
            text (str): The text to split. Whitespace-only strings return
                an empty list.

        Returns:
            Iterable[str]: A list of non-empty text chunks, each at most
                ``self.split_chars`` characters long.

        Example:
            >>> tts = TTS(split_chars=20)
            >>> list(tts._split_text("Hello world. This is a test."))
            ['Hello world.', 'This is a test.']
        """
        text = (text or "").strip()
        if not text:
            return []

        # Split at sentence-ending punctuation followed by whitespace.
        # The lookbehind `(?<=[.!?])` matches the position *after* punctuation
        # without consuming it, so the punctuation stays with the left piece.
        pieces = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            # Try to extend the current chunk with this piece.
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= self.split_chars:
                # Still within the limit — keep accumulating.
                current = candidate
                continue

            # Adding this piece would exceed the limit.
            if current:
                # Flush the current chunk first.
                chunks.append(current)

            if len(piece) <= self.split_chars:
                # The piece fits on its own — start a new chunk with it.
                current = piece
            else:
                # The piece is too long by itself — hard-split it.
                for i in range(0, len(piece), self.split_chars):
                    chunks.append(piece[i : i + self.split_chars])
                current = ""

        # Flush any remaining text.
        if current:
            chunks.append(current)

        return chunks

    def _play_audio(self, audio) -> None:
        """Play a single audio chunk through the default output device.

        Converts the model's output to a flat float32 NumPy array and plays
        it synchronously (blocks until playback finishes). Skips silently if:

        * ``self._interrupt`` is True (user clicked Stop).
        * ``audio`` is None.
        * The audio array is empty after conversion.

        Args:
            audio: Audio data from the Kokoro model. Can be a NumPy array,
                a list of floats, or any array-like that ``np.asarray``
                accepts.
        """
        # Abort if a stop() was requested.
        if self._interrupt:
            return
        if audio is None:
            return

        # Normalise to a flat float32 array for sounddevice.
        arr = np.asarray(audio).astype(np.float32)

        # Squeeze out any extra dimensions (e.g. shape (N, 1) → (N,)).
        if arr.ndim > 1:
            arr = arr.squeeze()

        # Skip truly empty arrays.
        if arr.size == 0:
            return

        # blocking=True means this call waits until the audio has finished
        # playing before returning, which gives us the chance to check
        # _interrupt between chunks.
        sd.play(arr, samplerate=self.sample_rate, blocking=True)

    def stop(self) -> None:
        """Interrupt any ongoing TTS playback immediately.

        Sets the ``_interrupt`` flag (checked by :meth:`speak` between chunks
        and by :meth:`_play_audio`) and calls ``sounddevice.stop()`` to halt
        the currently playing audio.

        This method is safe to call from any thread.

        Side effects:
            Logs an INFO event ``tts_interrupted``.
        """
        with self._lock:
            self._interrupt = True
        # Stop the PortAudio stream immediately (does not block).
        sd.stop()
        logger.info("event=tts_interrupted")

    def speak(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        pitch: float | None = None,
        style: str | None = None,
    ) -> None:
        """Synthesise ``text`` as speech and play it through the speakers.

        This is the main public method. It:

        1. Clears the interrupt flag.
        2. Loads the Kokoro model (first call only).
        3. Resolves the voice/speed/pitch from the ``style`` argument (if
           provided) or falls back to per-call and then global defaults.
        4. Splits the text into chunks via :meth:`_split_text`.
        5. For each chunk, calls ``model.generate()`` with retry + pitch
           fallback logic, then plays the resulting audio via
           :meth:`_play_audio`.

        Pitch fallback:
            Some Kokoro model versions do not accept a ``pitch`` parameter.
            If ``model.generate()`` raises ``TypeError``, we retry
            *without* ``pitch`` so the call still succeeds.

        Args:
            text (str): The text to speak. Empty or whitespace-only strings
                are silently ignored.
            voice (Optional[str]): Voice ID override for this call. Defaults
                to ``self.voice`` if not provided.
            speed (float): Speech rate multiplier for this call.
                Default: 1.0.
            pitch (Optional[float]): Pitch multiplier for this call.
                ``None`` means use the style's pitch or no pitch adjustment.
            style (Optional[str]): Name of a style from ``self.style_map``
                to apply. Style settings override individual parameters.

        Side effects:
            * Plays audio through the default output device.
            * Logs errors for failed synthesis chunks (does not raise).
        """
        text = (text or "").strip()
        if not text:
            return

        # Clear any previous interrupt so this new speak() runs to completion
        # unless stop() is called again.
        with self._lock:
            self._interrupt = False

        # Load (or retrieve cached) Kokoro model.
        try:
            model = self._load_model()
        except Exception as exc:
            logger.exception(
                "event=tts_model_load_failed model_id=%s",
                self.model_id,
                exc_info=exc,
            )
            return

        # --- Resolve voice/speed/pitch ---
        # Priority: per-call argument → style override → global default.
        selected_voice = voice or self.voice
        selected_speed = float(speed if speed is not None else self.default_speed)
        selected_pitch = pitch

        if style and style in self.style_map:
            style_cfg = self.style_map[style] or {}
            # Style overrides per-call voice/speed; pitch only if not set.
            selected_voice = style_cfg.get("voice", selected_voice)
            selected_speed = float(style_cfg.get("speed", selected_speed))
            if selected_pitch is None and "pitch" in style_cfg:
                selected_pitch = float(style_cfg.get("pitch"))

        # --- Synthesise and play each chunk ---
        for chunk in self._split_text(text):
            # Check interrupt between chunks to stop quickly when requested.
            if self._interrupt:
                logger.info("event=tts_aborted_before_chunk")
                return

            # Build the keyword arguments for model.generate().
            gen_kwargs = {
                "voice": selected_voice,
                "speed": selected_speed,
            }
            if selected_pitch is not None:
                gen_kwargs["pitch"] = float(selected_pitch)

            # Attempt generation with up to 3 tries (pitch fallback on TypeError).
            stream = None
            last_error = None
            for attempt in range(3):
                try:
                    stream = model.generate(chunk, **gen_kwargs)
                    break  # Success — exit the retry loop
                except TypeError:
                    # The model doesn't support the pitch kwarg — remove it
                    # and try again immediately.
                    gen_kwargs.pop("pitch", None)
                    try:
                        stream = model.generate(chunk, **gen_kwargs)
                        break
                    except Exception as exc:
                        last_error = exc
                except Exception as exc:
                    last_error = exc
                if attempt < 2:
                    # Brief pause before next attempt (0.35 s, 0.70 s).
                    time.sleep(0.35 * (2**attempt))

            if stream is None:
                # All attempts failed — log and move on to the next chunk.
                logger.error(
                    "event=tts_generate_failed model_id=%s chunk_chars=%s voice=%s speed=%s style=%s error=%r",
                    self.model_id,
                    len(chunk),
                    selected_voice,
                    selected_speed,
                    style,
                    last_error,
                )
                continue

            # Iterate the generator returned by model.generate() and play
            # each audio result. Check interrupt between results.
            for result in stream:
                if self._interrupt:
                    logger.info("event=tts_aborted_during_stream")
                    return
                self._play_audio(result.audio)

    def speak_text(
        self,
        text: str,
        voice: str | None = None,
        speed: float = 1.0,
        pitch: float | None = None,
    ) -> None:
        """Convenience wrapper around :meth:`speak` without style support.

        Provides a simpler signature for callers that don't need named styles.
        Delegates directly to :meth:`speak`.

        Args:
            text (str): Text to synthesise and play.
            voice (Optional[str]): Voice ID override. Default: ``None``.
            speed (float): Speech rate multiplier. Default: 1.0.
            pitch (Optional[float]): Pitch multiplier. Default: ``None``.
        """
        self.speak(text=text, voice=voice, speed=speed, pitch=pitch)
