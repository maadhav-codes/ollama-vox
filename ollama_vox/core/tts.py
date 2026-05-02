import re
import time
import logging
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional

import numpy as np
import sounddevice as sd
import yaml

logger = logging.getLogger(__name__)


class TTS:
    def __init__(
        self,
        voice: Optional[str] = None,
        rate: float = 1.0,
        model_id: Optional[str] = None,
        sample_rate: int = 24000,
        split_chars: int = 180,
        style_map: Optional[dict] = None,
    ):
        self.voice = voice
        self.default_speed = float(rate)
        self.model_id = model_id or self._load_model_id_from_config()
        self.sample_rate = sample_rate
        self.split_chars = split_chars
        self.style_map = style_map or {}

        self._model = None
        self._interrupt = False
        self._lock = Lock()

    @staticmethod
    def _load_model_id_from_config() -> str:
        try:
            from ollama_vox.core.config import AppConfig
            from pathlib import Path

            config_path = Path(__file__).parent.parent / "config.yaml"
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            config = AppConfig.from_dict(data)
            return config.tts.model
        except (
            OSError,
            yaml.YAMLError,
            ImportError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return "./kokoro/Kokoro-82M-4bit"

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise RuntimeError(
                "mlx-audio is not installed. Install with: pip install mlx-audio"
            ) from exc

        self._model = load_model(Path(self.model_id))
        return self._model

    def _split_text(self, text: str) -> Iterable[str]:
        text = (text or "").strip()
        if not text:
            return []

        pieces = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= self.split_chars:
                current = candidate
                continue

            if current:
                chunks.append(current)
            if len(piece) <= self.split_chars:
                current = piece
            else:
                for i in range(0, len(piece), self.split_chars):
                    chunks.append(piece[i : i + self.split_chars])
                current = ""

        if current:
            chunks.append(current)

        return chunks

    def _play_audio(self, audio):
        if self._interrupt:
            return
        if audio is None:
            return
        arr = np.asarray(audio).astype(np.float32)
        if arr.ndim > 1:
            arr = arr.squeeze()
        if arr.size == 0:
            return

        sd.play(arr, samplerate=self.sample_rate, blocking=True)

    def stop(self):
        with self._lock:
            self._interrupt = True
        sd.stop()
        logger.info("event=tts_interrupted")

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        pitch: Optional[float] = None,
        style: Optional[str] = None,
    ):
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._interrupt = False

        try:
            model = self._load_model()
        except Exception as exc:
            logger.exception(
                "event=tts_model_load_failed model_id=%s",
                self.model_id,
                exc_info=exc,
            )
            return

        selected_voice = voice or self.voice
        selected_speed = float(speed if speed is not None else self.default_speed)
        selected_pitch = pitch

        if style and style in self.style_map:
            style_cfg = self.style_map[style] or {}
            selected_voice = style_cfg.get("voice", selected_voice)
            selected_speed = float(style_cfg.get("speed", selected_speed))
            if selected_pitch is None and "pitch" in style_cfg:
                selected_pitch = float(style_cfg.get("pitch"))

        for chunk in self._split_text(text):
            if self._interrupt:
                logger.info("event=tts_aborted_before_chunk")
                return
            gen_kwargs = {
                "voice": selected_voice,
                "speed": selected_speed,
            }
            if selected_pitch is not None:
                gen_kwargs["pitch"] = float(selected_pitch)

            stream = None
            last_error = None
            for attempt in range(3):
                try:
                    stream = model.generate(chunk, **gen_kwargs)
                    break
                except TypeError:
                    gen_kwargs.pop("pitch", None)
                    try:
                        stream = model.generate(chunk, **gen_kwargs)
                        break
                    except Exception as exc:
                        last_error = exc
                except Exception as exc:
                    last_error = exc
                if attempt < 2:
                    time.sleep(0.35 * (2**attempt))

            if stream is None:
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

            for result in stream:
                if self._interrupt:
                    logger.info("event=tts_aborted_during_stream")
                    return
                self._play_audio(result.audio)

    def speak_text(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        pitch: Optional[float] = None,
    ):
        self.speak(text=text, voice=voice, speed=speed, pitch=pitch)
