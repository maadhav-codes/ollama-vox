import time
import logging

import mlx_whisper
import numpy as np

logger = logging.getLogger(__name__)


class _TerminalTranscriptPrinter:
    def __init__(self):
        self._use_rich = False
        self._console = None
        try:
            from rich.console import Console

            self._console = Console()
            self._use_rich = True
        except ImportError:
            pass

    def live(self, text):
        text = text.strip()
        if not text:
            return
        if self._use_rich:
            self._console.print(f"[cyan][LIVE][/cyan] {text}")
        else:
            print(f"\033[96m[LIVE]\033[0m {text}")

    def final(self, text):
        text = text.strip()
        if self._use_rich:
            self._console.print(f"[bold green][FINAL][/bold green] {text}")
        else:
            print(f"\033[1;32m[FINAL]\033[0m {text}")


class STT:
    def __init__(self, model, retries=2, backoff_seconds=0.35):
        self.model = model
        self.printer = _TerminalTranscriptPrinter()
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def _transcribe_audio(self, audio_data):
        last_error = None
        for attempt in range(self.retries + 1):
            try:
                result = mlx_whisper.transcribe(
                    audio_data,
                    path_or_hf_repo=self.model,
                )
                return result.get("text", "").strip()
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        logger.exception(
            "event=stt_transcribe_failed model=%s retries=%s",
            self.model,
            self.retries,
            exc_info=last_error,
        )
        return ""

    def transcribe(self, audio, sr):
        if audio is None or len(audio) == 0:
            self.printer.final("")
            return ""

        audio_1d = np.asarray(audio).reshape(-1)
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
                self.printer.live(" ".join(live_segments).strip())

        final_text = self._transcribe_audio(audio_1d)

        self.printer.final(final_text)
        return final_text
