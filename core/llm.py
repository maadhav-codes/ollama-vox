import requests
import time
import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        model,
        temperature,
        retries=2,
        backoff_seconds=0.5,
        fallback_message="Sorry, my brain glitched for a moment. Please try again.",
    ):
        self.model = model
        self.temperature = temperature
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.fallback_message = fallback_message

    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
        }
        last_error = None

        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("response", "").strip()
                if text:
                    return text
                raise RuntimeError("Ollama returned an empty response")
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "event=ollama_generate_retry attempt=%s max_attempts=%s model=%s error=%r",
                    attempt + 1,
                    self.retries + 1,
                    self.model,
                    exc,
                )
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))

        logger.error(
            "event=ollama_generate_failed model=%s retries=%s error=%r",
            self.model,
            self.retries,
            last_error,
        )
        return self.fallback_message

    def stream_generate(self, prompt: str) -> Iterable[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": True,
        }
        last_error = None
        emitted_any = False

        for attempt in range(self.retries + 1):
            try:
                with requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=120,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        data = requests.models.complexjson.loads(line)
                        token = data.get("response", "")
                        if token:
                            emitted_any = True
                            yield token
                    return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "event=ollama_stream_retry attempt=%s max_attempts=%s model=%s error=%r",
                    attempt + 1,
                    self.retries + 1,
                    self.model,
                    exc,
                )
                if emitted_any:
                    break
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))

        logger.error(
            "event=ollama_stream_failed model=%s retries=%s error=%r",
            self.model,
            self.retries,
            last_error,
        )
        if emitted_any:
            yield f" {self.fallback_message}"
        else:
            yield self.fallback_message

    @staticmethod
    def sentence_chunks(token_stream: Iterable[str]) -> Iterable[str]:
        buffer = ""
        for token in token_stream:
            buffer += token
            while True:
                m = re.search(r"[.!?](?:\s|$)", buffer)
                if not m:
                    break
                end = m.end()
                sentence = buffer[:end].strip()
                buffer = buffer[end:]
                if sentence:
                    yield sentence

        tail = buffer.strip()
        if tail:
            yield tail
