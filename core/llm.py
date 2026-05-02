import requests
import time
import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        endpoint,
        model,
        temperature,
        retries=2,
        backoff_seconds=0.5,
        fallback_message="Sorry, my brain glitched for a moment. Please try again.",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.fallback_message = fallback_message
        self.history_size = 5
        self.history = []

    def generate(self, prompt):
        self.history.append({"role": "user", "content": prompt})

        if len(self.history) > self.history_size * 2:
            self.history = self.history[-self.history_size * 2 :]

        payload = {
            "model": self.model,
            "messages": self.history,
            "temperature": self.temperature,
            "stream": False,
        }
        last_error = None

        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    f"{self.endpoint}/api/chat",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                if "application/json" not in content_type:
                    raise ValueError(f"Expected JSON response, got {content_type}")

                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Invalid response format: expected JSON object")

                message = data.get("message", {})
                text = message.get("content")
                if not isinstance(text, str):
                    raise ValueError(
                        "Invalid response format: missing or non-string 'content' field in 'message'"
                    )

                text = text.strip()
                if text:
                    self.history.append({"role": "assistant", "content": text})
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
        self.history.append({"role": "user", "content": prompt})

        if len(self.history) > self.history_size * 2:
            self.history = self.history[-self.history_size * 2 :]

        payload = {
            "model": self.model,
            "messages": self.history,
            "temperature": self.temperature,
            "stream": True,
        }
        last_error = None
        emitted_any = False
        full_response = ""

        for attempt in range(self.retries + 1):
            try:
                with requests.post(
                    f"{self.endpoint}/api/chat",
                    json=payload,
                    timeout=120,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            data = requests.models.complexjson.loads(line)
                        except ValueError as e:
                            logger.warning(
                                "event=ollama_stream_json_error error=%r line=%r",
                                e,
                                line,
                            )
                            continue

                        if not isinstance(data, dict):
                            continue

                        message = data.get("message", {})
                        token = message.get("content")
                        if isinstance(token, str) and token:
                            emitted_any = True
                            full_response += token
                            yield token

                    self.history.append({"role": "assistant", "content": full_response})
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
