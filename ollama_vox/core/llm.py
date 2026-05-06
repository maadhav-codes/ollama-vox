"""Ollama LLM client module for streaming and non-streaming chat completions.

This module provides :class:`OllamaClient`, which communicates with a locally
running Ollama server to generate AI responses based on the transcribed
user speech. It maintains a rolling conversation history so the model can
respond in context.

How it fits into the pipeline::

    STT → [text] → OllamaClient → [response sentences] → TTS

Key design decisions
---------------------
* **Streaming**: The main path uses :meth:`stream_generate` which yields
  tokens incrementally as Ollama produces them. This allows the TTS worker
  to start speaking the first sentence while the rest of the response is
  still being generated.
* **Sentence chunking**: :meth:`sentence_chunks` groups the raw token stream
  into complete sentences so that TTS receives natural speech units rather
  than individual words.
* **History window**: Only the last ``history_size * 2`` messages (user +
  assistant pairs) are kept to avoid unbounded memory growth and to prevent
  the LLM's context window from being exceeded.
* **Retry + fallback**: Both the streaming and non-streaming methods retry
  on failure with exponential backoff, and return a human-friendly fallback
  message if all retries fail.

Dependencies:
    * ``requests`` — HTTP communication with the Ollama server.
"""

import json
import logging
import re
import time
from collections.abc import Iterable

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """HTTP client for the Ollama local LLM server with conversation history.

    Sends the user's transcribed text (plus conversation history) to the
    Ollama ``/api/chat`` endpoint and retrieves the model's response. Supports
    both full (non-streaming) and token-by-token (streaming) modes.

    Conversation history
    --------------------
    The client maintains a list of ``{"role": ..., "content": ...}`` messages
    that is included with every request, giving the model memory of the
    current conversation. The list is capped at ``history_size * 2`` entries
    (user + assistant pairs) to prevent unbounded growth.

    Retry strategy
    --------------
    Both :meth:`generate` and :meth:`stream_generate` retry on any exception
    using exponential backoff. Specifically, attempt ``i`` sleeps for
    ``backoff_seconds * 2^i`` seconds before the next try.

    Args:
        endpoint (str): Base URL of the Ollama server, e.g.
            ``"http://localhost:11434"``. A trailing ``/`` is stripped
            automatically.
        model (str): The Ollama model tag to use, e.g.
            ``"llama3.2:1b-instruct-q4_K_M"``.
        temperature (float): Sampling temperature in [0.0, 1.0].
            Higher values produce more varied/creative responses.
        retries (int): Additional retry attempts after the first failure.
            Total = ``retries + 1``. Default: 2.
        backoff_seconds (float): Base sleep between retries (seconds).
            Default: 0.5.
        fallback_message (str): Human-friendly message returned/yielded when
            all attempts fail.

    Attributes:
        history (list[dict]): Conversation history in Ollama's message format.
        history_size (int): Max number of user-assistant pairs to retain.
    """

    def __init__(
        self,
        endpoint,
        model,
        temperature,
        retries=2,
        backoff_seconds=0.5,
        fallback_message="Sorry, my brain glitched for a moment. Please try again.",
    ):
        # Strip trailing slash so we can safely append paths like "/api/chat".
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.fallback_message = fallback_message

        # Keep the last 5 user+assistant pairs (= 10 messages) in memory.
        self.history_size = 5
        self.history = []  # list of {"role": "user"|"assistant", "content": str}

    def generate(self, prompt: str) -> str:
        """Send a prompt and return the full response as a single string.

        This is the non-streaming version. It waits for Ollama to finish
        generating the entire response before returning it. Useful for
        situations where you need the complete text before doing anything
        with it.

        The prompt is appended to the conversation history before sending,
        and the assistant's response is appended after a successful reply.

        Args:
            prompt (str): The user's message to send to the model.

        Returns:
            str: The model's full text response, stripped of whitespace.
                 Returns ``self.fallback_message`` if all retries fail.

        Side effects:
            * Appends prompt to ``self.history`` as a ``"user"`` message.
            * Appends the response to ``self.history`` as an ``"assistant"``
              message (only on success).
            * Trims history if it exceeds ``history_size * 2`` entries.

        Example:
            >>> client = OllamaClient(
            ...     "http://localhost:11434", "llama3.2:1b-instruct-q4_K_M", 0.7
            ... )
            >>> client.generate("What is the capital of France?")
            'The capital of France is Paris.'
        """
        # Add the new user message to history before sending.
        self.history.append({"role": "user", "content": prompt})

        # Trim history: keep only the most recent history_size*2 messages.
        # We multiply by 2 because each "turn" consists of a user message
        # AND an assistant message.
        if len(self.history) > self.history_size * 2:
            self.history = self.history[-self.history_size * 2 :]

        payload = {
            "model": self.model,
            "messages": self.history,
            "temperature": self.temperature,
            "stream": False,  # Non-streaming: wait for full response
        }
        last_error = None

        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    f"{self.endpoint}/api/chat",
                    json=payload,
                    timeout=60,  # Wait up to 60 s for a complete response
                )
                response.raise_for_status()

                # Validate that we actually got JSON back.
                content_type = response.headers.get("Content-Type", "")
                if "application/json" not in content_type:
                    raise ValueError(f"Expected JSON response, got {content_type}")

                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Invalid response format: expected JSON object")

                # Navigate the nested response: {"message": {"content": "..."}}
                message = data.get("message", {})
                text = message.get("content")
                if not isinstance(text, str):
                    raise ValueError(
                        "Invalid response format: missing or non-string 'content' field in 'message'"
                    )

                text = text.strip()
                if text:
                    # Persist the assistant's reply in conversation history.
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
                    # Exponential backoff before the next attempt.
                    time.sleep(self.backoff_seconds * (2**attempt))

        logger.error(
            "event=ollama_generate_failed model=%s retries=%s error=%r",
            self.model,
            self.retries,
            last_error,
        )
        return self.fallback_message

    def stream_generate(self, prompt: str, cancel_event=None) -> Iterable[str]:
        """Send a prompt and yield response tokens as they arrive (streaming).

        This is the preferred method for the live voice pipeline because it
        allows the TTS worker to begin speaking the first sentence while the
        LLM is still generating the rest of the response, dramatically
        reducing perceived latency.

        Ollama streams the response as a series of newline-delimited JSON
        objects, each containing a small ``"content"`` token. This method
        parses each line and yields the token string.

        Cancellation:
            Pass a ``threading.Event`` as ``cancel_event``. If the event is
            set (e.g. the user clicks Stop), the generator returns early
            without yielding further tokens.

        Args:
            prompt (str): The user's message.
            cancel_event (threading.Event or None): Optional cancellation
                signal. Checked before each token yield.

        Yields:
            str: Individual token strings as they arrive from Ollama.
                 Yields ``self.fallback_message`` (possibly prefixed with a
                 space) if all retries fail.

        Side effects:
            * Appends prompt to ``self.history`` as ``"user"``.
            * Appends the full concatenated response to ``self.history``
              as ``"assistant"`` after the stream ends successfully.
            * Trims history as in :meth:`generate`.

        Example:
            >>> for token in client.stream_generate("Tell me a joke"):
            ...     print(token, end="", flush=True)
            Why don't scientists trust atoms? Because they make up everything!
        """
        self.history.append({"role": "user", "content": prompt})

        if len(self.history) > self.history_size * 2:
            self.history = self.history[-self.history_size * 2 :]

        payload = {
            "model": self.model,
            "messages": self.history,
            "temperature": self.temperature,
            "stream": True,  # Streaming mode: receive tokens incrementally
        }
        last_error = None
        emitted_any = False  # Track whether we yielded at least one token
        full_response = ""  # Accumulate tokens to store in history

        for attempt in range(self.retries + 1):
            try:
                # stream=True tells requests not to read the body immediately,
                # so we can iterate over it line by line.
                with requests.post(
                    f"{self.endpoint}/api/chat",
                    json=payload,
                    timeout=120,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        # Respect cancellation requests between tokens.
                        if cancel_event and cancel_event.is_set():
                            return

                        # Skip blank lines (Ollama uses them as separators).
                        if not line:
                            continue

                        # Each line should be a valid JSON object.
                        try:
                            data = json.loads(line)
                        except ValueError as e:
                            logger.warning(
                                "event=ollama_stream_json_error error=%r line=%r",
                                e,
                                line,
                            )
                            continue

                        if not isinstance(data, dict):
                            continue

                        # Extract the token string from the response object.
                        message = data.get("message", {})
                        token = message.get("content")
                        if isinstance(token, str) and token:
                            emitted_any = True
                            full_response += token
                            yield token

                    # Stream ended cleanly — persist assistant reply in history.
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
                # If we already yielded tokens before the error, don't retry —
                # retrying from the start would cause duplicate speech.
                if emitted_any:
                    break
                if attempt < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))

        # All retries exhausted (or we had a mid-stream error).
        logger.error(
            "event=ollama_stream_failed model=%s retries=%s error=%r",
            self.model,
            self.retries,
            last_error,
        )
        # Append the fallback so the user hears *something* rather than silence.
        if emitted_any:
            # Add a leading space so the fallback blends into any partial text.
            yield f" {self.fallback_message}"
        else:
            yield self.fallback_message

    @staticmethod
    def sentence_chunks(token_stream: Iterable[str]) -> Iterable[str]:
        """Group a raw token stream into complete sentences.

        Tokens from the LLM arrive word-by-word (sometimes character-by-character).
        Passing individual tokens to TTS would produce choppy, unnatural speech.
        This method buffers tokens until it detects a sentence-ending
        punctuation mark (``"."``, ``"!"``, or ``"?"`` followed by whitespace
        or end-of-string) and then yields the complete sentence.

        Any remaining text after the last sentence boundary (e.g. a phrase
        without trailing punctuation) is yielded as a final chunk.

        Args:
            token_stream (Iterable[str]): Sequence of raw token strings,
                typically from :meth:`stream_generate`.

        Yields:
            str: Complete sentences or tail fragments, stripped of surrounding
                 whitespace.

        Example:
            >>> tokens = ["Hello ", "world.", " How ", "are ", "you?", " Good"]
            >>> list(OllamaClient.sentence_chunks(tokens))
            ['Hello world.', 'How are you?', 'Good']
        """
        buffer = ""

        for token in token_stream:
            buffer += token
            # Keep extracting complete sentences from the buffer.
            while True:
                # Look for a sentence-ending punctuation followed by
                # whitespace OR end-of-string. The regex `(?:\s|$)` matches
                # either a whitespace character or the end of the string.
                m = re.search(r"[.!?](?:\s|$)", buffer)
                if not m:
                    # No complete sentence yet — wait for more tokens.
                    break
                end = m.end()
                sentence = buffer[:end].strip()
                # Advance the buffer past the sentence we just extracted.
                buffer = buffer[end:]
                if sentence:
                    yield sentence

        # Yield any remaining text that didn't end with punctuation.
        tail = buffer.strip()
        if tail:
            yield tail
