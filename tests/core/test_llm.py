"""Unit tests for ``ollama_vox.core.llm.OllamaClient``.

These tests verify:
* Successful non-streaming generation updates conversation history.
* Non-JSON content-type responses trigger the fallback message.
* Streaming generation yields tokens and persists history.
* All retries exhausted → fallback message is yielded.
* ``sentence_chunks`` correctly splits token streams at punctuation.

All HTTP calls to the Ollama server are mocked with ``mocker.patch`` so the
tests run without a running Ollama instance.
"""

from unittest.mock import MagicMock

from ollama_vox.core.llm import OllamaClient


class _StreamResponse:
    """Fake ``requests.Response`` for streaming (``stream=True``) scenarios.

    Mimics the interface used by :meth:`OllamaClient.stream_generate`:
    ``raise_for_status()``, ``iter_lines()``, and the context manager protocol.
    """

    def __init__(self, lines):
        """
        Args:
            lines (list[str]): Pre-defined newline-delimited JSON strings to
                return from ``iter_lines()``.
        """
        self._lines = lines

    def raise_for_status(self):
        """No-op: fake response is always successful."""
        return None

    def iter_lines(self, decode_unicode=True):
        """Return lines as an iterator (mimics requests' streaming iterator)."""
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_generate_success_updates_history(mocker):
    """A successful generate() call returns stripped text and appends to history.

    The history should contain exactly two entries after one successful call:
    the user prompt and the assistant response.
    """
    resp = MagicMock()
    resp.headers = {"Content-Type": "application/json"}
    resp.json.return_value = {"message": {"content": " Hello "}}
    mock_post = mocker.patch("ollama_vox.core.llm.requests.post", return_value=resp)

    client = OllamaClient("http://test", "m", 0.5)
    out = client.generate("Hi")

    # Response must be stripped of surrounding whitespace.
    assert out == "Hello"
    # History: [user, assistant]
    assert [x["role"] for x in client.history] == ["user", "assistant"]
    # Only one HTTP request should have been made (no retries needed).
    assert mock_post.call_count == 1


def test_generate_non_json_content_type_falls_back(mocker):
    """A non-JSON Content-Type causes generate() to return the fallback message.

    Ollama should always return ``application/json``. If it doesn't (e.g. the
    server is behind a proxy that returns HTML errors), the client must not
    crash — it should return the human-friendly fallback string.
    """
    resp = MagicMock()
    resp.headers = {"Content-Type": "text/plain"}
    resp.raise_for_status.return_value = None
    mocker.patch("ollama_vox.core.llm.requests.post", return_value=resp)

    # retries=0 avoids waiting in sleep() during tests.
    client = OllamaClient("http://test", "m", 0.5, retries=0)
    assert client.generate("Hi") == client.fallback_message


def test_stream_generate_success_and_sentence_chunks(mocker):
    """stream_generate() yields individual tokens and stores the full response.

    After the stream ends, ``history[-1]["content"]`` must be the concatenation
    of all tokens. ``sentence_chunks`` applied to the same tokens must split
    them at punctuation boundaries.
    """
    lines = [
        '{"message": {"content": "Hello "}}',
        '{"message": {"content": "world."}}',
        '{"message": {"content": " Next"}}',
    ]
    mocker.patch(
        "ollama_vox.core.llm.requests.post", return_value=_StreamResponse(lines)
    )

    client = OllamaClient("http://test", "m", 0.5)
    tokens = list(client.stream_generate("Hi"))
    assert tokens == ["Hello ", "world.", " Next"]
    # Full concatenation must be stored as the assistant's message.
    assert client.history[-1]["content"] == "Hello world. Next"

    # sentence_chunks should split "Hello world." at the period.
    chunks = list(OllamaClient.sentence_chunks(tokens))
    assert chunks == ["Hello world.", "Next"]


def test_stream_generate_retry_then_fallback(mocker):
    """Repeated HTTP failures cause stream_generate() to yield the fallback.

    With ``retries=1`` (2 total attempts), every attempt raises an exception.
    The generator must exhaust its retries and then yield exactly the fallback
    message so the user hears something rather than silence.
    """
    mocker.patch("ollama_vox.core.llm.requests.post", side_effect=Exception("boom"))
    client = OllamaClient("http://test", "m", 0.5, retries=1, backoff_seconds=0)

    out = list(client.stream_generate("Hi"))
    assert out == [client.fallback_message]


def test_sentence_chunks_tail_without_punctuation():
    """Tokens that form no complete sentence must be yielded as one tail chunk.

    If the entire token stream contains no ``.``, ``!``, or ``?`` characters,
    ``sentence_chunks`` must still yield the joined text rather than silently
    discarding it.
    """
    chunks = list(OllamaClient.sentence_chunks(["a", " b", " c"]))
    assert chunks == ["a b c"]
