from unittest.mock import MagicMock

from ollama_vox.core.llm import OllamaClient


class _StreamResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_generate_success_updates_history(mocker):
    resp = MagicMock()
    resp.headers = {"Content-Type": "application/json"}
    resp.json.return_value = {"message": {"content": " Hello "}}
    mock_post = mocker.patch("ollama_vox.core.llm.requests.post", return_value=resp)

    client = OllamaClient("http://test", "m", 0.5)
    out = client.generate("Hi")

    assert out == "Hello"
    assert [x["role"] for x in client.history] == ["user", "assistant"]
    assert mock_post.call_count == 1


def test_generate_non_json_content_type_falls_back(mocker):
    resp = MagicMock()
    resp.headers = {"Content-Type": "text/plain"}
    resp.raise_for_status.return_value = None
    mocker.patch("ollama_vox.core.llm.requests.post", return_value=resp)

    client = OllamaClient("http://test", "m", 0.5, retries=0)
    assert client.generate("Hi") == client.fallback_message


def test_stream_generate_success_and_sentence_chunks(mocker):
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
    assert client.history[-1]["content"] == "Hello world. Next"

    chunks = list(OllamaClient.sentence_chunks(tokens))
    assert chunks == ["Hello world.", "Next"]


def test_stream_generate_retry_then_fallback(mocker):
    mocker.patch("ollama_vox.core.llm.requests.post", side_effect=Exception("boom"))
    client = OllamaClient("http://test", "m", 0.5, retries=1, backoff_seconds=0)

    out = list(client.stream_generate("Hi"))
    assert out == [client.fallback_message]


def test_sentence_chunks_tail_without_punctuation():
    chunks = list(OllamaClient.sentence_chunks(["a", " b", " c"]))
    assert chunks == ["a b c"]
