import pytest
from unittest.mock import MagicMock
from ollama_vox.core.llm import OllamaClient


@pytest.fixture
def mock_requests(mocker):
    return mocker.patch("requests.post")


def test_ollama_client_init():
    client = OllamaClient(
        endpoint="http://test:11434", model="test-model", temperature=0.5
    )
    assert client.endpoint == "http://test:11434"
    assert client.model == "test-model"
    assert client.temperature == 0.5
    assert len(client.history) == 0


def test_ollama_client_generate(mock_requests):
    mock_response = MagicMock()
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {"message": {"content": "Hello World"}}
    mock_requests.return_value = mock_response

    client = OllamaClient("http://test", "test", 0.5)
    response = client.generate("Hi")

    assert response == "Hello World"
    assert len(client.history) == 2
    assert client.history[0]["role"] == "user"
    assert client.history[1]["role"] == "assistant"


def test_ollama_client_generate_retry_on_failure(mock_requests):
    mock_requests.side_effect = Exception("Network Error")
    client = OllamaClient("http://test", "test", 0.5, retries=1, backoff_seconds=0.01)

    response = client.generate("Hi")

    assert response == client.fallback_message
    assert mock_requests.call_count == 2


def test_sentence_chunks():
    token_stream = ["Hello", " world", ". ", "How ", "are", " you? ", "I am", " fine"]
    chunks = list(OllamaClient.sentence_chunks(token_stream))
    assert chunks == ["Hello world.", "How are you?", "I am fine"]
