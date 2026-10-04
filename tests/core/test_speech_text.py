import pytest

from ollama_vox.core.llm import OllamaClient
from ollama_vox.core.speech_text import sanitize_for_speech, without_fenced_code


@pytest.mark.parametrize("size", [1, 2, 5, 100])
def test_streaming_code_markers_never_leak_when_tokens_are_split(size):
    raw = "**First answer.**\n```python\ndef secret_code():\n    print('hello.')\n```\nNext answer."
    tokens = [raw[i : i + size] for i in range(0, len(raw), size)]
    spoken = " ".join(
        sanitize_for_speech(sentence)
        for sentence in OllamaClient.sentence_chunks(without_fenced_code(tokens))
    )
    assert "First answer." in spoken
    assert "Next answer." in spoken
    assert "response panel" in spoken
    assert "secret_code" not in spoken
    assert "python" not in spoken
    assert "`" not in spoken
    assert "*" not in spoken


def test_unfinished_code_block_is_not_spoken():
    result = "".join(
        without_fenced_code(["Try this.\n``", "`python\n", "print('secret')"])
    )
    assert "Try this." in result
    assert "secret" not in result


def test_markdown_links_lists_and_urls_become_plain_speech():
    raw = "# Overview\n- **Use** `git status`.\n2. See [the documentation](https://example.com).\nVisit https://example.com/long/path for details."
    spoken = sanitize_for_speech(raw)
    assert "Overview" in spoken
    assert "Use git status." in spoken
    assert "the documentation" in spoken
    assert "https" not in spoken
    assert "example.com" not in spoken
    assert not any(marker in spoken for marker in ("#", "*", "`", "[", "]"))


def test_first_sentence_is_available_before_the_rest_of_the_stream():
    def tokens():
        yield "First sentence. "
        raise AssertionError("Reading the rest would delay the first sentence")

    chunks = OllamaClient.sentence_chunks(without_fenced_code(tokens()))
    assert next(chunks) == "First sentence."


def test_tts_defense_preserves_natural_language(mocker):
    from ollama_vox.core.tts import TTS

    tts = TTS(model_id="mock")
    mocker.patch.object(tts, "_voice_path", return_value="local-voice.safetensors")
    tts._model = mocker.Mock()
    tts._model.generate.return_value = []

    tts.speak("**Hello**.\n```python\ndef secret(): pass\n```")

    text = " ".join(call.args[0] for call in tts._model.generate.call_args_list)
    assert "Hello" in text
    assert "secret" not in text
    assert "response panel" in text
