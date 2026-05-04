import pytest
from unittest.mock import MagicMock
from ollama_vox.core.workers import Pipeline


@pytest.fixture
def mock_components():
    return MagicMock(), MagicMock(), MagicMock()


def test_pipeline_init(mock_components):
    stt, llm, tts = mock_components
    pipeline = Pipeline(stt, llm, tts, sample_rate=16000)

    assert pipeline.sr == 16000
    assert pipeline.audio_q.maxsize == 4
    assert pipeline.running is True


def test_pipeline_enqueue_audio(mock_components):
    stt, llm, tts = mock_components
    pipeline = Pipeline(stt, llm, tts, sample_rate=16000)

    success = pipeline.enqueue_audio([0.1, 0.2])
    assert success is True
    assert pipeline.audio_q.qsize() == 1
    assert pipeline.cancel_event.is_set() is False


def test_pipeline_interrupt_speaking(mock_components):
    stt, llm, tts = mock_components
    pipeline = Pipeline(stt, llm, tts, sample_rate=16000)

    pipeline.text_q.put("some text")
    pipeline.response_q.put("some response")

    pipeline.interrupt_speaking()

    assert pipeline.cancel_event.is_set() is True
    assert tts.stop.call_count == 1
    assert pipeline.text_q.empty() is True
    assert pipeline.response_q.empty() is True


def test_pipeline_start_stop(mock_components):
    stt, llm, tts = mock_components
    pipeline = Pipeline(stt, llm, tts, sample_rate=16000)

    pipeline.start()
    assert len(pipeline._threads) == 3
    for t in pipeline._threads:
        assert t.is_alive() is True

    pipeline.stop()
    assert pipeline.running is False
    for t in pipeline._threads:
        assert t.is_alive() is False
