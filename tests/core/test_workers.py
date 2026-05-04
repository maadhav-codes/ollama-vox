from unittest.mock import MagicMock

from ollama_vox.core.workers import Pipeline


def test_safe_put_drop_new_policy_drops_when_full():
    p = Pipeline(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        sample_rate=16000,
        queue_maxsize=1,
        drop_policy="drop_new",
    )
    assert p._safe_put(p.audio_q, "a", "audio") is True
    assert p._safe_put(p.audio_q, "b", "audio") is False
    assert p.audio_q.get_nowait() == "a"


def test_safe_put_drop_oldest_policy_replaces_when_full():
    p = Pipeline(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        sample_rate=16000,
        queue_maxsize=1,
        drop_policy="drop_oldest",
    )
    assert p._safe_put(p.audio_q, "a", "audio") is True
    assert p._safe_put(p.audio_q, "b", "audio") is True
    assert p.audio_q.get_nowait() == "b"


def test_enqueue_audio_clears_cancel_event():
    p = Pipeline(MagicMock(), MagicMock(), MagicMock(), sample_rate=16000)
    p.cancel_event.set()
    assert p.enqueue_audio([1]) is True
    assert p.cancel_event.is_set() is False


def test_interrupt_speaking_stops_tts_and_clears_queues():
    stt, llm, tts = MagicMock(), MagicMock(), MagicMock()
    p = Pipeline(stt, llm, tts, sample_rate=16000)
    p.text_q.put("x")
    p.response_q.put("y")

    p.interrupt_speaking()

    assert p.cancel_event.is_set() is True
    tts.stop.assert_called_once()
    assert p.text_q.empty()
    assert p.response_q.empty()


def test_update_metric_sets_last_and_smooth_avg():
    p = Pipeline(MagicMock(), MagicMock(), MagicMock(), sample_rate=16000)
    p._update_metric("stt_ms_last", "stt_ms_avg", 100.0)
    assert p.metrics["stt_ms_last"] == 100.0
    assert p.metrics["stt_ms_avg"] == 100.0

    p._update_metric("stt_ms_last", "stt_ms_avg", 50.0)
    assert p.metrics["stt_ms_avg"] == 90.0
