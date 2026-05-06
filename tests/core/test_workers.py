"""Unit tests for ``ollama_vox.core.workers.Pipeline``.

Verifies the queue management, cancellation, interrupt, and metrics update
logic without running the actual worker threads. STT, LLM, and TTS are
replaced with ``MagicMock`` objects so tests remain fast and deterministic.
"""

from unittest.mock import MagicMock

from ollama_vox.core.workers import Pipeline


def test_safe_put_drop_new_policy_drops_when_full():
    """_safe_put() with 'drop_new' discards incoming items when the queue is full.

    With ``queue_maxsize=1``, the first put succeeds. The second put (on a
    full queue with ``drop_new`` policy) must return ``False`` and leave the
    original item intact.
    """
    p = Pipeline(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        sample_rate=16000,
        queue_maxsize=1,
        drop_policy="drop_new",
    )
    # First item fits.
    assert p._safe_put(p.audio_q, "a", "audio") is True
    # Second item is dropped because the queue is full.
    assert p._safe_put(p.audio_q, "b", "audio") is False
    # Original item must still be in the queue.
    assert p.audio_q.get_nowait() == "a"


def test_safe_put_drop_oldest_policy_replaces_when_full():
    """_safe_put() with 'drop_oldest' evicts the oldest item to make room.

    With ``queue_maxsize=1``, the second put on a full queue (with
    ``drop_oldest``) must evict ``"a"`` and enqueue ``"b"``, returning True.
    """
    p = Pipeline(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        sample_rate=16000,
        queue_maxsize=1,
        drop_policy="drop_oldest",
    )
    assert p._safe_put(p.audio_q, "a", "audio") is True
    # Second put should evict "a" and insert "b".
    assert p._safe_put(p.audio_q, "b", "audio") is True
    # Only the newest item should remain.
    assert p.audio_q.get_nowait() == "b"


def test_enqueue_audio_clears_cancel_event():
    """enqueue_audio() clears the cancel_event before enqueuing new audio.

    If a previous conversation turn was interrupted (cancel_event set), the
    next enqueue must clear it so the new audio is fully processed rather
    than immediately skipped by the LLM worker.
    """
    p = Pipeline(MagicMock(), MagicMock(), MagicMock(), sample_rate=16000)
    p.cancel_event.set()
    assert p.enqueue_audio([1]) is True
    # cancel_event must be cleared so the LLM worker processes the new audio.
    assert p.cancel_event.is_set() is False


def test_interrupt_speaking_stops_tts_and_clears_queues():
    """interrupt_speaking() sets cancel_event, stops TTS, and drains queues.

    When the user starts speaking again, interrupt_speaking() must:
    1. Set cancel_event so the LLM worker exits its generation loop.
    2. Call tts.stop() to halt audio playback immediately.
    3. Drain text_q and response_q to discard stale in-flight work.
    """
    stt, llm, tts = MagicMock(), MagicMock(), MagicMock()
    p = Pipeline(stt, llm, tts, sample_rate=16000)

    # Seed the queues with stale items.
    p.text_q.put("x")
    p.response_q.put("y")

    p.interrupt_speaking()

    assert p.cancel_event.is_set() is True
    tts.stop.assert_called_once()
    # Both queues must be empty after interruption.
    assert p.text_q.empty()
    assert p.response_q.empty()


def test_update_metric_sets_last_and_smooth_avg():
    """_update_metric() stores the last value and computes an exponential moving average.

    First call: both ``_last`` and ``_avg`` should equal the measured value.
    Second call: ``_avg`` = old_avg * 0.8 + new_value * 0.2.
    With old_avg=100 and new_value=50: expected = 100*0.8 + 50*0.2 = 90.
    """
    p = Pipeline(MagicMock(), MagicMock(), MagicMock(), sample_rate=16000)

    p._update_metric("stt_ms_last", "stt_ms_avg", 100.0)
    assert p.metrics["stt_ms_last"] == 100.0
    assert p.metrics["stt_ms_avg"] == 100.0

    p._update_metric("stt_ms_last", "stt_ms_avg", 50.0)
    # EMA: 100 * 0.8 + 50 * 0.2 = 90.0
    assert p.metrics["stt_ms_avg"] == 90.0
