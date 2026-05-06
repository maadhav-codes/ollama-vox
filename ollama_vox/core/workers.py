"""Pipeline workers that glue the STT → LLM → TTS stages together.

This module provides :class:`Pipeline`, the central coordinator of the Ollama
Vox voice assistant. It runs three long-lived daemon threads — one per stage —
and connects them using thread-safe bounded queues.

How the pipeline works::

    [UI thread]
        ↓  enqueue_audio(audio_array)
    audio_q  →  stt_worker  →  text_q  →  llm_worker  →  response_q  →  tts_worker
                 (Thread 1)                  (Thread 2)                    (Thread 3)

Each worker loops forever (while ``self.running``), blocking on its input
queue with a 0.5-second timeout so it can cleanly exit when ``stop()`` is
called.

Queue overflow / drop policies
--------------------------------
Each queue has a fixed maximum size. When full, the ``drop_policy`` controls
what happens to newly arriving items:

* ``"drop_oldest"`` — evict the oldest item to make room (default). This
  ensures the pipeline always processes the *most recent* user input.
* ``"drop_new"``    — discard the incoming item and keep the queue unchanged.
* ``"block"``       — the producing thread waits until a slot opens.

Performance metrics
--------------------
The pipeline tracks per-stage latency (last and exponential moving average)
and other statistics in ``self.metrics``. These are forwarded to the UI via
an optional ``metrics_callback`` function.

Cancellation
------------
A ``threading.Event`` (``cancel_event``) lets the UI thread signal all three
workers to abandon their current work immediately, e.g. when the user starts
speaking again while the assistant is still responding.
"""

import logging
import time
from queue import Empty, Queue
from threading import Event, Thread

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the STT → LLM → TTS pipeline using three worker threads.

    :class:`Pipeline` creates and manages the three queues and three daemon
    threads that form the core processing loop of Ollama Vox. The UI enqueues
    audio; the pipeline transcribes it, generates a response, and speaks it —
    all without blocking the UI event loop.

    Args:
        stt: An :class:`~ollama_vox.core.stt.STT` instance for transcription.
        llm: An :class:`~ollama_vox.core.llm.OllamaClient` for LLM responses.
        tts: A :class:`~ollama_vox.core.tts.TTS` instance for speech output.
        sample_rate (int): Audio sample rate (Hz) passed to the STT worker.
        queue_maxsize (int): Maximum items in each of the three queues.
            Default: 4.
        drop_policy (str): Overflow strategy — ``"drop_oldest"``,
            ``"drop_new"``, or ``"block"``. Default: ``"drop_oldest"``.
        status_callback (callable or None): Called with a status string
            (``"idle"``, ``"busy"``, ``"speaking"``, ``"error"``) whenever
            the pipeline state changes. Runs on the worker thread.
        response_style (str): Named TTS style from ``config.yaml`` applied
            to every spoken response. Default: ``"neutral"``.
        metrics_callback (callable or None): Called with a copy of
            ``self.metrics`` dict after each pipeline stage completes.

    Attributes:
        audio_q (Queue): Receives raw audio arrays from the UI.
        text_q (Queue): Receives transcribed text strings from STT.
        response_q (Queue): Receives response sentences from the LLM.
        running (bool): ``True`` while worker threads should keep looping.
        cancel_event (threading.Event): Set to abort the current LLM/TTS cycle.
        metrics (dict): Latest performance and status data.
    """

    def __init__(
        self,
        stt,
        llm,
        tts,
        sample_rate,
        queue_maxsize=4,
        drop_policy="drop_oldest",
        status_callback=None,
        response_style="neutral",
        metrics_callback=None,
    ):
        # --- Inter-thread queues ---
        # Each queue connects one stage to the next in the pipeline.
        self.audio_q = Queue(maxsize=queue_maxsize)  # UI → STT worker
        self.text_q = Queue(maxsize=queue_maxsize)  # STT → LLM worker
        self.response_q = Queue(maxsize=queue_maxsize)  # LLM → TTS worker

        # --- Core components ---
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.sr = sample_rate
        self.drop_policy = drop_policy

        # --- Callbacks (set from the UI after construction) ---
        self.status_callback = status_callback
        self.metrics_callback = metrics_callback
        self.response_style = response_style

        # --- Control state ---
        self.running = True
        # cancel_event is set when the user interrupts the assistant mid-speech.
        # Workers check it before each expensive operation.
        self.cancel_event = Event()
        self._threads = []  # Holds the three worker Thread objects

        # --- Metrics dictionary ---
        # Tracks latency, response count, and error information for the UI.
        self.metrics = {
            "stt_ms_last": None,  # Most recent STT latency in ms
            "llm_ms_last": None,  # Most recent LLM latency in ms
            "tts_ms_last": None,  # Most recent TTS latency in ms
            "stt_ms_avg": None,  # Exponential moving average of STT latency
            "llm_ms_avg": None,  # Exponential moving average of LLM latency
            "tts_ms_avg": None,  # Exponential moving average of TTS latency
            "responses_count": 0,  # Total number of successful responses
            "last_response": "",  # Text of the most recent response
            "last_error": None,  # String description of the last error
            "last_error_time": None,  # HH:MM:SS timestamp of the last error
        }

    def set_status_callback(self, callback) -> None:
        """Register a function to receive pipeline status updates.

        The callback will be called with a single string argument whenever
        the pipeline changes state. Runs on the worker thread, so the UI
        must be careful to marshal updates to the main thread if needed
        (e.g. PySide6's ``Signal``/``Slot`` mechanism handles this).

        Args:
            callback (callable): A function accepting one ``str`` argument.
                Common values: ``"idle"``, ``"busy"``, ``"speaking"``,
                ``"error"``, ``"listening"``.
        """
        self.status_callback = callback

    def _set_status(self, status: str) -> None:
        """Invoke the status callback safely, catching and logging exceptions.

        Args:
            status (str): Status string to pass to the callback.
        """
        if self.status_callback:
            try:
                self.status_callback(status)
            except Exception as exc:
                logger.exception(
                    "event=status_callback_failed status=%s error=%r", status, exc
                )

    def _record_error(self, exc: Exception) -> None:
        """Persist error details in metrics and notify the UI of an error state.

        Stores the exception class name and message in ``self.metrics`` along
        with a human-readable timestamp, then triggers status and metrics
        callbacks so the UI can display the error.

        Args:
            exc (Exception): The exception to record.
        """
        import datetime

        # Format the error as "ExceptionType: message" for display.
        self.metrics["last_error"] = type(exc).__name__ + ": " + str(exc)
        self.metrics["last_error_time"] = datetime.datetime.now().strftime("%H:%M:%S")
        self._set_status("error")
        self._publish_metrics()

    def set_metrics_callback(self, callback) -> None:
        """Register a function to receive pipeline metrics updates.

        The callback is called after each pipeline stage with a snapshot of
        ``self.metrics``.

        Args:
            callback (callable): A function accepting one ``dict`` argument.
        """
        self.metrics_callback = callback

    def _update_metric(self, key_last: str, key_avg: str, elapsed_ms: float) -> None:
        """Update the last and exponential-moving-average latency for a stage.

        Uses an 80/20 exponential moving average (EMA):
        ``new_avg = old_avg * 0.8 + new_value * 0.2``

        This formula gives more weight to recent history (the 0.8 factor)
        while still smoothing out occasional spikes (the 0.2 factor).

        Args:
            key_last (str): Metrics dict key for the raw last value,
                e.g. ``"stt_ms_last"``.
            key_avg (str): Metrics dict key for the running average,
                e.g. ``"stt_ms_avg"``.
            elapsed_ms (float): The measured latency in milliseconds.
        """
        self.metrics[key_last] = round(elapsed_ms, 1)
        prev = self.metrics.get(key_avg)
        if prev is None:
            # First measurement — use it as the initial average.
            self.metrics[key_avg] = round(elapsed_ms, 1)
        else:
            # Exponential moving average: weight recent values at 20%.
            self.metrics[key_avg] = round((prev * 0.8) + (elapsed_ms * 0.2), 1)

    def _publish_metrics(self) -> None:
        """Send a copy of the current metrics dict to the metrics callback.

        We send a *copy* (``dict(self.metrics)``) so the callback receives a
        stable snapshot even if the pipeline continues updating ``self.metrics``
        on a worker thread.
        """
        if self.metrics_callback:
            try:
                self.metrics_callback(dict(self.metrics))
            except Exception as exc:
                logger.exception("event=metrics_callback_failed error=%r", exc)

    def _safe_put(self, q: Queue, item, name: str) -> bool:
        """Put an item into a queue, applying the configured drop policy.

        Args:
            q (Queue): The target queue.
            item: The item to enqueue.
            name (str): Human-readable queue name for log messages
                (e.g. ``"audio"``, ``"text"``, ``"response"``).

        Returns:
            bool: ``True`` if the item was successfully enqueued,
                  ``False`` if it was dropped (``"drop_new"`` policy).
        """
        if not q.full():
            # Queue has room — just put the item in.
            q.put(item)
            return True

        if self.drop_policy == "drop_new":
            # Discard the incoming item; keep what's already queued.
            logger.warning(
                "event=queue_drop queue=%s policy=%s", name, self.drop_policy
            )
            return False

        # "drop_oldest" (or "block" that somehow got here):
        # Remove the oldest item to make room, then enqueue the new one.
        try:
            q.get_nowait()
        except Empty:
            pass  # Queue emptied between the full check and here — that's fine.
        q.put(item)
        logger.warning("event=queue_drop queue=%s policy=%s", name, self.drop_policy)
        return True

    def interrupt_speaking(self) -> None:
        """Interrupt the current LLM generation and TTS playback immediately.

        Called by the UI when the user starts speaking again while the
        assistant is still responding. It:

        1. Sets ``cancel_event`` so the LLM worker stops generating.
        2. Calls ``tts.stop()`` to halt audio playback.
        3. Drains ``text_q`` and ``response_q`` to discard stale work.
        4. Signals the UI that the pipeline is now idle.

        Side effects:
            Empties ``text_q`` and ``response_q``.
        """
        self.cancel_event.set()

        # Stop the sounddevice output immediately.
        try:
            self.tts.stop()
        except Exception as exc:
            logger.exception("event=tts_interrupt_error error=%r", exc)

        # Drain any queued text and response items that are no longer needed.
        for q in [self.text_q, self.response_q]:
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break

        self._set_status("idle")

    def enqueue_audio(self, audio) -> bool:
        """Submit a recorded audio array to the pipeline for processing.

        Clears the cancellation flag (so a fresh conversation turn can
        proceed) and enqueues the audio for the STT worker.

        Args:
            audio: Float32 NumPy array of recorded audio samples.

        Returns:
            bool: ``True`` if the audio was successfully enqueued,
                  ``False`` if the queue was full and the drop policy
                  prevented insertion.
        """
        # Clear any previous cancellation so the new audio is fully processed.
        self.cancel_event.clear()
        return self._safe_put(self.audio_q, audio, "audio")

    def start(self) -> None:
        """Start all three worker threads.

        Creates daemon threads (they exit automatically when the main process
        exits) for STT, LLM, and TTS workers, then starts each one.

        Call :meth:`stop` to cleanly shut them down.
        """
        self._threads = [
            Thread(target=self.stt_worker, daemon=True),
            Thread(target=self.llm_worker, daemon=True),
            Thread(target=self.tts_worker, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        """Signal worker threads to stop and wait for them to exit.

        Sets ``self.running = False``, which causes each worker's ``while``
        loop to exit after the current operation. Each thread is joined
        with a 1-second timeout to avoid blocking the application on quit.
        """
        self.running = False
        for t in self._threads:
            t.join(timeout=1.0)

    def stt_worker(self) -> None:
        """Worker thread: dequeue audio and transcribe it with STT.

        Runs forever in a loop (while ``self.running``):
        1. Block on ``audio_q`` with a 0.5-second timeout.
        2. Call ``stt.transcribe()`` on the audio.
        3. If transcription produced text, put it in ``text_q``.
        4. Update STT latency metrics.

        Thread: Runs in its own daemon thread started by :meth:`start`.
        """
        while self.running:
            try:
                # Block for up to 0.5 s, then loop back to check self.running.
                audio = self.audio_q.get(timeout=0.5)
            except Empty:
                continue
            except Exception as exc:
                self._record_error(exc)
                logger.exception(
                    "event=stt_worker_error sample_rate=%s error=%r",
                    self.sr,
                    exc,
                )
                continue

            try:
                self._set_status("busy")
                # Time the transcription for performance metrics.
                start = time.perf_counter()
                text = self.stt.transcribe(audio, self.sr)
                self._update_metric(
                    "stt_ms_last",
                    "stt_ms_avg",
                    (time.perf_counter() - start) * 1000.0,
                )
                # Only forward non-empty transcriptions to the LLM.
                if text:
                    self._safe_put(self.text_q, text, "text")
                self._publish_metrics()
                self._set_status("idle")
            except Exception as exc:
                self._record_error(exc)
                logger.exception(
                    "event=stt_worker_error sample_rate=%s error=%r",
                    self.sr,
                    exc,
                )

    def llm_worker(self) -> None:
        """Worker thread: dequeue text and generate a streaming LLM response.

        Runs forever in a loop (while ``self.running``):
        1. Block on ``text_q`` with a 0.5-second timeout.
        2. Skip if ``cancel_event`` is set (stale request).
        3. Call ``llm.stream_generate()`` and group tokens into sentences via
           ``llm.sentence_chunks()``.
        4. Enqueue each complete sentence in ``response_q`` for the TTS worker.
        5. Update LLM latency metrics.

        Thread: Runs in its own daemon thread started by :meth:`start`.
        """
        while self.running:
            try:
                text = self.text_q.get(timeout=0.5)
            except Empty:
                continue
            except Exception as exc:
                self._record_error(exc)
                logger.exception("event=llm_worker_error error=%r", exc)
                continue

            # If a cancellation happened while we were waiting, skip this text.
            if self.cancel_event.is_set():
                continue

            try:
                self._set_status("busy")
                start = time.perf_counter()

                # Stream tokens from the LLM, passing cancel_event so the
                # generator can abort early if stop() is called.
                token_stream = self.llm.stream_generate(
                    text, cancel_event=self.cancel_event
                )

                # Group tokens into sentences before handing to TTS.
                response_parts = []
                for sentence in self.llm.sentence_chunks(token_stream):
                    if self.cancel_event.is_set():
                        break
                    response_parts.append(sentence)
                    # Enqueue each sentence independently so TTS can start
                    # speaking while the LLM generates the rest.
                    self._safe_put(self.response_q, sentence, "response")

                self._update_metric(
                    "llm_ms_last",
                    "llm_ms_avg",
                    (time.perf_counter() - start) * 1000.0,
                )

                if response_parts:
                    # Store the full response for display in the status panel.
                    self.metrics["last_response"] = " ".join(response_parts).strip()
                    self.metrics["responses_count"] += 1

                self._publish_metrics()
                self._set_status("idle")
            except Exception as exc:
                self._record_error(exc)
                logger.exception("event=llm_worker_error error=%r", exc)

    def tts_worker(self) -> None:
        """Worker thread: dequeue response sentences and synthesise speech.

        Runs forever in a loop (while ``self.running``):
        1. Block on ``response_q`` with a 0.5-second timeout.
        2. Call ``tts.speak()`` with the configured response style.
        3. Update TTS latency metrics.

        Thread: Runs in its own daemon thread started by :meth:`start`.
        """
        while self.running:
            try:
                response = self.response_q.get(timeout=0.5)
            except Empty:
                continue
            except Exception as exc:
                self._record_error(exc)
                logger.exception("event=tts_worker_error error=%r", exc)
                continue

            try:
                self._set_status("speaking")
                start = time.perf_counter()
                # Speak the sentence with the configured style preset.
                self.tts.speak(response, style=self.response_style)
                self._update_metric(
                    "tts_ms_last",
                    "tts_ms_avg",
                    (time.perf_counter() - start) * 1000.0,
                )
                self._publish_metrics()
                self._set_status("idle")
            except Exception as exc:
                self._record_error(exc)
                logger.exception("event=tts_worker_error error=%r", exc)
