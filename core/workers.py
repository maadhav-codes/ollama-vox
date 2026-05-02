from queue import Queue, Empty
from threading import Thread, Event
import logging
import time

logger = logging.getLogger(__name__)


class Pipeline:
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
        self.audio_q = Queue(maxsize=queue_maxsize)
        self.text_q = Queue(maxsize=queue_maxsize)
        self.response_q = Queue(maxsize=queue_maxsize)

        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.sr = sample_rate
        self.drop_policy = drop_policy
        self.status_callback = status_callback
        self.metrics_callback = metrics_callback
        self.response_style = response_style

        self.running = True
        self.cancel_event = Event()
        self._threads = []
        self.metrics = {
            "stt_ms_last": None,
            "llm_ms_last": None,
            "tts_ms_last": None,
            "stt_ms_avg": None,
            "llm_ms_avg": None,
            "tts_ms_avg": None,
            "responses_count": 0,
            "last_response": "",
            "last_error": None,
            "last_error_time": None,
        }

    def set_status_callback(self, callback):
        self.status_callback = callback

    def _set_status(self, status):
        if self.status_callback:
            try:
                self.status_callback(status)
            except Exception as exc:
                logger.exception("event=status_callback_failed status=%s error=%r", status, exc)

    def _record_error(self, exc):
        import datetime

        self.metrics["last_error"] = type(exc).__name__ + ": " + str(exc)
        self.metrics["last_error_time"] = datetime.datetime.now().strftime("%H:%M:%S")
        self._set_status("error")
        self._publish_metrics()

    def set_metrics_callback(self, callback):
        self.metrics_callback = callback

    def _update_metric(self, key_last, key_avg, elapsed_ms):
        self.metrics[key_last] = round(elapsed_ms, 1)
        prev = self.metrics.get(key_avg)
        if prev is None:
            self.metrics[key_avg] = round(elapsed_ms, 1)
        else:
            self.metrics[key_avg] = round((prev * 0.8) + (elapsed_ms * 0.2), 1)

    def _publish_metrics(self):
        if self.metrics_callback:
            try:
                self.metrics_callback(dict(self.metrics))
            except Exception as exc:
                logger.exception("event=metrics_callback_failed error=%r", exc)

    def _safe_put(self, q, item, name):
        if not q.full():
            q.put(item)
            return True
        if self.drop_policy == "drop_new":
            logger.warning(
                "event=queue_drop queue=%s policy=%s", name, self.drop_policy
            )
            return False
        try:
            q.get_nowait()
        except Empty:
            pass
        q.put(item)
        logger.warning("event=queue_drop queue=%s policy=%s", name, self.drop_policy)
        return True

    def interrupt_speaking(self):
        self.cancel_event.set()
        try:
            self.tts.stop()
        except Exception as exc:
            logger.exception("event=tts_interrupt_error error=%r", exc)

        for q in [self.text_q, self.response_q]:
            while not q.empty():
                try:
                    q.get_nowait()
                except Empty:
                    break
        self._set_status("idle")

    def enqueue_audio(self, audio):
        self.cancel_event.clear()
        return self._safe_put(self.audio_q, audio, "audio")

    def start(self):
        self._threads = [
            Thread(target=self.stt_worker, daemon=True),
            Thread(target=self.llm_worker, daemon=True),
            Thread(target=self.tts_worker, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        self.running = False
        for t in self._threads:
            t.join(timeout=1.0)

    def stt_worker(self):
        while self.running:
            try:
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
                start = time.perf_counter()
                text = self.stt.transcribe(audio, self.sr)
                self._update_metric(
                    "stt_ms_last",
                    "stt_ms_avg",
                    (time.perf_counter() - start) * 1000.0,
                )
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

    def llm_worker(self):
        while self.running:
            try:
                text = self.text_q.get(timeout=0.5)
            except Empty:
                continue
            except Exception as exc:
                self._record_error(exc)
                logger.exception("event=llm_worker_error error=%r", exc)
                continue

            if self.cancel_event.is_set():
                continue

            try:
                self._set_status("busy")
                start = time.perf_counter()
                token_stream = self.llm.stream_generate(
                    text, cancel_event=self.cancel_event
                )
                response_parts = []
                for sentence in self.llm.sentence_chunks(token_stream):
                    if self.cancel_event.is_set():
                        break
                    response_parts.append(sentence)
                    self._safe_put(self.response_q, sentence, "response")
                self._update_metric(
                    "llm_ms_last",
                    "llm_ms_avg",
                    (time.perf_counter() - start) * 1000.0,
                )
                if response_parts:
                    self.metrics["last_response"] = " ".join(response_parts).strip()
                    self.metrics["responses_count"] += 1
                self._publish_metrics()
                self._set_status("idle")
            except Exception as exc:
                self._record_error(exc)
                logger.exception("event=llm_worker_error error=%r", exc)

    def tts_worker(self):
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
