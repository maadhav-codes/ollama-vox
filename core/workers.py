from queue import Queue
from threading import Thread
import logging

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
        self.response_style = response_style

        self.running = True

    def set_status_callback(self, callback):
        self.status_callback = callback

    def _set_status(self, status):
        if self.status_callback:
            try:
                self.status_callback(status)
            except Exception:
                logger.exception("event=status_callback_failed status=%s", status)

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
        except Exception:
            pass
        q.put(item)
        logger.warning("event=queue_drop queue=%s policy=%s", name, self.drop_policy)
        return True

    def interrupt_speaking(self):
        try:
            self.tts.stop()
            self._set_status("idle")
        except Exception as exc:
            logger.exception("event=tts_interrupt_error error=%r", exc)

    def enqueue_audio(self, audio):
        return self._safe_put(self.audio_q, audio, "audio")

    def start(self):
        Thread(target=self.stt_worker, daemon=True).start()
        Thread(target=self.llm_worker, daemon=True).start()
        Thread(target=self.tts_worker, daemon=True).start()

    def stt_worker(self):
        while self.running:
            try:
                audio = self.audio_q.get()
                self._set_status("busy")
                text = self.stt.transcribe(audio, self.sr)
                if text:
                    self._safe_put(self.text_q, text, "text")
                self._set_status("idle")
            except Exception as exc:
                self._set_status("error")
                logger.exception(
                    "event=stt_worker_error sample_rate=%s error=%r",
                    self.sr,
                    exc,
                )

    def llm_worker(self):
        while self.running:
            try:
                text = self.text_q.get()
                self._set_status("busy")
                token_stream = self.llm.stream_generate(text)
                for sentence in self.llm.sentence_chunks(token_stream):
                    self._safe_put(self.response_q, sentence, "response")
                self._set_status("idle")
            except Exception as exc:
                self._set_status("error")
                logger.exception("event=llm_worker_error error=%r", exc)

    def tts_worker(self):
        while self.running:
            try:
                response = self.response_q.get()
                self._set_status("speaking")
                self.tts.speak(response, style=self.response_style)
                self._set_status("idle")
            except Exception as exc:
                self._set_status("error")
                logger.exception("event=tts_worker_error error=%r", exc)
