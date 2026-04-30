import rumps
from pynput import keyboard


class VoiceApp(rumps.App):
    def __init__(self, pipeline, recorder, hotkey=None):
        super().__init__("🎤")
        self.pipeline = pipeline
        self.recorder = recorder
        self.recording = False
        self.hotkey = hotkey or "cmd+shift"
        self._hotkey_listener = None
        self.status = "idle"
        self.metrics = {}

        self.menu = [
            "Start Listening",
            "Stop Listening",
            f"Hotkey: {self.hotkey}",
            "Status: idle",
            "Show Status",
            None,
            "Quit",
        ]
        self._status_item = self.menu["Status: idle"]

        self._start_hotkey_listener()

    def _latency_indicator(self):
        llm_avg = self.metrics.get("llm_ms_avg")
        if llm_avg is None:
            return ("⚪", "--")
        if llm_avg <= 1200:
            dot = "🟢"
        elif llm_avg <= 2600:
            dot = "🟡"
        else:
            dot = "🔴"
        return (dot, f"{int(round(llm_avg))}ms")

    def _render_title(self):
        icon_map = {
            "idle": "🎤",
            "listening": "🔴",
            "busy": "⚡",
            "speaking": "🗣️",
            "error": "❌",
        }
        dot, llm_ms = self._latency_indicator()
        self.title = f"{icon_map.get(self.status, '🎤')} {dot}{llm_ms}"

    @rumps.timer(0.2)
    def auto_stop_tick(self, _):
        if self.recording and self.recorder.should_auto_stop():
            self.stop(None)

    def _start_hotkey_listener(self):
        keys = [part.strip().lower() for part in self.hotkey.split("+") if part.strip()]
        if not keys:
            return

        pressed = set()

        def on_press(key):
            token = self._normalize_key(key)
            if token:
                pressed.add(token)
            if all(k in pressed for k in keys):
                self._toggle_recording()

        def on_release(key):
            token = self._normalize_key(key)
            if token and token in pressed:
                pressed.remove(token)

        self._hotkey_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()

    def _normalize_key(self, key):
        mapping = {
            keyboard.Key.cmd: "cmd",
            keyboard.Key.cmd_l: "cmd",
            keyboard.Key.cmd_r: "cmd",
            keyboard.Key.shift: "shift",
            keyboard.Key.shift_l: "shift",
            keyboard.Key.shift_r: "shift",
            keyboard.Key.ctrl: "ctrl",
            keyboard.Key.ctrl_l: "ctrl",
            keyboard.Key.ctrl_r: "ctrl",
            keyboard.Key.alt: "alt",
            keyboard.Key.alt_l: "alt",
            keyboard.Key.alt_r: "alt",
        }
        if key in mapping:
            return mapping[key]
        try:
            return key.char.lower() if key.char else None
        except AttributeError:
            return None

    def _toggle_recording(self):
        if self.recording:
            self.stop(None)
        else:
            self.start(None)

    def set_status(self, status):
        self.status = status
        self._render_title()
        if self._status_item is not None:
            self._status_item.title = f"Status: {status}"

    def set_metrics(self, metrics):
        self.metrics = metrics or {}
        self._render_title()

    def _fmt_ms(self, value):
        if value is None:
            return "n/a"
        return f"{value:.1f} ms"

    def _status_body(self):
        llm_model = getattr(self.pipeline.llm, "model", "unknown")
        stt_model = getattr(self.pipeline.stt, "model_path", None) or "configured"
        tts_model = getattr(self.pipeline.tts, "model_id", "configured")
        last_response = (
            self.metrics.get("last_response") or "No responses yet."
        ).strip()
        if len(last_response) > 220:
            last_response = last_response[:217] + "..."

        lines = [
            f"Status: {self.status}",
            f"LLM: {llm_model}",
            f"STT: {stt_model}",
            f"TTS: {tts_model}",
            "",
            "Latency",
            f"STT last/avg: {self._fmt_ms(self.metrics.get('stt_ms_last'))} / {self._fmt_ms(self.metrics.get('stt_ms_avg'))}",
            f"LLM last/avg: {self._fmt_ms(self.metrics.get('llm_ms_last'))} / {self._fmt_ms(self.metrics.get('llm_ms_avg'))}",
            f"TTS last/avg: {self._fmt_ms(self.metrics.get('tts_ms_last'))} / {self._fmt_ms(self.metrics.get('tts_ms_avg'))}",
            f"Responses: {self.metrics.get('responses_count', 0)}",
            "",
            f"Last response: {last_response}",
        ]
        return "\n".join(lines)

    @rumps.clicked("Start Listening")
    def start(self, _):
        if not self.recording:
            self.pipeline.interrupt_speaking()
            self.recording = True
            self.set_status("listening")
            self.recorder.start()

    @rumps.clicked("Stop Listening")
    def stop(self, _):
        if self.recording:
            self.recording = False
            self.set_status("busy")

            audio = self.recorder.stop()
            ok = self.pipeline.enqueue_audio(audio)
            if not ok:
                self.set_status("error")
                return

            self.set_status("idle")

    @rumps.clicked("Show Status")
    def show_status(self, _):
        rumps.alert(
            title="Voice Assistant Status",
            message=self._status_body(),
            ok="Close",
        )

    @rumps.clicked("Quit")
    def quit(self, _):
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()
        rumps.quit_application()
