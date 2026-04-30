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

        self.menu = [
            "Start Listening",
            "Stop Listening",
            f"Hotkey: {self.hotkey}",
            "Status: idle",
            None,
            "Quit",
        ]
        self._status_item = self.menu["Status: idle"]

        self._start_hotkey_listener()

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
        icon_map = {
            "idle": "🎤",
            "listening": "🔴",
            "busy": "⚡",
            "speaking": "🗣️",
            "error": "❌",
        }
        self.title = icon_map.get(status, "🎤")
        if self._status_item is not None:
            self._status_item.title = f"Status: {status}"

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

    @rumps.clicked("Quit")
    def quit(self, _):
        if self._hotkey_listener is not None:
            self._hotkey_listener.stop()
        rumps.quit_application()
