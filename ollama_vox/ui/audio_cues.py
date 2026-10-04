"""Short queued earcons on Qt's audio output, independent of TTS playback."""

from collections import deque
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, QUrl
from PySide6.QtMultimedia import QSoundEffect


class AudioCues(QObject):
    DURATION_MS = 120

    def __init__(self, enabled=True, volume=0.25, parent=None):
        super().__init__(parent)
        self.enabled = enabled
        self.pending = deque()
        self.playing = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._finished)
        self.effects = {}
        self.callback = None
        for name in ("start", "stop", "thinking"):
            effect = QSoundEffect(self)
            effect.setSource(
                QUrl.fromLocalFile(
                    str(
                        Path(__file__).resolve().parent.parent
                        / "assets/cues"
                        / f"{name}.wav"
                    )
                )
            )
            effect.setVolume(volume)
            self.effects[name] = effect

    def play(self, name: str, on_finished=None) -> None:
        if not self.enabled:
            if on_finished:
                on_finished()
            return
        self.pending.append((name, on_finished))
        if not self.playing:
            self._next()

    def _next(self) -> None:
        if not self.pending:
            self.playing = False
            return
        self.playing = True
        name, self.callback = self.pending.popleft()
        self.effects[name].play()
        # A bounded timer also handles unavailable audio devices gracefully.
        self.timer.start(self.DURATION_MS)

    def _finished(self) -> None:
        callback, self.callback = self.callback, None
        if callback:
            callback()
        self._next()

    def stop(self) -> None:
        self.timer.stop()
        self.pending.clear()
        self.callback = None
        self.playing = False
        for effect in self.effects.values():
            effect.stop()
