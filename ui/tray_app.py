from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any


from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QBrush, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

STATUS_COLOR = {
    "idle": "#34C759",
    "listening": "#FF3B30",
    "busy": "#FF9500",
    "speaking": "#007AFF",
    "error": "#FF2D55",
}

STATUS_SUB = {
    "idle": "Ready",
    "listening": "recording…",
    "busy": "thinking…",
    "speaking": "speaking…",
    "error": "something went wrong",
}

QSS = """
QWidget {
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #1c1c1e;
    background: #ffffff;
}
QLabel { background: transparent; }
QLabel#muted { color: #8e8e93; }
QPushButton {
    background: transparent;
    border: 1px solid #d1d1d6;
    border-radius: 7px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
    color: #1c1c1e;
}
QPushButton:hover { background: #f2f2f7; }
QPushButton#primary {
    background: #1c1c1e;
    border-color: transparent;
    color: #ffffff;
}
QPushButton#primary:hover { background: #3a3a3c; }
QPlainTextEdit {
    background: #f2f2f7;
    border: none;
    border-radius: 7px;
    padding: 8px;
    font-size: 12px;
    color: #3a3a3c;
    selection-background-color: #007AFF;
}
"""


def _info_row(label: str, value: str = "—") -> tuple[QWidget, QLabel]:
    w = QWidget()
    w.setStyleSheet("border-bottom: 1px solid #f2f2f7;")
    h = QHBoxLayout(w)
    h.setContentsMargins(16, 10, 16, 10)
    lbl = QLabel(label)
    lbl.setObjectName("muted")
    val = QLabel(value)
    val.setAlignment(Qt.AlignmentFlag.AlignRight)
    h.addWidget(lbl)
    h.addWidget(val)
    return w, val


class StatusPanel(QWidget):
    def __init__(self, app: "VoiceTrayApp") -> None:
        super().__init__()
        self.app = app
        self.setWindowTitle("Ollama Voice")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self.setFixedWidth(320)
        self.setStyleSheet(QSS)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QWidget()
        header.setStyleSheet("border-bottom: 1px solid #f2f2f7;")
        hh = QHBoxLayout(header)
        hh.setContentsMargins(16, 14, 16, 14)
        hh.setSpacing(10)
        self._dot = QLabel("●")
        self._dot.setStyleSheet(f"color: {STATUS_COLOR['idle']}; font-size: 10px;")
        self._status_lbl = QLabel("idle")
        self._status_lbl.setStyleSheet("font-size: 14px; font-weight: 600;")
        self._sub_lbl = QLabel(STATUS_SUB["idle"])
        self._sub_lbl.setObjectName("muted")
        hh.addWidget(self._dot)
        hh.addWidget(self._status_lbl)
        hh.addStretch()
        hh.addWidget(self._sub_lbl)
        root.addWidget(header)

        r_model, self._llm_val = _info_row("model")
        r_stt, self._stt_val = _info_row("stt")
        r_lat, self._lat_val = _info_row("latency", "—")
        r_cnt, self._cnt_val = _info_row("responses", "0")

        r_mic, self._mic_val = _info_row("mic active", "No")
        r_mod, self._mod_val = _info_row("model path exists", "—")
        r_ollama, self._ollama_val = _info_row("ollama reachable", "—")

        for r in (r_model, r_stt, r_lat, r_cnt, r_mic, r_mod, r_ollama):
            root.addWidget(r)

        self._error_wrap = QWidget()
        eh = QVBoxLayout(self._error_wrap)
        eh.setContentsMargins(16, 0, 16, 10)
        self._error_val = QLabel("")
        self._error_val.setStyleSheet(
            "color: #FF2D55; font-size: 11px; font-weight: 500;"
        )
        self._error_val.setWordWrap(True)
        eh.addWidget(self._error_val)
        self._error_wrap.hide()
        root.addWidget(self._error_wrap)

        resp_wrap = QWidget()
        resp_wrap.setStyleSheet("border-bottom: 1px solid #f2f2f7;")
        rv = QVBoxLayout(resp_wrap)
        rv.setContentsMargins(16, 10, 16, 10)
        self._response = QPlainTextEdit()
        self._response.setReadOnly(True)
        self._response.setPlaceholderText("No responses yet.")
        self._response.setFixedHeight(72)
        rv.addWidget(self._response)
        root.addWidget(resp_wrap)

        act = QWidget()
        ah = QHBoxLayout(act)
        ah.setContentsMargins(16, 10, 16, 12)
        ah.setSpacing(8)
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._copy)
        self._main_btn = QPushButton("Start")
        self._main_btn.setObjectName("primary")
        self._main_btn.clicked.connect(self._toggle)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        ah.addWidget(copy_btn)
        ah.addStretch()
        ah.addWidget(self._main_btn)
        ah.addWidget(close_btn)
        root.addWidget(act)

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._response.toPlainText())

    def _toggle(self) -> None:
        self.app.stop() if self.app.recording else self.app.start()

    def apply_status(self, status: str) -> None:
        color = STATUS_COLOR.get(status, STATUS_COLOR["idle"])
        self._dot.setStyleSheet(f"color: {color}; font-size: 10px;")
        self._status_lbl.setText(status)
        self._sub_lbl.setText(STATUS_SUB.get(status, ""))
        self._main_btn.setText("Stop" if status == "listening" else "Start")
        if hasattr(self, "_mic_val"):
            self._mic_val.setText("Yes" if self.app.recording else "No")

    def refresh(self, pipeline: Any, metrics: dict[str, Any]) -> None:
        self._llm_val.setText(str(getattr(pipeline.llm, "model", "—")))
        stt_path = str(getattr(pipeline.stt, "model", None) or "—")
        self._stt_val.setText(stt_path.split("/")[-1])
        last_ms = metrics.get("llm_ms_last")
        self._lat_val.setText("—" if last_ms is None else f"{last_ms:.0f} ms")
        self._cnt_val.setText(str(metrics.get("responses_count", 0)))
        last = (metrics.get("last_response") or "").strip()
        if last:
            self._response.setPlainText(last)

        err = metrics.get("last_error")
        if err:
            err_time = metrics.get("last_error_time", "")
            self._error_val.setText(f"Error ({err_time}): {err}")
            self._error_wrap.show()
        else:
            self._error_wrap.hide()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        import os
        import requests

        stt_model = getattr(self.app.pipeline.stt, "model", "")
        self._mod_val.setText("Yes" if os.path.exists(stt_model) else "No")
        endpoint = getattr(self.app.pipeline.llm, "endpoint", "http://localhost:11434")
        try:
            r = requests.get(f"{endpoint}/api/tags", timeout=0.5)
            self._ollama_val.setText("Yes" if r.ok else "No")
        except Exception:
            self._ollama_val.setText("No")


class VoiceTrayApp(QSystemTrayIcon):
    _status_signal = Signal(str)
    _metrics_signal = Signal(dict)

    def __init__(self, pipeline: Any, recorder: Any) -> None:
        self.qt_app = QApplication.instance() or QApplication(sys.argv)
        self.qt_app.setQuitOnLastWindowClosed(False)
        super().__init__()

        self.pipeline = pipeline
        self.recorder = recorder
        self.recording = False

        self.status = "idle"
        self.metrics: dict[str, Any] = {}

        self.panel = StatusPanel(self)
        self._status_signal.connect(self._apply_status)
        self._metrics_signal.connect(self._apply_metrics)

        self.menu = QMenu()
        self._start_a = QAction("Start Listening", self.menu)
        self._stop_a = QAction("Stop Listening", self.menu)

        self._status_a = QAction("Status: idle", self.menu)
        self._show_a = QAction("Show Panel", self.menu)
        self._change_model_a = QAction("Change Model...", self.menu)
        self._quit_a = QAction("Quit", self.menu)

        self._start_a.triggered.connect(self.start)
        self._stop_a.triggered.connect(self.stop)

        self._status_a.setEnabled(False)
        self._show_a.triggered.connect(self.show_status)
        self._change_model_a.triggered.connect(self.change_model)
        self._quit_a.triggered.connect(self.quit)

        for a in (
            self._start_a,
            self._stop_a,
            self._status_a,
            self._show_a,
            self._change_model_a,
        ):
            self.menu.addAction(a)
        self.menu.addSeparator()
        self.menu.addAction(self._quit_a)
        self.setContextMenu(self.menu)

        self._set_icon("idle")
        self._render_tooltip()

        self._auto_stop_t = QTimer()
        self._auto_stop_t.setInterval(200)
        self._auto_stop_t.timeout.connect(self._auto_stop_tick)
        self._auto_stop_t.start()

        self._pump = QTimer()
        self._pump.setInterval(100)
        self._pump.timeout.connect(lambda: None)
        self._pump.start()

        signal.signal(signal.SIGINT, lambda *_: QTimer.singleShot(0, self.quit))

        self._refresh_menu()

    def _set_icon(self, status: str) -> None:
        for name in (f"tray_{status}_template.png", "tray_template.png"):
            p = Path("assets/icons") / name
            if p.exists():
                self.setIcon(QIcon(str(p.resolve())))
                return
        self.setIcon(self._build_icon(status))

    def _build_icon(self, status: str) -> QIcon:
        SIZE = 44

        pix = QPixmap(SIZE, SIZE)
        pix.setDevicePixelRatio(2.0)
        pix.fill(Qt.GlobalColor.transparent)

        pa = QPainter(pix)
        pa.setRenderHint(QPainter.RenderHint.Antialiasing)

        pa.setPen(Qt.PenStyle.NoPen)
        pa.setBrush(QBrush(QColor("#ffffff")))
        pa.drawRoundedRect(QRectF(8, 3, 6, 11), 3, 3)

        stem_pen = QPen(QColor("#ffffff"), 1.5)
        stem_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pa.setPen(stem_pen)
        pa.setBrush(Qt.BrushStyle.NoBrush)
        pa.drawArc(QRectF(5, 8, 12, 9), 0, -180 * 16)
        pa.drawLine(QPointF(11, 17), QPointF(11, 20))
        pa.drawLine(QPointF(8, 20), QPointF(14, 20))

        badge_color = QColor(STATUS_COLOR.get(status, "#34C759"))
        pa.setPen(Qt.PenStyle.NoPen)
        pa.setBrush(QBrush(badge_color))
        pa.drawEllipse(QRectF(14, 14, 7, 7))

        pa.end()

        icon = QIcon()
        icon.addPixmap(pix)
        return icon

    def _toggle(self) -> None:
        self.stop() if self.recording else self.start()

    def _render_tooltip(self) -> None:
        avg = self.metrics.get("llm_ms_avg")
        lat = "—" if avg is None else f"{int(round(avg))}ms"
        self.setToolTip(f"{self.status} · {lat}")
        self._status_a.setText(f"Status: {self.status}")

    def _refresh_menu(self) -> None:
        self._start_a.setEnabled(not self.recording)
        self._stop_a.setEnabled(self.recording)

    def set_status(self, status: str) -> None:
        self._status_signal.emit(status)

    @Slot(str)
    def _apply_status(self, status: str) -> None:
        self.status = status
        self._set_icon(status)
        self._render_tooltip()
        self._refresh_menu()
        self.panel.apply_status(status)

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        self._metrics_signal.emit(metrics or {})

    @Slot(dict)
    def _apply_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics = metrics or {}
        self._render_tooltip()
        self.panel.refresh(self.pipeline, self.metrics)

    def _auto_stop_tick(self) -> None:
        if self.recording and self.recorder.should_auto_stop():
            self.stop()

    def start(self) -> None:
        if not self.recording:
            self.pipeline.interrupt_speaking()
            self.recording = True
            self.set_status("listening")
            self.recorder.start()

    def stop(self) -> None:
        if self.recording:
            self.recording = False
            self.set_status("busy")
            audio = self.recorder.stop()
            if not self.pipeline.enqueue_audio(audio):
                self.set_status("error")
                return
            self.set_status("idle")

    def show_status(self) -> None:
        self.panel.refresh(self.pipeline, self.metrics)
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    def change_model(self) -> None:
        from ui.model_setup import OllamaModelWizard
        import yaml
        from core.config import AppConfig

        base_dir = Path(__file__).parent.parent.resolve()
        config_path = base_dir / "config.yaml"
        if not config_path.exists():
            config_path = Path("config.yaml")

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            config = AppConfig.from_dict(data)
        except Exception:
            return

        wizard = OllamaModelWizard(config)
        success = wizard.run(force_setup=True)
        if success:
            self.pipeline.llm.model = config.ollama.model
            self.pipeline.llm.history = []
            self.panel.refresh(self.pipeline, self.metrics)

    def quit(self) -> None:
        self._auto_stop_t.stop()
        self._pump.stop()

        if self.recording:
            self.recorder.stop()
            self.recording = False
        self.pipeline.stop()
        self.hide()
        self.qt_app.quit()

    def run(self) -> None:
        self.show()
        try:
            self.qt_app.exec()
        except KeyboardInterrupt:
            self.quit()
