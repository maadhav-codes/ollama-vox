"""System-tray UI for Ollama Vox.

This module provides the main graphical interface: a macOS system-tray icon
(:class:`VoiceTrayApp`) with a floating status panel (:class:`StatusPanel`).

Architecture
------------
* :class:`VoiceTrayApp` subclasses ``QSystemTrayIcon`` and owns the Qt
  application lifetime, the tray icon, the context menu, and two QTimers.
* :class:`StatusPanel` is a floating ``QWidget`` that displays pipeline
  metrics, the last response, and Start/Stop controls.

Thread safety
-------------
Worker threads call :meth:`VoiceTrayApp.set_status` and
:meth:`VoiceTrayApp.set_metrics`, which emit Qt signals
(``_status_signal`` / ``_metrics_signal``). The corresponding slots
(``_apply_status`` / ``_apply_metrics``) execute on the main Qt thread,
keeping all UI updates thread-safe.

Constants:
    STATUS_COLOR (dict): Maps pipeline status strings to hex colour codes
        used for the dot indicator and tray icon badge.
    STATUS_SUB (dict): Maps pipeline status strings to short subtitle text
        shown next to the status label.
    QSS (str): Global Qt stylesheet (CSS-like) for the status panel.
"""

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

# Maps each pipeline status to a distinctive colour used in the UI dot badge
# and the programmatically drawn tray icon.
STATUS_COLOR = {
    "idle": "#34C759",  # Apple green — all good, ready to listen
    "listening": "#FF3B30",  # Apple red — microphone is active
    "busy": "#FF9500",  # Apple orange — transcribing or generating
    "speaking": "#007AFF",  # Apple blue — TTS is playing audio
    "error": "#FF2D55",  # Apple pink-red — something went wrong
}

# Short subtitle strings shown below the status label in the panel header.
STATUS_SUB = {
    "idle": "Ready",
    "listening": "recording…",
    "busy": "thinking…",
    "speaking": "speaking…",
    "error": "something went wrong",
}

# Qt Style Sheet for the StatusPanel — scoped to QWidget so child widgets
# inherit typography and colour defaults.
QSS = """
QWidget {
    font-family: -apple-system, "Helvetica", Arial, sans-serif;
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
    """Build a horizontal info row widget with a muted label and a value label.

    Used inside :class:`StatusPanel` to create rows like::

        model           llama3.2:1b-instruct-q4_K_M
        latency         142 ms

    Args:
        label (str): The left-hand descriptor text (displayed in muted grey).
        value (str): The initial right-hand value text. Default: ``"—"``.

    Returns:
        tuple[QWidget, QLabel]: The container widget and the value label.
            The caller keeps the ``QLabel`` reference to update the value
            later (e.g. ``self._lat_val.setText("200 ms")``).
    """
    w = QWidget()
    # Thin bottom border gives a table-row feel without a heavy grid.
    w.setStyleSheet("border-bottom: 1px solid #f2f2f7;")
    h = QHBoxLayout(w)
    h.setContentsMargins(16, 10, 16, 10)
    lbl = QLabel(label)
    lbl.setObjectName("muted")  # Matches the QLabel#muted QSS rule (grey)
    val = QLabel(value)
    val.setAlignment(Qt.AlignmentFlag.AlignRight)
    h.addWidget(lbl)
    h.addWidget(val)
    return w, val


class StatusPanel(QWidget):
    """Floating panel that shows pipeline status, metrics, and controls.

    Displayed when the user clicks "Show Panel" in the tray menu or presses
    the tray icon. Always stays on top of other windows (``WindowStaysOnTopHint``).

    Layout (top to bottom):
        * **Header** — coloured dot, status label, subtitle.
        * **Info rows** — model, STT name, latency, response count, mic
          active, model path exists, Ollama reachable.
        * **Error section** — hidden unless an error is recorded.
        * **Response area** — read-only text box with the last LLM response.
        * **Action bar** — Copy, Start/Stop, and Close buttons.

    Args:
        app (VoiceTrayApp): Parent tray application, used to delegate
            button presses (``start()``, ``stop()``) and to read state
            (``app.recording``).
    """

    def __init__(self, app: VoiceTrayApp) -> None:
        super().__init__()
        self.app = app
        self.setWindowTitle("Ollama Vox")
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

        # --- Header (status dot + label + subtitle) ---
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

        # --- Info rows (keep value labels as attributes for later updates) ---
        r_model, self._llm_val = _info_row("model")
        r_stt, self._stt_val = _info_row("stt")
        r_lat, self._lat_val = _info_row("latency", "—")
        r_cnt, self._cnt_val = _info_row("responses", "0")
        r_mic, self._mic_val = _info_row("mic active", "No")
        r_mod, self._mod_val = _info_row("model path exists", "—")
        r_ollama, self._ollama_val = _info_row("ollama reachable", "—")

        for r in (r_model, r_stt, r_lat, r_cnt, r_mic, r_mod, r_ollama):
            root.addWidget(r)

        # --- Error section (hidden until an error is recorded) ---
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

        # --- Response text area ---
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

        # --- Action buttons ---
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
        """Copy the last LLM response text to the system clipboard."""
        QApplication.clipboard().setText(self._response.toPlainText())

    def _toggle(self) -> None:
        """Toggle recording: stop if recording, start if idle."""
        self.app.stop() if self.app.recording else self.app.start()

    def apply_status(self, status: str) -> None:
        """Update all status-dependent UI elements to reflect ``status``.

        Updates the coloured dot, the status label text, the subtitle text,
        and the Start/Stop button label.

        Args:
            status (str): One of ``"idle"``, ``"listening"``, ``"busy"``,
                ``"speaking"``, or ``"error"``.
        """
        color = STATUS_COLOR.get(status, STATUS_COLOR["idle"])
        self._dot.setStyleSheet(f"color: {color}; font-size: 10px;")
        self._status_lbl.setText(status)
        self._sub_lbl.setText(STATUS_SUB.get(status, ""))
        # Button label reflects the *next* action (inverse of current state).
        self._main_btn.setText("Stop" if status == "listening" else "Start")
        if hasattr(self, "_mic_val"):
            self._mic_val.setText("Yes" if self.app.recording else "No")

    def refresh(self, pipeline: Any, metrics: dict[str, Any]) -> None:
        """Refresh all info rows with the latest pipeline state and metrics.

        Called by :meth:`VoiceTrayApp._apply_metrics` whenever new metrics
        arrive from the worker threads.

        Args:
            pipeline: The :class:`~ollama_vox.core.workers.Pipeline` instance,
                used to read ``pipeline.llm.model`` and ``pipeline.stt.model``.
            metrics (dict): Latest metrics snapshot from ``Pipeline.metrics``.
        """
        self._llm_val.setText(str(getattr(pipeline.llm, "model", "—")))
        # Show just the directory name for the STT model (not the full path).
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
        """Close the panel when the Escape key is pressed.

        Args:
            event (QKeyEvent): The key press event from Qt.
        """
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        """Refresh live health checks each time the panel is shown.

        Checks whether the STT model path exists on disk and whether the
        Ollama server is reachable, then updates the corresponding info rows.
        Using ``showEvent`` (rather than a timer) ensures checks are fresh
        every time the user opens the panel.

        Args:
            event (QShowEvent): Qt show event (passed to super).
        """
        super().showEvent(event)
        import os

        import requests

        stt_model = getattr(self.app.pipeline.stt, "model", "")
        self._mod_val.setText("Yes" if os.path.exists(stt_model) else "No")
        endpoint = getattr(self.app.pipeline.llm, "endpoint", "http://localhost:11434")
        try:
            r = requests.get(f"{endpoint}/api/tags", timeout=0.5)
            self._ollama_val.setText("Yes" if r.ok else "No")
        except requests.RequestException:
            self._ollama_val.setText("No")


class VoiceTrayApp(QSystemTrayIcon):
    """System-tray icon and main controller for Ollama Vox.

    Subclasses ``QSystemTrayIcon`` to live in the macOS menu bar. Owns:

    * A context menu with Start/Stop/Show Panel/Change Model/Quit actions.
    * A floating :class:`StatusPanel`.
    * Two QTimers:

      * ``_auto_stop_t`` — polls :meth:`~ollama_vox.core.audio.AudioRecorder.should_auto_stop`
        every 200 ms to auto-stop recording on VAD silence or max duration.
      * ``_pump`` — fires every 100 ms to keep the Qt event loop processing
        events (prevents UI hangs on macOS when no windows are open).

    Thread-safe callbacks
    ---------------------
    Worker threads call :meth:`set_status` and :meth:`set_metrics`, which
    emit Qt signals. The matching ``@Slot`` methods execute on the Qt main
    thread, ensuring all widget updates are thread-safe.

    Args:
        pipeline (Pipeline): The active processing pipeline.
        recorder (AudioRecorder): The microphone recorder.

    Signals:
        _status_signal (str): Internal signal — carries status string from
            worker thread to ``_apply_status`` slot on the main thread.
        _metrics_signal (dict): Internal signal — carries metrics dict from
            worker thread to ``_apply_metrics`` slot on the main thread.
    """

    _status_signal = Signal(str)
    _metrics_signal = Signal(dict)

    def __init__(self, pipeline: Any, recorder: Any) -> None:
        # Ensure there is exactly one QApplication instance.
        self.qt_app = QApplication.instance() or QApplication(sys.argv)
        # Keep the app running even when all windows are closed (tray app).
        self.qt_app.setQuitOnLastWindowClosed(False)
        super().__init__()

        self.pipeline = pipeline
        self.recorder = recorder
        self.recording = False  # True while the microphone is actively recording

        self.status = "idle"
        self.metrics: dict[str, Any] = {}

        # Create the floating status panel (hidden initially).
        self.panel = StatusPanel(self)

        # Connect internal signals to their main-thread slots.
        self._status_signal.connect(self._apply_status)
        self._metrics_signal.connect(self._apply_metrics)

        # --- Context menu ---
        self.menu = QMenu()
        self._start_a = QAction("Start Listening", self.menu)
        self._stop_a = QAction("Stop Listening", self.menu)
        self._status_a = QAction("Status: idle", self.menu)
        self._show_a = QAction("Show Panel", self.menu)
        self._change_model_a = QAction("Change Model...", self.menu)
        self._quit_a = QAction("Quit", self.menu)

        self._start_a.triggered.connect(self.start)
        self._stop_a.triggered.connect(self.stop)
        self._status_a.setEnabled(False)  # Non-interactive status display
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

        # Timer: checks AudioRecorder.should_auto_stop() every 200 ms.
        self._auto_stop_t = QTimer()
        self._auto_stop_t.setInterval(200)
        self._auto_stop_t.timeout.connect(self._auto_stop_tick)
        self._auto_stop_t.start()

        # Timer: keeps Qt event loop alive when no windows are visible.
        self._pump = QTimer()
        self._pump.setInterval(100)
        self._pump.timeout.connect(lambda: None)
        self._pump.start()

        # Handle Ctrl-C from the terminal gracefully by quitting via Qt.
        signal.signal(signal.SIGINT, lambda sig, frame: QTimer.singleShot(0, self.quit))

        self._refresh_menu()

    def _set_icon(self, status: str) -> None:
        """Set the tray icon for the given status.

        First tries to load a PNG asset from ``assets/icons/``. Falls back to
        :meth:`_build_icon` which draws the icon programmatically.

        Args:
            status (str): Current pipeline status string (e.g. ``"idle"``).
        """
        # Try status-specific asset first, then generic template.
        for name in (f"tray_{status}_template.png", "tray_template.png"):
            p = Path("assets/icons") / name
            if p.exists():
                self.setIcon(QIcon(str(p.resolve())))
                return
        # Fall back to the programmatically drawn icon.
        self.setIcon(self._build_icon(status))

    @staticmethod
    def _build_icon(status: str) -> QIcon:
        """Draw a microphone tray icon with a status-colour badge programmatically.

        Draws directly onto a ``QPixmap`` using ``QPainter``. The icon
        consists of:

        * A white rounded rectangle (microphone body).
        * White arc + vertical line (microphone stand).
        * A small filled circle in the status colour (badge).

        Args:
            status (str): Pipeline status used to select the badge colour from
                ``STATUS_COLOR``.

        Returns:
            QIcon: The constructed icon, suitable for ``setIcon()``.
        """
        size = 44

        pix = QPixmap(size, size)
        # devicePixelRatio=2.0 makes the icon crisp on Retina displays.
        pix.setDevicePixelRatio(2.0)
        pix.fill(Qt.GlobalColor.transparent)

        pa = QPainter(pix)
        pa.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw the microphone capsule (rounded rect).
        pa.setPen(Qt.PenStyle.NoPen)
        pa.setBrush(QBrush(QColor("#ffffff")))
        pa.drawRoundedRect(QRectF(8, 3, 6, 11), 3, 3)

        # Draw the microphone stand arc and stem.
        stem_pen = QPen(QColor("#ffffff"), 1.5)
        stem_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pa.setPen(stem_pen)
        pa.setBrush(Qt.BrushStyle.NoBrush)
        pa.drawArc(QRectF(5, 8, 12, 9), 0, -180 * 16)
        pa.drawLine(QPointF(11, 17), QPointF(11, 20))
        pa.drawLine(QPointF(8, 20), QPointF(14, 20))

        # Draw the status badge (small coloured circle).
        badge_color = QColor(STATUS_COLOR.get(status, "#34C759"))
        pa.setPen(Qt.PenStyle.NoPen)
        pa.setBrush(QBrush(badge_color))
        pa.drawEllipse(QRectF(14, 14, 7, 7))

        pa.end()

        icon = QIcon()
        icon.addPixmap(pix)
        return icon

    def _toggle(self) -> None:
        """Toggle between recording and idle states."""
        self.stop() if self.recording else self.start()

    def _render_tooltip(self) -> None:
        """Update the tray icon tooltip and the status menu action text."""
        avg = self.metrics.get("llm_ms_avg")
        lat = "—" if avg is None else f"{int(round(avg))}ms"
        self.setToolTip(f"{self.status} · {lat}")
        self._status_a.setText(f"Status: {self.status}")

    def _refresh_menu(self) -> None:
        """Enable/disable Start and Stop menu actions based on recording state."""
        self._start_a.setEnabled(not self.recording)
        self._stop_a.setEnabled(self.recording)

    def set_status(self, status: str) -> None:
        """Thread-safe entry point for status updates from worker threads.

        Emits ``_status_signal`` which is delivered to ``_apply_status``
        on the Qt main thread.

        Args:
            status (str): New pipeline status string.
        """
        self._status_signal.emit(status)

    @Slot(str)
    def _apply_status(self, status: str) -> None:
        """Apply a status update on the main thread (slot for ``_status_signal``).

        Args:
            status (str): New pipeline status string.
        """
        self.status = status
        self._set_icon(status)
        self._render_tooltip()
        self._refresh_menu()
        self.panel.apply_status(status)

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        """Thread-safe entry point for metrics updates from worker threads.

        Args:
            metrics (dict): Latest pipeline metrics snapshot.
        """
        self._metrics_signal.emit(metrics or {})

    @Slot(dict)
    def _apply_metrics(self, metrics: dict[str, Any]) -> None:
        """Apply a metrics update on the main thread (slot for ``_metrics_signal``).

        Args:
            metrics (dict): Latest pipeline metrics snapshot.
        """
        self.metrics = metrics or {}
        self._render_tooltip()
        self.panel.refresh(self.pipeline, self.metrics)

    def _auto_stop_tick(self) -> None:
        """Poll the recorder for auto-stop conditions (called every 200 ms).

        If the recorder signals it should auto-stop (VAD silence or max
        duration reached), delegate to :meth:`stop`.
        """
        if self.recording and self.recorder.should_auto_stop():
            self.stop()

    def start(self) -> None:
        """Begin recording from the microphone.

        Interrupts any ongoing TTS playback (so the assistant stops talking
        when the user starts speaking), then starts the audio recorder and
        updates the UI to "listening" state.

        No-op if already recording.
        """
        if not self.recording:
            # Cancel any in-progress LLM/TTS output before recording.
            self.pipeline.interrupt_speaking()
            self.recording = True
            self.set_status("listening")
            self.recorder.start()

    def stop(self) -> None:
        """Stop recording and enqueue the captured audio for transcription.

        Stops the audio recorder, retrieves the audio array, and submits it
        to the pipeline via :meth:`~ollama_vox.core.workers.Pipeline.enqueue_audio`.

        No-op if not currently recording.
        """
        if self.recording:
            self.recording = False
            self.set_status("busy")
            audio = self.recorder.stop()
            if not self.pipeline.enqueue_audio(audio):
                # Queue was full and the item was dropped — report an error.
                self.set_status("error")
                return
            self.set_status("idle")

    def show_status(self) -> None:
        """Refresh and raise the floating status panel."""
        self.panel.refresh(self.pipeline, self.metrics)
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    def change_model(self) -> None:
        """Open the Ollama model selection wizard to switch models at runtime.

        Reads the current ``config.yaml``, runs the model wizard, and if the
        user selects a new model, updates the pipeline's LLM client in-place
        (without restarting the app) and clears conversation history.
        """
        import yaml

        from ollama_vox.core.config import AppConfig
        from ollama_vox.ui.model_setup import OllamaModelWizard

        base_dir = Path(__file__).parent.parent.resolve()
        config_path = base_dir / "config.yaml"
        if not config_path.exists():
            config_path = Path("config.yaml")

        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            config = AppConfig.from_dict(data)
        except (
            OSError,
            yaml.YAMLError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ):
            return  # Silently abort if config can't be read

        wizard = OllamaModelWizard(config)
        success = wizard.run(force_setup=True)
        if success:
            # Hot-swap the model tag on the existing LLM client so no restart
            # is required, and clear history so the new model starts fresh.
            self.pipeline.llm.model = config.ollama.model
            self.pipeline.llm.history = []
            self.panel.refresh(self.pipeline, self.metrics)

    def quit(self) -> None:
        """Cleanly shut down the application.

        Stops both timers, stops any active recording, shuts down the
        pipeline worker threads, hides the tray icon, and exits the Qt
        event loop.
        """
        self._auto_stop_t.stop()
        self._pump.stop()

        if self.recording:
            self.recorder.stop()
            self.recording = False

        self.pipeline.stop()
        self.hide()
        self.qt_app.quit()

    def run(self) -> None:
        """Show the tray icon and enter the Qt event loop (blocks until quit).

        Catches ``KeyboardInterrupt`` (Ctrl-C from the terminal) and calls
        :meth:`quit` for a clean shutdown.
        """
        self.show()
        try:
            self.qt_app.exec()
        except KeyboardInterrupt:
            self.quit()
