"""Ollama model selection and download wizard for Ollama Vox.

This module handles everything related to choosing which Ollama LLM the
application will use. It is responsible for:

1. Verifying that the Ollama server is reachable, or offering to launch ``ollama serve``.
2. Listing already-downloaded models.
3. Letting the user choose a model (or download one if none exist).
4. Persisting the selection back to ``config.yaml``.
5. Cleanly shutting down locally managed Ollama server instances when the app quits.

Where it fits in startup::

    main() → AppSetupWizard.run()         (Whisper STT + Kokoro TTS download)
           → ensure_speech_dependencies() (spaCy English model)
           → OllamaModelWizard.run()      ← this module
           → prepare_speech()             (TTS silent pre-warm)
           → Pipeline + VoiceTrayApp

It is also accessible at runtime via the tray-icon's "Change Model…" menu
item, which calls :meth:`OllamaModelWizard.run` with ``force_setup=True``.

Threading model
---------------
Downloading an Ollama model (``ollama pull <model>``) can take several
minutes. :class:`OllamaPullThread` handles the download in a background
QThread and reports streaming JSON progress data back to the main thread via
Qt signals. A :class:`QEventLoop` in :meth:`OllamaModelWizard.pull_model`
blocks the calling code until the download finishes without freezing the Qt
event loop.

Dependencies:
    * ``PySide6``  — Qt GUI framework.
    * ``requests`` — HTTP communication with the Ollama API.
"""

import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from urllib.parse import urlparse
from pathlib import Path

import requests
from PySide6.QtCore import QEventLoop, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QVBoxLayout,
)


# Keep ownership across startup and runtime model wizards.
_started_servers: list[tuple[str, subprocess.Popen]] = []


def running_started_servers() -> list[tuple[str, subprocess.Popen]]:
    """Return live servers launched by this app, pruning exited processes."""
    _started_servers[:] = [
        (endpoint, process)
        for endpoint, process in _started_servers
        if process.poll() is None
    ]
    return list(_started_servers)


def is_local_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint.rstrip("/"))
    return (
        parsed.scheme == "http"
        and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
        and parsed.username is None
        and parsed.password is None
    )


def _process_identity(pid: int) -> str | None:
    """Read command and start time together to detect exited or reused PIDs."""
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
        capture_output=True,
        text=True,
        timeout=2,
    )
    if result.returncode == 1 and not result.stdout.strip():
        return None
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or "Could not inspect the Ollama process.")
    return result.stdout.strip() or None


def _local_listener_pids(port: int) -> set[int]:
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
        timeout=2,
    )
    if (
        result.returncode == 1
        and not result.stdout.strip()
        and not result.stderr.strip()
    ):
        return set()  # No server is listening anymore.
    if result.returncode != 0:
        raise OSError(
            result.stderr.strip() or "Could not find the local Ollama server."
        )
    return {int(line) for line in result.stdout.split()}


def _stop_existing_local_server(endpoint: str) -> None:
    """Stop verified ``ollama serve`` listeners at the configured local port."""
    if not is_local_endpoint(endpoint):
        raise OSError("Only local Ollama servers can be stopped by Ollama Vox.")
    port = urlparse(endpoint).port or 11434
    for pid in _local_listener_pids(port):
        identity = _process_identity(pid)
        if identity is None:
            continue
        # lstart has five fields (weekday, month, day, time, year).
        fields = identity.split(maxsplit=5)
        command = shlex.split(fields[5]) if len(fields) == 6 else []
        if (
            len(command) < 2
            or Path(command[0]).name != "ollama"
            or command[1] != "serve"
        ):
            raise OSError(
                f"The listener on port {port} is not a verified 'ollama serve' process."
            )
        for stop_signal in (signal.SIGTERM, signal.SIGKILL):
            if _process_identity(pid) != identity:
                break
            try:
                os.kill(pid, stop_signal)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if _process_identity(pid) != identity:
                    break
                time.sleep(0.1)
            else:
                continue
            break
        else:
            raise OSError(f"Ollama on port {port} did not stop.")
    if _local_listener_pids(port):
        raise OSError(
            "A server is still listening on the Ollama port. "
            "If Ollama is managed by a desktop app or service, stop it there, "
            "then retry, or choose No to leave it running."
        )


def _stop_server_process(process: subprocess.Popen) -> None:
    """Terminate an owned server, escalating to kill after a bounded wait."""
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return  # The process exited between poll() and terminate().
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


class OllamaStopThread(QThread):
    """Stop local servers without blocking the Qt GUI thread."""

    finished_stop = Signal(bool, str)

    def __init__(self, servers: list[tuple[str, subprocess.Popen]], endpoint: str = ""):
        super().__init__()
        self.servers = servers
        self.endpoint = endpoint

    def run(self) -> None:
        errors = []
        for endpoint, process in self.servers:
            try:
                _stop_server_process(process)
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"{endpoint}: {exc}")
        if self.endpoint and self.endpoint not in {
            endpoint for endpoint, _ in self.servers
        }:
            try:
                _stop_existing_local_server(self.endpoint)
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                errors.append(f"{self.endpoint}: {exc}")
        self.finished_stop.emit(not errors, "\n".join(errors))


def update_config_model(config_path: str, new_model: str) -> None:
    """Overwrite the ``model:`` field inside the ``ollama:`` section of a YAML file.

    This function uses a simple line-by-line text parser rather than loading
    the YAML as a Python object and re-serialising it. This approach
    **preserves all comments and formatting** in the original file — a full
    YAML round-trip would strip comments.

    Algorithm:
        1. Read all lines.
        2. Scan for the ``ollama:`` section header.
        3. Within that section, find the first ``model:`` line.
        4. Replace only that line, preserving its leading whitespace (indent).
        5. Write all lines back to the file.

    Args:
        config_path (str): Absolute or relative path to the ``config.yaml``
            file that should be updated.
        new_model (str): The new Ollama model tag to write, e.g.
            ``"llama3.2:1b-instruct-q4_K_M"``.

    Raises:
        OSError: If the file cannot be read or written (permissions, missing
            file, etc.). The caller is responsible for catching this.

    Example:
        >>> update_config_model("/app/config.yaml", "mistral")
        # config.yaml now has:  model: mistral  under the ollama: section
    """
    with open(config_path) as f:
        lines = f.readlines()

    in_ollama = False  # Track whether we're inside the `ollama:` block

    for i, line in enumerate(lines):
        if line.strip() == "ollama:":
            # Entering the ollama section.
            in_ollama = True

        elif in_ollama and line.strip().startswith("model:"):
            # Found the model line inside the ollama section.
            # Preserve leading whitespace so we don't break YAML indentation.
            leading_ws = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{leading_ws}model: {new_model}\n"
            break  # Done — no need to scan further

        elif in_ollama and not line.startswith(" ") and line.strip() != "":
            # A non-indented, non-empty line means we've left the ollama block.
            in_ollama = False

    with open(config_path, "w") as f:
        f.writelines(lines)


class OllamaPullThread(QThread):
    """Background thread that pulls (downloads) an Ollama model.

    Calls ``POST /api/pull`` on the Ollama server with ``stream: true`` and
    parses the newline-delimited JSON progress updates. Each update is
    forwarded to the main thread via the ``progress`` signal.

    Signals:
        progress (str, int): Emitted for each progress line received from
            Ollama. The ``str`` is a human-readable status message (e.g.
            ``"pulling manifest (42%)"``) and the ``int`` is the percentage
            0–100 (or 0 if the total size is unknown).
        finished_pull (bool, str): Emitted exactly once when the pull
            finishes. ``bool`` is ``True`` on success, ``False`` on failure
            or cancellation. ``str`` is empty on success, or the error/
            cancellation message on failure.
    """

    # Signal: (status_text, percentage_0_to_100)
    progress = Signal(str, int)
    # Signal: (success, error_message)
    finished_pull = Signal(bool, str)

    def __init__(self, endpoint: str, model: str):
        """Initialise the pull thread.

        Args:
            endpoint (str): Base URL of the Ollama server (no trailing slash),
                e.g. ``"http://localhost:11434"``.
            model (str): The model tag to pull, e.g. ``"llama3.2:1b-instruct-q4_K_M"``.
        """
        super().__init__()
        self.endpoint = endpoint
        self.model = model
        # _cancel is checked in the streaming loop so the download can be
        # aborted cleanly from the main thread.
        self._cancel = False

    def cancel(self) -> None:
        """Request cancellation of the ongoing pull.

        Sets the ``_cancel`` flag, which the ``run()`` streaming loop checks
        before processing each line. The thread will emit
        ``finished_pull(False, "Cancelled")`` and exit cleanly.

        Thread safety:
            This method is called from the main thread (connected to the
            progress dialog's Cancel button). Setting a bool is atomic in
            CPython, so no lock is needed.
        """
        self._cancel = True

    def run(self) -> None:
        """Execute the model pull and emit progress/completion signals.

        Sends ``POST /api/pull`` with ``stream=True`` and iterates over the
        newline-delimited JSON response. Each line contains a ``"status"``
        field and optionally ``"total"`` and ``"completed"`` byte counts
        used to compute a percentage.

        Side effects:
            * Emits ``progress`` for each received line.
            * Emits ``finished_pull(True, "")`` on success.
            * Emits ``finished_pull(False, reason)`` on cancellation or error.
        """
        try:
            url = f"{self.endpoint}/api/pull"
            payload = {"model": self.model, "stream": True}

            # Use requests' streaming mode to iterate lines as they arrive.
            with requests.post(url, json=payload, stream=True, timeout=120) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if self._cancel:
                        self.finished_pull.emit(False, "Cancelled")
                        return

                    if line:
                        data = json.loads(line)
                        status = data.get("status", "Pulling...")
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)

                        if total > 0:
                            # Compute percentage if byte counts are available.
                            pct = int((completed / total) * 100)
                            self.progress.emit(f"{status} ({pct}%)", pct)
                        else:
                            # Unknown total — emit 0% to keep dialog updated.
                            self.progress.emit(status, 0)

            self.finished_pull.emit(True, "")
        except Exception as e:
            self.finished_pull.emit(False, str(e))


class OllamaStartThread(QThread):
    """Start a local Ollama server and wait at most 15 seconds for its API.

    A successfully started server is tracked so the user can stop it on quit.
    Only the process created here is stopped if startup fails.
    """

    finished_start = Signal(bool, str)

    def __init__(self, endpoint: str):
        super().__init__()
        self.endpoint = endpoint

    def run(self) -> None:
        process = None
        try:
            executable = shutil.which("ollama")
            if not executable:
                raise RuntimeError(
                    "Ollama is not installed or is not on PATH. "
                    "Install it from https://ollama.com/download, then try again."
                )
            env = os.environ.copy()
            # Respect the configured local port rather than an inherited host.
            env["OLLAMA_HOST"] = self.endpoint
            process = subprocess.Popen(
                [executable, "serve"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            deadline = time.monotonic() + 15
            while (remaining := deadline - time.monotonic()) > 0:
                try:
                    response = requests.get(
                        f"{self.endpoint}/api/tags", timeout=min(1, remaining)
                    )
                    response.raise_for_status()
                    response.json()
                    if process.poll() is None:
                        _started_servers.append((self.endpoint, process))
                    self.finished_start.emit(True, "")
                    return
                except (requests.RequestException, ValueError):
                    pass
                if process.poll() is not None:
                    raise RuntimeError(
                        f"ollama serve exited with code {process.returncode}. "
                        "Run 'ollama serve' in a terminal to see its error output."
                    )
                self.msleep(250)
            raise RuntimeError("Ollama did not become ready within 15 seconds.")
        except Exception as exc:
            error = str(exc)
            try:
                if process is not None:
                    _stop_server_process(process)
            except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                error += f"\nCould not stop the failed server: {cleanup_error}"
            self.finished_start.emit(False, error)


class OllamaModelWizard:
    """Interactive wizard for selecting or downloading an Ollama LLM model.

    This wizard guides the user through choosing which model Ollama Vox will
    use for conversation. It handles four scenarios:

    1. Local Ollama is not running → ask permission to start ``ollama serve``.
    2. No models are installed → offer to download the default model.
    3. Models are installed → show a combo-box selection dialog.
    4. ``force_setup=True`` → always show the selection/download dialog.

    Args:
        config (AppConfig): Validated application config. Used to read
            ``config.ollama.endpoint`` and ``config.ollama.model``.

    Example::

        wizard = OllamaModelWizard(config)
        if not wizard.run():
            sys.exit(1)  # User cancelled or Ollama is not reachable
    """

    def __init__(self, config):
        self.config = config
        # Strip trailing slash from endpoint for clean URL construction.
        self.endpoint = config.ollama.endpoint.rstrip("/")

        # Determine the "recommended" / default model.
        # Use the model currently in config if available; otherwise fall back
        # to the hard-coded default.
        self.default_model = "llama3.2:1b-instruct-q4_K_M"
        if getattr(config.ollama, "model", None):
            self.default_model = config.ollama.model

    def run(self, force_setup: bool = False) -> bool:
        """Check Ollama, query available models, and prompt the user to select one.

        Args:
            force_setup (bool): When ``True``, always show the model selection
                dialog even if a valid model is already configured. Default: False.

        Returns:
            bool: ``True`` if a model was successfully selected (and the app
                  can proceed). ``False`` if the user cancelled.
        """
        # --- Step 1: Verify Ollama is running, offering local startup once ---
        try:
            try:
                models = self._available_models()
            except (requests.ConnectionError, requests.Timeout):
                if not self.start_local_server():
                    return False
                models = self._available_models()
        except Exception as exc:
            self._show_connection_error(str(exc))
            return False

        # --- Step 2: Determine if model selection is needed ---
        current_model = self.config.ollama.model
        # needs_model is True if: no models exist, setup is forced, or the
        # currently configured model is not in the available list.
        needs_model = not models or force_setup or current_model not in models

        if not needs_model and not force_setup:
            # Everything is already set up — nothing to do.
            return True

        # --- Step 3: No models installed → offer to download default ---
        if not models:
            msg = QMessageBox()
            msg.setWindowTitle("No Models Detected")
            msg.setText(
                f"No models found in Ollama.\nWould you like to download the default model ({self.default_model})?"
            )
            msg.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No  # type: ignore
            )
            ans = msg.exec()

            if ans == QMessageBox.StandardButton.Yes:  # type: ignore
                success = self.pull_model(self.default_model)
                if success:
                    self.save_model_selection(self.default_model)
                    return True
                return False
            else:
                # User declined to download — cannot proceed without a model.
                return False

        # --- Step 4: Models exist → show selection dialog ---
        # Build display labels. Mark the recommended model with a suffix.
        items = []
        default_index = 0
        for i, m in enumerate(models):
            display = m
            if m == self.default_model:
                display += " (recommended)"
                default_index = i
            items.append(display)

        # Build a simple modal dialog with a combo-box.
        dialog = QDialog()
        dialog.setWindowTitle("Select Model")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Choose an Ollama model to use:"))

        combo = QComboBox()
        combo.addItems(items)
        combo.setCurrentIndex(default_index)
        layout.addWidget(combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel  # type: ignore
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Strip the "(recommended)" suffix before saving.
            selected_model = combo.currentText().replace(" (recommended)", "")
            self.save_model_selection(selected_model)
            return True

        # User clicked Cancel.
        return False

    def _available_models(self) -> list[str]:
        response = requests.get(f"{self.endpoint}/api/tags", timeout=2)
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]

    def _show_connection_error(self, error: str) -> None:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Ollama Unavailable")
        msg.setText(f"Could not connect to Ollama at {self.endpoint}.")
        msg.setInformativeText(error)
        msg.exec()

    def start_local_server(self) -> bool:
        """Ask before starting a local server; never launch for a remote URL."""
        if not is_local_endpoint(self.endpoint):
            self._show_connection_error(
                "Start Ollama on the configured host and try again. "
                "Ollama Vox can only start a local HTTP server."
            )
            return False

        msg = QMessageBox()
        msg.setWindowTitle("Start Ollama?")
        msg.setText(f"Ollama is not running at {self.endpoint}. Start it now?")
        msg.setInformativeText(
            "This runs 'ollama serve' in the background. "
            "When you quit Ollama Vox, you can stop the server or leave it running."
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        if msg.exec() != QMessageBox.StandardButton.Yes:
            return False

        progress = QProgressDialog("Starting Ollama...", "", 0, 0, None)
        progress.setWindowTitle("Starting Ollama")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)
        progress.show()

        thread = OllamaStartThread(self.endpoint)
        loop = QEventLoop()
        result = [False, ""]

        def on_finished(success: bool, error: str) -> None:
            result[:] = [success, error]
            loop.quit()

        thread.finished_start.connect(on_finished)
        thread.start()
        loop.exec()
        # Keep the QThread alive until run() has returned after emitting.
        thread.wait()
        progress.close()
        if not result[0]:
            self._show_connection_error(result[1])
        return result[0]

    def pull_model(self, model_name: str) -> bool:
        """Download an Ollama model with a cancellable progress dialog.

        Creates a :class:`QProgressDialog`, starts an :class:`OllamaPullThread`,
        and blocks (via a :class:`QEventLoop`) until the download completes.

        Args:
            model_name (str): The Ollama model tag to pull, e.g.
                ``"llama3.2:1b-instruct-q4_K_M"``.

        Returns:
            bool: ``True`` if the model was pulled successfully, ``False`` if
                  the user cancelled or an error occurred.
        """
        # Create an auto-closing progress dialog (0–100 range).
        progress_dialog = QProgressDialog(
            f"Pulling {model_name}...", "Cancel", 0, 100, None
        )
        progress_dialog.setWindowTitle("Downloading Model")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)
        progress_dialog.show()

        # Start the background pull thread.
        thread = OllamaPullThread(self.endpoint, model_name)

        def update_progress(status: str, val: int) -> None:
            """Update the dialog label and progress bar from the thread signal."""
            progress_dialog.setLabelText(status)
            progress_dialog.setValue(val)

        thread.progress.connect(update_progress)

        # Use QEventLoop to block without freezing Qt's event loop.
        loop = QEventLoop()
        success_result = [False]  # Mutable container for closure capture
        error_msg = [""]

        def on_finished(success: bool, err: str) -> None:
            """Capture thread result and unblock the event loop."""
            success_result[0] = success
            error_msg[0] = err
            loop.quit()

        thread.finished_pull.connect(on_finished)
        # If the user clicks Cancel on the dialog, signal the thread to stop.
        progress_dialog.canceled.connect(thread.cancel)

        thread.start()
        loop.exec()  # Block here until on_finished() calls loop.quit()

        # --- Handle cancellation ---
        if progress_dialog.wasCanceled():
            warn_msg = QMessageBox()
            warn_msg.setIcon(QMessageBox.Icon.Warning)
            warn_msg.setWindowTitle("Cancelled")
            warn_msg.setText("Model download was cancelled.")
            warn_msg.exec()
            return False

        # --- Handle failure ---
        if not success_result[0] and error_msg[0] != "Cancelled":
            err_msg = QMessageBox()
            err_msg.setIcon(QMessageBox.Icon.Critical)
            err_msg.setWindowTitle("Error")
            err_msg.setText(f"Failed to pull model:\n{error_msg[0]}")
            err_msg.exec()
            return False

        # --- Success ---
        info_msg = QMessageBox()
        info_msg.setIcon(QMessageBox.Icon.Information)
        info_msg.setWindowTitle("Success")
        info_msg.setText(f"Successfully pulled {model_name}.")
        info_msg.exec()
        return True

    def save_model_selection(self, model_name: str) -> None:
        """Persist the chosen model name in memory and in ``config.yaml``.

        Updates ``self.config.ollama.model`` so that the rest of the
        application immediately uses the new model, and writes the change
        to disk so it persists across restarts.

        Args:
            model_name (str): The Ollama model tag to save.

        Side effects:
            * Mutates ``self.config.ollama.model``.
            * Overwrites the ``model:`` line in ``config.yaml`` via
              :func:`update_config_model`.
            * Prints an error message to stdout if the file write fails
              (does not raise so that the application can continue even if
              the config cannot be persisted).
        """
        # Update the in-memory config so the pipeline uses the new model
        # without needing a restart.
        self.config.ollama.model = model_name

        from pathlib import Path

        # Locate the config file (same logic as in main.load_config).
        base_dir = Path(__file__).parent.parent.resolve()
        config_path = base_dir / "config.yaml"
        if not config_path.exists():
            config_path = Path("config.yaml")

        try:
            update_config_model(str(config_path), model_name)
        except Exception as e:
            # Non-fatal: model is updated in memory even if disk write fails.
            print(f"Failed to update config.yaml: {e}")
