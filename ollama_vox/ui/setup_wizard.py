"""First-run setup wizard: downloads Whisper STT and Kokoro TTS models.

This module handles the very first time a user runs Ollama Vox. It checks
whether the required AI model files are already present on disk. If they
are not, it presents a GUI dialog asking the user for permission to download
them from HuggingFace Hub, then runs the download in a background thread
with a progress dialog.

Why a background thread?
-------------------------
Downloading models can take minutes (the Kokoro TTS model alone is several
hundred MB). Running the download on the main Qt thread would freeze the
entire UI. Instead, we use :class:`SetupDownloadThread` (a QThread subclass)
to do the work in the background while a :class:`QProgressDialog` keeps the
user informed. A :class:`QEventLoop` bridges the thread completion back to
the blocking ``run()`` method.

Where it fits in startup::

    main() → AppSetupWizard.run()   (STT + TTS models)
           → OllamaModelWizard.run()  (Ollama LLM model)
           → Pipeline + UI

Dependencies:
    * ``PySide6``         — Qt GUI framework.
    * ``huggingface_hub`` — ``snapshot_download`` for model downloads.
"""

import os
import sys

from huggingface_hub import snapshot_download
from PySide6.QtCore import QEventLoop, Qt, QThread, Signal
from PySide6.QtWidgets import QMessageBox, QProgressDialog


class SetupDownloadThread(QThread):
    """Background thread that downloads the Whisper and Kokoro model files.

    Emits Qt signals to report progress and completion back to the main thread.
    The main thread uses these signals to update the progress dialog and
    react to success or failure.

    Signals:
        progress (str, int): Emitted during the download with a human-readable
            status message and a percentage (0–100). The percentage is not
            always meaningful (HuggingFace Hub doesn't always report it), so
            the progress dialog uses indeterminate mode (max=0).
        finished_pull (bool, str): Emitted exactly once when the thread
            finishes. ``bool`` is ``True`` on success, ``False`` on failure.
            ``str`` is an empty string on success or the error message on
            failure.
    """

    # Signal signatures: (status_text, percentage)
    progress = Signal(str, int)
    # Signal signatures: (success, error_message)
    finished_pull = Signal(bool, str)

    def __init__(self, config):
        """Initialise the download thread with the application configuration.

        Args:
            config (AppConfig): The validated application config object.
                Used to read ``config.stt.model`` (Whisper path),
                ``config.tts.model`` (Kokoro path), and ``config.tts.voice``
                (voice file name for selective downloading).
        """
        super().__init__()
        self.config = config

    @staticmethod
    def _ensure_parent(path: str) -> None:
        """Create all parent directories of ``path`` if they don't exist.

        For example, if ``path`` is ``"./whisper/whisper-small.en-mlx-q4"``,
        this ensures the ``./whisper/`` directory exists before
        ``snapshot_download`` tries to write into it.

        Args:
            path (str): The target path whose parent directories should exist.
        """
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    def run(self) -> None:
        """Download missing STT and TTS model files from HuggingFace Hub.

        This method runs on the background thread (invoked by Qt when
        ``thread.start()`` is called). It:

        1. Checks whether the Whisper STT model directory already exists.
           If not, creates its parent directory and downloads it.
        2. Checks whether the Kokoro TTS model directory already exists.
           If not, downloads only the files needed for the configured voice
           (skipping unused voice files to save disk space).
        3. Emits ``finished_pull(True, "")`` on success, or
           ``finished_pull(False, error_message)`` if any step fails.

        Side effects:
            * May create directories on disk.
            * Downloads several hundred MB from HuggingFace Hub.
            * Emits ``progress`` and ``finished_pull`` signals.
        """
        try:
            stt_path = self.config.stt.model
            tts_path = self.config.tts.model
            voice = self.config.tts.voice

            # --- Download Whisper STT model if missing ---
            if not os.path.exists(stt_path):
                self._ensure_parent(stt_path)
                self.progress.emit("Downloading STT model (Whisper)...", 0)
                snapshot_download(
                    repo_id="mlx-community/whisper-small.en-mlx-q4",
                    local_dir=stt_path,
                )

            # --- Download Kokoro TTS model if missing ---
            if not os.path.exists(tts_path):
                self._ensure_parent(tts_path)
                self.progress.emit("Downloading TTS model (Kokoro)...", 0)

                # Only download files relevant to the configured voice.
                # This avoids downloading all ~20 voice files (saves ~500 MB).
                allow_patterns = [
                    "*.json",  # Model config files
                    "*.md",  # Documentation
                    "*.safetensors",  # Model weights (all)
                    "*.pth",  # PyTorch checkpoint (if any)
                    f"voices/{voice}.safetensors",  # Voice-specific weights
                    f"voices/{voice}.pt",  # Voice-specific checkpoint
                ]
                snapshot_download(
                    repo_id="mlx-community/Kokoro-82M-4bit",
                    local_dir=tts_path,
                    allow_patterns=allow_patterns,
                )

            # Both models are present — signal success.
            self.finished_pull.emit(True, "")
        except Exception as e:
            # Signal failure with the exception message for display in the UI.
            self.finished_pull.emit(False, str(e))


class AppSetupWizard:
    """Wizard that ensures the STT and TTS model files are downloaded.

    On the first run (or when ``--setup`` is passed on the CLI), this wizard
    checks for the Whisper and Kokoro model directories. If either is missing
    it asks the user's permission to download them, then runs the download
    in a background thread with a progress dialog.

    Args:
        config (AppConfig): Validated application configuration. Used to
            determine model paths and the configured voice.

    Example::

        wizard = AppSetupWizard(config)
        success = wizard.run()  # blocks until done
        if not success:
            sys.exit(1)
    """

    def __init__(self, config):
        self.config = config

    def run(self, force_setup: bool = False) -> bool:
        """Check for models and download them if needed.

        Logic overview:

        1. If both model directories already exist:
           - If ``force_setup=True`` (called via ``--setup`` flag), show an
             informational dialog saying models are already present, then
             return ``True``.
           - Otherwise, return ``True`` immediately (nothing to do).
        2. If either model is missing:
           - Ask the user for permission. If they say No, show an error and
             call ``sys.exit(1)`` (the app cannot run without models).
           - If they say Yes, run the background download with a progress
             dialog. Exit on failure; show a success message and return
             ``True`` on success.

        Args:
            force_setup (bool): When ``True``, always show the "already
                ready" dialog even if models exist. Used by ``--setup`` mode.

        Returns:
            bool: ``True`` if models are available (already existed or were
                  downloaded successfully). Never returns ``False`` — the
                  method either returns ``True`` or calls ``sys.exit(1)``.
        """
        stt_path = self.config.stt.model
        tts_path = self.config.tts.model

        # Check whether both model directories already exist on disk.
        needs_download = not os.path.exists(stt_path) or not os.path.exists(tts_path)

        if not needs_download:
            if force_setup:
                # Models already present — inform the user and return.
                info_msg = QMessageBox()
                info_msg.setIcon(QMessageBox.Icon.Information)
                info_msg.setWindowTitle("Ready")
                info_msg.setText(
                    "Required models (STT and TTS) are already available to use."
                )
                info_msg.exec()
            return True

        # --- At least one model is missing — ask the user ---
        msg = QMessageBox()
        msg.setWindowTitle("Download Required Models")
        msg.setText(
            "Ollama Vox needs to download required STT (Whisper) and TTS (Kokoro) models. Download now?"
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No  # type: ignore
        )
        ans = msg.exec()

        if ans == QMessageBox.StandardButton.No:
            # Without models the app cannot function — inform and exit.
            err_msg = QMessageBox()
            err_msg.setIcon(QMessageBox.Icon.Critical)
            err_msg.setWindowTitle("Setup Cancelled")
            err_msg.setText(
                "It is recommended to download these models. The program cannot run without them."
            )
            err_msg.exec()
            sys.exit(1)

        # Suppress HuggingFace's own CLI progress bars (they conflict with
        # our QProgressDialog).
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        # --- Show progress dialog ---
        # max=0 creates an "indeterminate" (bouncing) progress bar because
        # we don't always get reliable percentage data from snapshot_download.
        progress_dialog = QProgressDialog("Downloading models...", "", 0, 0, None)
        progress_dialog.setWindowTitle("Downloading Models")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setAutoClose(True)
        # Remove the Cancel button — downloads are required and cannot be skipped.
        progress_dialog.setCancelButton(None)
        progress_dialog.show()

        # --- Start the background download thread ---
        thread = SetupDownloadThread(self.config)

        def update_progress(status, _):
            """Update the dialog label text (percentage is unused here)."""
            progress_dialog.setLabelText(status)

        thread.progress.connect(update_progress)

        # Use a QEventLoop to block the calling code (run()) until the thread
        # finishes, without freezing the Qt event loop itself.
        loop = QEventLoop()
        success_result = [False]  # Mutable container to capture result from closure
        error_msg = [""]

        def on_finished(success: bool, err: str) -> None:
            """Capture the thread result and quit the event loop."""
            success_result[0] = success
            error_msg[0] = err
            loop.quit()

        thread.finished_pull.connect(on_finished)
        thread.start()
        # Block here until on_finished() calls loop.quit().
        loop.exec()

        if not success_result[0]:
            # Download failed — show error and exit (models are required).
            err_msg = QMessageBox()
            err_msg.setIcon(QMessageBox.Icon.Critical)
            err_msg.setWindowTitle("Error")
            err_msg.setText(f"Failed to download models:\n{error_msg[0]}")
            err_msg.exec()
            sys.exit(1)

        # Download succeeded — inform the user and proceed.
        info_msg = QMessageBox()
        info_msg.setIcon(QMessageBox.Icon.Information)
        info_msg.setWindowTitle("Success")
        info_msg.setText("Models downloaded successfully.")
        info_msg.exec()
        return True
