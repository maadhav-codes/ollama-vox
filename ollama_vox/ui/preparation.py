"""Prepare local speech synthesis before the user can record their first turn."""

import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

from PySide6.QtCore import QEventLoop, QThread
from PySide6.QtWidgets import QMessageBox, QProgressDialog


def english_model_installed() -> bool:
    try:
        version("en-core-web-sm")
        return True
    except PackageNotFoundError:
        return False


class SpacyDownloadThread(QThread):
    def __init__(self):
        super().__init__()
        self.error = None

    def run(self):
        try:
            environment = os.environ.copy()
            # spaCy falls back to `uv pip` when pip is absent. Keep that
            # installer in the same environment as this running application.
            environment["UV_PYTHON"] = sys.executable
            if sys.prefix != sys.base_prefix:
                environment["VIRTUAL_ENV"] = sys.prefix
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if not english_model_installed():
                raise RuntimeError(
                    "spaCy finished but the English model is still missing."
                )
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or "").strip()
            self.error = f"spaCy model installation failed.\n{details[-3000:]}"
        except subprocess.TimeoutExpired:
            self.error = "spaCy model installation timed out. Check your connection and retry setup."
        except Exception as exc:
            self.error = str(exc)


def ensure_speech_dependencies() -> bool:
    if english_model_installed():
        return True
    answer = QMessageBox.question(
        None,
        "Install English Speech Support",
        "Kokoro needs spaCy's English model (en_core_web_sm). Install it now "
        "using `python -m spacy download en_core_web_sm`?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if answer != QMessageBox.StandardButton.Yes:
        return False
    dialog = QProgressDialog("Installing spaCy English model…", "", 0, 0)
    dialog.setWindowTitle("Setting Up Speech Support")
    dialog.setCancelButton(None)
    dialog.show()
    thread = SpacyDownloadThread()
    loop = QEventLoop()
    thread.finished.connect(loop.quit)
    thread.start()
    loop.exec()
    thread.wait()
    dialog.close()
    if thread.error:
        QMessageBox.critical(None, "Speech Setup Failed", thread.error)
        return False
    return True


class SpeechPreparationThread(QThread):
    def __init__(self, tts):
        super().__init__()
        self.tts = tts
        self.error = None

    def run(self):
        try:
            self.tts.prepare()
        except Exception as exc:
            self.error = str(exc)


def prepare_speech(tts) -> bool:
    dialog = QProgressDialog("Preparing local speech synthesis…", "", 0, 0)
    dialog.setWindowTitle("Preparing Ollama Vox")
    dialog.setCancelButton(None)
    # Keep initialization visible and the GUI responsive; no microphone/audio playback.
    dialog.show()
    thread = SpeechPreparationThread(tts)
    loop = QEventLoop()
    thread.finished.connect(loop.quit)
    thread.start()
    loop.exec()
    thread.wait()
    dialog.close()
    if thread.error:
        QMessageBox.critical(None, "Speech Setup Failed", thread.error)
        return False
    return True
