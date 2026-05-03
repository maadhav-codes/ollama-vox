import os
import sys
from PySide6.QtCore import Qt, QThread, Signal, QEventLoop
from PySide6.QtWidgets import QMessageBox, QProgressDialog
from huggingface_hub import snapshot_download


class SetupDownloadThread(QThread):
    progress = Signal(str, int)
    finished_pull = Signal(bool, str)

    def __init__(self, config):
        super().__init__()
        self.config = config

    @staticmethod
    def _ensure_parent(path):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    def run(self):
        try:
            stt_path = self.config.stt.model
            tts_path = self.config.tts.model
            voice = self.config.tts.voice

            if not os.path.exists(stt_path):
                self._ensure_parent(stt_path)
                self.progress.emit("Downloading STT model (Whisper)...", 0)
                snapshot_download(
                    repo_id="mlx-community/whisper-small.en-mlx-q4",
                    local_dir=stt_path,
                )

            if not os.path.exists(tts_path):
                self._ensure_parent(tts_path)
                self.progress.emit("Downloading TTS model (Kokoro)...", 0)
                allow_patterns = [
                    "*.json",
                    "*.md",
                    "*.safetensors",
                    "*.pth",
                    f"voices/{voice}.safetensors",
                    f"voices/{voice}.pt",
                ]
                snapshot_download(
                    repo_id="mlx-community/Kokoro-82M-4bit",
                    local_dir=tts_path,
                    allow_patterns=allow_patterns,
                )

            self.finished_pull.emit(True, "")
        except Exception as e:
            self.finished_pull.emit(False, str(e))


class AppSetupWizard:
    def __init__(self, config):
        self.config = config

    def run(self, force_setup=False) -> bool:
        stt_path = self.config.stt.model
        tts_path = self.config.tts.model

        needs_download = not os.path.exists(stt_path) or not os.path.exists(tts_path)

        if not needs_download:
            if force_setup:
                info_msg = QMessageBox()
                info_msg.setIcon(QMessageBox.Icon.Information)
                info_msg.setWindowTitle("Ready")
                info_msg.setText(
                    "Required models (STT and TTS) are already available to use."
                )
                info_msg.exec()
            return True

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
            err_msg = QMessageBox()
            err_msg.setIcon(QMessageBox.Icon.Critical)
            err_msg.setWindowTitle("Setup Cancelled")
            err_msg.setText(
                "It is recommended to download these models. The program cannot run without them."
            )
            err_msg.exec()
            sys.exit(1)

        # Disable HuggingFace CLI progress bars
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        progress_dialog = QProgressDialog("Downloading models...", "", 0, 0, None)
        progress_dialog.setWindowTitle("Downloading Models")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.setCancelButton(None)  # Disable cancel
        progress_dialog.show()

        thread = SetupDownloadThread(self.config)

        def update_progress(status, _):
            progress_dialog.setLabelText(status)

        thread.progress.connect(update_progress)

        loop = QEventLoop()
        success_result = [False]
        error_msg = [""]

        def on_finished(success, err):
            success_result[0] = success
            error_msg[0] = err
            loop.quit()

        thread.finished_pull.connect(on_finished)
        thread.start()
        loop.exec()

        if not success_result[0]:
            err_msg = QMessageBox()
            err_msg.setIcon(QMessageBox.Icon.Critical)
            err_msg.setWindowTitle("Error")
            err_msg.setText(f"Failed to download models:\n{error_msg[0]}")
            err_msg.exec()
            sys.exit(1)

        info_msg = QMessageBox()
        info_msg.setIcon(QMessageBox.Icon.Information)
        info_msg.setWindowTitle("Success")
        info_msg.setText("Models downloaded successfully.")
        info_msg.exec()
        return True
