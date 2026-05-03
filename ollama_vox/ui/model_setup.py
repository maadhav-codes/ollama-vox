import json
import requests
from PySide6.QtCore import Qt, QThread, Signal, QEventLoop
from PySide6.QtWidgets import (
    QMessageBox,
    QProgressDialog,
    QDialog,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QDialogButtonBox,
)


def update_config_model(config_path: str, new_model: str) -> None:
    with open(config_path, "r") as f:
        lines = f.readlines()

    in_ollama = False
    for i, line in enumerate(lines):
        if line.strip() == "ollama:":
            in_ollama = True
        elif in_ollama and line.strip().startswith("model:"):
            # preserve leading whitespace
            leading_ws = line[: len(line) - len(line.lstrip())]
            lines[i] = f"{leading_ws}model: {new_model}\n"
            break
        elif in_ollama and not line.startswith(" ") and line.strip() != "":
            # out of ollama block
            in_ollama = False

    with open(config_path, "w") as f:
        f.writelines(lines)


class OllamaPullThread(QThread):
    progress = Signal(str, int)  # status text, progress percentage (0-100)
    finished_pull = Signal(bool, str)  # success, error_message

    def __init__(self, endpoint: str, model: str):
        super().__init__()
        self.endpoint = endpoint
        self.model = model
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            url = f"{self.endpoint}/api/pull"
            payload = {"model": self.model, "stream": True}
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
                            pct = int((completed / total) * 100)
                            self.progress.emit(f"{status} ({pct}%)", pct)
                        else:
                            self.progress.emit(status, 0)
            self.finished_pull.emit(True, "")
        except Exception as e:
            self.finished_pull.emit(False, str(e))


class OllamaModelWizard:
    def __init__(self, config):
        self.config = config
        self.endpoint = config.ollama.endpoint.rstrip("/")
        self.default_model = "llama3.2:1b-instruct-q4_K_M"
        if getattr(config.ollama, "model", None):
            self.default_model = config.ollama.model

    def run(self, force_setup=False) -> bool:
        # 1. Check if Ollama is running
        while True:
            try:
                r = requests.get(f"{self.endpoint}/api/tags", timeout=2)
                r.raise_for_status()
                data = r.json()
                models = [m["name"] for m in data.get("models", [])]
                break
            except Exception as e:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setText("Ollama is not reachable.")
                msg.setInformativeText(
                    f"Please ensure Ollama is running at {self.endpoint}.\n\nError: {e}"
                )
                msg.setStandardButtons(
                    QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel  # type: ignore
                )
                if msg.exec() == QMessageBox.StandardButton.Cancel:  # type: ignore
                    return False

        # 2. Check if we have models or if setup is forced
        current_model = self.config.ollama.model
        needs_model = not models or force_setup or current_model not in models

        if not needs_model and not force_setup:
            return True

        # 3. If no models, prompt to download default
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
                return False

        # 4. If models exist, show selection dialog
        items = []
        default_index = 0
        for i, m in enumerate(models):
            display = m
            if m == self.default_model:
                display += " (recommended)"
                default_index = i
            items.append(display)

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
            selected_model = combo.currentText().replace(" (recommended)", "")
            self.save_model_selection(selected_model)
            return True

        return False

    def pull_model(self, model_name: str) -> bool:
        progress_dialog = QProgressDialog(
            f"Pulling {model_name}...", "Cancel", 0, 100, None
        )
        progress_dialog.setWindowTitle("Downloading Model")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)
        progress_dialog.show()

        thread = OllamaPullThread(self.endpoint, model_name)

        def update_progress(status, val):
            progress_dialog.setLabelText(status)
            progress_dialog.setValue(val)

        thread.progress.connect(update_progress)

        loop = QEventLoop()
        success_result = [False]
        error_msg = [""]

        def on_finished(success, err):
            success_result[0] = success
            error_msg[0] = err
            loop.quit()

        thread.finished_pull.connect(on_finished)

        progress_dialog.canceled.connect(thread.cancel)

        thread.start()
        loop.exec()

        if progress_dialog.wasCanceled():
            warn_msg = QMessageBox()
            warn_msg.setIcon(QMessageBox.Icon.Warning)
            warn_msg.setWindowTitle("Cancelled")
            warn_msg.setText("Model download was cancelled.")
            warn_msg.exec()
            return False

        if not success_result[0] and error_msg[0] != "Cancelled":
            err_msg = QMessageBox()
            err_msg.setIcon(QMessageBox.Icon.Critical)
            err_msg.setWindowTitle("Error")
            err_msg.setText(f"Failed to pull model:\n{error_msg[0]}")
            err_msg.exec()
            return False

        info_msg = QMessageBox()
        info_msg.setIcon(QMessageBox.Icon.Information)
        info_msg.setWindowTitle("Success")
        info_msg.setText(f"Successfully pulled {model_name}.")
        info_msg.exec()
        return True

    def save_model_selection(self, model_name: str) -> None:
        self.config.ollama.model = model_name

        from pathlib import Path

        base_dir = Path(__file__).parent.parent.resolve()
        config_path = base_dir / "config.yaml"
        if not config_path.exists():
            config_path = Path("config.yaml")

        try:
            update_config_model(str(config_path), model_name)
        except Exception as e:
            print(f"Failed to update config.yaml: {e}")
