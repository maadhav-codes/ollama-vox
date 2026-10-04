from unittest.mock import Mock

from ollama_vox.ui.preparation import SpeechPreparationThread


def test_speech_preparation_runs_silently():
    tts = Mock()
    thread = SpeechPreparationThread(tts)
    thread.run()
    tts.prepare.assert_called_once_with()
    tts.speak.assert_not_called()
    assert thread.error is None


def test_speech_preparation_keeps_failure_for_startup_dialog():
    tts = Mock()
    tts.prepare.side_effect = RuntimeError("Missing English text-processing model")
    thread = SpeechPreparationThread(tts)
    thread.run()
    assert thread.error == "Missing English text-processing model"


def test_spacy_download_uses_current_python_and_official_cli(mocker):
    import sys
    from ollama_vox.ui import preparation

    run = mocker.patch.object(preparation.subprocess, "run")
    mocker.patch.object(preparation, "english_model_installed", return_value=True)
    thread = preparation.SpacyDownloadThread()
    thread.run()
    assert thread.error is None
    assert run.call_args.args[0] == [
        sys.executable,
        "-m",
        "spacy",
        "download",
        "en_core_web_sm",
    ]
    assert run.call_args.kwargs["env"]["UV_PYTHON"] == sys.executable
    assert run.call_args.kwargs["timeout"] == 180
    assert run.call_args.kwargs["check"] is True


def test_installed_spacy_model_never_prompts_or_downloads(mocker):
    from ollama_vox.ui import preparation

    mocker.patch.object(preparation, "english_model_installed", return_value=True)
    question = mocker.patch.object(preparation.QMessageBox, "question")
    thread = mocker.patch.object(preparation, "SpacyDownloadThread")
    assert preparation.ensure_speech_dependencies()
    question.assert_not_called()
    thread.assert_not_called()


def test_declining_spacy_download_blocks_setup_without_starting_installer(mocker):
    from ollama_vox.ui import preparation

    mocker.patch.object(preparation, "english_model_installed", return_value=False)
    mocker.patch.object(
        preparation.QMessageBox,
        "question",
        return_value=preparation.QMessageBox.StandardButton.No,
    )
    thread = mocker.patch.object(preparation, "SpacyDownloadThread")
    assert not preparation.ensure_speech_dependencies()
    thread.assert_not_called()


def test_failed_spacy_download_is_reported(mocker):
    import subprocess
    from ollama_vox.ui import preparation

    mocker.patch.object(
        preparation.subprocess,
        "run",
        side_effect=subprocess.CalledProcessError(1, "spacy", stderr="Download failed"),
    )
    thread = preparation.SpacyDownloadThread()
    thread.run()
    assert "Download failed" in thread.error
