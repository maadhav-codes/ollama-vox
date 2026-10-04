"""Existing model files are reused before asking for a download."""

import pytest

from ollama_vox.core.config import AppConfig
from ollama_vox.core.model_assets import discover_model_paths, model_files_available


def make_model(path, kind):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / ("weights.npz" if kind == "stt" else "model.safetensors")).write_bytes(
        b"weights"
    )
    if kind == "tts":
        (path / "voices").mkdir(exist_ok=True)
        (path / "voices/af_bella.pt").write_bytes(b"voice")
    return path


@pytest.fixture(autouse=True)
def locations(tmp_path, mocker):
    root = tmp_path / "Documents/ollama-vox/models"
    mocker.patch(
        "ollama_vox.core.config.user_documents_path",
        return_value=tmp_path / "Documents",
    )
    mocker.patch(
        "ollama_vox.core.config.model_location_file",
        return_value=tmp_path / "settings.json",
    )
    mocker.patch("ollama_vox.core.model_assets.model_search_roots", return_value=[root])
    mocker.patch("ollama_vox.core.model_assets.HF_HUB_CACHE", str(tmp_path / "cache"))
    return root


@pytest.mark.parametrize("kind", ["stt", "tts"])
def test_empty_or_partial_directories_are_not_ready(tmp_path, kind):
    path = tmp_path / kind
    path.mkdir()
    assert not model_files_available(path, kind)
    (path / "config.json").write_text("{}")
    assert not model_files_available(path, kind)
    weights = path / ("weights.npz" if kind == "stt" else "model.safetensors")
    weights.touch()
    assert not model_files_available(path, kind)
    make_model(path, kind)
    assert model_files_available(path, kind)


def test_tts_requires_configured_voice(tmp_path):
    path = make_model(tmp_path / "tts", "tts")
    assert not model_files_available(path, "tts", "af_sky")


@pytest.mark.parametrize(
    "layout", ["project", "Desktop/My Models", "Documents/ollama-vox/models"]
)
def test_existing_models_are_discovered_and_remembered(tmp_path, mocker, layout):
    root = tmp_path / layout
    stt = make_model(root / "whisper/whisper-small.en-mlx-q4", "stt")
    tts = make_model(root / "kokoro/Kokoro-82M-4bit", "tts")
    mocker.patch("ollama_vox.core.model_assets.model_search_roots", return_value=[root])
    config = AppConfig.from_dict({})

    discover_model_paths(config)

    assert config.stt.model == str(stt)
    assert config.tts.model == str(tts)
    # A later launch keeps using the discovered locations.
    later = AppConfig.from_dict({})
    assert later.stt.model == str(stt)
    assert later.tts.model == str(tts)


def test_existing_models_skip_picker_and_download_prompt(tmp_path, mocker):
    from ollama_vox.ui.setup_wizard import AppSetupWizard

    root = tmp_path / "project"
    make_model(root / "whisper/whisper-small.en-mlx-q4", "stt")
    make_model(root / "kokoro/Kokoro-82M-4bit", "tts")
    mocker.patch("ollama_vox.core.model_assets.model_search_roots", return_value=[root])
    picker = mocker.patch("ollama_vox.ui.setup_wizard.QFileDialog")
    message = mocker.patch("ollama_vox.ui.setup_wizard.QMessageBox")
    download = mocker.patch("ollama_vox.ui.setup_wizard.SetupDownloadThread")

    assert AppSetupWizard(AppConfig.from_dict({})).run()

    picker.assert_not_called()
    message.assert_not_called()
    download.assert_not_called()


def test_selected_existing_folder_is_loaded_without_download_offer(tmp_path, mocker):
    from ollama_vox.ui.setup_wizard import AppSetupWizard

    root = tmp_path / "elsewhere"
    make_model(root / "whisper/whisper-small.en-mlx-q4", "stt")
    make_model(root / "kokoro/Kokoro-82M-4bit", "tts")
    mocker.patch(
        "ollama_vox.ui.setup_wizard.QFileDialog.getExistingDirectory",
        return_value=str(root),
    )
    message = mocker.patch("ollama_vox.ui.setup_wizard.QMessageBox")
    download = mocker.patch("ollama_vox.ui.setup_wizard.SetupDownloadThread")

    assert AppSetupWizard(AppConfig.from_dict({})).run()

    message.assert_not_called()
    download.assert_not_called()


def test_picker_keeps_a_model_already_found_in_another_location(tmp_path, mocker):
    from ollama_vox.ui.setup_wizard import AppSetupWizard

    stt = make_model(tmp_path / "existing/whisper-small.en-mlx-q4", "stt")
    mocker.patch(
        "ollama_vox.core.model_assets.model_search_roots", return_value=[stt.parent]
    )
    config = AppConfig.from_dict({})
    discover_model_paths(config)
    chosen = tmp_path / "new"
    mocker.patch(
        "ollama_vox.ui.setup_wizard.QFileDialog.getExistingDirectory",
        return_value=str(chosen),
    )

    AppSetupWizard(config).choose_model_directory()

    assert config.stt.model == str(stt)
    assert config.tts.model == str(chosen / "kokoro/Kokoro-82M-4bit")


def test_hugging_face_cache_is_reused(tmp_path):
    root = tmp_path / "cache"
    stt = make_model(
        root / "models--mlx-community--whisper-small.en-mlx-q4/snapshots/revision",
        "stt",
    )
    tts = make_model(
        root / "models--mlx-community--Kokoro-82M-4bit/snapshots/revision", "tts"
    )
    config = AppConfig.from_dict({})

    discover_model_paths(config)

    assert config.stt.model == str(stt)
    assert config.tts.model == str(tts)


def test_only_missing_model_is_downloaded_and_verified(tmp_path, mocker):
    from ollama_vox.ui.setup_wizard import SetupDownloadThread

    stt = make_model(tmp_path / "whisper", "stt")
    tts = tmp_path / "kokoro"
    tts.mkdir()  # An interrupted download must be retried.
    config = AppConfig.from_dict(
        {"stt": {"model": str(stt)}, "tts": {"model": str(tts)}}
    )
    download = mocker.patch(
        "ollama_vox.ui.setup_wizard.snapshot_download",
        side_effect=lambda **kwargs: make_model(tts, "tts"),
    )
    thread = SetupDownloadThread(config)
    finished = mocker.Mock()
    thread.finished_pull.connect(finished)

    thread.run()

    download.assert_called_once()
    assert download.call_args.kwargs["local_dir"] == str(tts)
    finished.assert_called_once_with(True, "")


def test_incomplete_download_does_not_report_success(mocker):
    from ollama_vox.ui.setup_wizard import SetupDownloadThread

    mocker.patch("ollama_vox.ui.setup_wizard.snapshot_download")
    thread = SetupDownloadThread(AppConfig.from_dict({}))
    finished = mocker.Mock()
    thread.finished_pull.connect(finished)

    thread.run()

    assert finished.call_args.args[0] is False
    assert "incomplete" in finished.call_args.args[1]


def test_folder_is_checked_before_download_permission(tmp_path, mocker):
    from ollama_vox.ui import setup_wizard

    events = []
    mocker.patch.object(
        setup_wizard.QFileDialog,
        "getExistingDirectory",
        side_effect=lambda *args: events.append("folder") or str(tmp_path / "chosen"),
    )
    buttons = setup_wizard.QMessageBox.StandardButton
    messages = mocker.patch.object(setup_wizard, "QMessageBox")
    messages.StandardButton = buttons
    messages.return_value.exec.side_effect = lambda: (
        events.append("permission") or buttons.No
    )
    download = mocker.patch.object(setup_wizard, "SetupDownloadThread")

    with pytest.raises(SystemExit):
        setup_wizard.AppSetupWizard(AppConfig.from_dict({})).run()

    assert events[:2] == ["folder", "permission"]
    download.assert_not_called()
