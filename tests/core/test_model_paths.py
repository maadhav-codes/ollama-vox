"""Model lookup and downloads must be independent of the launch directory."""

from pathlib import Path

import pytest

from ollama_vox.core.config import AppConfig, ConfigValidationError


@pytest.fixture(autouse=True)
def model_storage(tmp_path, mocker):
    documents = tmp_path / "Documents"
    mocker.patch("ollama_vox.core.config.user_documents_path", return_value=documents)
    mocker.patch(
        "ollama_vox.core.config.model_location_file",
        return_value=tmp_path / "settings.json",
    )
    mocker.patch(
        "ollama_vox.core.model_assets.model_search_roots",
        return_value=[documents / "ollama-vox" / "models"],
    )
    mocker.patch(
        "ollama_vox.core.model_assets.HF_HUB_CACHE", str(tmp_path / "hf-cache")
    )
    return documents / "ollama-vox" / "models"


def test_defaults_stay_the_same_across_launch_directories(
    tmp_path, monkeypatch, model_storage
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(first)
    initial = AppConfig.from_dict({})
    monkeypatch.chdir(second)
    subsequent = AppConfig.from_dict({})

    assert (
        initial.stt.model
        == subsequent.stt.model
        == str(model_storage / "whisper/whisper-small.en-mlx-q4")
    )
    assert (
        initial.tts.model
        == subsequent.tts.model
        == str(model_storage / "kokoro/Kokoro-82M-4bit")
    )
    assert not model_storage.exists()  # Reading config doesn't create directories.


def test_custom_relative_paths_use_user_storage(model_storage):
    config = AppConfig.from_dict(
        {"stt": {"model": "custom/whisper"}, "tts": {"model": "custom/kokoro"}}
    )

    assert config.stt.model == str(model_storage / "custom/whisper")
    assert config.tts.model == str(model_storage / "custom/kokoro")


def test_absolute_and_home_paths_are_supported(tmp_path):
    config = AppConfig.from_dict(
        {
            "stt": {"model": str(tmp_path / "whisper")},
            "tts": {"model": "~/models/kokoro"},
        }
    )

    assert config.stt.model == str(tmp_path / "whisper")
    assert config.tts.model == str(Path.home() / "models/kokoro")


def test_project_root_models_are_not_automatically_used(
    tmp_path, model_storage, monkeypatch
):
    source = tmp_path / "checkout"
    relative_paths = ["whisper/whisper-small.en-mlx-q4", "kokoro/Kokoro-82M-4bit"]
    for path in relative_paths:
        (source / path).mkdir(parents=True)
    monkeypatch.chdir(source)

    config = AppConfig.from_dict({})

    assert config.stt.model == str(model_storage / relative_paths[0])
    assert config.tts.model == str(model_storage / relative_paths[1])


def test_selected_directory_persists_across_launch_directories(tmp_path, monkeypatch):
    from ollama_vox.core.config import save_model_data_dir

    chosen = tmp_path / "Desktop" / "My Models"
    save_model_data_dir(chosen)
    initial = AppConfig.from_dict({})
    monkeypatch.chdir(tmp_path)
    later = AppConfig.from_dict({})

    assert (
        initial.stt.model
        == later.stt.model
        == str(chosen / "whisper/whisper-small.en-mlx-q4")
    )
    assert (
        initial.tts.model == later.tts.model == str(chosen / "kokoro/Kokoro-82M-4bit")
    )


@pytest.mark.parametrize("select_folder", [True, False])
def test_setup_folder_picker_remembers_choice_or_documents_default(
    tmp_path, model_storage, mocker, select_folder
):
    from ollama_vox.core.config import saved_model_data_dir
    from ollama_vox.ui.setup_wizard import AppSetupWizard

    chosen = tmp_path / "Desktop" / "Models"
    picker = mocker.patch(
        "ollama_vox.ui.setup_wizard.QFileDialog.getExistingDirectory",
        return_value=str(chosen) if select_folder else "",
    )
    config = AppConfig.from_dict({})

    AppSetupWizard(config).choose_model_directory()

    expected = chosen if select_folder else model_storage
    assert saved_model_data_dir() == expected
    assert config.stt.model == str(expected / "whisper/whisper-small.en-mlx-q4")
    assert config.tts.model == str(expected / "kokoro/Kokoro-82M-4bit")
    assert AppConfig.from_dict({}).stt.model == config.stt.model
    picker.assert_called_once()


def test_saved_choice_skips_picker(tmp_path, mocker):
    from ollama_vox.core.config import save_model_data_dir
    from ollama_vox.ui.setup_wizard import AppSetupWizard

    save_model_data_dir(tmp_path / "chosen")
    picker = mocker.patch("ollama_vox.ui.setup_wizard.QFileDialog.getExistingDirectory")

    AppSetupWizard(AppConfig.from_dict({})).choose_model_directory()

    picker.assert_not_called()


def test_folder_choice_preserves_explicit_absolute_paths(tmp_path, mocker):
    from ollama_vox.ui.setup_wizard import AppSetupWizard

    explicit = tmp_path / "existing-whisper"
    chosen = tmp_path / "chosen"
    config = AppConfig.from_dict({"stt": {"model": str(explicit)}})
    mocker.patch(
        "ollama_vox.ui.setup_wizard.QFileDialog.getExistingDirectory",
        return_value=str(chosen),
    )

    AppSetupWizard(config).choose_model_directory()

    assert config.stt.model == str(explicit)
    assert config.tts.model == str(chosen / "kokoro/Kokoro-82M-4bit")


def test_cannot_save_choice_exits_before_download(mocker):
    from ollama_vox.ui.setup_wizard import AppSetupWizard

    mocker.patch(
        "ollama_vox.ui.setup_wizard.QFileDialog.getExistingDirectory", return_value=""
    )
    mocker.patch(
        "ollama_vox.ui.setup_wizard.save_model_data_dir",
        side_effect=PermissionError("denied"),
    )
    error = mocker.patch("ollama_vox.ui.setup_wizard.QMessageBox.critical")

    with pytest.raises(SystemExit):
        AppSetupWizard(AppConfig.from_dict({})).choose_model_directory()

    error.assert_called_once()


@pytest.mark.parametrize("section", ["stt", "tts"])
@pytest.mark.parametrize("path", [None, "", "   "])
def test_empty_model_paths_fail_validation(section, path):
    with pytest.raises(ConfigValidationError, match=f"{section}.model"):
        AppConfig.from_dict({section: {"model": path}})


def test_first_run_downloads_to_stable_storage_from_elsewhere(
    tmp_path, model_storage, mocker, monkeypatch
):
    from ollama_vox.ui.setup_wizard import SetupDownloadThread

    monkeypatch.chdir(tmp_path)
    config = AppConfig.from_dict({})
    download = mocker.patch("ollama_vox.ui.setup_wizard.snapshot_download")

    SetupDownloadThread(config).run()

    assert [Path(entry.kwargs["local_dir"]) for entry in download.call_args_list] == [
        model_storage / "whisper/whisper-small.en-mlx-q4",
        model_storage / "kokoro/Kokoro-82M-4bit",
    ]
    assert not (tmp_path / "whisper").exists()
    assert not (tmp_path / "kokoro").exists()


@pytest.mark.parametrize("failure", ["missing", "invalid"])
def test_tts_fallback_also_uses_stable_storage(model_storage, mocker, failure):
    from ollama_vox.core.tts import TTS

    if failure == "missing":
        mocker.patch("builtins.open", side_effect=FileNotFoundError())
    else:
        mocker.patch(
            "builtins.open", mocker.mock_open(read_data="tts:\n  model: null\n")
        )

    assert TTS().model_id == str(model_storage / "kokoro/Kokoro-82M-4bit")
