"""Startup must complete model setup before launching the voice pipeline."""

import sys
from unittest.mock import Mock, call

import pytest

from ollama_vox.core.config import AppConfig


@pytest.fixture
def startup(mocker):
    # Skip the optional espeak compatibility patch; it is unrelated to startup
    # ordering and importing it after Qt can trigger third-party import hooks.
    mocker.patch.dict(sys.modules, {"dateutil.parser": None})
    # Import after the hardware dependency fixtures have been applied.
    from ollama_vox import main

    config = AppConfig.from_dict({})
    mocker.patch.object(main, "load_config", return_value=config)
    mocker.patch.object(main, "configure_logging")
    mocker.patch("PySide6.QtWidgets.QApplication")
    setup = mocker.patch("ollama_vox.ui.setup_wizard.AppSetupWizard")
    setup.return_value.run.return_value = True
    ollama = mocker.patch.object(main, "OllamaModelWizard")
    ollama.return_value.run.return_value = True
    health = mocker.patch.object(main, "run_startup_health_checks")
    recorder = mocker.patch.object(main, "AudioRecorder")
    mocker.patch.object(main, "STT")
    mocker.patch.object(main, "TTS")
    mocker.patch.object(main, "OllamaClient")
    pipeline = mocker.patch.object(main, "Pipeline")
    app = mocker.patch.object(main, "VoiceApp")

    events = Mock()
    for name, mock in (
        ("setup", setup),
        ("ollama", ollama),
        ("health", health),
        ("recorder", recorder),
        ("pipeline", pipeline),
        ("app", app),
    ):
        events.attach_mock(mock, name)

    return main, config, events


def test_normal_startup_completes_setup_before_launch(startup, monkeypatch):
    main, config, events = startup
    monkeypatch.setattr(sys, "argv", ["ollama-vox"])

    main.main()

    # Missing models are handled by the setup wizard before health checks
    # or any component that could try to use them.
    assert events.mock_calls[:5] == [
        call.setup(config),
        call.setup().run(force_setup=False),
        call.ollama(config),
        call.ollama().run(),
        call.health(config),
    ]
    events.pipeline.return_value.start.assert_called_once_with()
    events.app.return_value.run.assert_called_once_with()


@pytest.mark.parametrize("failure", [False, SystemExit(1)])
@pytest.mark.parametrize("setup_only", [False, True])
def test_unsuccessful_setup_blocks_launch(startup, monkeypatch, failure, setup_only):
    main, _, events = startup
    monkeypatch.setattr(
        sys, "argv", ["ollama-vox"] + (["--setup"] if setup_only else [])
    )
    if isinstance(failure, SystemExit):
        events.setup.return_value.run.side_effect = failure
    else:
        events.setup.return_value.run.return_value = failure

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 1
    events.ollama.assert_not_called()
    events.health.assert_not_called()
    events.recorder.assert_not_called()
    events.pipeline.assert_not_called()
    events.app.assert_not_called()


def test_setup_only_forces_wizards_and_exits_without_launch(startup, monkeypatch):
    main, config, events = startup
    monkeypatch.setattr(sys, "argv", ["ollama-vox", "--setup"])

    main.main()

    assert events.mock_calls == [
        call.setup(config),
        call.setup().run(force_setup=True),
        call.ollama(config),
        call.ollama().run(force_setup=True),
    ]


def test_existing_models_skip_downloads_and_dialogs(tmp_path, mocker):
    from ollama_vox.ui.setup_wizard import AppSetupWizard

    stt = tmp_path / "whisper"
    tts = tmp_path / "kokoro"
    stt.mkdir()
    tts.mkdir()
    (stt / "config.json").write_text("{}")
    (stt / "weights.npz").write_bytes(b"weights")
    (tts / "config.json").write_text("{}")
    (tts / "model.safetensors").write_bytes(b"weights")
    (tts / "voices").mkdir()
    (tts / "voices/af_bella.pt").write_bytes(b"voice")
    config = AppConfig.from_dict(
        {"stt": {"model": str(stt)}, "tts": {"model": str(tts)}}
    )
    dialog = mocker.patch("ollama_vox.ui.setup_wizard.QMessageBox")
    download = mocker.patch("ollama_vox.ui.setup_wizard.SetupDownloadThread")

    assert AppSetupWizard(config).run() is True

    dialog.assert_not_called()
    download.assert_not_called()
