"""Local Ollama startup requires permission and has a bounded readiness wait."""

import subprocess

import pytest
import requests

from ollama_vox.core.config import AppConfig


@pytest.fixture
def wizard(mocker):
    from ollama_vox.ui import model_setup

    mocker.patch.object(model_setup, "_started_servers", [])
    config = AppConfig.from_dict({})
    instance = model_setup.OllamaModelWizard(config)
    buttons = model_setup.QMessageBox.StandardButton
    icons = model_setup.QMessageBox.Icon
    messages = mocker.patch.object(model_setup, "QMessageBox")
    messages.StandardButton = buttons
    messages.Icon = icons
    messages.return_value.exec.return_value = buttons.Yes
    mocker.patch.object(model_setup, "QProgressDialog")
    mocker.patch.object(model_setup, "QEventLoop")
    return model_setup, instance, messages


def test_running_server_needs_no_prompt_or_process(wizard, mocker):
    module, instance, messages = wizard
    get = mocker.patch.object(module.requests, "get")
    get.return_value.json.return_value = {
        "models": [{"name": instance.config.ollama.model}]
    }
    launch = mocker.patch.object(module.subprocess, "Popen")

    assert instance.run() is True

    messages.assert_not_called()
    launch.assert_not_called()


@pytest.mark.parametrize("answer", ["No", "Cancel"])
def test_declining_or_closing_prompt_never_starts_server(wizard, mocker, answer):
    module, instance, messages = wizard
    get = mocker.patch.object(
        module.requests, "get", side_effect=requests.ConnectionError("offline")
    )
    messages.return_value.exec.return_value = getattr(messages.StandardButton, answer)
    thread = mocker.patch.object(module, "OllamaStartThread")

    assert instance.run() is False

    get.assert_called_once()
    messages.return_value.exec.assert_called_once()
    thread.assert_not_called()


@pytest.mark.parametrize("success", [True, False])
def test_approved_start_waits_then_continues_or_reports_failure(
    wizard, mocker, success
):
    module, instance, messages = wizard
    response = mocker.Mock()
    response.json.return_value = {"models": [{"name": instance.config.ollama.model}]}
    get = mocker.patch.object(
        module.requests, "get", side_effect=[requests.ConnectionError(), response]
    )
    thread = mocker.patch.object(module, "OllamaStartThread").return_value

    def finish():
        callback = thread.finished_start.connect.call_args.args[0]
        callback(success, "startup failed" if not success else "")

    module.QEventLoop.return_value.exec.side_effect = finish

    assert instance.run() is success

    thread.start.assert_called_once_with()
    thread.wait.assert_called_once_with()
    module.QProgressDialog.return_value.close.assert_called_once_with()
    assert get.call_count == (2 if success else 1)
    assert messages.return_value.exec.call_count == (1 if success else 2)


@pytest.mark.parametrize(
    "endpoint",
    ["http://remote:11434", "https://localhost:11434", "http://localhost/api"],
)
def test_nonlocal_or_unsupported_endpoint_never_launches(wizard, mocker, endpoint):
    module, instance, messages = wizard
    instance.endpoint = endpoint
    get = mocker.patch.object(
        module.requests, "get", side_effect=requests.ConnectionError()
    )
    thread = mocker.patch.object(module, "OllamaStartThread")

    assert instance.run() is False

    thread.assert_not_called()
    get.assert_called_once()
    messages.return_value.setWindowTitle.assert_called_with("Ollama Unavailable")


def test_http_error_is_reported_without_starting_another_server(wizard, mocker):
    module, instance, messages = wizard
    get = mocker.patch.object(module.requests, "get")
    get.return_value.raise_for_status.side_effect = requests.HTTPError("unauthorized")
    thread = mocker.patch.object(module, "OllamaStartThread")

    assert instance.run() is False

    thread.assert_not_called()
    messages.return_value.exec.assert_called_once()


@pytest.fixture
def server_start(mocker):
    from ollama_vox.ui import model_setup

    mocker.patch.object(model_setup, "_started_servers", [])
    thread = model_setup.OllamaStartThread("http://127.0.0.1:11500")
    finished = mocker.Mock()
    thread.finished_start.connect(finished)
    mocker.patch.object(model_setup.shutil, "which", return_value="/usr/bin/ollama")
    launch = mocker.patch.object(model_setup.subprocess, "Popen")
    launch.return_value.poll.return_value = None
    get = mocker.patch.object(model_setup.requests, "get")
    mocker.patch.object(thread, "msleep")
    return model_setup, thread, launch, get, finished


def test_launch_uses_configured_host_and_waits_for_api(server_start, mocker):
    module, thread, launch, get, finished = server_start
    mocker.patch.dict(module.os.environ, {"OLLAMA_HOST": "http://remote:9999"})
    get.side_effect = [requests.ConnectionError(), mocker.Mock()]

    thread.run()

    finished.assert_called_once_with(True, "")
    assert launch.call_args.args == (["/usr/bin/ollama", "serve"],)
    assert launch.call_args.kwargs["env"]["OLLAMA_HOST"] == thread.endpoint
    assert launch.call_args.kwargs["start_new_session"] is True
    assert get.call_count == 2
    launch.return_value.terminate.assert_not_called()
    assert module.running_started_servers() == [(thread.endpoint, launch.return_value)]


@pytest.mark.parametrize(
    "failure", ["not_installed", "spawn_error", "exited", "timeout"]
)
def test_start_failure_is_reported_and_own_process_cleaned_up(
    server_start, mocker, failure
):
    module, thread, launch, get, finished = server_start
    get.side_effect = requests.ConnectionError()
    if failure == "not_installed":
        module.shutil.which.return_value = None
    elif failure == "spawn_error":
        launch.side_effect = OSError("cannot execute")
    elif failure == "exited":
        launch.return_value.poll.return_value = 1
        launch.return_value.returncode = 1
    else:
        mocker.patch.object(module.time, "monotonic", side_effect=[0, 0, 16])

    thread.run()

    finished.assert_called_once()
    assert finished.call_args.args[0] is False
    assert finished.call_args.args[1]
    assert module.running_started_servers() == []
    if failure == "not_installed":
        launch.assert_not_called()
        assert "Install" in finished.call_args.args[1]
    elif failure == "timeout":
        launch.return_value.terminate.assert_called_once_with()
        launch.return_value.wait.assert_called_once_with(timeout=2)


def test_timed_out_process_is_killed_if_it_will_not_terminate(server_start, mocker):
    module, thread, launch, get, finished = server_start
    get.side_effect = requests.ConnectionError()
    mocker.patch.object(module.time, "monotonic", side_effect=[0, 16])
    launch.return_value.wait.side_effect = [subprocess.TimeoutExpired("ollama", 2), 0]

    thread.run()

    launch.return_value.kill.assert_called_once_with()
    finished.assert_called_once_with(
        False, "Ollama did not become ready within 15 seconds."
    )
