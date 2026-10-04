"""Quit confirmation stops local servers before exiting the application."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def tray(mocker):
    from ollama_vox.ui import tray_app

    app = SimpleNamespace(
        _quitting=False,
        _shutdown_complete=False,
        recording=True,
        _auto_stop_t=Mock(),
        _pump=Mock(),
        recorder=Mock(),
        pipeline=Mock(),
        hide=Mock(),
        qt_app=Mock(),
    )
    app.quit = MethodType(tray_app.VoiceTrayApp.quit, app)
    app.pipeline.llm.endpoint = "http://localhost:11434"
    app._confirm_server_shutdown = MethodType(
        tray_app.VoiceTrayApp._confirm_server_shutdown, app
    )
    buttons = tray_app.QMessageBox.StandardButton
    icons = tray_app.QMessageBox.Icon
    messages = mocker.patch.object(tray_app, "QMessageBox")
    messages.StandardButton = buttons
    messages.Icon = icons
    messages.return_value.exec.return_value = buttons.Yes
    mocker.patch.object(tray_app, "QProgressDialog")
    mocker.patch.object(tray_app, "QEventLoop")
    servers = mocker.patch.object(
        tray_app,
        "running_started_servers",
        return_value=[("http://localhost:11434", Mock())],
    )
    thread = mocker.patch.object(tray_app, "OllamaStopThread")

    def finish():
        callback = thread.return_value.finished_stop.connect.call_args.args[0]
        callback(True, "")

    tray_app.QEventLoop.return_value.exec.side_effect = finish
    return tray_app, app, messages, servers, thread


def test_yes_waits_for_server_shutdown_before_app_exit(tray):
    module, app, messages, servers, thread = tray
    events = Mock()
    events.attach_mock(thread.return_value.wait, "server_stopped")
    events.attach_mock(app.pipeline.stop, "pipeline_stopped")
    events.attach_mock(app.qt_app.quit, "app_exited")

    app.quit()

    thread.assert_called_once_with(
        servers.return_value, endpoint="http://localhost:11434"
    )
    assert [entry[0] for entry in events.mock_calls] == [
        "server_stopped",
        "pipeline_stopped",
        "app_exited",
    ]
    app.recorder.stop.assert_called_once_with()
    module.QProgressDialog.return_value.close.assert_called_once_with()
    assert app._shutdown_complete is True
    app.quit()
    messages.return_value.exec.assert_called_once()
    app.qt_app.quit.assert_called_once_with()


def test_no_leaves_server_running_and_exits(tray):
    _, app, messages, _, thread = tray
    messages.return_value.exec.return_value = messages.StandardButton.No

    app.quit()

    thread.assert_not_called()
    app.pipeline.stop.assert_called_once_with()
    app.qt_app.quit.assert_called_once_with()


def test_cancel_keeps_app_and_server_running(tray):
    _, app, messages, _, thread = tray
    messages.return_value.exec.return_value = messages.StandardButton.Cancel

    app.quit()

    thread.assert_not_called()
    app._auto_stop_t.stop.assert_not_called()
    app.recorder.stop.assert_not_called()
    app.pipeline.stop.assert_not_called()
    app.qt_app.quit.assert_not_called()
    assert app._quitting is False
    assert app._shutdown_complete is False


def test_already_running_local_server_still_prompts(tray):
    _, app, messages, servers, thread = tray
    servers.return_value = []

    app.quit()

    messages.return_value.exec.assert_called_once()
    thread.assert_called_once_with([], endpoint="http://localhost:11434")
    app.qt_app.quit.assert_called_once_with()


def test_remote_server_exits_without_stop_prompt(tray):
    _, app, messages, servers, thread = tray
    servers.return_value = []
    app.pipeline.llm.endpoint = "http://remote:11434"

    app.quit()

    messages.assert_not_called()
    thread.assert_not_called()
    app.qt_app.quit.assert_called_once_with()


def test_shutdown_failure_keeps_app_open(tray):
    module, app, messages, _, thread = tray

    def fail():
        callback = thread.return_value.finished_stop.connect.call_args.args[0]
        callback(False, "permission denied")

    module.QEventLoop.return_value.exec.side_effect = fail

    app.quit()

    app.pipeline.stop.assert_not_called()
    app.qt_app.quit.assert_not_called()
    messages.return_value.setInformativeText.assert_called_with("permission denied")
    assert app._quitting is False


def test_nested_quit_does_not_open_another_prompt(tray):
    _, app, messages, _, thread = tray
    app._quitting = True

    app.quit()

    messages.assert_not_called()
    thread.assert_not_called()
    app.qt_app.quit.assert_not_called()


def test_native_quit_routes_through_confirmation(tray, mocker):
    module, app, _, _, _ = tray
    timer = mocker.patch.object(module.QTimer, "singleShot")
    event = module.QEvent(module.QEvent.Type.Quit)

    assert module.VoiceTrayApp.eventFilter(app, app.qt_app, event) is True

    timer.assert_called_once_with(0, app.quit)


def test_server_registry_prunes_exited_processes(mocker):
    from ollama_vox.ui import model_setup

    live, exited = Mock(), Mock()
    live.poll.return_value = None
    exited.poll.return_value = 0
    mocker.patch.object(
        model_setup, "_started_servers", [("local", live), ("old", exited)]
    )

    assert model_setup.running_started_servers() == [("local", live)]
    assert model_setup._started_servers == [("local", live)]


def test_stop_thread_attempts_all_owned_servers_and_reports_errors():
    from ollama_vox.ui.model_setup import OllamaStopThread

    first, second = Mock(), Mock()
    first.poll.return_value = second.poll.return_value = None
    first.terminate.side_effect = OSError("permission denied")
    thread = OllamaStopThread([("first", first), ("second", second)])
    finished = Mock()
    thread.finished_stop.connect(finished)

    thread.run()

    second.terminate.assert_called_once_with()
    second.wait.assert_called_once_with(timeout=2)
    finished.assert_called_once_with(False, "first: permission denied")


@pytest.fixture
def existing_server(mocker):
    from ollama_vox.ui import model_setup

    inspect = mocker.patch.object(model_setup.subprocess, "run")
    inspect.return_value.returncode = 0
    inspect.return_value.stdout = "123\n"
    inspect.return_value.stderr = ""
    identity = mocker.patch.object(model_setup, "_process_identity")
    identity.return_value = "Sun Oct 4 17:55:10 2026 /usr/local/bin/ollama serve"
    kill = mocker.patch.object(model_setup.os, "kill")
    mocker.patch.object(model_setup.time, "sleep")
    return model_setup, inspect, identity, kill


def test_stop_preexisting_server_targets_configured_port_and_verified_pid(
    existing_server,
):
    module, inspect, identity, kill = existing_server
    marker = identity.return_value
    identity.side_effect = [marker, marker, None]
    inspect.side_effect = [
        inspect.return_value,
        Mock(returncode=1, stdout="", stderr=""),
    ]

    module._stop_existing_local_server("http://localhost:11500")

    assert "-iTCP:11500" in inspect.call_args.args[0]
    kill.assert_called_once_with(123, module.signal.SIGTERM)


def test_unrelated_listener_is_never_stopped(existing_server):
    module, _, identity, kill = existing_server
    identity.return_value = "Sun Oct 4 17:55:10 2026 /usr/bin/python server.py"

    with pytest.raises(OSError, match="not a verified"):
        module._stop_existing_local_server("http://localhost:11434")

    kill.assert_not_called()


def test_reused_pid_is_never_stopped(existing_server):
    module, inspect, identity, kill = existing_server
    identity.side_effect = [identity.return_value, "a different process"]
    inspect.side_effect = [
        inspect.return_value,
        Mock(returncode=1, stdout="", stderr=""),
    ]

    module._stop_existing_local_server("http://localhost:11434")

    kill.assert_not_called()


def test_stopped_server_needs_no_signal(existing_server):
    module, inspect, _, kill = existing_server
    inspect.return_value.returncode = 1
    inspect.return_value.stdout = ""

    module._stop_existing_local_server("http://localhost:11434")

    kill.assert_not_called()


def test_external_server_shutdown_escalates_after_timeout(existing_server, mocker):
    module, inspect, identity, kill = existing_server
    marker = identity.return_value
    identity.side_effect = [marker, marker, marker, None]
    inspect.side_effect = [
        inspect.return_value,
        Mock(returncode=1, stdout="", stderr=""),
    ]
    mocker.patch.object(module.time, "monotonic", side_effect=[0, 3, 3, 3])

    module._stop_existing_local_server("http://localhost:11434")

    assert [entry.args for entry in kill.call_args_list] == [
        (123, module.signal.SIGTERM),
        (123, module.signal.SIGKILL),
    ]


def test_stop_thread_discovers_server_left_running_from_an_earlier_session(mocker):
    from ollama_vox.ui import model_setup

    stop = mocker.patch.object(model_setup, "_stop_existing_local_server")
    thread = model_setup.OllamaStopThread([], endpoint="http://localhost:11434")
    finished = Mock()
    thread.finished_stop.connect(finished)

    thread.run()

    stop.assert_called_once_with("http://localhost:11434")
    finished.assert_called_once_with(True, "")


def test_restarted_server_is_reported_instead_of_claiming_shutdown(existing_server):
    module, _, identity, kill = existing_server
    marker = identity.return_value
    identity.side_effect = [marker, marker, None]

    with pytest.raises(OSError, match="still listening"):
        module._stop_existing_local_server("http://localhost:11434")

    kill.assert_called_once_with(123, module.signal.SIGTERM)
