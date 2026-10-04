from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def app():
    from ollama_vox.ui.tray_app import VoiceTrayApp

    app = SimpleNamespace(
        _quitting=False,
        _shutdown_complete=False,
        recording=False,
        _start_pending=None,
        _ptt_held=False,
        _recording_source=None,
        _thinking_cued=False,
        status="idle",
        _cues=Mock(),
        recorder=Mock(),
        pipeline=Mock(),
        panel=Mock(),
        _refresh_menu=Mock(),
        _set_icon=Mock(),
        _render_tooltip=Mock(),
    )
    for name in (
        "start",
        "stop",
        "_begin_recording",
        "_push_to_talk_changed",
        "_auto_stop_tick",
        "_apply_status",
    ):
        setattr(app, name, MethodType(getattr(VoiceTrayApp, name), app))
    app.set_status = Mock(side_effect=app._apply_status)
    app.pipeline.enqueue_audio.return_value = True
    return app


def finish_start_cue(app):
    assert app._cues.play.call_args.args[0] == "start"
    app._cues.play.call_args.kwargs["on_finished"]()


def test_press_waits_for_cue_and_release_stops_before_playing_stop_cue(app):
    events = Mock()
    events.attach_mock(app.recorder.stop, "microphone_stopped")
    events.attach_mock(app._cues.play, "cue")
    app._push_to_talk_changed(True)
    app.recorder.start.assert_not_called()
    app.pipeline.interrupt_speaking.assert_called_once_with()
    finish_start_cue(app)
    app.recorder.start.assert_called_once_with()
    assert app.recording and app.status == "listening"

    events.reset_mock()
    app._push_to_talk_changed(False)

    assert not app.recording
    assert [call[0] for call in events.mock_calls] == [
        "microphone_stopped",
        "cue",
        "cue",
    ]
    assert [call.args[0] for call in app._cues.play.call_args_list][-2:] == [
        "stop",
        "thinking",
    ]
    assert app.status == "busy"
    app.pipeline.enqueue_audio.assert_called_once_with(app.recorder.stop.return_value)


def test_quick_release_during_start_cue_never_opens_microphone(app):
    app._push_to_talk_changed(True)
    callback = app._cues.play.call_args.kwargs["on_finished"]
    app._push_to_talk_changed(False)
    callback()
    app.recorder.start.assert_not_called()
    app.pipeline.enqueue_audio.assert_not_called()
    assert app._start_pending is None


def test_manual_recording_is_not_stopped_by_shortcut_release(app):
    app.start()
    finish_start_cue(app)
    app._push_to_talk_changed(True)
    app._push_to_talk_changed(False)
    assert app.recording
    app.recorder.stop.assert_not_called()


def test_thinking_cue_is_not_repeated_at_each_pipeline_stage(app):
    app._apply_status("busy")
    app._apply_status("idle")
    app._apply_status("busy")
    app._cues.play.assert_called_once_with("thinking")


def test_worker_status_updates_do_not_overwrite_listening_state(app):
    app.recording = True
    app.status = "listening"
    app._apply_status("idle")
    app._apply_status("busy")
    assert app.status == "listening"
    app._cues.play.assert_not_called()


def test_max_duration_can_stop_a_held_shortcut(app):
    app.recording = True
    app._recording_source = "ptt"
    app._ptt_held = True
    app.recorder.should_auto_stop.return_value = True
    app._auto_stop_tick()
    app.recorder.should_auto_stop.assert_called_once_with(allow_silence=False)
    app.recorder.stop.assert_called_once_with()


def test_release_while_quit_prompt_is_open_still_stops_recording(app):
    app.recording = True
    app._recording_source = "ptt"
    app._ptt_held = True
    app._quitting = True
    app._push_to_talk_changed(False)
    assert not app.recording
    assert not app._ptt_held


def test_shutdown_cancels_pending_start_callback(app):
    app.start()
    callback = app._cues.play.call_args.kwargs["on_finished"]
    app._shutdown_complete = True
    callback()
    app.recorder.start.assert_not_called()


def test_audio_cues_are_queued_and_can_be_disabled(mocker):
    from ollama_vox.ui.audio_cues import AudioCues

    timer = mocker.patch("ollama_vox.ui.audio_cues.QTimer")
    effects = [Mock(), Mock(), Mock()]
    mocker.patch("ollama_vox.ui.audio_cues.QSoundEffect", side_effect=effects)
    cues = AudioCues()
    ready = Mock()
    cues.play("start", on_finished=ready)
    cues.play("stop")
    effects[0].play.assert_called_once_with()
    effects[1].play.assert_not_called()
    ready.assert_not_called()
    cues._finished()
    ready.assert_called_once_with()
    effects[1].play.assert_called_once_with()
    cues.stop()
    timer.return_value.stop.assert_called_once_with()
    cues.enabled = False
    ready.reset_mock()
    cues.play("start", on_finished=ready)
    ready.assert_called_once_with()
