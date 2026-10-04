import ctypes as C
from unittest.mock import Mock

import pytest

from ollama_vox.core.hotkey import (
    PushToTalkHotkey,
    _HotKeyID,
    _SIGNATURE,
    _parse_shortcut,
)


@pytest.fixture
def carbon(mocker):
    native = Mock()

    def install(*args):
        C.cast(args[-1], C.POINTER(C.c_void_p))[0] = 10
        return 0

    def register(*args):
        C.cast(args[-1], C.POINTER(C.c_void_p))[0] = 20
        return 0

    def parameter(*args):
        C.cast(args[-1], C.POINTER(_HotKeyID))[0] = _HotKeyID(_SIGNATURE, 1)
        return 0

    native.InstallEventHandler.side_effect = install
    native.RegisterEventHotKey.side_effect = register
    native.GetEventParameter.side_effect = parameter
    mocker.patch("ollama_vox.core.hotkey._load_carbon", return_value=native)
    return native


def test_exact_shortcut_registration_and_repeat_release(carbon):
    changed = Mock()
    hotkey = PushToTalkHotkey("<ctrl>+<alt>+<space>", changed)
    hotkey.start()
    hotkey.start()
    args = carbon.RegisterEventHotKey.call_args.args
    assert args[:2] == (49, (1 << 12) | (1 << 11))
    carbon.RegisterEventHotKey.assert_called_once()
    for kind in [5, 5, 6, 6]:
        carbon.GetEventKind.return_value = kind
        assert hotkey._callback(None, None, None) == 0
    assert [call.args[0] for call in changed.call_args_list] == [True, False]
    hotkey.stop()
    hotkey.stop()
    carbon.UnregisterEventHotKey.assert_called_once()
    carbon.RemoveEventHandler.assert_called_once()


def test_other_shortcuts_and_invalid_events_are_not_handled(carbon):
    changed = Mock()
    hotkey = PushToTalkHotkey("<ctrl>+<space>", changed)
    hotkey.start()

    def other_identity(*args):
        C.cast(args[-1], C.POINTER(_HotKeyID))[0] = _HotKeyID(_SIGNATURE, 99)
        return 0

    carbon.GetEventParameter.side_effect = other_identity
    assert hotkey._callback(None, None, None) == -9874
    carbon.GetEventParameter.side_effect = None
    carbon.GetEventParameter.return_value = -1
    assert hotkey._callback(None, None, None) == -9874
    changed.assert_not_called()


def test_registration_failure_cleans_up_handler(carbon):
    carbon.RegisterEventHotKey.side_effect = None
    carbon.RegisterEventHotKey.return_value = -9878
    hotkey = PushToTalkHotkey("<ctrl>+<space>", Mock())
    with pytest.raises(RuntimeError, match="Choose another"):
        hotkey.start()
    carbon.RemoveEventHandler.assert_called_once()
    assert not hotkey._handler.value
    assert not hotkey._hotkey.value


def test_disabling_hotkey_releases_active_recording(carbon):
    changed = Mock()
    hotkey = PushToTalkHotkey("<ctrl>+<space>", changed)
    hotkey.start()
    carbon.GetEventKind.return_value = 5
    hotkey._callback(None, None, None)
    hotkey.stop()
    assert [call.args[0] for call in changed.call_args_list] == [True, False]


@pytest.mark.parametrize("shortcut", ["space", "<alt>+a", "<ctrl>+a+b", "<ctrl>+<fn>"])
def test_invalid_shortcuts_never_register(shortcut, carbon):
    with pytest.raises(ValueError):
        PushToTalkHotkey(shortcut, Mock()).start()
    carbon.RegisterEventHotKey.assert_not_called()


def test_command_alias_and_function_key():
    assert _parse_shortcut("<command>+<shift>+<f18>") == (79, (1 << 8) | (1 << 9))


def test_callback_exception_does_not_escape_native_boundary(carbon):
    hotkey = PushToTalkHotkey("<ctrl>+<space>", Mock(side_effect=RuntimeError))
    hotkey.start()
    carbon.GetEventKind.return_value = 5
    assert hotkey._callback(None, None, None) == 0
