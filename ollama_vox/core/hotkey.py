"""Register one macOS shortcut, without listening to the keyboard globally.

Carbon delivers shortcut press/release events on the application's Qt event loop.
No event tap, Accessibility permission, or Input Monitoring permission is used.
"""

import ctypes as C
import logging
import sys


class _EventType(C.Structure):
    _fields_ = [("event_class", C.c_uint32), ("kind", C.c_uint32)]


class _HotKeyID(C.Structure):
    _fields_ = [("signature", C.c_uint32), ("id", C.c_uint32)]


_CALLBACK = C.CFUNCTYPE(C.c_int32, C.c_void_p, C.c_void_p, C.c_void_p)
_KEYBOARD = int.from_bytes(b"keyb", "big")
_SIGNATURE = int.from_bytes(b"OVox", "big")
_MODIFIERS = {
    "ctrl": 1 << 12,
    "control": 1 << 12,
    "alt": 1 << 11,
    "option": 1 << 11,
    "cmd": 1 << 8,
    "command": 1 << 8,
    "shift": 1 << 9,
}
# Physical macOS virtual key codes. Letter shortcuts use US key positions.
_KEYS = dict(zip("asdfhgzxcv", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
_KEYS.update(dict(zip("bqwerty", [11, 12, 13, 14, 15, 16, 17])))
_KEYS.update(dict(zip("123465=97-80", range(18, 30))))
_KEYS.update(dict(zip("ouip", [31, 32, 34, 35])))
_KEYS.update(dict(zip("ljknm", [37, 38, 40, 45, 46])))
_KEYS.update(
    {
        "space": 49,
        "tab": 48,
        "enter": 36,
        "return": 36,
        "esc": 53,
        "escape": 53,
        "backspace": 51,
        "left": 123,
        "right": 124,
        "down": 125,
        "up": 126,
    }
)
_KEYS.update(
    dict(
        zip(
            (f"f{i}" for i in range(1, 21)),
            [
                122,
                120,
                99,
                118,
                96,
                97,
                98,
                100,
                101,
                109,
                103,
                111,
                105,
                107,
                113,
                106,
                64,
                79,
                80,
                90,
            ],
        )
    )
)


def _parse_shortcut(shortcut: str) -> tuple[int, int]:
    modifiers = 0
    keys = []
    for part in shortcut.lower().split("+"):
        key = part.strip().removeprefix("<").removesuffix(">")
        if key in _MODIFIERS:
            modifiers |= _MODIFIERS[key]
        elif key in _KEYS:
            keys.append(_KEYS[key])
        else:
            raise ValueError(f"Unsupported push-to-talk key: {part!r}")
    if len(keys) != 1 or not modifiers & ((1 << 8) | (1 << 12)):
        raise ValueError(
            "Push-to-talk requires Control or Command and exactly one key."
        )
    return keys[0], modifiers


def _load_carbon():
    if sys.platform != "darwin":
        raise RuntimeError("Global push-to-talk is supported on macOS only.")
    carbon = C.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
    signatures = {
        "GetApplicationEventTarget": (C.c_void_p, []),
        "InstallEventHandler": (
            C.c_int32,
            [
                C.c_void_p,
                _CALLBACK,
                C.c_uint32,
                C.POINTER(_EventType),
                C.c_void_p,
                C.POINTER(C.c_void_p),
            ],
        ),
        "RegisterEventHotKey": (
            C.c_int32,
            [
                C.c_uint32,
                C.c_uint32,
                _HotKeyID,
                C.c_void_p,
                C.c_uint32,
                C.POINTER(C.c_void_p),
            ],
        ),
        "UnregisterEventHotKey": (C.c_int32, [C.c_void_p]),
        "RemoveEventHandler": (C.c_int32, [C.c_void_p]),
        "GetEventKind": (C.c_uint32, [C.c_void_p]),
        "GetEventParameter": (
            C.c_int32,
            [
                C.c_void_p,
                C.c_uint32,
                C.c_uint32,
                C.c_void_p,
                C.c_uint32,
                C.c_void_p,
                C.c_void_p,
            ],
        ),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(carbon, name)
        function.restype = result
        function.argtypes = arguments
    return carbon


class PushToTalkHotkey:
    def __init__(self, shortcut: str, on_change):
        self.shortcut = shortcut
        self.on_change = on_change
        self.active = False
        self._carbon = None
        self._handler = C.c_void_p()
        self._hotkey = C.c_void_p()
        # Keep the C callback alive until its native event handler is removed.
        self._callback = _CALLBACK(self._event)

    def start(self) -> None:
        if self._hotkey.value:
            return
        keycode, modifiers = _parse_shortcut(self.shortcut)
        self._carbon = _load_carbon()
        target = self._carbon.GetApplicationEventTarget()
        events = (_EventType * 2)(_EventType(_KEYBOARD, 5), _EventType(_KEYBOARD, 6))
        status = self._carbon.InstallEventHandler(
            target, self._callback, 2, events, None, C.byref(self._handler)
        )
        if status:
            raise RuntimeError(
                f"Could not install the shortcut handler (macOS {status})."
            )
        status = self._carbon.RegisterEventHotKey(
            keycode,
            modifiers,
            _HotKeyID(_SIGNATURE, 1),
            target,
            1,  # kEventHotKeyExclusive: report conflicts rather than sharing a chord.
            C.byref(self._hotkey),
        )
        if status:
            self.stop()
            raise RuntimeError(
                f"Could not register {self.shortcut} (macOS {status}). "
                "Choose another interaction.push_to_talk_hotkey in config.yaml; "
                "another app may already use this shortcut."
            )

    def _event(self, _next_handler, event, _user_data) -> int:
        identity = _HotKeyID()
        status = self._carbon.GetEventParameter(
            event,
            int.from_bytes(b"----", "big"),
            int.from_bytes(b"hkid", "big"),
            None,
            C.sizeof(identity),
            None,
            C.byref(identity),
        )
        if status or (identity.signature, identity.id) != (_SIGNATURE, 1):
            return -9874  # eventNotHandledErr: let other shortcut handlers run.
        kind = self._carbon.GetEventKind(event)
        if kind not in (5, 6):
            return -9874
        held = kind == 5
        if held != self.active:
            self.active = held
            try:
                self.on_change(held)
            except Exception:
                logging.exception("Push-to-talk callback failed")
        return 0

    def stop(self) -> None:
        if self._hotkey.value:
            self._carbon.UnregisterEventHotKey(self._hotkey)
            self._hotkey = C.c_void_p()
        if self._handler.value:
            self._carbon.RemoveEventHandler(self._handler)
            self._handler = C.c_void_p()
        if self.active:
            self.active = False
            self.on_change(False)
