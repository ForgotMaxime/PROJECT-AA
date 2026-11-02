"""Mouse movement backends.

- `win32`: legacy `win32api.mouse_event` (flagged by some anti-cheats, e.g. EQU8).
- `sendinput`: `user32.SendInput` relative move. Still user-mode (not a driver),
  but avoids the win32 mouse_event hook path. Default for `main_v8.py`.
"""
import ctypes

MOUSEEVENTF_MOVE = 0x0001


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("mi", _MOUSEINPUT)]


def move_sendinput(dx, dy):
    inp = _INPUT()
    inp.type = 0  # INPUT_MOUSE
    inp.mi.dx = int(dx)
    inp.mi.dy = int(dy)
    inp.mi.mouseData = 0
    inp.mi.dwFlags = MOUSEEVENTF_MOVE
    inp.mi.time = 0
    inp.mi.dwExtraInfo = None
    n = ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    if n != 1:
        raise OSError(f"SendInput moved {n}/1 inputs")


def move_win32(dx, dy):
    import win32api
    import win32con
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)


def move_relative(dx, dy, backend="sendinput"):
    b = str(backend).lower()
    if b == "sendinput":
        return move_sendinput(dx, dy)
    if b in ("win32", "legacy"):
        return move_win32(dx, dy)
    raise ValueError(f'unknown mouse backend: {backend!r} (use "sendinput" or "win32")')
