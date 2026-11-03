"""Mouse movement backends.

- `win32`: legacy `win32api.mouse_event` (flagged by some anti-cheats, e.g. EQU8).
- `sendinput`: `user32.SendInput` relative move. Still user-mode (not a driver),
  but avoids the win32 mouse_event hook path. Default for `main_v8.py`.
- `makcu`: MAKCU USB HID bridge over serial (`km.move(dx,dy)\\r`). The device
  shows up as real hardware mouse input. Needs `aaMakcuPort`/`aaMakcuBaud`.
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


# Cached MAKCU serial handles keyed by (port, baud).
_MAKCU_CONNS = {}


def makcu_connect(port="COM3", baud=115200):
    """Open (or reuse) the MAKCU serial handle. Raises OSError on failure."""
    key = (str(port), int(baud))
    ser = _MAKCU_CONNS.get(key)
    if ser is not None and getattr(ser, "is_open", False):
        return ser
    try:
        import serial
        ser = serial.serial_for_url(str(port), baudrate=int(baud),
                                    timeout=0, write_timeout=1)
        ser.reset_input_buffer()
    except Exception as e:
        raise OSError(
            f"MAKCU: cannot open {port} @ {baud} ({e}). "
            "Check CH343 driver, COM port number (<10), baud toggle on the "
            "device (115200 vs 4M), and close the AIO/test tool first."
        ) from e
    _MAKCU_CONNS[key] = ser
    return ser


def close_makcu(port=None):
    """Close cached MAKCU handle(s)."""
    for key in [k for k in _MAKCU_CONNS if port is None or k[0] == str(port)]:
        try:
            _MAKCU_CONNS.pop(key).close()
        except Exception:
            pass


def move_makcu(dx, dy, port="COM3", baud=115200, segments=1, bezier=None):
    """Relative move via MAKCU (`km.move(dx,dy[seg,cx1,cy1,cx2,cy2])\\r`).

    Zero moves are skipped. segments=1 with no bezier sends the legacy
    single-segment command (byte-identical). bezier=(cx1,cy1,cx2,cy2) adds
    a cubic curve, paced device-side over `segments` USB reports.
    """
    dx, dy = int(dx), int(dy)
    if dx == 0 and dy == 0:
        return
    seg = max(1, int(segments or 1))
    if bezier is not None and seg > 1:
        cx1, cy1, cx2, cy2 = (int(v) for v in bezier)
        cmd = f"km.move({dx},{dy},{seg},{cx1},{cy1},{cx2},{cy2})\r"
    elif seg > 1:
        cmd = f"km.move({dx},{dy},{seg})\r"
    else:
        cmd = f"km.move({dx},{dy})\r"
    ser = makcu_connect(port, baud)
    try:
        ser.write(cmd.encode("ascii"))
    except Exception as e:
        raise OSError(f"MAKCU: write failed on {port} ({e})") from e


def move_relative(dx, dy, backend="sendinput", port=None, baud=None,
                  segments=1, bezier=None):
    b = str(backend).lower()
    if b == "sendinput":
        return move_sendinput(dx, dy)
    if b in ("win32", "legacy"):
        return move_win32(dx, dy)
    if b == "makcu":
        return move_makcu(dx, dy, port=port or "COM3", baud=baud or 115200,
                          segments=segments, bezier=bezier)
    raise ValueError(f'unknown mouse backend: {backend!r} (use "sendinput", "win32" or "makcu")')
