"""Live DearPyGui tuning panel for main_v8.py.

Pumped inside the aim loop (same thread, no locks): each tick calls
frame() which renders one DPG frame. Slider moves write straight into
the `config` module, which the loop re-reads every tick.
Headless-testable: build() without show_viewport().
"""
import time

import dearpygui.dearpygui as dpg
import win32api

import config

# (tag, label, config attr, min, max)
SLIDERS = [
    ("g_amp", "amp", "aaMovementAmp", 0.1, 1.5),
    ("g_smooth", "smooth", "aaSmooth", 1.0, 4.0),
    ("g_maxstep", "max step", "aaMaxStep", 10.0, 120.0),
    ("g_conf", "conf", "v8Conf", 0.1, 0.6),
    ("g_iou", "iou", "v8Iou", 0.3, 0.8),
    ("g_lead", "lead s", "aaLeadT", 0.0, 0.1),
    ("g_maxbox", "max box h", "aaMaxBoxH", 0.0, 300.0),
    ("g_recover", "recover px", "aaRecoverPx", 0.0, 150.0),
    ("g_sticky", "sticky px", "aaStickyPx", 0.0, 150.0),
    ("g_dzenter", "dz enter", "aaDzEnter", 0.0, 10.0),
    ("g_jitter", "jitter", "aaJitter", 0.0, 5.0),
    ("g_dz", "deadzone", "aaDeadzone", 0.0, 5.0),
]
# Note: shift/crop stay static (capture region is built once at startup).

# (tag, label, config attr)
CHECKS = [
    ("g_predict", "predict (Kalman+lead)", "aaPredict"),
    ("g_mono", "mono target", "aaMonoTarget"),
    ("g_mask", "use mask", "useMask"),
    ("g_visuals", "cv feed", "visuals"),
]

_quit = False
_built = False
_capturing = False
_cap_deadline = 0.0

_VK_NAMES = {
    0x01: "LMB", 0x02: "RMB", 0x04: "MMB", 0x05: "X1", 0x06: "X2",
    0x08: "BACK", 0x09: "TAB", 0x0D: "ENTER", 0x10: "SHIFT", 0x11: "CTRL",
    0x12: "ALT", 0x14: "CAPS", 0x1B: "ESC", 0x20: "SPACE",
}
_VK_SCAN = (list(range(0x01, 0x07)) + list(range(0x08, 0x5B))
            + list(range(0x70, 0x7C)))


def vk_name(vk):
    try:
        vk = int(vk)
    except Exception:
        return "?"
    if vk in _VK_NAMES:
        return _VK_NAMES[vk]
    if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
        return chr(vk)
    return f"VK {vk:#04x}"


def _set_float(sender, value, attr):
    setattr(config, attr, float(value))


def _set_bool(sender, value, attr):
    setattr(config, attr, bool(value))


def _on_quit(sender, value):
    global _quit
    _quit = True


def start_key_capture(timeout=10.0):
    """Arm hold-key capture: next key/mouse button down becomes aaHoldKey."""
    global _capturing, _cap_deadline
    _capturing = True
    _cap_deadline = time.perf_counter() + float(timeout)
    if dpg.does_item_exist("g_holdkey"):
        dpg.set_value("g_holdkey", "press a key...")


def _poll_key_capture():
    """Call each tick while armed. Returns True while still capturing."""
    global _capturing
    if not _capturing:
        return False
    if time.perf_counter() > _cap_deadline:
        _capturing = False
        if dpg.does_item_exist("g_holdkey"):
            dpg.set_value("g_holdkey", f"hold: {vk_name(config.aaHoldKey)}")
        return False
    for vk in _VK_SCAN:
        try:
            down = bool(win32api.GetAsyncKeyState(vk) & 0x8000)
        except Exception:
            continue
        if down:
            config.aaHoldKey = vk
            _capturing = False
            if dpg.does_item_exist("g_holdkey"):
                dpg.set_value("g_holdkey", f"hold: {vk_name(vk)}")
            return False
    return True


def build():
    """Create context + panel items. Idempotent, no viewport (test-friendly)."""
    global _built
    if _built:
        return
    _built = True
    dpg.create_context()
    with dpg.window(label="PROJECT AA live", width=330, height=600):
        for tag, label, attr, lo, hi in SLIDERS:
            try:
                cur = float(getattr(config, attr))
            except Exception:
                cur = lo
            dpg.add_slider_float(tag=tag, label=label, default_value=cur,
                                 min_value=lo, max_value=hi,
                                 callback=_set_float, user_data=attr)
        for tag, label, attr in CHECKS:
            try:
                cur = bool(getattr(config, attr))
            except Exception:
                cur = False
            dpg.add_checkbox(tag=tag, label=label, default_value=cur,
                             callback=_set_bool, user_data=attr)
        dpg.add_text("", tag="g_stats")
        dpg.add_text(f"hold: {vk_name(config.aaHoldKey)}", tag="g_holdkey")
        dpg.add_button(label="Capture hold key", tag="g_capkey",
                       callback=lambda s, v: start_key_capture())
        dpg.add_button(label="Quit", tag="g_quit", callback=_on_quit)


def init(show=True):
    """Full GUI: context + panel + viewport. show=False for headless tests."""
    build()
    if show:
        dpg.create_viewport(title="PROJECT AA live", width=350, height=620)
        dpg.setup_dearpygui()
        dpg.show_viewport()


def frame(stats_text):
    if dpg.does_item_exist("g_stats"):
        dpg.set_value("g_stats", stats_text)
    _poll_key_capture()
    dpg.render_dearpygui_frame()


def should_quit():
    return _quit or not dpg.is_dearpygui_running()


def stop():
    try:
        if dpg.is_dearpygui_running():
            dpg.stop_dearpygui()
    except Exception:
        pass
    try:
        dpg.destroy_context()
    except Exception:
        pass
