"""Headless smoke test for the threaded main_v8 pipeline.

Fake camera cycles a real detection crop; mouse backend points at a
serial loopback so nothing moves on screen. Asserts: clean run to
max_ticks, inference ran, mouse thread sent framed moves, threads exit.

Usage: python tools/smoke_threads.py
"""
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeWin32:
    def __init__(self):
        self.sent_quit = False

    def GetKeyState(self, vk):
        return 1  # CAPS "held"

    def GetAsyncKeyState(self, vk):
        return 0  # never quit via key


class FakeCam:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0
        self.stopped = False

    def get_latest_frame(self):
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        time.sleep(0.005)
        return f

    def stop(self):
        self.stopped = True


def main():
    import config
    import main_v8
    import mouse

    main_v8.win32api = FakeWin32()
    config.aaMouseBackend = "makcu"
    config.aaMakcuPort = "loop://"
    config.visuals = False
    config.v8Track = False  # detect path; tracker covered live
    config.v8Conf = 0.05  # ensure the test crop fires
    config.aaIgnoreRects = []  # bypass gun blackout for the test

    data = np.frombuffer(
        Path("diag/shot_crop_320.png").read_bytes(), np.uint8)
    crop_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    blank = np.zeros_like(crop)
    cam = FakeCam([crop, crop, blank, crop])

    before = set(t.name for t in threading.enumerate())
    t0 = time.perf_counter()
    main_v8.close_makcu = lambda *a, **k: None  # keep handles open for inspection
    main_v8.main(source=(cam, 130, 160), max_ticks=30)
    dt = time.perf_counter() - t0

    assert cam.stopped, "camera.stop() not called"
    key = ("loop://", int(config.aaMakcuBaud))
    ser = mouse._MAKCU_CONNS.get(key)
    assert ser is not None, "mouse thread never opened loopback"
    pending = ser.read(1024)
    assert pending.startswith(b"km.move("), pending[:40]
    leaked = [t.name for t in threading.enumerate() if t.name not in before]
    assert not leaked, f"thread leak: {leaked}"
    print(f"smoke ok: 30 ticks in {dt:.1f}s, mouse bytes: {pending[:40]!r}")
    mouse.close_makcu()


if __name__ == "__main__":
    main()
