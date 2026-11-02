"""Diagnostic: capture N frames from a window, run V8 inference, save annotated PNGs.

Usage: python tools/diag_capture.py --window Fortnite --n 10
Saves diag/frame_*.png + prints per-frame detections (no mouse movement).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygetwindow

import config
from aim import apply_ignore_rects
from inference_v8 import V8Detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="Fortnite")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="diag")
    args = ap.parse_args()

    needle = args.window.lower()
    wins = [w for w in pygetwindow.getAllWindows() if needle in (w.title or "").lower()]
    if not wins:
        print(f'No window matching "{args.window}"')
        return
    win = wins[0]
    print(f"Window: {win.title!r} at ({win.left},{win.top}) size {win.width}x{win.height}")

    sw, sh = config.screenShotWidth, config.screenShotHeight
    left = ((win.left + win.right) // 2) - (sw // 2)
    top = win.top + (win.height - sh) // 2
    region = (left, top, left + sw, top + sh)
    cw, ch = sw // 2, sh // 2
    print(f"Region: {region} center=({cw},{ch})")

    import bettercam
    cam = bettercam.create(region=region, output_color="BGRA", max_buffer_len=512)
    if cam is None:
        print("bettercam failed")
        return
    cam.start(target_fps=30, video_mode=True)
    time.sleep(1.0)

    det = V8Detector(model=config.v8Model, imgsz=config.v8Imgsz,
                     conf=config.v8Conf, iou=config.v8Iou, device=config.v8Device)
    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)
    for i in range(args.n):
        frame = np.array(cam.get_latest_frame())
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        apply_ignore_rects(frame, getattr(config, "aaIgnoreRects", []))
        dets = det.infer(frame)
        view = frame.copy()
        for (rx, ry, rw, rh) in getattr(config, "aaIgnoreRects", []) or []:
            cv2.rectangle(view, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 1)
        print(f"--- frame {i}: {dets.shape[0]} dets ---")
        for (x1, y1, x2, y2, c, cls) in dets.tolist():
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            h = y2 - y1
            off = h * 0.38
            print(f"  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) conf={c:.2f} "
                  f"mid=({xm:.0f},{ym:.0f}) aim_d=({xm - cw:.0f},{ym - off - ch:.0f})")
            cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(view, (int(xm), int(ym)), 4, (0, 0, 255), -1)
            cv2.circle(view, (int(xm), int(ym - off)), 4, (255, 0, 0), -1)
        cv2.drawMarker(view, (cw, ch), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
        p = outdir / f"frame_{i}.png"
        cv2.imwrite(str(p), view)
        print(f"  saved {p}")
        time.sleep(0.5)
    cam.stop()


if __name__ == "__main__":
    main()
