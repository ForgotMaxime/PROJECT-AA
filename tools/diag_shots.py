"""Offline detection check on user-provided fullscreen screenshots."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference_v8 import V8Detector

shots = sorted(Path(r"C:\Users\humax\Desktop\FFA\Image").glob("*.png"))


def imread_unicode(path):
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


img = imread_unicode(shots[0])
print("full:", img.shape, shots[0].name)
crop = img[380:700, 880:1140]  # same box as live capture
print("crop:", crop.shape)
for imgsz in (320, 640):
    d = V8Detector("yolov8s.pt", imgsz=imgsz, conf=0.25, device=0)
    t = time.perf_counter()
    dets = d.infer(crop)
    dt = (time.perf_counter() - t) * 1000
    print(f"imgsz={imgsz}: {dets.shape[0]} dets in {dt:.0f}ms")
    view = crop.copy()
    for x1, y1, x2, y2, c, _ in dets.tolist():
        print(f"  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) conf={c:.2f} h={y2 - y1:.0f}")
        cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.imwrite(f"diag/shot_crop_{imgsz}.png", view)
print("saved diag/shot_crop_*.png")
