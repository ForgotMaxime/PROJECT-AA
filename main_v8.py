import time

import cv2
import numpy as np
import win32api
import win32con

import gameSelection
from inference_v8 import V8Detector
from mouse import move_relative


def _cfg(name, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


def pick_target(dets, cWidth, cHeight, centerOfScreen, last_mid):
    """dets: (N,6) x1,y1,x2,y2,conf,cls in pixels. Returns (xMid,yMid,h) or None."""
    if dets.shape[0] == 0:
        return None
    mid = np.stack([(dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0], axis=1)
    order = np.arange(dets.shape[0])
    if centerOfScreen:
        dist = np.hypot(mid[:, 0] - cWidth, mid[:, 1] - cHeight)
        order = np.argsort(dist, kind="stable")
    elif last_mid is not None:
        dist = np.hypot(mid[:, 0] - last_mid[0], mid[:, 1] - last_mid[1])
        order = np.argsort(dist, kind="stable")
    i = int(order[0])
    h = float(dets[i, 3] - dets[i, 1])
    return float(mid[i, 0]), float(mid[i, 1]), h


def main():
    camera, cWidth, cHeight = gameSelection.gameSelection()

    useMask = _cfg("useMask", False)
    maskSide = str(_cfg("maskSide", "left")).lower()
    maskWidth = int(_cfg("maskWidth", 80))
    maskHeight = int(_cfg("maskHeight", 200))
    aaMovementAmp = float(_cfg("aaMovementAmp", 0.4))
    aaQuitKey = str(_cfg("aaQuitKey", "Q"))
    headshot_mode = bool(_cfg("headshot_mode", True))
    cpsDisplay = bool(_cfg("cpsDisplay", True))
    visuals = bool(_cfg("visuals", False))
    centerOfScreen = bool(_cfg("centerOfScreen", True))

    model = _cfg("v8Model", "yolov8s.pt")
    imgsz = int(_cfg("v8Imgsz", 320))
    conf = float(_cfg("v8Conf", _cfg("confidence", 0.4)))
    iou = float(_cfg("v8Iou", 0.45))
    device = _cfg("v8Device", 0)
    mouse_backend = _cfg("aaMouseBackend", "sendinput")

    detector = V8Detector(model=model, imgsz=imgsz, conf=conf, iou=iou, device=device)

    colors = np.random.uniform(0, 255, size=(2, 3))
    last_mid = None
    count = 0
    sTime = time.time()

    while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:
        frame = np.array(camera.get_latest_frame())
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if useMask:
            if maskSide == "right":
                frame[-maskHeight:, -maskWidth:] = 0
            elif maskSide == "left":
                frame[-maskHeight:, :maskWidth] = 0
            else:
                raise ValueError('maskSide must be "left" or "right"')

        dets = detector.infer(frame)
        tgt = pick_target(dets, cWidth, cHeight, centerOfScreen, last_mid)

        if tgt is not None:
            xMid, yMid, box_h = tgt
            offset = box_h * (0.38 if headshot_mode else 0.2)
            dx = int((xMid - cWidth) * aaMovementAmp)
            dy = int((yMid - offset - cHeight) * aaMovementAmp)
            if win32api.GetKeyState(0x14):
                move_relative(dx, dy, backend=mouse_backend)
            last_mid = (xMid, yMid)
        else:
            last_mid = None

        if visuals:
            view = frame
            for x1, y1, x2, y2, c, _ in dets.tolist():
                cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), colors[0].tolist(), 2)
                cv2.putText(view, f"Human: {c * 100:.1f}%", (int(x1), max(15, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0].tolist(), 2)
            cv2.imshow("Live Feed v8", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        count += 1
        if (time.time() - sTime) > 1:
            if cpsDisplay:
                print(f"CPS: {count}")
            count = 0
            sTime = time.time()

    camera.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exception(e)
        print("ERROR: " + str(e))
