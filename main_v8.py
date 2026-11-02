import time

import cv2
import numpy as np
import win32api
import win32con

import gameSelection
from aim import select_target, smooth_move
from inference_v8 import V8Detector
from mouse import close_makcu, move_relative


def _cfg(name, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


def pick_target(dets, cWidth, cHeight, centerOfScreen, last_mid):
    """dets: (N,6) x1,y1,x2,y2,conf,cls in pixels. Returns (xMid,yMid,h) or None."""
    return select_target(dets, cWidth, cHeight, center=centerOfScreen, last=last_mid)


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
    if isinstance(device, str) and device.lower() == "auto":
        try:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    mouse_backend = _cfg("aaMouseBackend", "sendinput")
    makcu_port = _cfg("aaMakcuPort", "COM3")
    makcu_baud = int(_cfg("aaMakcuBaud", 115200))
    hold_key = int(_cfg("aaHoldKey", 0x14))
    smooth = float(_cfg("aaSmooth", 2.0))
    max_step = int(_cfg("aaMaxStep", 40))
    deadzone = int(_cfg("aaDeadzone", 1))
    jitter = float(_cfg("aaJitter", 0))
    sticky_px = int(_cfg("aaStickyPx", 80))
    fovX = int(_cfg("aaFovX", 0))
    fovY = int(_cfg("aaFovY", 0))
    fov = (fovX, fovY) if fovX and fovY else None

    detector = V8Detector(model=model, imgsz=imgsz, conf=conf, iou=iou, device=device)

    colors = np.random.uniform(0, 255, size=(2, 3))
    last_mid = None
    count = 0
    infer_ms = 0.0
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

        t0 = time.perf_counter()
        dets = detector.infer(frame)
        infer_ms += ((time.perf_counter() - t0) * 1000.0 - infer_ms) * 0.1
        tgt = select_target(dets, cWidth, cHeight, center=centerOfScreen,
                            last=last_mid, sticky_px=sticky_px, fov=fov)

        if tgt is not None:
            xMid, yMid, box_h = tgt
            offset = box_h * (0.38 if headshot_mode else 0.2)
            raw_dx = (xMid - cWidth) * aaMovementAmp
            raw_dy = (yMid - offset - cHeight) * aaMovementAmp
            holding = bool(win32api.GetKeyState(hold_key))
            dx, dy = smooth_move(raw_dx, raw_dy, smooth=smooth, max_step=max_step,
                                 deadzone=deadzone, jitter=jitter if holding else 0)
            if holding:
                move_relative(dx, dy, backend=mouse_backend,
                              port=makcu_port, baud=makcu_baud)
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
                print(f"CPS: {count} infer: {infer_ms:.1f}ms")
            count = 0
            sTime = time.time()

    camera.stop()
    try:
        close_makcu()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exception(e)
        print("ERROR: " + str(e))
