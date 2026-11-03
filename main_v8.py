import time

import cv2
import numpy as np
import win32api
import win32con

import gameSelection
from aim import (Kalman2D, apply_ignore_rects, headshot_offset, select_locked,
                 select_target, smooth_move)
from inference_v8 import V8Detector
from mouse import close_makcu, move_relative


def _cfg(name, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


def pick_target(dets, cWidth, cHeight, centerOfScreen, last_mid, max_box_h=0):
    """dets: (N,6) x1,y1,x2,y2,conf,cls in pixels. Returns (xMid,yMid,h) or None."""
    return select_target(dets, cWidth, cHeight, center=centerOfScreen,
                         last=last_mid, max_box_h=max_box_h)


def main():
    import sys
    title_filter = None
    args = sys.argv[1:]
    if "--window" in args:
        title_filter = args[args.index("--window") + 1]
    shift_x = int(_cfg("aaShiftX", 0))
    shift_y = int(_cfg("aaShiftY", 0))
    crop_left = int(_cfg("aaCropLeft", 0))
    crop_right = int(_cfg("aaCropRight", 0))
    crop_top = int(_cfg("aaCropTop", 0))
    crop_bottom = int(_cfg("aaCropBottom", 0))
    selection = gameSelection.gameSelection(
        title_filter, shift_x, shift_y,
        crop_left, crop_right, crop_top, crop_bottom)
    if selection is None:
        print("Game selection failed, exiting.")
        return
    camera, cWidth, cHeight = selection

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
    use_tracker = bool(_cfg("v8Track", True))
    tracker_yaml = _cfg("v8Tracker", "trackers/bytetrack-aim.yaml")
    track_min_conf = float(_cfg("aaTrackMinConf", 0.3))
    track_buffer = int(_cfg("aaTrackBuffer", 10))
    hold_key = int(_cfg("aaHoldKey", 0x14))
    smooth = float(_cfg("aaSmooth", 2.0))
    max_step = int(_cfg("aaMaxStep", 40))
    deadzone = int(_cfg("aaDeadzone", 1))
    jitter = float(_cfg("aaJitter", 0))
    sticky_px = int(_cfg("aaStickyPx", 80))
    fovX = int(_cfg("aaFovX", 0))
    fovY = int(_cfg("aaFovY", 0))
    fov = (fovX, fovY) if fovX and fovY else None
    ignore_rects = _cfg("aaIgnoreRects", [])
    max_box_h = float(_cfg("aaMaxBoxH", 0))
    dz_enter = float(_cfg("aaDzEnter", 2.5))
    dz_exit = float(_cfg("aaDzExit", 1.0))
    aa_predict = bool(_cfg("aaPredict", True))
    lead_t = float(_cfg("aaLeadT", 0.05))
    lead_max = float(_cfg("aaLeadMax", 30.0))
    gate_px = float(_cfg("aaGatePx", 60.0))
    kalman_qx = float(_cfg("aaKalmanQX", 4e6))
    kalman_qy = float(_cfg("aaKalmanQY", 16e6))
    kalman_r = float(_cfg("aaKalmanR", 50.0))
    coast_max = int(_cfg("aaCoastMax", 5))
    lock_hold = int(_cfg("aaLockHold", 8))
    lead_vmax = (lead_max / lead_t) if lead_t > 0 else 0.0

    detector = V8Detector(model=model, imgsz=imgsz, conf=conf, iou=iou,
                         device=device,
                         tracker=tracker_yaml if use_tracker else None)

    colors = np.random.uniform(0, 255, size=(2, 3))
    last_mid = None
    ema_h = None
    engaged = False
    locked_id = None
    lock_misses = 0
    tracker_ok = use_tracker
    kf = None
    kf_id = None
    kf_coast = 0
    last_t = None
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
        if ignore_rects:
            apply_ignore_rects(frame, ignore_rects)

        now = time.perf_counter()
        dt = min(0.2, max(0.005, now - last_t)) if last_t is not None else 1 / 30.0
        last_t = now
        coasted = False

        t0 = time.perf_counter()
        if tracker_ok:
            try:
                tr = detector.track(frame)
            except Exception as e:
                print(f"tracker failed, falling back to detect-only: {e}")
                tracker_ok = False
                tr = None
            if tracker_ok and tr is not None:
                dets = tr[:, :6]
                ids = [int(v) for v in tr[:, 6].tolist()]
                t4, new_lock = select_locked(
                    dets, ids, locked_id, cWidth, cHeight,
                    center=centerOfScreen, last=last_mid,
                    sticky_px=sticky_px, fov=fov, max_box_h=max_box_h,
                    min_conf=track_min_conf)
                if t4 is None:
                    lock_misses += 1
                    if lock_misses > track_buffer:
                        locked_id = None
                        lock_misses = 0
                        kf = None
                        kf_id = None
                    tgt = None
                elif (locked_id is not None and new_lock != locked_id
                        and lock_misses < lock_hold and kf is not None):
                    # Locked target briefly unseen: hold the lock and coast-aim
                    # at its prediction instead of flicking to another target.
                    lock_misses += 1
                    kf.predict(dt)
                    coasted = True
                    ax, ay = kf.ahead(lead_t, lead_vmax)
                    tgt = (ax, ay, ema_h if ema_h is not None else t4[2])
                else:
                    lock_misses = 0
                    locked_id = new_lock
                    tgt = t4[:3]
            else:
                dets = detector.infer(frame)
                tgt = select_target(dets, cWidth, cHeight,
                                    center=centerOfScreen, last=last_mid,
                                    sticky_px=sticky_px, fov=fov,
                                    max_box_h=max_box_h)
        else:
            dets = detector.infer(frame)
            tgt = select_target(dets, cWidth, cHeight,
                                center=centerOfScreen, last=last_mid,
                                sticky_px=sticky_px, fov=fov,
                                max_box_h=max_box_h)
        infer_ms += ((time.perf_counter() - t0) * 1000.0 - infer_ms) * 0.1

        aim_dbg = None

        if tgt is not None:
            xMid, yMid, box_h = tgt
            ema_h = box_h if ema_h is None else 0.3 * box_h + 0.7 * ema_h
            offset = headshot_offset(ema_h, headshot_mode)
            if aa_predict and not coasted:
                key = locked_id if use_tracker else True
                if kf is None or kf_id != key:
                    kf = Kalman2D(xMid, yMid, qx=kalman_qx, qy=kalman_qy)
                    kf_id = key
                    kf_coast = 0
                else:
                    kf.predict(dt)
                    kf.update(xMid, yMid, kalman_r, gate_px)
                    kf_coast = 0
                ax, ay = kf.ahead(lead_t, lead_vmax)
                aim_dbg = (ax, ay)
            else:
                ax, ay = xMid, yMid
                aim_dbg = (ax, ay)
            # aaShiftX/Y: same shift as the capture region, so aim stays exact
            # while the selection center moves (ex-aaRightShift).
            raw_dx = (ax + shift_x - cWidth) * aaMovementAmp
            raw_dy = (ay + shift_y - offset - cHeight) * aaMovementAmp
            mag = float(np.hypot(raw_dx, raw_dy))
            # Deadzone hysteresis: engage above enter, release below exit.
            if (engaged and mag < dz_exit) or (not engaged and mag < dz_enter):
                engaged = False
            else:
                engaged = True
                holding = bool(win32api.GetKeyState(hold_key))
                dx, dy = smooth_move(raw_dx, raw_dy, smooth=smooth, max_step=max_step,
                                     deadzone=deadzone, jitter=jitter if holding else 0)
                if holding:
                    move_relative(dx, dy, backend=mouse_backend,
                                  port=makcu_port, baud=makcu_baud)
            last_mid = (xMid, yMid)
        else:
            if kf is not None and kf_coast < coast_max:
                kf.predict(dt)
                kf_coast += 1
            last_mid = None
            ema_h = None
            engaged = False

        if visuals:
            view = frame
            for x1, y1, x2, y2, c, _ in dets.tolist():
                cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), colors[0].tolist(), 2)
                cv2.putText(view, f"Human: {c * 100:.1f}%", (int(x1), max(15, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0].tolist(), 2)
            if aim_dbg is not None:
                cv2.drawMarker(view, (int(aim_dbg[0]), int(aim_dbg[1])),
                               (0, 255, 255), cv2.MARKER_DIAMOND, 12, 2)
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
