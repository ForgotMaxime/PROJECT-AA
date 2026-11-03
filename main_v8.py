import queue
import threading
import time

import cv2
import numpy as np
import win32api
import win32con

import gameSelection
from aim import (Kalman2D, apply_ignore_rects, headshot_offset, reject_own,
                 select_locked, select_target, smooth_move)
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


def _capture_loop(camera, frame_q, stop_ev):
    """Pump latest frames; drop stale so inference always sees fresh pixels."""
    while not stop_ev.is_set():
        try:
            frame = camera.get_latest_frame()
        except Exception:
            continue
        if frame is None or stop_ev.is_set():
            continue
        try:
            frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            frame_q.put_nowait(frame)
        except queue.Full:
            pass


def _mouse_loop(mouse_q, stop_ev, backend, port, baud, hold_key_live, stats,
                stats_lock, segments=1, bezier=None):
    """Send latest queued move; drop stale so the cursor never lags frames."""
    while not stop_ev.is_set():
        try:
            dx, dy = mouse_q.get(timeout=0.05)
        except queue.Empty:
            continue
        if stop_ev.is_set():
            break
        try:
            if win32api.GetKeyState(hold_key_live["vk"]):
                move_relative(dx, dy, backend=backend, port=port, baud=baud,
                              segments=segments, bezier=bezier)
                with stats_lock:
                    stats["mouse"] += 1
        except Exception:
            pass


def _queue_latest(q, item):
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def main(source=None, max_ticks=0):
    """Run the aimbot. source=(camera, cWidth, cHeight) injects a fake camera
    (headless tests); max_ticks>0 stops after N inference ticks (test hook)."""
    import sys
    title_filter = None
    args = sys.argv[1:]
    if "--window" in args:
        title_filter = args[args.index("--window") + 1]
    gui_on = "--gui" in args
    shift_x = int(_cfg("aaShiftX", 0))
    shift_y = int(_cfg("aaShiftY", 0))
    crop_left = int(_cfg("aaCropLeft", 0))
    crop_right = int(_cfg("aaCropRight", 0))
    crop_top = int(_cfg("aaCropTop", 0))
    crop_bottom = int(_cfg("aaCropBottom", 0))
    if source is None:
        selection = gameSelection.gameSelection(
            title_filter, shift_x, shift_y,
            crop_left, crop_right, crop_top, crop_bottom)
        if selection is None:
            print("Game selection failed, exiting.")
            return
        camera, cWidth, cHeight = selection
    else:
        camera, cWidth, cHeight = source

    useMask = _cfg("useMask", False)
    maskSide = str(_cfg("maskSide", "left")).lower()
    maskWidth = int(_cfg("maskWidth", 80))
    maskHeight = int(_cfg("maskHeight", 200))
    aaMovementAmp = float(_cfg("aaMovementAmp", 0.4))
    aaQuitKey = str(_cfg("aaQuitKey", "Q"))
    headshot_mode = bool(_cfg("headshot_mode", True))
    aim_offset_cfg = _cfg("aaAimOffset", None)
    cpsDisplay = bool(_cfg("cpsDisplay", True))
    visuals = bool(_cfg("visuals", False))
    centerOfScreen = bool(_cfg("centerOfScreen", True))

    model = _cfg("v8Model", "yolo11s.engine")
    imgsz = int(_cfg("v8Imgsz", 320))
    conf = float(_cfg("v8Conf", _cfg("confidence", 0.25)))
    iou = float(_cfg("v8Iou", 0.5))
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
    makcu_segments = int(_cfg("aaMakcuSegments", 1))
    _bez = _cfg("aaMakcuBezier", None)
    makcu_bezier = tuple(int(v) for v in _bez) if _bez else None
    mono_target = bool(_cfg("aaMonoTarget", True))
    use_tracker = bool(_cfg("v8Track", True))
    tracker_yaml = _cfg("v8Tracker", "trackers/bytetrack-aim.yaml")
    track_min_conf = float(_cfg("aaTrackMinConf", 0.3))
    track_buffer = int(_cfg("aaTrackBuffer", 10))
    recover_px = float(_cfg("aaRecoverPx", 60))
    hold_key = int(_cfg("aaHoldKey", 0x14))
    hold_key_live = {"vk": hold_key}  # shared with mouse thread for GUI rebind
    smooth = float(_cfg("aaSmooth", 1.3))
    max_step = int(_cfg("aaMaxStep", 60))
    deadzone = int(_cfg("aaDeadzone", 1))
    jitter = float(_cfg("aaJitter", 0))
    sticky_px = int(_cfg("aaStickyPx", 80))
    fovX = int(_cfg("aaFovX", 0))
    fovY = int(_cfg("aaFovY", 0))
    fov = (fovX, fovY) if fovX and fovY else None
    ignore_rects = _cfg("aaIgnoreRects", [])
    max_box_h = float(_cfg("aaMaxBoxH", 0))
    own_reject = bool(_cfg("aaOwnReject", True))
    own_edge = float(_cfg("aaOwnEdgePx", 15))
    own_corner = bool(_cfg("aaOwnCorner", True))
    own_sides = tuple(_cfg("aaOwnSides", ["left"]))
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

    # ORT/TRT sessions build lazily on first predict (multi-second freeze).
    # Warm up now so the first CAPS press aims immediately.
    print(f"Warming up {model} ...", flush=True)
    try:
        _warm = np.zeros((160, 160, 3), dtype=np.uint8)
        if use_tracker:
            detector.track(_warm)
        else:
            detector.infer(_warm)
    except Exception as e:
        print(f"Warmup failed (will retry live): {e}")
    print("Warmup done.", flush=True)
    if gui_on:
        import gui_live
        gui_live.init()

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
    prev_holding = False
    mono_armed = True
    count = 0
    ticks = 0
    infer_ms = 0.0
    sTime = time.time()
    stats_txt = ""
    stats = {"mouse": 0}
    stats_lock = threading.Lock()
    frame_q = queue.Queue()
    mouse_q = queue.Queue()
    stop_ev = threading.Event()
    cap_t = threading.Thread(target=_capture_loop, args=(camera, frame_q, stop_ev),
                             daemon=True, name="aa-capture")
    mouse_t = threading.Thread(target=_mouse_loop,
                               args=(mouse_q, stop_ev, mouse_backend,
                                     makcu_port, makcu_baud, hold_key_live,
                                     stats, stats_lock,
                                     makcu_segments, makcu_bezier),
                               daemon=True, name="aa-mouse")
    cap_t.start()
    mouse_t.start()

    while (not stop_ev.is_set()
           and win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0
           and (max_ticks <= 0 or ticks < max_ticks)
           and not (gui_on and gui_live.should_quit())):
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue
        ticks += 1
        frame = np.array(frame)
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

        hold_key = hold_key_live["vk"] = int(_cfg("aaHoldKey", 0x14))
        holding = bool(win32api.GetKeyState(hold_key))
        _trk = _cfg("aaTrackHoldKey", None)
        track_key = int(_trk) if _trk else 0
        track_holding = bool(track_key and win32api.GetKeyState(track_key))
        if mono_target:
            if holding and not prev_holding:
                # Fresh CAPS press: drop everything, acquire anew.
                mono_armed = True
                locked_id = None
                lock_misses = 0
                kf = None
                kf_id = None
            if not holding:
                mono_armed = False
            if track_holding:
                mono_armed = True
        prev_holding = holding

        # Live-tunable subset: the GUI writes config attrs directly, the loop
        # re-reads them every tick (single thread, no locks needed).
        aaMovementAmp = float(_cfg("aaMovementAmp", 0.7))
        smooth = float(_cfg("aaSmooth", 1.3))
        max_step = int(_cfg("aaMaxStep", 60))
        conf = float(_cfg("v8Conf", 0.25))
        iou = float(_cfg("v8Iou", 0.5))
        lead_t = float(_cfg("aaLeadT", 0.03))
        max_box_h = float(_cfg("aaMaxBoxH", 160))
        recover_px = float(_cfg("aaRecoverPx", 60))
        sticky_px = int(_cfg("aaStickyPx", 80))
        dz_enter = float(_cfg("aaDzEnter", 2.5))
        jitter = float(_cfg("aaJitter", 0))
        deadzone = int(_cfg("aaDeadzone", 1))
        aa_predict = bool(_cfg("aaPredict", True))
        mono_target = bool(_cfg("aaMonoTarget", True))
        cpsDisplay = bool(_cfg("cpsDisplay", True))
        visuals = bool(_cfg("visuals", True))
        useMask = bool(_cfg("useMask", False))
        detector.conf = conf
        detector.iou = iou
        lead_vmax = (lead_max / lead_t) if lead_t > 0 else 0.0

        t0 = time.perf_counter()
        tgt = None
        dets = np.zeros((0, 6), dtype=np.float32)
        if mono_target and not mono_armed:
            # Disarmed: fresh CAPS press to re-acquire. No inference spent,
            # just a calm spin so the feed stays live without burning CPU.
            time.sleep(0.02)
        elif tracker_ok:
            try:
                tr = detector.track(frame)
            except Exception as e:
                print(f"tracker failed, falling back to detect-only: {e}")
                tracker_ok = False
                tr = None
            if tracker_ok and tr is not None:
                if own_reject and len(tr):
                    fw, fh = frame.shape[1], frame.shape[0]
                    tr = tr[reject_own(tr[:, :6], fw, fh, own_edge,
                                       own_corner, own_sides)]
                dets = tr[:, :6]
                ids = [int(v) for v in tr[:, 6].tolist()]
                t4, new_lock = select_locked(
                    dets, ids, locked_id, cWidth, cHeight,
                    center=centerOfScreen, last=last_mid,
                    sticky_px=sticky_px, fov=fov, max_box_h=max_box_h,
                    min_conf=track_min_conf, recover_px=recover_px)
                if t4 is None:
                    lock_misses += 1
                    if lock_misses > track_buffer and not track_holding:
                        # Total loss: drop the lock but stay armed while held,
                        # so a reappearance auto-acquires without re-press.
                        locked_id = None
                        lock_misses = 0
                        kf = None
                        kf_id = None
                    elif (track_holding and kf is not None
                            and locked_id is not None):
                        # Absolute lock: coast-aim at the prediction through
                        # total loss instead of going idle.
                        kf.predict(dt)
                        coasted = True
                        ax, ay = kf.ahead(lead_t, lead_vmax)
                        tgt = (ax, ay, ema_h if ema_h is not None else 0.0)
                elif (locked_id is not None and new_lock != locked_id
                        and (lock_misses < lock_hold or track_holding)
                        and kf is not None):
                    # Locked target briefly unseen: hold the lock and coast-aim
                    # at its prediction instead of flicking to another target.
                    # Track-hold extends the coast indefinitely (absolute lock).
                    lock_misses += 1
                    kf.predict(dt)
                    coasted = True
                    ax, ay = kf.ahead(lead_t, lead_vmax)
                    tgt = (ax, ay, ema_h if ema_h is not None else t4[2])
                elif track_holding and locked_id is not None and last_mid is not None:
                    # Absolute lock without predictor: freeze on last known pos.
                    lock_misses += 1
                    tgt = (last_mid[0], last_mid[1],
                           ema_h if ema_h is not None else t4[2])
                else:
                    # Committed switch (hold expired) or first lock: one active
                    # target, never idle while anything is visible.
                    lock_misses = 0
                    locked_id = new_lock
                    tgt = t4[:3]
            else:
                dets = detector.infer(frame)
                if own_reject and len(dets):
                    fw, fh = frame.shape[1], frame.shape[0]
                    dets = dets[reject_own(dets, fw, fh, own_edge,
                                           own_corner, own_sides)]
                if mono_target and not mono_armed:
                    pass
                else:
                    tgt = select_target(dets, cWidth, cHeight,
                                        center=centerOfScreen, last=last_mid,
                                        sticky_px=sticky_px, fov=fov,
                                        max_box_h=max_box_h)
        else:
            dets = detector.infer(frame)
            if own_reject and len(dets):
                fw, fh = frame.shape[1], frame.shape[0]
                dets = dets[reject_own(dets, fw, fh, own_edge,
                                       own_corner, own_sides)]
            if mono_target and mono_armed:
                # Mono without IDs: acquire once, then sticky-only, never jump.
                # Total loss just clears last_mid: reappearance auto-acquires.
                if last_mid is None:
                    tgt = select_target(dets, cWidth, cHeight,
                                        center=centerOfScreen, last=None,
                                        sticky_px=sticky_px, fov=fov,
                                        max_box_h=max_box_h)
                else:
                    tgt = select_target(dets, cWidth, cHeight,
                                        center=False, last=last_mid,
                                        sticky_px=sticky_px, fov=fov,
                                        max_box_h=max_box_h)
            else:
                tgt = select_target(dets, cWidth, cHeight,
                                    center=centerOfScreen, last=last_mid,
                                    sticky_px=sticky_px, fov=fov,
                                    max_box_h=max_box_h)
        infer_ms += ((time.perf_counter() - t0) * 1000.0 - infer_ms) * 0.1

        aim_dbg = None

        if tgt is not None:
            xMid, yMid, box_h = tgt
            ema_h = box_h if ema_h is None else 0.3 * box_h + 0.7 * ema_h
            if aim_offset_cfg is None:
                offset = headshot_offset(ema_h, headshot_mode)
            else:
                offset = ema_h * float(aim_offset_cfg)
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
                dx, dy = smooth_move(raw_dx, raw_dy, smooth=smooth, max_step=max_step,
                                     deadzone=deadzone, jitter=jitter if holding else 0)
                if holding:
                    _queue_latest(mouse_q, (dx, dy))
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
            if mono_target:
                _mstate = f"mono {'ARMED' if mono_armed else 'idle'} lock:{locked_id}"
                if track_holding:
                    _mstate += " TRACK!"
                cv2.putText(view, _mstate,
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imshow("Live Feed v8", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        count += 1
        if (time.time() - sTime) > 1:
            if cpsDisplay:
                with stats_lock:
                    m = stats["mouse"]
                    stats["mouse"] = 0
                print(f"CPS inf: {count} mouse: {m} infer: {infer_ms:.1f}ms")
                stats_txt = (f"inf {count} | mouse {m} | {infer_ms:.1f}ms | "
                             f"lock {locked_id} | dets {len(dets)}")
            count = 0
            sTime = time.time()

        if gui_on:
            gui_live.frame(stats_txt)

    stop_ev.set()
    cap_t.join(timeout=3)
    mouse_t.join(timeout=3)
    if gui_on:
        gui_live.stop()
    try:
        camera.stop()
    except Exception:
        pass
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
