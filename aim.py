"""Aim helpers for main_v8.py: target selection + smoothed movement.

All functions are pure/numpy so they can be unit-tested without a game.
"""
import math
import random

import numpy as np


def apply_ignore_rects(frame, rects):
    """Black out (x, y, w, h) pixel rects in-place (own 3rd-person model, HUD).

    Pure frame preprocessing, no aim math: not covered by the aim-rs port.
    """
    h, w = frame.shape[:2]
    for (x, y, rw, rh) in rects or []:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w, x0 + int(rw)), min(h, y0 + int(rh))
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = 0
    return frame


def headshot_offset(h, headshot_mode=True):
    """Vertical offset from box center toward the head, by box height.

    Fixed 0.38*h jitters vertically as h fluctuates; bins stabilize it.
    Pure math: mirrored by the aim-rs port.
    """
    h = float(h)
    if not headshot_mode:
        return h * 0.2
    k = 0.42 if h < 80.0 else (0.38 if h <= 150.0 else 0.33)
    return h * k


def select_target(dets, cWidth, cHeight, center=True, last=None, sticky_px=80,
                  fov=None, max_box_h=0):
    """Pick one target from dets (N,6) x1,y1,x2,y2,conf,cls (pixels).

    - fov: None or (fx, fy) half-extents around screen center; targets whose
      midpoint falls outside are ignored (0/None = no filter).
    - sticky: if last is not None, keep tracking the target nearest to last
      while it stays within sticky_px; otherwise fall back to nearest-center.
    - max_box_h: ignore boxes taller than this (own close-up body/gun, 0 = off).
    Returns (xMid, yMid, h) or None.
    """
    if dets.shape[0] == 0:
        return None
    mid = np.stack([(dets[:, 0] + dets[:, 2]) / 2.0,
                    (dets[:, 1] + dets[:, 3]) / 2.0], axis=1)
    idx = np.arange(dets.shape[0])
    if fov is not None and fov[0] and fov[1]:
        fx, fy = float(fov[0]), float(fov[1])
        keep = (np.abs(mid[:, 0] - cWidth) <= fx) & (np.abs(mid[:, 1] - cHeight) <= fy)
        if not bool(np.any(keep)):
            return None
        mid = mid[keep]
        idx = idx[keep]
    if max_box_h:
        heights = dets[idx, 3] - dets[idx, 1]
        keep = heights <= float(max_box_h)
        if not bool(np.any(keep)):
            return None
        mid = mid[keep]
        idx = idx[keep]
    if last is not None and sticky_px:
        d_last = np.hypot(mid[:, 0] - last[0], mid[:, 1] - last[1])
        j = int(np.argmin(d_last))
        if float(d_last[j]) <= float(sticky_px):
            i = int(idx[j])
            return float(mid[j, 0]), float(mid[j, 1]), float(dets[i, 3] - dets[i, 1])
    if center:
        d_c = np.hypot(mid[:, 0] - cWidth, mid[:, 1] - cHeight)
        j = int(np.argmin(d_c))
    elif last is not None:
        d_last = np.hypot(mid[:, 0] - last[0], mid[:, 1] - last[1])
        j = int(np.argmin(d_last))
    else:
        j = 0
    i = int(idx[j])
    return float(mid[j, 0]), float(mid[j, 1]), float(dets[i, 3] - dets[i, 1])


def _norm_tid(tid):
    if tid is None:
        return None
    tid = int(tid)
    return None if tid < 0 else tid


def select_locked(dets, ids, locked_id, cWidth, cHeight, center=True, last=None,
                  sticky_px=80, fov=None, max_box_h=0, min_conf=0.3,
                  recover_px=0):
    """ID-stable selection over dets (N,6) with parallel ids (-1/None = no id).

    If locked_id is present with conf >= min_conf, keep it (highest conf on ties).
    With recover_px > 0 and a last position, spatial continuity wins over the
    id on disagreement (tracker swaps on crossings): nearest conf-gated row
    within recover_px of last is adopted. Else fall back to select_target.
    Pure: mirrored by the aim-rs port.
    Returns ((xMid, yMid, h, tid), new_locked_id); tid None when unknown.
    """
    arr = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
    ids = list(ids)
    locked_id = _norm_tid(locked_id)
    if locked_id is not None and len(arr):
        if last is not None and recover_px:
            lx, ly = float(last[0]), float(last[1])

            def _dist(i):
                r = arr[i]
                return math.hypot((float(r[0]) + float(r[2])) / 2 - lx,
                                  (float(r[1]) + float(r[3])) / 2 - ly)

            # ID match first: only override on clear disagreement, else the
            # lock would flap between two nearby targets every tick.
            best, best_c = -1, -1.0
            for i, tid in enumerate(ids):
                tid = _norm_tid(tid)
                c = float(arr[i, 4])
                if tid == locked_id and c >= min_conf and c > best_c:
                    best, best_c = i, c
            if best >= 0 and _dist(best) <= float(recover_px):
                r = arr[best]
                return ((float((r[0] + r[2]) / 2), float((r[1] + r[3]) / 2),
                         float(r[3] - r[1]), locked_id), locked_id)
            di, dd = -1, float(recover_px)
            for i, r in enumerate(arr):
                if float(r[4]) < min_conf:
                    continue
                d = _dist(i)
                if d < dd or (di < 0 and d <= dd):
                    di, dd = i, d
            if di >= 0:
                r = arr[di]
                tid = _norm_tid(ids[di]) if di < len(ids) else None
                return ((float((r[0] + r[2]) / 2), float((r[1] + r[3]) / 2),
                         float(r[3] - r[1]), tid), tid)
            # Nothing near: fall through to id-keep, then center fallback.
        best, best_c = -1, -1.0
        for i, tid in enumerate(ids):
            tid = _norm_tid(tid)
            c = float(arr[i, 4])
            if tid == locked_id and c >= min_conf and c > best_c:
                best, best_c = i, c
        if best >= 0:
            r = arr[best]
            return ((float((r[0] + r[2]) / 2), float((r[1] + r[3]) / 2),
                     float(r[3] - r[1]), locked_id), locked_id)
    t = select_target(arr, cWidth, cHeight, center=center, last=last,
                      sticky_px=sticky_px, fov=fov, max_box_h=max_box_h)
    if t is None:
        return None, None
    xm, ym, h = t
    tid = None
    for i, r in enumerate(arr):
        if (abs((r[0] + r[2]) / 2 - xm) < 1e-9
                and abs((r[1] + r[3]) / 2 - ym) < 1e-9
                and abs((r[3] - r[1]) - h) < 1e-9):
            tid = _norm_tid(ids[i]) if i < len(ids) else None
            break
    return ((xm, ym, h, tid), tid)


def fov_to_counts(dx_px, dy_px, screen_w, screen_h, fov_x, fov_y,
                  sens=1.0, yaw=0.022, pitch=0.022, speed=1.0,
                  base_fov=0.0, fov_scaled=False):
    """SunOne-style pixels -> mouse-counts conversion, fixed for center-crop.

    SunOne divides by the capture size; we divide by the full game-window
    size instead: the capture is a 1:1 center crop, so 1 capture px =
    1 screen px and deg/px = game FOV / full window width. Using the
    capture size here overestimates the move ~6x (1920/320).
    Returns (dx, dy) float counts; (0.0, 0.0) when disabled/invalid.
    Pure math: mirrored by the aim-rs port (same op order).
    """
    dx_px, dy_px = float(dx_px), float(dy_px)
    sw, sh = float(screen_w), float(screen_h)
    fx, fy = float(fov_x), float(fov_y)
    sens, yaw, pitch = float(sens), float(yaw), float(pitch)
    speed = float(speed)
    if not (math.isfinite(dx_px) and math.isfinite(dy_px)
            and math.isfinite(sw) and math.isfinite(sh)):
        return 0.0, 0.0
    if fx <= 0.0 or fy <= 0.0 or sens == 0.0 or yaw == 0.0 or pitch == 0.0:
        return 0.0, 0.0
    bf = float(base_fov)
    scale = fx / bf if (fov_scaled and bf > 1.0) else 1.0
    den_x = sens * yaw * scale
    den_y = sens * pitch * scale
    if den_x == 0.0 or den_y == 0.0:
        return 0.0, 0.0
    dx = dx_px * (fx / max(1.0, sw)) / den_x * speed
    dy = dy_px * (fy / max(1.0, sh)) / den_y * speed
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return 0.0, 0.0
    return dx, dy


def speed_curve(dist_px, snap_px=0.0, near_px=25.0, smin=0.5, smax=1.0,
               exp=3.0):
    """SunOne-style speed factor by true-pixel distance (continuous ramp).

    - snap zone (dist < snap_px) is handled by the caller (boost + smooth
      bypass); here the ramp starts at snap_px with smin and reaches smax
      at near_px: smin + (smax-smin) * (1-(1-t)^exp), t in 0..1.
    - At/after near_px (or near <= snap): flat smax.
    Pure math: mirrored by the aim-rs port.
    """
    d = float(dist_px)
    sn = float(snap_px)
    nr = float(near_px)
    smin, smax = float(smin), float(smax)
    exp = float(exp)
    if not math.isfinite(d) or d < 0.0:
        return float(smin)
    if not math.isfinite(nr) or nr <= max(sn, 0.0) or d >= nr:
        return float(smax)
    span = nr - max(sn, 0.0)
    t = min(1.0, max(0.0, (d - max(sn, 0.0)) / span)) if span > 0 else 1.0
    return smin + (smax - smin) * (1.0 - (1.0 - t) ** exp)


def smooth_move(dx, dy, smooth=2.0, max_step=40, deadzone=1, jitter=0, rng=None,
                quantize=True):
    """Humanize a raw (dx, dy) delta. Returns (mx, my) ints.

    - smooth >= 1: divisor (1 = instant, 2 = half speed...).
    - max_step: clamp per-tick magnitude, direction preserved.
    - deadzone: ignore moves smaller than this (pixels).
    - jitter: add uniform(-jitter, +jitter) noise on each axis.
    - quantize: round to ints (legacy/mouse units). False keeps floats for
      loop-side callers that convert units afterwards (FOV px -> counts):
      sub-px corrections survive to the final rounding instead of dying here.
    """
    dx, dy = float(dx), float(dy)
    mag = math.hypot(dx, dy)
    if mag < max(float(deadzone), 0.0):
        return 0, 0
    s = max(float(smooth), 1.0)
    dx, dy = dx / s, dy / s
    mag = math.hypot(dx, dy)
    ms = max(int(max_step), 1)
    if mag > ms:
        k = ms / mag
        dx, dy = dx * k, dy * k
    if jitter:
        r = rng or random
        dx += r.uniform(-jitter, jitter)
        dy += r.uniform(-jitter, jitter)
    if not quantize:
        return dx, dy
    return int(round(dx)), int(round(dy))


class KalmanCV:
    """1D constant-velocity Kalman filter, plain floats only.

    No numpy inside: op order is written to match the aim-rs port bit-exactly.
    State [x, v], P symmetric [[p00, p01], [p01, p11]].
    Damping (0 = off): velocity decay per predict, bounds drift through
    gated/missed updates (SunOne `kalman_velocity_damping`).
    """

    def __init__(self, x0=0.0, p_pos=25.0, p_vel=900.0, q=4e6, damping=0.0):
        self.x = float(x0)
        self.v = 0.0
        self.p00 = float(p_pos)
        self.p01 = 0.0
        self.p11 = float(p_vel)
        self.q = float(q)
        self.damping = float(damping)

    def predict(self, dt):
        dt = float(dt)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        self.v = self.v * (1.0 - self.damping)
        self.x = self.x + self.v * dt
        np00 = self.p00 + dt * (self.p01 + self.p01) + dt2 * self.p11 + self.q * dt4 / 4.0
        np01 = self.p01 + dt * self.p11 + self.q * dt3 / 2.0
        np11 = self.p11 + self.q * dt2
        self.p00, self.p01, self.p11 = np00, np01, np11

    def update(self, z, r, gate=0.0):
        """Fuse measurement z (variance r). Returns False if gated as outlier."""
        z = float(z)
        if gate and abs(z - self.x) > float(gate):
            return False
        s = self.p00 + float(r)
        k0 = self.p00 / s
        k1 = self.p01 / s
        y = z - self.x
        self.x = self.x + k0 * y
        self.v = self.v + k1 * y
        p00, p01 = self.p00, self.p01
        self.p00 = p00 - k0 * p00
        self.p01 = p01 - k0 * p01
        self.p11 = self.p11 - k1 * p01
        return True

    def ahead(self, t, vmax):
        v = max(-float(vmax), min(float(vmax), self.v))
        return self.x + v * float(t)


class Kalman2D:
    """Two independent KalmanCV axes, with separate process noise.

    qy > qx lets vertical velocity adapt in a few ticks (bounces, jumps)
    while keeping horizontal steady. Mirrored by the aim-rs port.
    """

    def __init__(self, x0=0.0, y0=0.0, p_pos=25.0, p_vel=900.0, qx=4e6, qy=4e6,
                 damping=0.0):
        self.kx = KalmanCV(x0, p_pos, p_vel, qx, damping)
        self.ky = KalmanCV(y0, p_pos, p_vel, qy, damping)

    def predict(self, dt):
        self.kx.predict(dt)
        self.ky.predict(dt)

    def update(self, zx, zy, r, gate=0.0):
        return (self.kx.update(zx, r, gate),
                self.ky.update(zy, r, gate))

    def ahead(self, t, vmax):
        return (self.kx.ahead(t, vmax), self.ky.ahead(t, vmax))


def reject_own(dets, frame_w, frame_h, edge_px=15, corner=True, sides=("left",)):
    """Lunar-style dynamic own-player rejection (third-person shoulder view).

    A detection is dropped when it touches a shoulder-side screen edge
    (x1 < edge) or sits in the bottom shoulder corner
    (x1 < 20% width and y2 > 80% height) — where your own model renders
    regardless of camera rotation. Returns a bool keep-mask.
    Pure: mirrored by the aim-rs port.
    """
    n = len(dets)
    keep = np.ones(n, dtype=bool)
    left = "left" in sides
    right = "right" in sides
    for i in range(n):
        x1, _, x2, y2 = (float(dets[i, 0]), float(dets[i, 1]),
                         float(dets[i, 2]), float(dets[i, 3]))
        if left and (x1 < edge_px
                     or (corner and x1 < frame_w * 0.2 and y2 > frame_h * 0.8)):
            keep[i] = False
        elif right and ((frame_w - x2) < edge_px
                        or (corner and (frame_w - x2) < frame_w * 0.2
                            and y2 > frame_h * 0.8)):
            keep[i] = False
    return keep


def mini_iou(a, b):
    """IoU of two (x, y, w, h) boxes. Pure: mirrored by the aim-rs port."""
    ax, ay, aw, ah = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bx, by, bw, bh = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    ua = aw * ah + bw * bh - inter
    if ua <= 1e-6:
        return 0.0
    return inter / ua


class MiniTrack:
    """SunOne-style multi-target tracker (mirror of `AimbotTarget.cpp`).

    - Association on VELOCITY-PREDICTED boxes/pivots (not instant IoU like
      ByteTrack), so identity survives crossings and fast motion.
    - The locked track gets privileges: gate x1.6, +8 missed frames, slower
      velocity decay, match-score bonus.
    - Greedy assignment by ascending cost instead of Hungarian (deviation:
      equivalent for the few boxes per frame we see; deterministic order
      (cost, track, det) mirrored by the aim-rs port).
    - Explicit float time `t` (no wall clock): deterministic, vector-tested.
    - Deviation: their frameDtMean `now`-rewrite on invalid dt is skipped
      (all dts are clamped anyway, same effect).

    Dets: (N,6) x1,y1,x2,y2,conf,cls. Only cls == class_player is tracked.
    """

    _UNASSIGNED = 2.0

    def __init__(self, class_player=0, max_missed=6):
        self.class_player = int(class_player)
        self.max_missed = int(max_missed)
        self.tracks = []
        self.next_id = 1
        self.locked = None
        self.frame_w = 320.0
        self.frame_h = 320.0
        self.last_ids = []

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.locked = None
        self.last_ids = []

    def _allowed(self, t):
        return self.max_missed + (8 if t["id"] == self.locked else 0)

    def _match_cost(self, t, box, cls, px, py, now):
        if int(cls) != int(t["cls"]):
            return None
        dt = min(0.25, max(1e-4, float(now) - float(t["last"])))
        pred_cx = t["box"][0] + t["box"][2] * 0.5 + t["vx"] * dt
        pred_cy = t["box"][1] + t["box"][3] * 0.5 + t["vy"] * dt
        pred = [pred_cx - t["box"][2] * 0.5, pred_cy - t["box"][3] * 0.5,
                t["box"][2], t["box"][3]]
        det_cx = box[0] + box[2] * 0.5
        det_cy = box[1] + box[3] * 0.5
        box_dist = math.hypot(det_cx - pred_cx, det_cy - pred_cy)
        pred_px = t["pivot"][0] + t["vx"] * dt
        pred_py = t["pivot"][1] + t["vy"] * dt
        pivot_dist = math.hypot(float(px) - pred_px, float(py) - pred_py)
        dist = min(box_dist, pivot_dist)
        diag = math.hypot(t["box"][2], t["box"][3])
        speed = math.hypot(t["vx"], t["vy"])
        max_dist = (max(24.0, diag * 1.15 + 10.0)
                    + speed * dt * (1.8 + t["missed"] * 0.35)
                    + t["missed"] * max(14.0, diag * 0.18))
        if t["id"] == self.locked:
            max_dist = max_dist * 1.6
        if dist > max_dist:
            return None
        overlap = mini_iou(pred, box)
        cost = ((dist / max_dist) + (1.0 - overlap) * 0.30
                + t["missed"] * 0.025 - min(6, t["hits"]) * 0.01
                - (0.10 if t["id"] == self.locked else 0.0))
        return cost

    def update(self, dets, t, keep_lock=True, frame_w=320.0, frame_h=320.0):
        """Associate dets at time `t`, coast misses, (re)acquire the lock.

        `last_ids` is input-aligned (one id per input det, -1 when filtered
        or untracked).
        """
        self.frame_w, self.frame_h = float(frame_w), float(frame_h)
        arr = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
        boxes, clses, confs = [], [], []
        filt_of_input = []
        for ii, r in enumerate(arr):
            if int(r[5]) != self.class_player or r[2] <= r[0] or r[3] <= r[1]:
                filt_of_input.append(-1)
                continue
            filt_of_input.append(len(boxes))
            w, h = float(r[2] - r[0]), float(r[3] - r[1])
            boxes.append([float(r[0]), float(r[1]), w, h])
            clses.append(int(r[5]))
            confs.append(float(r[4]))
        pivots = [(b[0] + b[2] * 0.5, b[1] + b[3] * 0.5) for b in boxes]

        cands = []
        for ti, tr in enumerate(self.tracks):
            for di in range(len(boxes)):
                c = self._match_cost(tr, boxes[di], clses[di],
                                     pivots[di][0], pivots[di][1], t)
                if c is not None and c < self._UNASSIGNED:
                    cands.append((c, ti, di))
        cands.sort(key=lambda e: (e[0], e[1], e[2]))
        track_hit = [-1] * len(self.tracks)
        det_hit = [-1] * len(boxes)
        for c, ti, di in cands:
            if track_hit[ti] < 0 and det_hit[di] < 0:
                track_hit[ti] = di
                det_hit[di] = ti

        vmax = max(self.frame_w, self.frame_h) * 3.5
        for ti, tr in enumerate(self.tracks):
            di = track_hit[ti]
            if di >= 0:
                dt = min(0.2, max(1e-4, float(t) - float(tr["last"])))
                rvx = (pivots[di][0] - tr["pivot"][0]) / dt
                rvy = (pivots[di][1] - tr["pivot"][1]) / dt
                spd = math.hypot(rvx, rvy)
                if spd > vmax and spd > 1e-4:
                    k = vmax / spd
                    rvx, rvy = rvx * k, rvy * k
                blend = 0.45 if tr["id"] == self.locked else 0.35
                tr["vx"] = tr["vx"] * (1.0 - blend) + rvx * blend
                tr["vy"] = tr["vy"] * (1.0 - blend) + rvy * blend
                tr["box"] = boxes[di]
                tr["pivot"] = [pivots[di][0], pivots[di][1]]
                tr["cls"] = clses[di]
                tr["conf"] = confs[di]
                tr["hits"] += 1
                tr["missed"] = 0
                tr["last"] = float(t)
            else:
                dt = min(0.2, max(0.0, float(t) - float(tr["last"])))
                tr["box"][0] += tr["vx"] * dt
                tr["box"][1] += tr["vy"] * dt
                tr["pivot"][0] += tr["vx"] * dt
                tr["pivot"][1] += tr["vy"] * dt
                decay = 0.90 if tr["id"] == self.locked else 0.84
                tr["vx"] *= decay
                tr["vy"] *= decay
                tr["missed"] += 1
                tr["last"] = float(t)

        self.last_ids = [-1] * len(arr)
        for di in range(len(boxes)):
            if det_hit[di] >= 0:
                fid = self.tracks[det_hit[di]]["id"]
            else:
                fid = self.next_id
                self.next_id += 1
                self.tracks.append({
                    "id": fid, "box": boxes[di],
                    "pivot": [pivots[di][0], pivots[di][1]],
                    "vx": 0.0, "vy": 0.0, "cls": clses[di],
                    "conf": confs[di], "hits": 1, "missed": 0,
                    "last": float(t)})
            for ii, fi in enumerate(filt_of_input):
                if fi == di:
                    self.last_ids[ii] = fid
                    break

        self.tracks = [tr for tr in self.tracks
                       if tr["missed"] <= self._allowed(tr)]
        if not any(tr["id"] == self.locked for tr in self.tracks):
            self.locked = None
        if not keep_lock or self.locked is None:
            best, best_s = -1, float("inf")
            cx, cy = self.frame_w * 0.5, self.frame_h * 0.5
            for ti, tr in enumerate(self.tracks):
                if tr["missed"] > self._allowed(tr):
                    continue
                s = (math.hypot(tr["pivot"][0] - cx, tr["pivot"][1] - cy)
                     + tr["missed"] * 50.0 - min(5, tr["hits"]) * 4.0)
                if s < best_s:
                    best, best_s = ti, s
            self.locked = self.tracks[best]["id"] if best >= 0 else None

    def get_locked(self):
        """Returns (xMid, yMid, h, tid, observed, missed, vx, vy) or None."""
        for tr in self.tracks:
            if tr["id"] == self.locked:
                if tr["missed"] > self._allowed(tr):
                    return None
                cx = tr["box"][0] + tr["box"][2] * 0.5
                cy = tr["box"][1] + tr["box"][3] * 0.5
                return (cx, cy, tr["box"][3], tr["id"],
                        tr["missed"] == 0, tr["missed"], tr["vx"], tr["vy"])
        return None
