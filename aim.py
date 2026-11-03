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
                  sticky_px=80, fov=None, max_box_h=0, min_conf=0.3):
    """ID-stable selection over dets (N,6) with parallel ids (-1/None = no id).

    If locked_id is present with conf >= min_conf, keep it (highest conf on ties).
    Else fall back to select_target. Pure: mirrored by the aim-rs port.
    Returns ((xMid, yMid, h, tid), new_locked_id); tid None when unknown.
    """
    arr = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
    ids = list(ids)
    locked_id = _norm_tid(locked_id)
    if locked_id is not None and len(arr):
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


def smooth_move(dx, dy, smooth=2.0, max_step=40, deadzone=1, jitter=0, rng=None):
    """Humanize a raw (dx, dy) delta. Returns (mx, my) ints.

    - smooth >= 1: divisor (1 = instant, 2 = half speed...).
    - max_step: clamp per-tick magnitude, direction preserved.
    - deadzone: ignore moves smaller than this (pixels).
    - jitter: add uniform(-jitter, +jitter) noise on each axis.
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
    return int(round(dx)), int(round(dy))


class KalmanCV:
    """1D constant-velocity Kalman filter, plain floats only.

    No numpy inside: op order is written to match the aim-rs port bit-exactly.
    State [x, v], P symmetric [[p00, p01], [p01, p11]].
    """

    def __init__(self, x0=0.0, p_pos=25.0, p_vel=900.0, q=4e6):
        self.x = float(x0)
        self.v = 0.0
        self.p00 = float(p_pos)
        self.p01 = 0.0
        self.p11 = float(p_vel)
        self.q = float(q)

    def predict(self, dt):
        dt = float(dt)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
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
    """Two independent KalmanCV axes. Mirrored by the aim-rs port."""

    def __init__(self, x0=0.0, y0=0.0, p_pos=25.0, p_vel=900.0, q=4e6):
        self.kx = KalmanCV(x0, p_pos, p_vel, q)
        self.ky = KalmanCV(y0, p_pos, p_vel, q)

    def predict(self, dt):
        self.kx.predict(dt)
        self.ky.predict(dt)

    def update(self, zx, zy, r, gate=0.0):
        return (self.kx.update(zx, r, gate),
                self.ky.update(zy, r, gate))

    def ahead(self, t, vmax):
        return (self.kx.ahead(t, vmax), self.ky.ahead(t, vmax))
