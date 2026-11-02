"""Aim helpers for main_v8.py: target selection + smoothed movement.

All functions are pure/numpy so they can be unit-tested without a game.
"""
import math
import random

import numpy as np


def select_target(dets, cWidth, cHeight, center=True, last=None, sticky_px=80,
                  fov=None):
    """Pick one target from dets (N,6) x1,y1,x2,y2,conf,cls (pixels).

    - fov: None or (fx, fy) half-extents around screen center; targets whose
      midpoint falls outside are ignored (0/None = no filter).
    - sticky: if last is not None, keep tracking the target nearest to last
      while it stays within sticky_px; otherwise fall back to nearest-center.
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
