"""Generate differential-test vectors for aim-rs from the Python reference (aim.py).

Writes aim-rs/tests/vectors/vectors.json — deterministic, no randomness
(all smooth cases use jitter=0; jitter is covered by Rust property tests).
Run from the repo root:  python tools/dump_vectors.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aim import headshot_offset, select_target, smooth_move  # noqa: E402

FRAME_A = [  # two persons; mids (140,160) d=20 and (20,30) d~184 from (160,160)
    [100.0, 100.0, 180.0, 220.0, 0.9, 0.0],
    [10.0, 10.0, 30.0, 50.0, 0.8, 0.0],
]
TIE = [  # equidistant from x=160 -> first wins (argmin)
    [120.0, 140.0, 160.0, 180.0, 0.9, 0.0],  # mid (140,160)
    [160.0, 140.0, 200.0, 180.0, 0.8, 0.0],  # mid (180,160)
]


def select_cases():
    import numpy as np
    cases = []

    def add(name, dets, cw=160.0, ch=160.0, center=True, last=None,
            sticky=80.0, fov=None, max_box_h=0.0):
        arr = np.array(dets, dtype=np.float64).reshape(-1, 6)
        got = select_target(arr, cw, ch, center=center,
                            last=tuple(last) if last else None,
                            sticky_px=sticky, fov=tuple(fov) if fov else None,
                            max_box_h=max_box_h)
        cases.append({"kind": "select", "name": name,
                      "input": {"dets": dets, "cw": cw, "ch": ch, "center": center,
                                "last": last, "sticky_px": sticky, "fov": fov,
                                "max_box_h": max_box_h},
                      "expected": list(got) if got else None})

    add("center", FRAME_A)
    add("sticky-keep", FRAME_A, last=[20.0, 30.0])
    add("sticky-fallback", FRAME_A, last=[500.0, 500.0])
    add("no-center-no-last", FRAME_A, center=False)
    add("no-center-with-last", FRAME_A, center=False, last=[0.0, 0.0])
    add("sticky-zero-means-off", FRAME_A, last=[20.0, 30.0], sticky=0.0)
    add("fov-none", FRAME_A, fov=[5.0, 5.0])
    add("fov-one", FRAME_A, cw=20.0, ch=30.0, fov=[15.0, 15.0])
    add("fov-zero-disables", FRAME_A, fov=[0.0, 50.0])
    add("empty", [])
    add("single", [FRAME_A[0]])
    add("tie-first-wins", TIE)
    add("maxbox-kills-big", FRAME_A, max_box_h=100.0)  # h=120,40 -> only small kept
    add("maxbox-kills-all", FRAME_A, max_box_h=30.0)
    add("maxbox-keeps-all", FRAME_A, max_box_h=200.0)
    return cases


def smooth_cases():
    cases = []

    def add(name, dx, dy, smooth=2.0, max_step=1000, deadzone=0.0):
        got = smooth_move(dx, dy, smooth=smooth, max_step=max_step,
                          deadzone=deadzone, jitter=0)
        cases.append({"kind": "smooth", "name": name,
                      "input": {"dx": dx, "dy": dy, "smooth": smooth,
                                "max_step": max_step, "deadzone": deadzone},
                      "expected": list(got)})

    add("instant", 100.0, 0.0, smooth=1.0)
    add("divisor", 100.0, 0.0, smooth=2.0)
    add("clamp", 1000.0, 0.0, smooth=1.0, max_step=40)
    add("clamp-diagonal", 30.0, 40.0, smooth=1.0, max_step=25)
    add("deadzone-zero", 0.0, 0.0, deadzone=1.0)
    add("deadzone-small", 0.5, 0.5, deadzone=1.0)
    add("plain", 3.0, 4.0, smooth=1.0)
    add("bankers-even", 5.0, 0.0, smooth=2.0)    # 2.5 -> 2
    add("bankers-odd", 7.0, 0.0, smooth=2.0)     # 3.5 -> 4
    add("bankers-negative", -7.0, 0.0, smooth=2.0)  # -3.5 -> -4
    add("smooth-floor", 10.0, 0.0, smooth=0.5)   # smooth<1 -> 1
    add("maxstep-floor", 10.0, 0.0, smooth=1.0, max_step=0)
    return cases


def offset_cases():
    cases = []
    for h in (20.0, 79.9, 80.0, 100.0, 150.0, 150.1, 300.0):
        for mode in (True, False):
            cases.append({"kind": "offset", "name": f"h{h}-m{int(mode)}",
                          "input": {"h": h, "headshot_mode": mode},
                          "expected": headshot_offset(h, mode)})
    return cases


def fov_cases():
    from aim import fov_to_counts
    cases = []

    def add(name, dx_px, dy_px, sw=1920.0, sh=1080.0, fx=80.0, fy=60.0,
            sens=1.0, yaw=0.022, pitch=0.022, speed=0.7):
        got = fov_to_counts(dx_px, dy_px, sw, sh, fx, fy,
                            sens, yaw, pitch, speed)
        cases.append({"kind": "fov", "name": name,
                      "input": {"dx_px": dx_px, "dy_px": dy_px,
                                "screen_w": sw, "screen_h": sh,
                                "fov_x": fx, "fov_y": fy, "sens": sens,
                                "yaw": yaw, "pitch": pitch, "speed": speed},
                      "expected": list(got)})

    add("centered", 0.0, 0.0)
    add("right-100", 100.0, 0.0)
    add("up-50", 0.0, -50.0)
    add("diag", -80.0, 40.0)
    add("full-speed", 100.0, 0.0, speed=1.0)
    add("sens-2", 100.0, 0.0, sens=2.0)
    add("off-fov", 100.0, 50.0, fx=0.0, fy=0.0)
    add("zero-yaw", 100.0, 0.0, yaw=0.0)
    add("zero-sens", 100.0, 0.0, sens=0.0)
    return cases


def speed_cases():
    from aim import speed_curve
    cases = []

    def add(name, dist, snap=4.0, near=25.0, smin=0.5, smax=1.0, exp=3.0):
        got = speed_curve(dist, snap, near, smin, smax, exp)
        cases.append({"kind": "speed", "name": name,
                      "input": {"dist_px": dist, "snap_px": snap,
                                "near_px": near, "smin": smin,
                                "smax": smax, "exp": exp},
                      "expected": got})

    add("at-zero", 0.0)
    add("mid-ramp", 14.5)
    add("at-near", 25.0)
    add("beyond-near", 100.0)
    add("far-corner", 200.0)
    add("no-near-flat", 100.0, near=0.0)
    add("no-snap", 2.0, snap=0.0)
    add("exp-1-linear", 14.5, exp=1.0)
    add("exp-2", 14.5, exp=2.0)
    add("negative-dist", -5.0)
    return cases


def iou_cases():
    from aim import mini_iou
    cases = []

    def add(name, a, b):
        cases.append({"kind": "iou", "name": name,
                      "input": {"a": list(a), "b": list(b)},
                      "expected": mini_iou(a, b)})

    add("identical", [10.0, 10.0, 40.0, 80.0], [10.0, 10.0, 40.0, 80.0])
    add("half-overlap", [10.0, 10.0, 40.0, 80.0], [30.0, 10.0, 40.0, 80.0])
    add("disjoint", [10.0, 10.0, 40.0, 80.0], [100.0, 100.0, 40.0, 80.0])
    add("contained", [10.0, 10.0, 100.0, 100.0], [30.0, 30.0, 20.0, 20.0])
    add("touching", [10.0, 10.0, 40.0, 80.0], [50.0, 10.0, 40.0, 80.0])
    return cases


def mini_cases():
    from aim import MiniTrack
    cases = []

    def run(name, frames, dt=1 / 30.0, w=320.0, h=320.0, keep_lock=True):
        import numpy as np
        mt = MiniTrack()
        locks, ids = [], []
        for i, dets in enumerate(frames):
            arr = np.array(dets, dtype=np.float64).reshape(-1, 6)
            mt.update(arr, t=i * dt, keep_lock=keep_lock,
                      frame_w=w, frame_h=h)
            lk = mt.get_locked()
            locks.append(lk[3] if lk else None)
            ids.append(list(mt.last_ids))
        cases.append({"kind": "mini", "name": name,
                      "input": {"frames": frames, "dt": dt, "w": w, "h": h,
                                "keep_lock": keep_lock},
                      "expected": {"locks": locks, "ids": ids}})

    def box(cx, cy, w=40.0, h=80.0, conf=0.9, cls=0.0):
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, conf, cls]

    # Straight crossing: A left->right, B right->left, exact overlap at f4.
    cross = [[box(40.0 + i * 30.0, 160.0), box(280.0 - i * 30.0, 160.0)]
             for i in range(9)]
    run("cross-keeps-lock", cross)
    # Same but the tracker may re-pick every frame (keep_lock=False).
    run("cross-repick", cross, keep_lock=False)
    # Occlusion: B vanishes f3..f5, returns; lock must survive on A.
    occ = [[box(100.0, 160.0), box(220.0, 160.0)] for _ in range(3)]
    occ += [[box(100.0 + i * 5.0, 160.0)] for i in range(3)]
    occ += [[box(115.0 + i * 5.0, 160.0), box(220.0, 160.0)] for i in range(3)]
    run("occlusion-survives", occ)
    # Non-person decoy (cls 2) is filtered, never tracked.
    dec = [[box(100.0, 160.0), box(200.0, 160.0, conf=0.8, cls=2.0)]
           for _ in range(3)]
    run("decoy-filtered", dec)
    return cases


def klead_cases():
    from aim import Kalman2D
    cases = []

    def add(name, zs, dt=1 / 30.0, r=50.0, qx=4e6, qy=4e6, gate=60.0,
            t_lead=0.05, vmax=30.0, damping=0.0):
        k = Kalman2D(zs[0][0], zs[0][1], qx=qx, qy=qy, damping=damping)
        for (zx, zy) in zs[1:]:
            k.predict(dt)
            k.update(zx, zy, r, gate)
        ax, ay = k.ahead(t_lead, vmax)
        cases.append({"kind": "klead", "name": name,
                      "input": {"zs": zs, "dt": dt, "r": r, "qx": qx, "qy": qy,
                                "gate": gate, "t_lead": t_lead, "vmax": vmax,
                                "damping": damping},
                      "expected": [ax, ay]})

    add("static", [[100.0, 200.0]] * 6)
    add("const-vel-x", [[100.0 + 10 * i, 200.0] for i in range(6)])
    add("const-vel-diag", [[100.0 + 8 * i, 200.0 - 6 * i] for i in range(6)])
    add("lead-unclamped", [[100.0 + 10 * i, 200.0] for i in range(6)], vmax=1000.0)
    add("bounce-qy", [[500.0, 500.0 - 20 * i] for i in range(6)], qy=16e6)
    add("stop", [[100.0 + 10 * i, 200.0] for i in range(4)] + [[140.0, 200.0]] * 3)
    add("outlier", [[100.0 + 10 * i, 200.0] for i in range(3)] + [[600.0, 200.0]] + [[130.0, 200.0]] * 2)
    add("damped-vel", [[100.0 + 10 * i, 200.0] for i in range(6)], damping=0.08)
    add("damped-stop", [[100.0 + 10 * i, 200.0] for i in range(4)] + [[140.0, 200.0]] * 3,
        damping=0.08)
    return cases


def own_cases():
    from aim import reject_own
    import numpy as np
    cases = []

    def add(name, dets, w=260.0, h=320.0, edge=15.0, corner=True,
            sides=("left",)):
        arr = np.array(dets, dtype=np.float64).reshape(-1, 6)
        keep = reject_own(arr, w, h, edge, corner, sides)
        cases.append({"kind": "own", "name": name,
                      "input": {"dets": dets, "w": w, "h": h, "edge": edge,
                                "corner": corner, "sides": list(sides)},
                      "expected": [bool(v) for v in keep]})

    gun = [[0.9, 243.4, 28.6, 319.6, 0.33, 0.0]]
    bot = [[149.0, 157.0, 170.0, 210.0, 0.73, 0.0]]
    add("gun-edge-rejected", gun)
    add("bot-kept", bot)
    add("corner-rule", [[30.0, 200.0, 80.0, 300.0, 0.5, 0.0]])
    add("corner-off-keeps", [[30.0, 200.0, 80.0, 300.0, 0.5, 0.0]], corner=False)
    add("right-off-by-default", [[200.0, 100.0, 259.0, 300.0, 0.9, 0.0]])
    add("right-on-rejects", [[200.0, 100.0, 259.0, 300.0, 0.9, 0.0]],
        sides=("left", "right"))
    add("empty", [])
    return cases


def locked_cases():
    from aim import select_locked
    import numpy as np
    cases = []

    def add(name, dets, ids, locked, cw=160.0, ch=160.0, center=True,
            last=None, sticky=80.0, fov=None, max_box_h=0.0, min_conf=0.3,
            recover_px=0.0):
        arr = np.array(dets, dtype=np.float64).reshape(-1, 6)
        tgt, new_lock = select_locked(
            arr, ids, locked, cw, ch, center=center,
            last=tuple(last) if last else None,
            sticky_px=sticky, fov=tuple(fov) if fov else None,
            max_box_h=max_box_h, min_conf=min_conf, recover_px=recover_px)
        cases.append({"kind": "locked", "name": name,
                      "input": {"dets": dets, "ids": ids, "locked": locked,
                                "cw": cw, "ch": ch, "center": center,
                                "last": last, "sticky_px": sticky, "fov": fov,
                                "max_box_h": max_box_h, "min_conf": min_conf,
                                "recover_px": recover_px},
                      "expected": {
                          "tgt": ({"x": tgt[0], "y": tgt[1], "h": tgt[2],
                                   "tid": tgt[3]} if tgt else None),
                          "lock": new_lock}})

    add("keep-lock", FRAME_A, [7, 3], 3)
    add("keep-lock-lowconf", FRAME_A, [7, 3], 7, min_conf=0.95)
    add("relock-other", FRAME_A, [7, 3], 99)
    add("no-lock-picks-center", FRAME_A, [7, 3], None)
    add("no-id-fallback", FRAME_A, [None, None], 7)
    add("neg-id-means-none", FRAME_A, [-1, -1], -1)
    add("empty", [], [], 5)
    add("recover-near-adopts", FRAME_A, [7, 3], 99, last=[22.0, 32.0],
        recover_px=60.0)
    add("recover-swap-heals", FRAME_A, [7, 3], 3, last=[140.0, 160.0],
        recover_px=60.0)
    add("recover-near-tie-keeps-id",
        [[140.0, 150.0, 160.0, 170.0, 0.9, 0.0],
         [160.0, 150.0, 180.0, 170.0, 0.9, 0.0]],
        [7, 3], 7, last=[168.0, 160.0], recover_px=60.0)
    add("recover-far-falls-back", FRAME_A, [7, 3], 99, last=[500.0, 500.0],
        recover_px=60.0)
    add("recover-lowconf-skips", FRAME_A, [7, 3], 99, last=[22.0, 32.0],
        min_conf=0.95, recover_px=60.0)
    return cases


def chain_cases():
    """Full main_v8.py tick math: select -> headshot offset -> amp -> smooth."""
    import numpy as np
    cases = []

    def add(name, dets, cw=160.0, ch=160.0, center=True, last=None,
            sticky=80.0, fov=None, max_box_h=0.0, headshot=True, amp=0.4,
            smooth=2.0, max_step=40, deadzone=1.0):
        arr = np.array(dets, dtype=np.float64).reshape(-1, 6)
        tgt = select_target(arr, cw, ch, center=center,
                            last=tuple(last) if last else None,
                            sticky_px=sticky, fov=tuple(fov) if fov else None,
                            max_box_h=max_box_h)
        if tgt is None:
            mv, new_last = None, None
        else:
            xm, ym, h = tgt
            off = headshot_offset(h, headshot)  # single-tick EMA init = h
            mv = list(smooth_move((xm - cw) * amp, (ym - off - ch) * amp,
                                  smooth=smooth, max_step=max_step,
                                  deadzone=deadzone, jitter=0))
            new_last = [xm, ym]
        cases.append({"kind": "chain", "name": name,
                      "input": {"dets": dets, "cw": cw, "ch": ch, "center": center,
                                "last": last, "sticky_px": sticky, "fov": fov,
                                "max_box_h": max_box_h,
                                "headshot_mode": headshot, "amp": amp,
                                "smooth": smooth, "max_step": max_step,
                                "deadzone": deadzone},
                      "expected": {"mv": mv, "last": new_last}})

    add("headshot", FRAME_A)
    add("body", FRAME_A, headshot=False)
    add("sticky", FRAME_A, last=[20.0, 30.0])
    add("none", [])
    add("fov-gate", FRAME_A, fov=[5.0, 5.0])
    return cases


def main():
    vectors = (select_cases() + smooth_cases() + offset_cases()
               + klead_cases() + own_cases() + locked_cases() + chain_cases()
               + fov_cases() + speed_cases() + iou_cases() + mini_cases())
    out = ROOT / "aim-rs" / "tests" / "vectors" / "vectors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vectors, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(vectors)} vectors -> {out}")


if __name__ == "__main__":
    main()
