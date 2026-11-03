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
    vectors = select_cases() + smooth_cases() + offset_cases() + chain_cases()
    out = ROOT / "aim-rs" / "tests" / "vectors" / "vectors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vectors, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(vectors)} vectors -> {out}")


if __name__ == "__main__":
    main()
