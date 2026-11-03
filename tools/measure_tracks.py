"""Track quality metrics over recorded (or replayed) frame sequences.

Analyze a diag folder (track.jsonl):
    python tools/measure_tracks.py diag

Replay raw frames through another tracker/model and compare:
    python tools/measure_tracks.py diag --replay --tracker ocsort.yaml --model yolo11s.engine

Metrics (no ground truth needed):
- det_rate: fraction of frames with >= 1 detection
- switches: times the nearest-center target's id changed between
  consecutive non-empty frames (proxy for ID instability)
- jitter: std of nearest-center midpoint displacement per tick (px)
- ids: distinct track ids seen
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def nearest_center(dets, cw, ch):
    best, bd = -1, float("inf")
    for i, r in enumerate(dets):
        d = math.hypot((r[0] + r[2]) / 2 - cw, (r[1] + r[3]) / 2 - ch)
        if d < bd:
            best, bd = i, d
    return best


def metrics(rows, cw, ch):
    n = len(rows)
    non_empty = [r for r in rows if len(r["dets"])]
    det_rate = len(non_empty) / n if n else 0.0
    switches, jumps, all_ids = 0, [], set()
    prev_id, prev_mid = None, None
    for r in rows:
        if not len(r["dets"]):
            continue
        j = nearest_center(r["dets"], cw, ch)
        tid = r["ids"][j] if j < len(r["ids"]) else -1
        if tid is not None and tid >= 0:
            all_ids.add(int(tid))
        if (prev_id is not None and prev_id >= 0 and tid is not None
                and tid >= 0 and tid != prev_id):
            switches += 1
        mid = ((r["dets"][j][0] + r["dets"][j][2]) / 2,
               (r["dets"][j][1] + r["dets"][j][3]) / 2)
        if prev_mid is not None:
            jumps.append(math.hypot(mid[0] - prev_mid[0], mid[1] - prev_mid[1]))
        prev_id, prev_mid = tid, mid
    import statistics
    jitter = statistics.pstdev(jumps) if len(jumps) > 1 else 0.0
    return {"frames": n, "det_rate": round(det_rate, 3), "switches": switches,
            "jitter_px": round(jitter, 2), "ids": sorted(all_ids)}


def replay(folder, tracker, model, imgsz=320, conf=0.25):
    from inference_v8 import V8Detector
    import cv2
    import numpy as np

    def imread_unicode(p):
        return cv2.imdecode(np.frombuffer(Path(p).read_bytes(), np.uint8),
                            cv2.IMREAD_COLOR)

    raws = sorted(Path(folder).glob("frame_raw_*.png"))
    if not raws:
        raise SystemExit(f"no frame_raw_*.png in {folder} (record with diag_capture.py)")
    det = V8Detector(model=model, imgsz=imgsz, conf=conf, device=0, tracker=tracker)
    out = []
    for p in raws:
        fr = imread_unicode(p)
        tr = det.track(fr)
        out.append({"t": 0.0,
                    "dets": tr[:, :6].tolist(),
                    "ids": [int(v) for v in tr[:, 6].tolist()]})
    tag = Path(tracker).stem
    rep = Path(folder) / f"replay-{tag}.jsonl"
    rep.write_text("\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")
    return out, str(rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", default="diag")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--tracker", default="trackers/bytetrack-aim.yaml")
    ap.add_argument("--model", default="weights/yolo11s.engine")
    ap.add_argument("--cw", type=float, default=130.0)
    ap.add_argument("--ch", type=float, default=160.0)
    args = ap.parse_args()

    if args.replay:
        rows, where = replay(args.folder, args.tracker, args.model)
        print(f"replayed {len(rows)} frames -> {where}")
    else:
        rows = load_jsonl(Path(args.folder) / "track.jsonl")
    m = metrics(rows, args.cw, args.ch)
    print(f"frames={m['frames']} det_rate={m['det_rate']} "
          f"switches={m['switches']} jitter={m['jitter_px']}px ids={m['ids']}")


if __name__ == "__main__":
    main()
