"""Auto-label Fortnite captures with the COCO model (bootstrap for fine-tune).

Reads PNGs from a folder (e.g. data/fn_raw/ recorded live), runs yolo11s at
low confidence, and writes YOLO-format labels (person class 0 only) plus
annotated previews for a quick visual triage (delete bad *_labeled.png pairs).

Usage:
    python tools/autolabel.py data/fn_raw --conf 0.10
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def imread_unicode(p):
    return cv2.imdecode(np.frombuffer(Path(p).read_bytes(), np.uint8),
                        cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder of raw PNGs")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--model", default="weights/yolo11s.pt")
    ap.add_argument("--person-only", action="store_true", default=True)
    args = ap.parse_args()

    from ultralytics import YOLO
    folder = Path(args.folder)
    pngs = sorted(folder.glob("*.png"))
    pngs = [p for p in pngs if "_labeled" not in p.name]
    if not pngs:
        raise SystemExit(f"no PNGs in {folder}")
    lab_dir = folder / "labels"
    prev_dir = folder / "preview"
    lab_dir.mkdir(exist_ok=True)
    prev_dir.mkdir(exist_ok=True)

    m = YOLO(args.model)
    kept_imgs, kept_boxes = 0, 0
    for p in pngs:
        img = imread_unicode(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        r = m.predict(img, imgsz=320, conf=args.conf, verbose=False)[0]
        lines, n = [], 0
        for b in (r.boxes or []):
            cls = int(b.cls)
            if args.person_only and cls != 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            n += 1
        (lab_dir / (p.stem + ".txt")).write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
        cv2.imwrite(str(prev_dir / (p.stem + "_labeled.png")), img)
        kept_imgs += 1
        kept_boxes += n
    print(f"labeled {kept_imgs} imgs, {kept_boxes} person boxes "
          f"(conf {args.conf}) -> {lab_dir}")
    print(f"triage previews in {prev_dir} (delete bad pairs before training)")


if __name__ == "__main__":
    main()
