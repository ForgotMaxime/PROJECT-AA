"""Fine-tune yolo11s on Fortnite captures (person class).

Layout expected (see tools/autolabel.py):
    data/fn/
        images/   <- triaged PNGs (rename from fn_raw after review)
        labels/   <- matching *.txt YOLO labels (class 0)

Usage:
    python tools/train_fortnite.py data/fn --epochs 50

Trains at imgsz 320 (runtime size) on cuda:0, splits 90/10 train/val
automatically, then prints the best.pt path. Export it with:
    python export_v8.py weights/finetune/fn1/weights/best.pt 320 onnx
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="folder with images/ + labels/")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--base", default="weights/yolo11s.pt")
    ap.add_argument("--name", default="fn1")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    root = Path(args.dataset)
    imgs = sorted((root / "images").glob("*.png"))
    labs = [(root / "labels" / (p.stem + ".txt")) for p in imgs]
    pairs = [(p, l) for p, l in zip(imgs, labs) if l.exists()]
    if len(pairs) < 20:
        raise SystemExit(f"only {len(pairs)} labeled pairs in {root} "
                         f"(need >= 20, aim 200+)")
    rng = random.Random(7)
    rng.shuffle(pairs)
    cut = max(1, int(len(pairs) * 0.9))

    work = root / "yolo"
    for split in ("train", "val"):
        (work / "images" / split).mkdir(parents=True, exist_ok=True)
        (work / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i, (p, l) in enumerate(pairs):
        split = "train" if i < cut else "val"
        shutil.copy(p, work / "images" / split / p.name)
        shutil.copy(l, work / "labels" / split / l.name)
    (work / "fn.yaml").write_text(
        f"path: {work.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names: {{0: person}}\n", encoding="utf-8")
    print(f"split: {cut} train / {len(pairs) - cut} val")

    from ultralytics import YOLO
    m = YOLO(args.base)
    m.train(data=str(work / "fn.yaml"), epochs=args.epochs, imgsz=320,
            batch=args.batch, device=0, project="weights/finetune",
            name=args.name, verbose=True)
    print(f"best: weights/finetune/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
