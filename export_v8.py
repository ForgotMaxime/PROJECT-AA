"""Export a YOLOv8 .pt model to ONNX or TensorRT engine.

ONNX (default): `python export_v8.py yolov8s.pt 320 onnx`
Engine (needs NVIDIA TensorRT installed):
    `python export_v8.py yolov8s.pt 320 engine`
"""
import sys


def main(model="yolo11s.pt", imgsz=320, fmt="onnx"):
    from ultralytics import YOLO
    m = YOLO(model)
    fmt = str(fmt).lower()
    if fmt == "onnx":
        out = m.export(format="onnx", imgsz=[int(imgsz), int(imgsz)], opset=12, simplify=True)
    elif fmt == "engine":
        try:
            import tensorrt  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "TensorRT python package not installed; cannot build .engine locally. "
                "Install TensorRT 10.x for your CUDA/Python first, then retry."
            ) from e
        out = m.export(format="engine", imgsz=[int(imgsz), int(imgsz)], half=True, workspace=4)
    else:
        raise SystemExit(f'unknown format: {fmt!r} (use "onnx" or "engine")')
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "yolo11s.pt",
         int(sys.argv[2]) if len(sys.argv) > 2 else 320,
         sys.argv[3] if len(sys.argv) > 3 else "onnx")
