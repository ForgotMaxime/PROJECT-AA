"""Export a YOLOv8 .pt model to ONNX (CUDA12 / ORT ready)."""
import sys


def main(model="yolov8s.pt", imgsz=320):
    from ultralytics import YOLO
    m = YOLO(model)
    out = m.export(format="onnx", imgsz=[int(imgsz), int(imgsz)], opset=12, simplify=True)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "yolov8s.pt",
         int(sys.argv[2]) if len(sys.argv) > 2 else 320)
