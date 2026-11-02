"""Ultralytics YOLOv8 detector (.pt and .onnx) for PROJECT AA.

Uses `ultralytics.YOLO` for both backends so NMS/preprocessing stay in one
place. `.onnx` runs through ONNX Runtime (CUDAExecutionProvider when
device=0, else CPU).

Note: onnxruntime-gpu >=1.30 needs CUDA 13 + cuDNN 9 for GPU ORT. Without
them it falls back to CPU (~250ms/320px). Default `v8Model=yolov8s.pt`
uses torch CUDA12 (~16ms on RTX 3060) with no extra setup.
"""
import numpy as np


class V8Detector:
    def __init__(self, model="yolov8s.pt", imgsz=320, conf=0.4, iou=0.45, device=0):
        from ultralytics import YOLO
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.model = YOLO(model)

    def infer(self, frame_bgr):
        """Run detection on a BGR HWC uint8 frame.

        Returns float32 array (N, 6): x1, y1, x2, y2, conf, cls (pixel coords).
        Person-only (class 0).
        """
        results = self.model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            classes=[0], device=self.device, verbose=False,
        )
        if not results:
            return np.zeros((0, 6), dtype=np.float32)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        data = boxes.data
        try:
            import torch
            if isinstance(data, torch.Tensor):
                data = data.detach().to("cpu").numpy()
        except ImportError:
            pass
        return np.asarray(data, dtype=np.float32).reshape(-1, 6)
