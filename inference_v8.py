"""Ultralytics YOLOv8 detector (.pt and .onnx) for PROJECT AA.

Uses `ultralytics.YOLO` for both backends so NMS/preprocessing stay in one
place. `.onnx` runs through ONNX Runtime (CUDAExecutionProvider when
device=0, else CPU). `.engine` (TensorRT) also loads via `YOLO()` when
TensorRT is installed; building needs `python export_v8.py <model> 320 engine`.

Note: onnxruntime-gpu >=1.30 needs CUDA 13 + cuDNN 9 for GPU ORT, loaded in
__init__ via ort.preload_dlls(). Without those DLLs it warns and falls back
to CPU (~240ms/320px).
"""
import numpy as np


class V8Detector:
    def __init__(self, model="yolo11s.onnx", imgsz=320, conf=0.25, iou=0.5,
                 device=0, tracker=None):
        # ORDER MATTERS: torch (CUDA12) must load BEFORE the CUDA13/cuDNN9
        # pip DLLs are preloaded for ORT, or torch fails to import (WinError 127).
        import torch  # noqa: F401
        from ultralytics import YOLO
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.tracker = tracker
        if str(model).endswith((".onnx", ".engine")):
            # Lets ORT find the pip-installed CUDA13/cuDNN9 DLLs; without this
            # it falls back to CPU even when the provider build exists.
            try:
                import onnxruntime as _ort
                _ort.preload_dlls()
            except Exception:
                pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
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

    def track(self, frame_bgr):
        """Like infer but with ByteTrack IDs (needs same instance across frames).

        Returns float32 array (N, 7): x1, y1, x2, y2, conf, cls, tid (-1 = none).
        Person-only (class 0). Requires `tracker` yaml set at init.
        """
        results = self.model.track(
            frame_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            classes=[0], device=self.device, verbose=False,
            persist=True, tracker=self.tracker,
        )
        if not results:
            return np.zeros((0, 7), dtype=np.float32)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 7), dtype=np.float32)
        data = boxes.data
        try:
            import torch
            if isinstance(data, torch.Tensor):
                data = data.detach().to("cpu").numpy()
        except ImportError:
            pass
        data = np.asarray(data, dtype=np.float32).reshape(-1, data.shape[-1])
        # Track mode inserts the id as 5th column: (x1,y1,x2,y2,id,conf,cls).
        col_id = data[:, 4].copy() if data.shape[1] == 7 else None
        if col_id is not None:
            data = data[:, [0, 1, 2, 3, 5, 6]]
        tids = boxes.id
        if tids is None:
            tids = col_id
        if tids is None:
            ids = np.full((len(data), 1), -1.0, dtype=np.float32)
        else:
            try:
                import torch
                if isinstance(tids, torch.Tensor):
                    tids = tids.detach().to("cpu").numpy()
            except ImportError:
                pass
            ids = np.asarray(tids, dtype=np.float32).reshape(-1, 1)
        return np.concatenate([data, ids], axis=1)
