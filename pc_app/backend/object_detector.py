"""
object_detector.py - 目标检测模块

模型: valorant.onnx (YOLO11s, 256×256 输入, 5 类)
预处理: 中心裁剪 256×256 (不 resize)
后处理: 置信度阈值 0.25, NMS IoU 0.45
"""

import logging
import os
import time
from typing import List, Optional

import cv2
import numpy as np

from trajectory_calculator import Detection

logger = logging.getLogger(__name__)


class ObjectDetector:
    """目标检测器 (valorant.onnx, center crop 256×256)"""

    CONFIDENCE_THRESHOLD = 0.25
    NMS_IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 50

    INPUT_SIZE = 256          # 模型输入尺寸 = 裁剪尺寸
    MODEL_PATH = 'valorant.onnx'

    # 类别名称 (valorant.onnx metadata: body, head, teammate, breakable, dodge)
    CLASS_NAMES = ['body', 'head', 'teammate', 'breakable', 'dodge']

    def __init__(self):
        self._session = None
        self._model_path = None
        self._loaded = False
        self._inference_time = 0.0
        self._inference_count = 0
        self._crop_offset = (0, 0)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_path(self) -> Optional[str]:
        return self._model_path

    @property
    def avg_inference_time_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return (self._inference_time / self._inference_count) * 1000

    def load_model(self, model_path: str):
        ext = os.path.splitext(model_path)[1].lower()
        logger.info(f"Loading model: {model_path}")

        if ext == '.onnx':
            self._load_onnx(model_path)
        else:
            raise ValueError(f"Unsupported model format: {ext}")

        self._model_path = model_path
        self._loaded = True
        logger.info(f"Model loaded ({model_path})")

    def _load_onnx(self, model_path: str):
        import onnxruntime
        available = onnxruntime.get_available_providers()
        preferred = ['DmlExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = [p for p in preferred if p in available]
        opts = onnxruntime.SessionOptions()
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        self._session = onnxruntime.InferenceSession(model_path, opts, providers=providers)
        for inp in self._session.get_inputs():
            logger.info(f"  Input: {inp.name} -> {inp.shape}")
        for out in self._session.get_outputs():
            logger.info(f"  Output: {out.name} -> {out.shape}")

    def detect(self, frame: np.ndarray, original_shape: Optional[tuple] = None) -> List[Detection]:
        if not self._loaded:
            return []
        t_start = time.perf_counter()
        detections = self._detect_onnx(frame, original_shape)
        self._inference_time += time.perf_counter() - t_start
        self._inference_count += 1
        return detections

    def _detect_onnx(self, frame: np.ndarray, original_shape: Optional[tuple]) -> List[Detection]:
        """ONNX 推理: 中心裁剪 256×256 (不 resize)"""
        orig_h, orig_w = original_shape or frame.shape[:2]
        SZ = self.INPUT_SIZE

        # 中心裁剪 256×256
        cx, cy = orig_w // 2, orig_h // 2
        half = SZ // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(orig_w, x1 + SZ)
        y2 = min(orig_h, y1 + SZ)
        crop = frame[y1:y2, x1:x2]
        # 补齐边缘 (如果画面边缘不足 256)
        ch, cw = crop.shape[:2]
        if ch != SZ or cw != SZ:
            crop = cv2.copyMakeBorder(crop, 0, SZ - ch, 0, SZ - cw,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self._crop_offset = (x1, y1)

        # BGR→RGB + /255, 无需 resize
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(np.transpose(rgb, (2, 0, 1)), axis=0).astype(np.float32)

        # 推理
        outputs = self._session.run(['output0'], {'images': input_tensor})
        output = np.squeeze(outputs[0], axis=0)  # [9, 1344]

        # 解析: ch0-3 bbox, ch4-8 cls(已sigmoid)
        cx = output[0, :]; cy = output[1, :]
        w = output[2, :]; h = output[3, :]
        cls_scores = output[4:9, :]  # [5, 1344]
        conf = np.max(cls_scores, axis=0)
        cls_ids = np.argmax(cls_scores, axis=0)

        mask = conf >= self.CONFIDENCE_THRESHOLD
        if not np.any(mask):
            return []

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        conf = conf[mask]; cls_ids = cls_ids[mask]

        # 坐标映射: 256 空间 → 原图 (1:1, 无缩放)
        off_x, off_y = self._crop_offset
        cx_img = cx + off_x
        cy_img = cy + off_y
        # w, h 在 256 空间 = 像素尺寸 (1:1 mapping)
        w_img = w; h_img = h

        x1 = (cx_img - w_img / 2).clip(0, orig_w)
        y1 = (cy_img - h_img / 2).clip(0, orig_h)
        x2 = (cx_img + w_img / 2).clip(0, orig_w)
        y2 = (cy_img + h_img / 2).clip(0, orig_h)

        # NMS
        indices = np.argsort(-conf)
        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            if len(indices) == 1:
                break
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_j = (x2[indices[1:]] - x1[indices[1:]]) * (y2[indices[1:]] - y1[indices[1:]])
            iou = inter / (area_i + area_j - inter + 1e-6)
            indices = indices[1:][iou < self.NMS_IOU_THRESHOLD]
            if len(keep) >= self.MAX_DETECTIONS:
                break

        detections = [Detection.from_bbox(
            x=float(x1[i]), y=float(y1[i]),
            w=float(x2[i] - x1[i]), h=float(y2[i] - y1[i]),
            confidence=float(conf[i]), class_id=int(cls_ids[i]),
        ) for i in keep]

        detections.sort(key=lambda d: d.confidence, reverse=True)

        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        self._debug_count += 1
        if self._debug_count % 30 == 0:
            print(f'[TIMING] infer: {self.avg_inference_time_ms:.1f}ms | dets={len(detections)}')

        return detections[:self.MAX_DETECTIONS]
