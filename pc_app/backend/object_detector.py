"""
object_detector.py - 目标检测模块

支持双格式:
- .onnx: ONNX Runtime 推理 (自定义模型)
- .pt:   Ultralytics YOLOv8 推理 (keremberke/yolov8n-valorant-detection)
"""

import logging
import os
import time
from typing import List, Optional

import numpy as np

from trajectory_calculator import Detection

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    目标检测器

    自动识别模型格式:
    - *.onnx → ONNX Runtime
    - *.pt   → Ultralytics YOLOv8
    """

    CONFIDENCE_THRESHOLD = 0.30
    NMS_IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 50

    # ONNX 模型参数
    INPUT_NAME = "images"
    OUTPUT_NAME = "output0"
    INPUT_SIZE = 640

    def __init__(self):
        self._session = None
        self._model_path = None
        self._loaded = False
        self._model_type = None  # 'onnx' or 'pt'
        self._yolo_model = None  # ultralytics YOLO 实例
        self._inference_time = 0.0
        self._inference_count = 0

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
        """
        加载模型 (自动识别格式)

        Args:
            model_path: .onnx 或 .pt 文件路径
        """
        ext = os.path.splitext(model_path)[1].lower()
        logger.info(f"Loading model: {model_path} (format: {ext})")

        if ext == '.pt':
            self._load_pt(model_path)
        elif ext == '.onnx':
            self._load_onnx(model_path)
        else:
            raise ValueError(f"Unsupported model format: {ext} (supported: .onnx, .pt)")

        self._model_path = model_path
        self._loaded = True
        logger.info(f"Model loaded ({self._model_type})")

    def _load_onnx(self, model_path: str):
        """加载 ONNX 模型"""
        import onnxruntime

        available = onnxruntime.get_available_providers()
        preferred = ['DmlExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = [p for p in preferred if p in available]

        self._session = onnxruntime.InferenceSession(model_path, providers=providers)
        self._model_type = 'onnx'

        for inp in self._session.get_inputs():
            logger.info(f"  Input: {inp.name} -> {inp.shape}")
        for out in self._session.get_outputs():
            logger.info(f"  Output: {out.name} -> {out.shape}")

    def _load_pt(self, model_path: str):
        """加载 YOLOv8 PyTorch 模型"""
        from ultralytics import YOLO

        self._yolo_model = YOLO(model_path)

        # 如果设备支持, 用 GPU
        import torch
        if torch.cuda.is_available():
            self._yolo_model.to('cuda')
            logger.info("YOLO model loaded on CUDA")
        else:
            logger.info("YOLO model loaded on CPU")

        self._model_type = 'pt'

    def detect(self, frame: np.ndarray, original_shape: Optional[tuple] = None) -> List[Detection]:
        """
        对帧进行目标检测

        Args:
            frame: 输入帧 (BGR, HWC)
            original_shape: 原始帧尺寸 (H, W), 用于坐标映射

        Returns:
            检测结果列表
        """
        if not self._loaded:
            return []

        t_start = time.perf_counter()

        if self._model_type == 'onnx':
            detections = self._detect_onnx(frame, original_shape)
        elif self._model_type == 'pt':
            detections = self._detect_pt(frame, original_shape)
        else:
            return []

        self._inference_time += time.perf_counter() - t_start
        self._inference_count += 1

        return detections

    # ================================================================
    # ONNX 推理
    # ================================================================

    def _detect_onnx(self, frame: np.ndarray, original_shape: Optional[tuple]) -> List[Detection]:
        """ONNX 模型推理 (自动适配 8/9 通道输出)"""
        import cv2

        orig_h, orig_w = original_shape or frame.shape[:2]

        # 预处理
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0).astype(np.float32)

        # 推理
        outputs = self._session.run([self.OUTPUT_NAME], {self.INPUT_NAME: input_tensor})
        output = np.squeeze(outputs[0], axis=0)  # [N, 1344]  N=8 或 9

        num_channels = output.shape[0]
        cx = output[0, :]
        cy = output[1, :]
        w = output[2, :]
        h = output[3, :]

        # 自动适配输出格式:
        # 9 通道: [cx, cy, w, h, objectness, cls1, cls2, cls3, cls4]
        # 8 通道: [cx, cy, w, h, cls1, cls2, cls3, cls4]  (标准 YOLOv8)
        if num_channels == 9:
            conf = 1.0 / (1.0 + np.exp(-output[4, :]))
            cls_scores = output[5:9, :]
        else:
            conf = 1.0  # 无 objectness, 直接用类别分
            cls_scores = output[4:8, :]

        cls_ids = np.argmax(cls_scores, axis=0)
        cls_conf = 1.0 / (1.0 + np.exp(-np.max(cls_scores, axis=0)))
        final_conf = conf * cls_conf

        mask = final_conf >= self.CONFIDENCE_THRESHOLD
        if not np.any(mask):
            return []

        cx = cx[mask]
        cy = cy[mask]
        w = w[mask]
        h = h[mask]
        final_conf = final_conf[mask]
        cls_ids = cls_ids[mask]

        # 映射回原始帧尺寸
        scale_x = orig_w / self.INPUT_SIZE
        scale_y = orig_h / self.INPUT_SIZE

        detections = []
        for i in range(len(cx)):
            x = (cx[i] - w[i] / 2) * scale_x
            y = (cy[i] - h[i] / 2) * scale_y
            bw = w[i] * scale_x
            bh = h[i] * scale_y
            detections.append(Detection.from_bbox(
                x=float(x), y=float(y), w=float(bw), h=float(bh),
                confidence=float(final_conf[i]),
                class_id=int(cls_ids[i]),
            ))

        detections = self._nms(detections)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:self.MAX_DETECTIONS]

    # ================================================================
    # YOLOv8 (PyTorch) 推理
    # ================================================================

    def _detect_pt(self, frame: np.ndarray, original_shape: Optional[tuple]) -> List[Detection]:
        """YOLOv8 PyTorch 模型推理"""
        orig_h, orig_w = original_shape or frame.shape[:2]

        # YOLO 推理 (返回 Results 对象列表)
        results = self._yolo_model(frame, verbose=False)

        if not results or len(results) == 0:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections = []
        boxes = result.boxes.xyxy.cpu().numpy()   # [N, 4]  xyxy 格式
        confs = result.boxes.conf.cpu().numpy()    # [N]
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N]

        for i in range(len(boxes)):
            if confs[i] < self.CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = boxes[i]
            w = x2 - x1
            h = y2 - y1

            detections.append(Detection.from_bbox(
                x=float(x1), y=float(y1), w=float(w), h=float(h),
                confidence=float(confs[i]),
                class_id=int(cls_ids[i]),
            ))

        # YOLO 自带 NMS, 但再加一道保险
        detections = self._nms(detections)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:self.MAX_DETECTIONS]

    # ================================================================
    # 通用后处理
    # ================================================================

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """非极大值抑制"""
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)
            sorted_dets = [d for d in sorted_dets if self._iou(best, d) < self.NMS_IOU_THRESHOLD]

        return keep

    def _iou(self, a: Detection, b: Detection) -> float:
        """计算两个检测框的 IoU"""
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.w, b.x + b.w)
        y2 = min(a.y + a.h, b.y + b.h)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = a.w * a.h
        area_b = b.w * b.h
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0