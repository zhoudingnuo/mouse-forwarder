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
    INPUT_SIZE = 640  # 模型内部处理尺寸 (实际输入为 1080x1920, 由模型内置 resize 处理)

    def __init__(self):
        self._session = None
        self._model_path = None
        self._loaded = False
        self._model_type = None  # 'onnx' or 'pt'
        self._yolo_model = None  # ultralytics YOLO 实例
        self._inference_time = 0.0
        self._inference_count = 0
        self._last_crop_offset = (0, 0)  # 中心裁剪偏移 (x, y)
        self._preproc_buffer = None  # 预分配预处理 buffer

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
        """ONNX 模型推理 (中心裁剪 640×640, 无需 resize)"""
        import time

        orig_h, orig_w = original_shape or frame.shape[:2]

        # 中心裁剪 640×640 + HWC→CHW + normalize (预分配 buffer, 避免重复分配)
        t0 = time.perf_counter()
        crop_size = self.INPUT_SIZE
        cx, cy = orig_w // 2, orig_h // 2
        x1 = cx - crop_size // 2
        y1 = cy - crop_size // 2
        crop = frame[y1:y1 + crop_size, x1:x1 + crop_size]  # [640, 640, 3] BGR
        # 预分配 buffer, 复用 (640×640×3 × float32 = 4.9MB)
        if self._preproc_buffer is None or self._preproc_buffer.shape[1] != crop_size:
            self._preproc_buffer = np.empty((1, 3, crop_size, crop_size), dtype=np.float32)
        buf = self._preproc_buffer
        scale = np.float32(1.0 / 255.0)
        # BGR→RGB + HWC→CHW + float32/255 写入预分配 buffer
        buf[0, 0] = crop[:, :, 2] * scale  # R
        buf[0, 1] = crop[:, :, 1] * scale  # G
        buf[0, 2] = crop[:, :, 0] * scale  # B
        input_tensor = buf
        self._last_crop_offset = (x1, y1)
        t1 = time.perf_counter()

        # DML 推理 (~8ms)
        outputs = self._session.run([self.OUTPUT_NAME], {self.INPUT_NAME: input_tensor})
        output = np.squeeze(outputs[0], axis=0)
        t2 = time.perf_counter()

        num_channels = output.shape[0]
        cx = output[0, :]
        cy = output[1, :]
        w = output[2, :]
        h = output[3, :]

        # 自动适配输出格式
        if num_channels == 9:
            conf = 1.0 / (1.0 + np.exp(-output[4, :]))
            cls_scores = output[5:9, :]
        else:
            conf = 1.0
            cls_scores = output[4:8, :]

        cls_ids = np.argmax(cls_scores, axis=0)
        cls_conf = 1.0 / (1.0 + np.exp(-np.max(cls_scores, axis=0)))
        final_conf = conf * cls_conf

        mask = final_conf >= self.CONFIDENCE_THRESHOLD
        if not np.any(mask):
            self._debug_count = getattr(self, '_debug_count', 0) + 1
            if self._debug_count % 30 == 0:
                pre_ms = (t1 - t0) * 1000
                inf_ms = (t2 - t1) * 1000
                print(f'[TIMING] pre={pre_ms:.1f}ms | infer={inf_ms:.1f}ms | post=0ms | total={(t2-t0)*1000:.1f}ms | dets=0')
            return []

        cx = cx[mask]
        cy = cy[mask]
        w = w[mask]
        h = h[mask]
        final_conf = final_conf[mask]
        cls_ids = cls_ids[mask]

        # 映射回原始帧尺寸 (中心裁剪, 加上裁剪偏移)
        off_x, off_y = self._last_crop_offset
        x1 = off_x + (cx - w / 2)
        y1 = off_y + (cy - h / 2)
        x2 = off_x + (cx + w / 2)
        y2 = off_y + (cy + h / 2)

        # 快速 NMS (向量化计算)
        indices = np.argsort(-final_conf)
        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            if len(indices) == 1:
                break
            # 计算 IOU 矩阵
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

        detections = []
        for i in keep:
            detections.append(Detection.from_bbox(
                x=float(x1[i]), y=float(y1[i]),
                w=float(x2[i] - x1[i]), h=float(y2[i] - y1[i]),
                confidence=float(final_conf[i]),
                class_id=int(cls_ids[i]),
            ))
        t3 = time.perf_counter()

        detections.sort(key=lambda d: d.confidence, reverse=True)

        # 每 30 帧打印一次各阶段耗时
        self._debug_count = getattr(self, '_debug_count', 0) + 1
        if self._debug_count % 30 == 0:
            pre_ms = (t1 - t0) * 1000
            inf_ms = (t2 - t1) * 1000
            post_ms = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000
            print(f'[TIMING] pre={pre_ms:.1f}ms | infer={inf_ms:.1f}ms | post={post_ms:.1f}ms | total={total_ms:.1f}ms | dets={len(detections)}/{len(keep)}')

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