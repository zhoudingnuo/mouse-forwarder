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
    """目标检测器 (支持任意 ONNX YOLO 模型, 动态读取输入尺寸/类别)"""

    CONFIDENCE_THRESHOLD = 0.25
    NMS_IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 50

    INPUT_SIZE = 256          # 模型输入尺寸 (加载时从模型覆盖)
    MODEL_PATH = 'valorant.onnx'

    # 默认类别名称 (加载时从模型 metadata 覆盖)
    CLASS_NAMES = ['body', 'head', 'teammate', 'breakable', 'dodge']

    def __init__(self):
        self._session = None
        self._model_path = None
        self._loaded = False
        self._inference_time = 0.0
        self._inference_count = 0
        self._crop_offset = (0, 0)
        self._num_classes = len(self.CLASS_NAMES)
        # 中心裁剪尺寸 (0 = 默认 INPUT_SIZE, 即裁剪 256×256 不缩放)
        # >256 时先中心裁剪 N×N 再 resize 到 256×256 喂模型 (视野更大, 目标更小)
        # 坐标映射会按比例缩回原图
        self.center_crop_size = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_path(self) -> Optional[str]:
        return self._model_path

    @property
    def class_names(self) -> list:
        return self.CLASS_NAMES

    @property
    def input_size(self) -> int:
        return self.INPUT_SIZE

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

        # 动态适配模型参数 (输入尺寸 / 类别数 / 类别名)
        try:
            # 1. 输入尺寸: [1, 3, H, W]
            in_shape = self._session.get_inputs()[0].shape
            if len(in_shape) == 4 and in_shape[2] and in_shape[3]:
                self.INPUT_SIZE = int(in_shape[2])
                logger.info(f"  Input size: {self.INPUT_SIZE}x{self.INPUT_SIZE}")

            # 2. 类别数: 输出通道数 - 4 (cx, cy, w, h)
            out_shape = self._session.get_outputs()[0].shape
            if len(out_shape) == 3:
                channels = int(out_shape[1])
                self._num_classes = channels - 4
                logger.info(f"  Classes: {self._num_classes}")

            # 3. 类别名: 从模型 metadata 读取
            import onnx
            onnx_model = onnx.load(model_path)
            for entry in onnx_model.metadata_props:
                if entry.key == 'names':
                    names = eval(entry.value)  # {0: 'name', 1: 'name', ...}
                    if isinstance(names, dict):
                        sorted_names = [names[k] for k in sorted(names.keys())]
                        if len(sorted_names) == self._num_classes:
                            self.CLASS_NAMES = sorted_names
                            logger.info(f"  Class names: {self.CLASS_NAMES}")
                    break
        except Exception as e:
            logger.warning(f"Model param detection failed (using defaults): {e}")

    def detect(self, frame: np.ndarray, original_shape: Optional[tuple] = None) -> List[Detection]:
        if not self._loaded:
            return []
        t_start = time.perf_counter()
        detections = self._detect_onnx(frame, original_shape)
        self._inference_time += time.perf_counter() - t_start
        self._inference_count += 1
        return detections

    def _detect_onnx(self, frame: np.ndarray, original_shape: Optional[tuple]) -> List[Detection]:
        """ONNX 推理: 中心裁剪 (可调尺寸) + 缩放 256×256"""
        orig_h, orig_w = original_shape or frame.shape[:2]
        SZ = self.INPUT_SIZE
        # 裁剪边长: 0=默认 SZ; 否则用配置的 center_crop_size (裁剪后 resize 回 SZ)
        crop_len = self.center_crop_size if self.center_crop_size > 0 else SZ

        # 中心裁剪 crop_len×crop_len
        cx, cy = orig_w // 2, orig_h // 2
        half = crop_len // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(orig_w, x1 + crop_len)
        y2 = min(orig_h, y1 + crop_len)
        crop = frame[y1:y2, x1:x2]
        # 补齐边缘 (如果画面边缘不足)
        ch, cw = crop.shape[:2]
        if ch != crop_len or cw != crop_len:
            crop = cv2.copyMakeBorder(crop, 0, crop_len - ch, 0, crop_len - cw,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self._crop_offset = (x1, y1)

        # 裁剪区域 != 模型输入尺寸时 resize 到 SZ (视野缩放)
        # resize 缩放因子: 原图像素 / 模型像素
        self._crop_scale = crop_len / SZ
        if crop_len != SZ:
            crop = cv2.resize(crop, (SZ, SZ), interpolation=cv2.INTER_LINEAR)

        # BGR→RGB + 单次 float32 转换 (复用 dst 缓冲, 避免多次分配)
        # 原实现 cvtColor→astype(float32)→transpose 三次分配; 这里合并为两次
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # HWC→CHW + /255 单步完成 (np 原地转置后除, 减少一次拷贝)
        input_tensor = np.transpose(rgb, (2, 0, 1))
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32) / 255.0
        input_tensor = input_tensor[np.newaxis, ...]

        # 推理
        outputs = self._session.run(['output0'], {'images': input_tensor})
        output = np.squeeze(outputs[0], axis=0)  # [4+num_cls, anchors]

        # 解析: ch0-3 bbox, ch4+ cls
        cx = output[0, :]; cy = output[1, :]
        w = output[2, :]; h = output[3, :]
        cls_scores = output[4:4 + self._num_classes, :]
        conf = np.max(cls_scores, axis=0)
        cls_ids = np.argmax(cls_scores, axis=0)

        mask = conf >= self.CONFIDENCE_THRESHOLD
        if not np.any(mask):
            return []

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        conf = conf[mask]; cls_ids = cls_ids[mask]

        # 坐标映射: 256 空间 → 原图
        # 1) 模型输出在 SZ 空间 → 乘 crop_scale 回到裁剪区 (crop_len) 空间
        # 2) 加裁剪偏移 (x1, y1) → 原图空间
        scale = self._crop_scale
        off_x, off_y = self._crop_offset
        cx_img = cx * scale + off_x
        cy_img = cy * scale + off_y
        w_img = w * scale
        h_img = h * scale

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

        return detections[:self.MAX_DETECTIONS]
