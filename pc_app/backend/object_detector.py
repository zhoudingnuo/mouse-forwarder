"""
object_detector.py - ONNX 目标检测模块

加载 Valorant 目标检测 ONNX 模型, 对采集卡帧进行推理,
输出检测到的目标边界框。

模型信息:
- 输入: [1, 3, 256, 256] (RGB, normalized)
- 输出: [1, 9, 1344] (检测结果)
- 每个候选框 9 个值: [cx, cy, w, h, conf, cls1, cls2, cls3, cls4]
- 共 1344 个候选框

注意: 此模块提供框架, 具体后处理逻辑需要根据模型输出格式调整。
"""

import logging
import time
from typing import List, Optional

import numpy as np

from trajectory_calculator import Detection

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    ONNX 目标检测器

    加载并运行 Valorant 目标检测模型,
    输出检测到的目标列表。
    """

    # 模型输入参数
    INPUT_NAME = "images"
    OUTPUT_NAME = "output0"
    INPUT_SIZE = 256  # 模型输入尺寸 256x256
    INPUT_SHAPE = (1, 3, INPUT_SIZE, INPUT_SIZE)

    # 后处理参数
    CONFIDENCE_THRESHOLD = 0.30
    NMS_IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 50

    def __init__(self):
        self._session = None
        self._model_path = None
        self._input_shape = None
        self._loaded = False
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
        加载 ONNX 模型

        Args:
            model_path: ONNX 模型文件路径
        """
        import onnxruntime

        logger.info(f"Loading ONNX model: {model_path}")

        # 优先使用 DirectML / CUDA 提供程序
        available_providers = onnxruntime.get_available_providers()
        preferred_providers = ['DmlExecutionProvider', 'CUDAExecutionProvider',
                               'CPUExecutionProvider']

        providers = [p for p in preferred_providers if p in available_providers]

        logger.info(f"Available providers: {available_providers}")
        logger.info(f"Using providers: {providers}")

        self._session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self._model_path = model_path
        self._input_shape = self.INPUT_SHAPE
        self._loaded = True

        # 打印模型信息
        for inp in self._session.get_inputs():
            logger.info(f"  Input: {inp.name} -> {inp.shape}")
        for out in self._session.get_outputs():
            logger.info(f"  Output: {out.name} -> {out.shape}")

        logger.info("Model loaded successfully")

    def detect(self, frame: np.ndarray, original_shape: Optional[tuple] = None) -> List[Detection]:
        """
        对帧进行目标检测

        Args:
            frame: 输入帧 (BGR, HWC)
            original_shape: 原始帧尺寸 (H, W), 用于坐标映射

        Returns:
            检测结果列表
        """
        if not self._loaded or self._session is None:
            logger.warning("Model not loaded")
            return []

        orig_h, orig_w = original_shape or frame.shape[:2]
        t_start = time.perf_counter()

        # 1. 预处理
        input_tensor = self._preprocess(frame)

        # 2. 推理
        outputs = self._session.run(
            [self.OUTPUT_NAME],
            {self.INPUT_NAME: input_tensor},
        )

        t_infer = time.perf_counter()
        self._inference_time += (t_infer - t_start)
        self._inference_count += 1

        # 3. 后处理
        detections = self._postprocess(
            outputs[0],  # [1, 9, 1344]
            orig_w, orig_h,
        )

        logger.debug(f"Inference: {len(detections)} detections in "
                      f"{(t_infer - t_start) * 1000:.1f}ms")

        return detections

    # ============ 预处理 ============

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理帧用于模型推理

        BGR → RGB → Resize 到 256x256 → Normalize → CHW → Batch
        """
        import cv2

        # BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize 到 256x256
        resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)

        # Normalize 到 [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # HWC → CHW
        chw = np.transpose(normalized, (2, 0, 1))

        # 添加 batch 维度
        batch = np.expand_dims(chw, axis=0).astype(np.float32)

        return batch

    # ============ 后处理 ============

    def _postprocess(self, output: np.ndarray, orig_w: int, orig_h: int) -> List[Detection]:
        """
        解析模型输出

        output shape: [1, 9, 1344]
        每个候选框 9 个值: [cx, cy, w, h, conf, cls1, cls2, cls3, cls4]

        Args:
            output: 模型输出张量
            orig_w: 原始帧宽度
            orig_h: 原始帧高度

        Returns:
            检测结果列表
        """
        # 移除 batch 维度 → [9, 1344]
        scores = np.squeeze(output, axis=0)

        # 解析: [cx, cy, w, h, conf, cls1, cls2, cls3, cls4]
        cx = scores[0, :]     # 中心 X (归一化到 0-1 或网格坐标)
        cy = scores[1, :]     # 中心 Y
        w = scores[2, :]      # 宽度
        h = scores[3, :]      # 高度
        conf = scores[4, :]   # 目标置信度
        cls_scores = scores[5:9, :]  # 4 个类别得分

        # 应用 sigmoid 到置信度
        conf = 1.0 / (1.0 + np.exp(-conf))

        # 获取每个候选框的类别
        cls_ids = np.argmax(cls_scores, axis=0)
        cls_conf = 1.0 / (1.0 + np.exp(-np.max(cls_scores, axis=0)))

        # 最终置信度 = 目标置信度 * 类别置信度
        final_conf = conf * cls_conf

        # 置信度过滤
        mask = final_conf >= self.CONFIDENCE_THRESHOLD
        if not np.any(mask):
            return []

        cx = cx[mask]
        cy = cy[mask]
        w = w[mask]
        h = h[mask]
        final_conf = final_conf[mask]
        cls_ids = cls_ids[mask]

        # 将坐标从 256x256 映射回原始帧尺寸
        scale_x = orig_w / self.INPUT_SIZE
        scale_y = orig_h / self.INPUT_SIZE

        cx_abs = cx * scale_x
        cy_abs = cy * scale_y
        w_abs = w * scale_x
        h_abs = h * scale_y

        # 计算边界框左上角
        x_abs = cx_abs - w_abs / 2
        y_abs = cy_abs - h_abs / 2

        # 构建检测列表
        detections = []
        for i in range(len(cx_abs)):
            detections.append(Detection.from_bbox(
                x=float(x_abs[i]),
                y=float(y_abs[i]),
                w=float(w_abs[i]),
                h=float(h_abs[i]),
                confidence=float(final_conf[i]),
                class_id=int(cls_ids[i]),
            ))

        # NMS 过滤重叠框
        detections = self._nms(detections)

        # 按置信度排序
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections[:self.MAX_DETECTIONS]

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """
        非极大值抑制 (Non-Maximum Suppression)

        去除重叠度高的检测框
        """
        if not detections:
            return []

        # 按置信度降序排序
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)

            # 过滤与 best 重叠过高的框
            sorted_dets = [
                d for d in sorted_dets
                if self._iou(best, d) < self.NMS_IOU_THRESHOLD
            ]

        return keep

    def _iou(self, a: Detection, b: Detection) -> float:
        """计算两个检测框的 IoU"""
        # 交集
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.w, b.x + b.w)
        y2 = min(a.y + a.h, b.y + b.h)

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h

        # 并集
        area_a = a.w * a.h
        area_b = b.w * b.h
        union = area_a + area_b - inter

        if union == 0:
            return 0.0

        return inter / union