"""
capture_inference.py - 采集卡视频捕获 + ONNX 推理引擎

功能:
1. 自动发现 MS2130 采集卡摄像头设备
2. 实时视频帧捕获 (OpenCV)
3. 基于 valorant.onnx 的 YOLO 目标检测推理
4. NMS 后处理, 结果通过回调发送

输出格式 (YOLO): [1, 9, 1344]
  - 行 0-3: cx, cy, w, h (归一化到 0-256)
  - 行 4: objectness (置信度)
  - 行 5-8: 4 个类别的概率
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger('capture')

# 模型参数
MODEL_PATH = 'valorant.onnx'  # 相对于项目根目录
INPUT_SIZE = 256  # 模型输入尺寸 256x256
CONF_THRESHOLD = 0.25  # 置信度阈值
NMS_THRESHOLD = 0.45  # NMS IoU 阈值

# 类别名称 (4 类, 根据 Valorant 场景)
CLASS_NAMES = ['head', 'body', 'weapon', 'unknown']

# 类别颜色 (BGR)
CLASS_COLORS = [
    (0, 255, 0),    # head - 绿色
    (0, 200, 255),  # body - 橙色
    (255, 100, 0),  # weapon - 蓝色
    (200, 200, 200),# unknown - 灰色
]


@dataclass
class Detection:
    """单个检测结果"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ''

    def __post_init__(self):
        self.class_name = CLASS_NAMES[self.class_id] if 0 <= self.class_id < len(CLASS_NAMES) else 'unknown'

    def to_dict(self, img_w: int, img_h: int) -> dict:
        """转换为可序列化的字典 (像素坐标)"""
        return {
            'x1': round(self.x1 * img_w),
            'y1': round(self.y1 * img_h),
            'x2': round(self.x2 * img_w),
            'y2': round(self.y2 * img_h),
            'confidence': round(float(self.confidence), 4),
            'class_id': self.class_id,
            'class_name': self.class_name,
        }


@dataclass
class CaptureResult:
    """一次捕获+推理的结果"""
    detections: list = field(default_factory=list)
    fps: float = 0.0
    frame_jpeg: Optional[bytes] = None  # JPEG 压缩帧用于前端显示


class CaptureInferenceEngine:
    """
    采集卡捕获 + 推理引擎

    在独立线程中运行, 通过回调将结果传回主事件循环。
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        camera_index: Optional[int] = None,
        on_result: Optional[Callable] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.model_path = model_path
        self.camera_index = camera_index  # None = 自动检测
        self.on_result = on_result  # async callback: on_result(result: CaptureResult)
        self.loop = loop

        # 状态
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._session = None
        self._thread = None
        self._fps_counter = 0
        self._fps_timer = 0.0
        self._current_fps = 0.0

        # 发送帧到前端的间隔 (降低带宽)
        self._frame_interval = 0.1  # 每 100ms 发一帧
        self._last_frame_time = 0.0

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载 ONNX 模型"""
        import onnxruntime as ort

        logger.info(f"Loading model: {self.model_path}")
        try:
            self._session = ort.InferenceSession(
                self.model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"CUDA not available, falling back to CPU: {e}")
            self._session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            logger.info("Model loaded on CPU")

    def find_camera(self) -> int:
        """
        自动发现 MS2130 采集卡摄像头

        策略:
        1. 尝试通过名称匹配 (DShow 后端)
        2. 回退: 遍历索引, 找到非集成摄像头的高分辨率设备
        """
        if self.camera_index is not None:
            return self.camera_index

        logger.info("Auto-detecting MS2130 capture card...")

        # 尝试用 DShow 枚举所有设备
        found_indices = []
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    # 尝试读取一帧
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        # 获取设备后端名称 (仅 DShow 有效)
                        backend = cap.getBackendName()
                        logger.info(f"  Camera[{i}]: {w}x{h} backend={backend}")
                        found_indices.append(i)
                    cap.release()
            except Exception:
                continue

        if not found_indices:
            raise RuntimeError("未找到任何摄像头设备")

        # 如果有多个摄像头, 优先选择非 0 索引 (通常 0 是集成摄像头)
        # 或者尝试设置高分辨率来识别 MS2130
        ms2130_index = None
        for i in found_indices:
            if i == 0:
                continue  # 跳过索引 0, 通常是集成摄像头
            # 尝试打开并设为 1080p, MS2130 支持
            try:
                test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if test_cap.isOpened():
                    test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        if w >= 1280:
                            ms2130_index = i
                            test_cap.release()
                            break
                    test_cap.release()
            except Exception:
                continue

        if ms2130_index is not None:
            logger.info(f"MS2130 capture card detected at index {ms2130_index}")
            self.camera_index = ms2130_index
            return ms2130_index

        # 回退: 使用找到的第一个非 0 索引
        for i in found_indices:
            if i != 0:
                logger.info(f"Using camera index {i} (non-default)")
                self.camera_index = i
                return i

        # 最后的回退
        logger.info(f"Using camera index {found_indices[0]}")
        self.camera_index = found_indices[0]
        return found_indices[0]

    def open_camera(self, preferred_width=1920, preferred_height=1080) -> cv2.VideoCapture:
        """
        打开采集卡摄像头

        先尝试 1080p, 不满足则降级到 720p。
        """
        idx = self.find_camera()
        logger.info(f"Opening camera [{idx}] with preferred {preferred_width}x{preferred_height}")

        # 用 DShow 后端打开 (Windows 低延迟)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 [{idx}]")

        # 尝试设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_height)
        cap.set(cv2.CAP_PROP_FPS, 60)

        # 等待几帧稳定
        for _ in range(5):
            cap.read()

        # 读取实际分辨率
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 如果达不到目标分辨率, 降级到 720p
        if w < preferred_width or h < preferred_height:
            logger.info(f"Target resolution not available (got {w:.0f}x{h:.0f}), falling back to 720p...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _ in range(5):
                cap.read()
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: actual {w:.0f}x{h:.0f} @ {actual_fps:.1f}fps")

        return cap

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理帧用于模型推理

        1. 保持宽高比的 resize + padding 到 256x256
        2. 归一化到 [0, 1]
        3. 转换为 CHW 格式
        """
        h, w = frame.shape[:2]

        # 计算缩放比例, 保持宽高比
        scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # padding 到正方形
        pad_h = INPUT_SIZE - new_h
        pad_w = INPUT_SIZE - new_w
        padded = cv2.copyMakeBorder(
            resized,
            pad_h // 2, pad_h - pad_h // 2,
            pad_w // 2, pad_w - pad_w // 2,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # BGR -> RGB, 归一化, CHW
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
        chw = np.expand_dims(chw, axis=0)   # -> NCHW

        return chw.astype(np.float32)

    def postprocess(
        self,
        output: np.ndarray,
        frame_h: int,
        frame_w: int,
    ) -> list:
        """
        后处理 YOLO 输出

        Args:
            output: [1, 9, 1344] - YOLO 原始输出
            frame_h: 原始帧高度
            frame_w: 原始帧宽度

        Returns:
            list[Detection]: 过滤和 NMS 后的检测结果
        """
        # 取 batch 0
        pred = output[0]  # [9, 1344]

        # 转置: [1344, 9]
        pred = pred.T

        # 计算缩放比例 (与 preprocess 一致)
        scale = min(INPUT_SIZE / frame_h, INPUT_SIZE / frame_w)
        new_h, new_w = int(frame_h * scale), int(frame_w * scale)
        pad_h = INPUT_SIZE - new_h
        pad_w = INPUT_SIZE - new_w

        # 从 padding 坐标系转换回原始图像坐标系
        cx = (pred[:, 0] - pad_w / 2) / scale
        cy = (pred[:, 1] - pad_h / 2) / scale
        w = pred[:, 2] / scale
        h = pred[:, 3] / scale

        # 转为 xyxy 格式
        x1 = (cx - w / 2).clip(0, frame_w)
        y1 = (cy - h / 2).clip(0, frame_h)
        x2 = (cx + w / 2).clip(0, frame_w)
        y2 = (cy + h / 2).clip(0, frame_h)

        # 置信度
        obj_conf = pred[:, 4]

        # 类别概率 (5 个 softmax 类别)
        cls_probs = pred[:, 5:9]
        cls_ids = np.argmax(cls_probs, axis=1)
        cls_scores = np.max(cls_probs, axis=1)

        # 综合置信度 = objectness * class_score
        scores = obj_conf * cls_scores

        # 过滤低置信度
        mask = scores > CONF_THRESHOLD
        if not np.any(mask):
            return []

        x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        # 归一化到 [0, 1] 用于 to_dict
        x1_norm = x1 / frame_w
        y1_norm = y1 / frame_h
        x2_norm = x2 / frame_w
        y2_norm = y2 / frame_h

        # NMS
        detections = []
        # 按类别分组 NMS
        unique_cls = np.unique(cls_ids)
        for cls_id in unique_cls:
            cls_mask = cls_ids == cls_id
            cls_x1 = x1_norm[cls_mask]
            cls_y1 = y1_norm[cls_mask]
            cls_x2 = x2_norm[cls_mask]
            cls_y2 = y2_norm[cls_mask]
            cls_scores = scores[cls_mask]

            # 按置信度排序
            order = np.argsort(-cls_scores)
            keep = []

            while len(order) > 0:
                i = order[0]
                keep.append(i)
                if len(order) == 1:
                    break

                # 计算 IoU
                xx1 = np.maximum(cls_x1[i], cls_x1[order[1:]])
                yy1 = np.maximum(cls_y1[i], cls_y1[order[1:]])
                xx2 = np.minimum(cls_x2[i], cls_x2[order[1:]])
                yy2 = np.minimum(cls_y2[i], cls_y2[order[1:]])

                w_int = np.maximum(0, xx2 - xx1)
                h_int = np.maximum(0, yy2 - yy1)
                area_int = w_int * h_int

                area_i = (cls_x2[i] - cls_x1[i]) * (cls_y2[i] - cls_y1[i])
                area_j = (cls_x2[order[1:]] - cls_x1[order[1:]]) * (cls_y2[order[1:]] - cls_y1[order[1:]])
                iou = area_int / (area_i + area_j - area_int + 1e-7)

                order = order[1:][iou < NMS_THRESHOLD]

            for idx in keep:
                detections.append(Detection(
                    x1=float(cls_x1[idx]),
                    y1=float(cls_y1[idx]),
                    x2=float(cls_x2[idx]),
                    y2=float(cls_y2[idx]),
                    confidence=float(cls_scores[idx]),
                    class_id=int(cls_id),
                ))

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """在帧上绘制检测框"""
        h, w = frame.shape[:2]
        for det in detections:
            d = det.to_dict(w, h)
            color = CLASS_COLORS[det.class_id] if det.class_id < len(CLASS_COLORS) else (200, 200, 200)
            x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制标签背景
            label = f"{det.class_name} {det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 6), (x1 + label_w + 6, y1), color, -1)

            # 绘制标签文字
            cv2.putText(
                frame, label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

        # 绘制 FPS
        fps_text = f"FPS: {self._current_fps:.1f}"
        cv2.putText(
            frame, fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2, cv2.LINE_AA
        )

        return frame

    def _run_capture_loop(self):
        """
        捕获+推理主循环 (在独立线程中运行)
        """
        logger.info("Capture loop started")
        self._running = True

        try:
            self._cap = self.open_camera()
        except Exception as e:
            logger.error(f"Failed to open camera: {e}")
            self._running = False
            return

        self._fps_timer = time.time()
        self._fps_counter = 0
        self._current_fps = 0.0
        self._last_frame_time = 0.0

        while self._running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame")
                time.sleep(0.01)
                continue

            self._fps_counter += 1
            now = time.time()

            # 每秒更新 FPS
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._current_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_timer = now

            # 推理
            try:
                input_tensor = self.preprocess(frame)
                output = self._session.run(['output0'], {'images': input_tensor})[0]
                h, w = frame.shape[:2]
                detections = self.postprocess(output, h, w)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                detections = []

            # 绘制检测框
            annotated = self.draw_detections(frame, detections)

            # 按间隔压缩帧发送到前端
            send_frame = False
            if now - self._last_frame_time >= self._frame_interval:
                self._last_frame_time = now
                send_frame = True

            frame_jpeg = None
            if send_frame:
                # JPEG 压缩
                ret_jpg, buf = cv2.imencode('.jpg', annotated, [
                    cv2.IMWRITE_JPEG_QUALITY, 70
                ])
                if ret_jpg:
                    frame_jpeg = buf.tobytes()

            # 回调通知
            if self.on_result and self.loop:
                result = CaptureResult(
                    detections=detections,
                    fps=self._current_fps,
                    frame_jpeg=frame_jpeg,
                )
                asyncio.run_coroutine_threadsafe(
                    self.on_result(result),
                    self.loop
                )

        # 清理
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Capture loop stopped")

    def start(self, loop: asyncio.AbstractEventLoop):
        """
        在后台线程启动捕获循环

        Args:
            loop: 主事件循环引用 (用于回调)
        """
        if self._running:
            logger.warning("Capture already running")
            return

        self.loop = loop
        import threading
        self._thread = threading.Thread(
            target=self._run_capture_loop,
            daemon=True,
            name='capture-inference',
        )
        self._thread.start()
        logger.info("Capture engine started")

    def stop(self):
        """停止捕获"""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Capture engine stopped")

    @property
    def is_running(self) -> bool:
        return self._running


# ============================================================
# 便捷工具函数
# ============================================================

def draw_detections_on_frame(frame: np.ndarray, detections: list, fps: float = 0.0) -> np.ndarray:
    """
    在帧上绘制检测框 (独立函数, 供外部调用)

    支持两种 Detection 格式:
    - capture_inference.Detection (有 to_dict 方法)
    - trajectory_calculator.Detection (有 x, y, w, h, cx, cy, confidence, class_id 属性)

    Args:
        frame: 原始 BGR 帧
        detections: Detection 对象列表
        fps: 当前 FPS 显示

    Returns:
        带标注的帧
    """
    h, w = frame.shape[:2]
    for det in detections:
        color = CLASS_COLORS[det.class_id] if det.class_id < len(CLASS_COLORS) else (200, 200, 200)

        # 兼容两种 Detection 格式
        if hasattr(det, 'to_dict'):
            # capture_inference.Detection 格式
            d = det.to_dict(w, h)
            x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
            class_name = det.class_name
            confidence = det.confidence
        else:
            # trajectory_calculator.Detection 格式 (x, y, w, h, cx, cy, confidence, class_id)
            x1 = int(det.x)
            y1 = int(det.y)
            x2 = int(det.x + det.w)
            y2 = int(det.y + det.h)
            class_name = CLASS_NAMES[det.class_id] if det.class_id < len(CLASS_NAMES) else 'unknown'
            confidence = det.confidence

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = f"{class_name} {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_h - 6), (x1 + label_w + 6, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    # 绘制 FPS 或推理时间
    if fps > 0:
        info_text = f"FPS: {fps:.1f}" if fps < 100 else f"Inference: {fps:.1f}ms"
        cv2.putText(
            frame, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )

    return frame


def encode_frame_jpeg(frame: np.ndarray, quality: int = 70) -> bytes:
    """将帧压缩为 JPEG 字节"""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def list_cameras() -> list:
    """
    列出所有可用摄像头

    Returns:
        list[dict]: 每个摄像头的信息
    """
    cameras = []
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    cameras.append({
                        'index': i,
                        'width': w,
                        'height': h,
                    })
                cap.release()
        except Exception:
            continue
    return cameras