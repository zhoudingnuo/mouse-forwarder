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
import colorsys
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger('capture')

# 模型参数
MODEL_PATH = 'valorant.onnx'  # 默认模型
INPUT_SIZE = 256              # 模型输入尺寸 256x256
CROP_SIZE = 640               # 中心裁剪尺寸
CONF_THRESHOLD = 0.25         # 置信度阈值
NMS_THRESHOLD = 0.45          # NMS IoU 阈值

# 类别名称: 从 ONNX 模型 metadata 读取, 不存在则用默认值
_MODEL_PATH = None
for _p in [
    os.path.join(os.path.dirname(__file__), '..', '..', 'valorant.onnx'),
    os.path.join(os.path.dirname(__file__), '..', '..', 'best.onnx'),
    os.path.join(os.path.dirname(__file__), '..', 'valorant.onnx'),
    os.path.join(os.path.dirname(__file__), '..', 'best.onnx'),
    os.path.join(os.path.dirname(__file__), 'valorant.onnx'),
    os.path.join(os.path.dirname(__file__), 'best.onnx'),
]:
    if os.path.exists(_p):
        _MODEL_PATH = _p
        break


def _load_class_names_from_onnx(onnx_path: str) -> list:
    """从 ONNX 模型 metadata 读取类别名称"""
    try:
        import onnx
        model = onnx.load(onnx_path)
        for p in model.metadata_props:
            if p.key == "classes":
                return [n.strip() for n in p.value.split(",")]
            if p.key == "names":
                import ast
                try:
                    names_dict = ast.literal_eval(p.value)
                    if isinstance(names_dict, dict):
                        return [names_dict[i] for i in sorted(names_dict.keys())]
                except Exception:
                    pass
    except Exception:
        pass
    return ['body', 'head', 'teammate', 'breakable', 'dodge']


CLASS_NAMES = _load_class_names_from_onnx(_MODEL_PATH) if os.path.exists(_MODEL_PATH) else ['body', 'head', 'teammate', 'breakable', 'dodge']


def update_class_names(model_path: str):
    """更新类别名称 (在模型加载后调用, 确保与实际模型匹配)"""
    global CLASS_NAMES, CLASS_COLORS
    new_names = _load_class_names_from_onnx(model_path)
    if new_names and len(new_names) > 0:
        CLASS_NAMES = new_names
        # 重新生成颜色
        CLASS_COLORS.clear()
        for i in range(len(CLASS_NAMES)):
            hue = i / len(CLASS_NAMES)
            r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.8)
            CLASS_COLORS.append((int(b * 255), int(g * 255), int(r * 255)))


# 类别颜色 (BGR) - 动态生成
CLASS_COLORS = []
for i in range(len(CLASS_NAMES)):
    hue = i / len(CLASS_NAMES)
    r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.8)
CLASS_COLORS.append((int(b * 255), int(g * 255), int(r * 255)))


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
    inf_fps: float = 0.0  # 推理 FPS
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
        self._inf_counter = 0  # 推理帧数计数
        self._inf_timer = 0.0  # 推理计时器
        self._current_inf_fps = 0.0  # 当前推理 FPS

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

    def preprocess(self, frame: np.ndarray):
        """
        预处理帧用于模型推理

        中心裁剪 640×640 → resize 到 256×256 + /255 归一化
        返回 (input_tensor, crop_offset)
        """
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        half = CROP_SIZE // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        crop = frame[y1:y1 + CROP_SIZE, x1:x1 + CROP_SIZE]

        resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.expand_dims(np.transpose(rgb, (2, 0, 1)), axis=0).astype(np.float32)
        return chw, (x1, y1)

    def postprocess(
        self,
        output: np.ndarray,
        frame_h: int,
        frame_w: int,
        crop_offset,
    ) -> list:
        """
        后处理 YOLO 输出

        Args:
            output: [1, 9, 1344] - YOLO 原始输出 (ch0-3 bbox, ch4-8 cls 已 sigmoid)
            frame_h: 原始帧高度
            frame_w: 原始帧宽度
            crop_offset: (x1, y1) 裁剪偏移

        Returns:
            list[Detection]: 过滤和 NMS 后的检测结果
        """
        off_x, off_y = crop_offset
        pred = np.squeeze(output, axis=0)  # [9, 1344]
        pred = pred.T  # [1344, 9]

        # bbox: 模型输出在 256×256 空间 → 640×640 裁剪空间 → 原图
        scale_crop = CROP_SIZE / INPUT_SIZE  # 2.5
        cx = pred[:, 0] * scale_crop + off_x
        cy = pred[:, 1] * scale_crop + off_y
        w = pred[:, 2] * scale_crop
        h = pred[:, 3] * scale_crop

        x1 = (cx - w / 2).clip(0, frame_w)
        y1 = (cy - h / 2).clip(0, frame_h)
        x2 = (cx + w / 2).clip(0, frame_w)
        y2 = (cy + h / 2).clip(0, frame_h)

        # 类别概率: ch4-ch8 已 sigmoid (5 类: body, head, teammate, breakable, dodge)
        cls_scores = pred[:, 4:9]  # [1344, 5]
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = np.max(cls_scores, axis=1)

        # 过滤低置信度
        mask = scores >= CONF_THRESHOLD
        if not np.any(mask):
            return []

        x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        # NMS (按类别分组)
        detections = []
        unique_cls = np.unique(cls_ids)
        for cls_id in unique_cls:
            cls_mask = cls_ids == cls_id
            cls_x1 = x1[cls_mask]
            cls_y1 = y1[cls_mask]
            cls_x2 = x2[cls_mask]
            cls_y2 = y2[cls_mask]
            cls_scores = scores[cls_mask]

            order = np.argsort(-cls_scores)
            keep = []

            while len(order) > 0:
                i = order[0]
                keep.append(i)
                if len(order) == 1:
                    break

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
        self._inf_timer = time.time()
        self._inf_counter = 0
        self._current_inf_fps = 0.0
        self._last_frame_time = 0.0

        while self._running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Failed to read frame")
                time.sleep(0.01)
                continue

            self._fps_counter += 1
            now = time.time()

            # 每秒更新管线 FPS
            elapsed = now - self._fps_timer
            if elapsed >= 1.0:
                self._current_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_timer = now

            # 推理
            try:
                t0 = time.perf_counter()
                input_tensor, crop_offset = self.preprocess(frame)
                output = self._session.run(['output0'], {'images': input_tensor})[0]
                h, w = frame.shape[:2]
                detections = self.postprocess(output, h, w, crop_offset)
                self._inf_counter += 1

                # 每秒更新推理 FPS
                inf_elapsed = now - self._inf_timer
                if inf_elapsed >= 1.0:
                    self._current_inf_fps = self._inf_counter / inf_elapsed
                    self._inf_counter = 0
                    self._inf_timer = now
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
                    inf_fps=self._current_inf_fps,
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

def draw_detections_on_frame(frame: np.ndarray, detections: list, fps: float = 0.0,
                              inference_ms: float = 0.0,
                              target_cx: float = None, target_cy: float = None,
                              aim_x: float = 0, aim_y: float = 0,
                              is_settled: bool = False) -> np.ndarray:
    """
    在帧上绘制检测框 + 瞄准点 + 高亮选中目标 (独立函数, 供外部调用)

    支持两种 Detection 格式:
    - capture_inference.Detection (有 to_dict 方法)
    - trajectory_calculator.Detection (有 x, y, w, h, cx, cy, confidence, class_id 属性)

    Args:
        frame: 原始 BGR 帧
        detections: Detection 对象列表
        fps: 当前 FPS 显示
        target_cx, target_cy: 选中目标的中心坐标 (用于高亮)
        aim_x, aim_y: 瞄准点坐标 (中心 + 偏移)
        is_settled: 是否已对准 (对准后框变绿)

    Returns:
        带标注的帧
    """
    h, w = frame.shape[:2]

    # 判断是否为选中目标 (中心点匹配)
    def _is_target(det) -> bool:
        if target_cx is None or target_cy is None:
            return False
        if hasattr(det, 'cx'):
            d_cx, d_cy = det.cx, det.cy
        else:
            d = det.to_dict(w, h)
            d_cx, d_cy = d['cx'], d['cy']
        return abs(d_cx - target_cx) < 2 and abs(d_cy - target_cy) < 2

    for det in detections:
        is_target = _is_target(det)

        # 颜色: 选中目标用亮红/亮绿, 其他用类别色
        if is_target and is_settled:
            color = (0, 255, 0)      # 对准 → 绿色
            thickness = 3
        elif is_target:
            color = (0, 0, 255)      # 选中未对准 → 亮红
            thickness = 3
        else:
            color = CLASS_COLORS[det.class_id] if det.class_id < len(CLASS_COLORS) else (200, 200, 200)
            thickness = 2

        # 兼容两种 Detection 格式
        if hasattr(det, 'to_dict'):
            # capture_inference.Detection 格式
            d = det.to_dict(w, h)
            x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
            class_name = det.class_name
            confidence = det.confidence
        else:
            # trajectory_calculator.Detection 格式
            x1 = int(det.x)
            y1 = int(det.y)
            x2 = int(det.x + det.w)
            y2 = int(det.y + det.h)
            class_name = CLASS_NAMES[det.class_id] if det.class_id < len(CLASS_NAMES) else 'unknown'
            confidence = det.confidence

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 绘制标签 (选中目标加 "🎯" 标记)
        if is_target:
            label = f"🎯 {class_name} {confidence:.2f}"
        else:
            label = f"{class_name} {confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - label_h - 6), (x1 + label_w + 6, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )

    # ── 绘制瞄准点 (十字准星) ──
    if aim_x > 0 and aim_y > 0:
        ax, ay = int(aim_x), int(aim_y)
        crosshair_color = (0, 255, 0) if is_settled else (0, 255, 255)
        crosshair_size = 12
        # 十字线
        cv2.line(frame, (ax - crosshair_size, ay), (ax + crosshair_size, ay), crosshair_color, 2)
        cv2.line(frame, (ax, ay - crosshair_size), (ax, ay + crosshair_size), crosshair_color, 2)
        # 中心点
        cv2.circle(frame, (ax, ay), 3, crosshair_color, -1)
        # 圆圈
        cv2.circle(frame, (ax, ay), crosshair_size, crosshair_color, 1)

    # 绘制推理区域 (中心 640×640 裁剪框, 绿色边框)
    fh, fw = frame.shape[:2]
    s = 640
    if fw >= s and fh >= s:
        rx1 = fw // 2 - s // 2
        ry1 = fh // 2 - s // 2
        cv2.rectangle(frame, (rx1, ry1), (rx1 + s, ry1 + s), (0, 255, 0), 2)

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