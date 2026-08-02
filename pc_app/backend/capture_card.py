"""
capture_card.py - 采集卡帧捕获模块

从采集卡 (DirectShow 摄像头) 捕获视频帧,
供目标检测管道消费。

注意: 此模块为框架实现, 具体采集卡参数需要
根据实际硬件调整 camera_index 和分辨率。
"""

import asyncio
import logging
import os
import re
import subprocess
import time
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 采集卡设备名关键词 (按优先级)
CAPTURE_CARD_KEYWORDS = [
    # AVerMedia 采集卡
    ['avermedia', 'gamer', 'ultra', 'live'],
    # MS2130 采集卡
    ['ms2130', 'usb3.0'],
]


def list_dshow_video_devices() -> list:
    """用 ffmpeg 枚举 DirectShow 视频设备名 (按顺序, 与 OpenCV 索引一致)"""
    try:
        r = subprocess.run(
            ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=10)
        devices = []
        for line in r.stderr.split('\n'):
            m = re.search(r'"([^"]+)" \(video\)', line)
            if m:
                devices.append(m.group(1))
        return devices
    except Exception:
        return []


def wake_device(camera_index: int) -> bool:
    """用 ffmpeg 打开一次设备 (唤醒), 解决 AVerMedia DSHOW 打不开问题"""
    try:
        devices = list_dshow_video_devices()
        if camera_index >= len(devices):
            return False
        dev_name = devices[camera_index]
        r = subprocess.run(
            ['ffmpeg', '-y', '-f', 'dshow', '-video_size', '1280x720',
             '-i', f'video={dev_name}', '-vframes', '1',
             os.devnull, '-loglevel', 'error'],
            capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


class CaptureCard:
    """
    采集卡帧捕获器

    使用 OpenCV 的 DirectShow 后端读取采集卡画面,
    在独立线程中循环拉帧, 通过回调将帧传递给检测管道。
    """

    def __init__(self, camera_index: int = 0, target_fps: int = 240):
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._target_width = 1920
        self._target_height = 1080
        self._format = 'mjpeg'  # 'mjpeg' or 'yuv'
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._buffers: list = [None, None]  # 双缓冲 (免拷贝)
        self._write_idx = 0
        self._read_idx = 1
        self._frame_lock = asyncio.Lock()
        self._on_frame: Optional[Callable] = None
        self._frame_count = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def capture_format(self) -> str:
        return self._format

    def set_format(self, fmt: str):
        """设置采集格式: 'mjpeg' 或 'yuv'"""
        if fmt.lower() in ('mjpeg', 'yuv', 'yuyv', 'yuv422'):
            self._format = fmt.lower()
            # 统一为 fourcc 格式
            if self._format == 'yuv' or self._format == 'yuyv':
                self._format = 'yuv'
            logger.info(f"Capture format set to {self._format}")
        else:
            logger.warning(f"Unknown format: {fmt}, keeping {self._format}")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def set_callback(self, callback: Callable[[np.ndarray], None]):
        """设置新帧到达回调 (将在异步上下文中调用)"""
        self._on_frame = callback

    def set_camera_index(self, index: int):
        """设置摄像头索引"""
        self._camera_index = index

    async def start(self):
        """启动采集卡捕获"""
        if self._running:
            logger.warning("CaptureCard already running")
            return

        self._loop = asyncio.get_running_loop()

        # 短暂延迟, 让 DShow 后端释放
        await asyncio.sleep(0.2)

        logger.info(f"Opening capture card at camera index {self._camera_index}...")
        # 优先 DSHOW, 失败回退 MSMF (AVerMedia 卡 MSMF 兼容性更好)
        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not self._cap or not self._cap.isOpened():
            logger.warning("DSHOW backend failed, trying ffmpeg wake + retry...")
            wake_device(self._camera_index)
            time.sleep(1)
            self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)

        if not self._cap or not self._cap.isOpened():
            logger.warning("DSHOW still failed, trying MSMF...")
            self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_MSMF)

        if not self._cap.isOpened():
            logger.error(f"Failed to open capture card at index {self._camera_index} (try replug USB)")
            self._cap = None
            return False

        # 先读几帧让摄像头稳定 (AVerMedia 需要稳定后才能切分辨率)
        for _ in range(5):
            self._cap.read()
            time.sleep(0.03)

        # 设置采集格式 (MJPEG=压缩, YUV422=无压缩更低延迟)
        if self._format == 'yuv':
            # 常见 YUV 格式: YUY2, YUYV, UYVY
            for fourcc_str in [('Y','U','Y','2'), ('Y','U','Y','V'), ('U','Y','V','Y')]:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                if self._cap.set(cv2.CAP_PROP_FOURCC, fourcc):
                    logger.info(f"  YUV FOURCC set to {''.join(fourcc_str)}")
                    break
            else:
                logger.warning("  Could not set YUV format, falling back to default")
        else:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            logger.info("  Format: MJPEG (compressed, lower bandwidth)")

        # 尝试设置分辨率: 优先目标分辨率, 回退 1280x720
        target_res = [(self._target_width, self._target_height), (1920, 1080), (1280, 720)]
        for (w, h) in target_res:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最低缓冲, 降低延迟
            time.sleep(0.3)
            # 读帧验证
            got_frame = False
            for _ in range(8):
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    got_frame = True
                    break
                time.sleep(0.1)
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if got_frame and actual_width >= 1280:
                logger.info(f"Resolution set to {actual_width}x{actual_height}")
                break
            logger.info(f"  {w}x{h} not available (got {actual_width}x{actual_height}), trying next...")

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Capture card started: {actual_width}x{actual_height} @ {actual_fps:.1f}fps (target {self._target_fps})")
        self._running = True

        # 启动帧捕获循环 (在独立线程中)
        asyncio.get_running_loop().run_in_executor(None, self._capture_loop)

        return True

    async def stop(self):
        """停止采集卡捕获"""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Capture card stopped")

    async def read_frame(self) -> Optional[np.ndarray]:
        """读取最新帧 (非阻塞, 双缓冲免拷贝)

        返回上一次写入的缓冲区引用 (0ms 开销)。
        主循环处理期间, 捕获线程写入另一个缓冲区,
        处理速度跟不上时才会覆盖。
        """
        return self._buffers[self._read_idx]

    def _capture_loop(self):
        """帧捕获循环 (在独立线程中运行)"""
        while self._running and self._cap:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame from capture card")
                continue

            self._frame_count += 1

            # 双缓冲: 写当前缓冲区, 读指针切到刚写完的
            self._buffers[self._write_idx] = frame
            self._write_idx ^= 1
            self._read_idx = self._write_idx ^ 1

            # 回调通知
            if self._on_frame and self._loop:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._dispatch_frame(frame),
                        self._loop
                    )
                    # 不等待结果, 避免阻塞
                except Exception as e:
                    logger.error(f"Frame dispatch error: {e}")

    async def _dispatch_frame(self, frame: np.ndarray):
        """分发帧到回调"""
        if self._on_frame:
            try:
                self._on_frame(frame)
            except Exception as e:
                logger.error(f"Frame callback error: {e}")

    @staticmethod
    def list_cameras(max_index: int = 10) -> list:
        """
        安全列出可用摄像头设备 (不开高风险操作)

        Returns:
            [(index, name, is_capture_card), ...] 列表
        """
        # 获取设备名 (与 OpenCV 索引顺序一致)
        devices = list_dshow_video_devices()
        available = []
        for i in range(max_index):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        # 按设备名识别采集卡 (AVerMedia/MS2130)
                        dev_name = devices[i] if i < len(devices) else f"Camera {i}"
                        is_capture = any(
                            any(k in dev_name.lower() for k in kw)
                            for kw in CAPTURE_CARD_KEYWORDS
                        )
                        # 回退: 分辨率 >= 1920 也可能是采集卡
                        if not is_capture:
                            is_capture = (w >= 1920)
                        suffix = " - 采集卡" if is_capture else f" - 摄像头 ({w}x{h})"
                        name = f"[{i}] {dev_name}" + suffix
                        available.append((i, name, is_capture))
                    cap.release()
                    time.sleep(0.1)
            except Exception:
                continue
        # 如果索引探测漏掉了采集卡, 按设备名补上
        for i, dev_name in enumerate(devices):
            if i >= max_index:
                break
            if not any(a[0] == i for a in available):
                is_capture = any(
                    any(k in dev_name.lower() for k in kw)
                    for kw in CAPTURE_CARD_KEYWORDS
                )
                if is_capture:
                    name = f"[{i}] {dev_name} - 采集卡"
                    available.append((i, name, True))
        return available

    @staticmethod
    def find_capture_card() -> int:
        """
        自动查找采集卡 (AVerMedia/MS2130)

        按设备名匹配 (不实际打开设备, 避免 DShow 崩溃)。
        名称匹配失败时用 MSMF 后端安全测试 (MSMF 不会崩溃)。
        """
        # 1. 按设备名匹配 (最可靠, 不打开设备)
        devices = list_dshow_video_devices()
        for idx, name in enumerate(devices):
            lower = name.lower()
            for kw in CAPTURE_CARD_KEYWORDS:
                if any(k in lower for k in kw):
                    logger.info(f"Capture card found by name: [{idx}] {name}")
                    return idx

        # 2. 按分辨率测试 (用 MSMF 后端, 避免 DShow 崩溃)
        for idx in range(10):
            cap = None
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
                if cap is None or not cap.isOpened():
                    continue
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.release()
                    continue
                h, w = frame.shape[:2]
                cap.release()
                time.sleep(0.2)
                if w >= 1920:
                    logger.info(f"Capture card found at index {idx} ({w}x{h})")
                    return idx
            except Exception:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                continue

        # 回退: 返回 index 1 (常用采集卡位置)
        logger.warning("Capture card not detected by name/resolution, falling back to index 1")
        return 1

    # 兼容旧名称
    find_ms2130 = find_capture_card