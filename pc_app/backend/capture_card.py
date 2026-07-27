"""
capture_card.py - 采集卡帧捕获模块

从采集卡 (DirectShow 摄像头) 捕获视频帧,
供目标检测管道消费。

注意: 此模块为框架实现, 具体采集卡参数需要
根据实际硬件调整 camera_index 和分辨率。
"""

import asyncio
import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CaptureCard:
    """
    采集卡帧捕获器

    使用 OpenCV 的 DirectShow 后端读取采集卡画面,
    在独立线程中循环拉帧, 通过回调将帧传递给检测管道。
    """

    def __init__(self, camera_index: int = 0, target_fps: int = 30):
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = asyncio.Lock()
        self._on_frame: Optional[Callable] = None
        self._frame_count = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_running(self) -> bool:
        return self._running

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
        """启动采集卡捕获 (自动优先使用 MS2130 采集卡)"""
        if self._running:
            logger.warning("CaptureCard already running")
            return

        self._loop = asyncio.get_running_loop()

        # 始终优先查找 MS2130 采集卡, 覆盖任何传入的 camera_index
        ms2130_idx = CaptureCard.find_ms2130()
        if ms2130_idx >= 0:
            if ms2130_idx != self._camera_index:
                logger.info(f"Auto-detected MS2130 at index {ms2130_idx}, overriding camera_index {self._camera_index}")
            self._camera_index = ms2130_idx

        # 短暂延迟, 让 DShow 后端释放
        await asyncio.sleep(0.2)

        logger.info(f"Opening capture card at camera index {self._camera_index}...")
        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            logger.error(f"Failed to open capture card at index {self._camera_index}")
            self._cap = None
            return False

        # 设置采集卡参数 - 尝试 1080p, 失败则回退到 720p
        # 先尝试用户指定的分辨率, 默认 1080p
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最低缓冲, 降低延迟
        
        # 等待几帧让摄像头稳定
        for _ in range(5):
            self._cap.read()
        
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 如果实际分辨率达不到 1080p, 降级到 720p
        if actual_width < 1920 or actual_height < 1080:
            logger.info(f"1080p not available (got {actual_width}x{actual_height}), falling back to 720p...")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(5):
                self._cap.read()
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Capture card started: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
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
        """读取最新帧 (非阻塞)"""
        async with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
            return None

    def _capture_loop(self):
        """帧捕获循环 (在独立线程中运行)"""
        while self._running and self._cap:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame from capture card")
                continue

            self._frame_count += 1

            # 更新最新帧
            self._frame = frame

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
            [(index, name, is_ms2130), ...] 列表
        """
        available = []
        for i in range(max_index):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        # 标记: 如果默认分辨率不是 640x480, 可能是采集卡
                        # 注意: 不在这里设置高分辨率, 避免 segfault
                        is_ms2130 = (w >= 1280)
                        name = f"Camera {i}" + (" - MS2130" if is_ms2130 else " - 内置摄像头")
                        available.append((i, name, is_ms2130))
                    cap.release()
            except Exception:
                continue
        return available

    @staticmethod
    def find_ms2130() -> int:
        """
        自动查找 MS2130 采集卡

        逐个测试摄像头, 每个测试完立即释放, 避免 DShow 后端冲突。
        先测非 0 索引, 跳过内置摄像头。
        """
        candidates = [1, 2, 8, 3, 4, 5, 6, 7, 9]

        for idx in candidates:
            try:
                # 打开并测试
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    time.sleep(0.1)  # 让 DShow 后端释放
                    continue

                # 读一帧确认能工作
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.release()
                    time.sleep(0.1)
                    continue

                # 尝试设置 720p — MS2130 支持, 内置摄像头不支持
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                for _ in range(3):
                    cap.read()
                    time.sleep(0.02)

                test_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                test_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                time.sleep(0.1)  # 让 DShow 后端释放

                if test_w >= 1280:
                    logger.info(f"MS2130 found at index {idx} ({test_w}x{test_h})")
                    return idx
                else:
                    logger.info(f"Camera {idx} = {test_w}x{test_h} (not MS2130)")

            except Exception as e:
                logger.warning(f"Camera {idx} test failed: {e}")
                try:
                    cap.release()
                except Exception:
                    pass
                continue

        # 回退: 直接返回 index 1 (最可能的 MS2130 位置)
        logger.warning("MS2130 not detected, falling back to camera 1")
        return 1
        """
        自动查找 MS2130 采集卡

        Returns:
            camera_index 或 -1 (未找到)
        """
        cameras = CaptureCard.list_cameras()
        for idx, name, is_ms2130 in cameras:
            if is_ms2130:
                logger.info(f"MS2130 capture card found at index {idx}: {name}")
                return idx
        # 回退: 用第一个非 0 索引的摄像头
        for idx, name, is_ms2130 in cameras:
            if idx != 0:
                logger.info(f"No MS2130 found, using camera {idx} as fallback")
                return idx
        if cameras:
            logger.info(f"No MS2130 found, using camera {cameras[0][0]} as fallback")
            return cameras[0][0]
        return -1