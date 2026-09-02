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
import threading
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

    def __init__(self, camera_index: int = 0, target_fps: int = 240,
                 use_ffmpeg: bool = True, crop_size: int = 640,
                 capture_format: str = 'nv12'):
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._target_width = 1920
        self._target_height = 1080
        # 默认 NV12 (未压缩, 无需解码); 用户可通过 set_format 切换
        self._format = capture_format if capture_format in ('mjpeg', 'yuv', 'nv12') else 'nv12'
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._buffers: list = [None, None]  # 双缓冲 (免拷贝)
        self._write_idx = 0
        self._read_idx = 1
        self._frame_lock = asyncio.Lock()
        self._on_frame: Optional[Callable] = None
        self._frame_count = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._actual_width = 0
        self._actual_height = 0
        # 帧序号 (采集线程自增) + 采集时刻 (用于去重和延迟分析)
        self._frame_seq = 0
        # 消费端最近读到的帧序号 (背压控制: 采集端避免超前转换浪费 CPU)
        self._last_consumed_seq = -1
        self._frame_ts = 0.0
        # 采集实际帧率统计
        self._fps = 0.0
        self._fps_count = 0
        self._fps_timer = 0.0
        # ffmpeg 硬解模式 (NVIDIA cuvid 解码 + 中心裁剪, 输出 nv12 管道)
        self._use_ffmpeg = use_ffmpeg
        self._crop_size = crop_size  # 0 = 不裁剪
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._crop_x = 0
        self._crop_y = 0

    @property
    def crop_offset_x(self) -> int:
        """裁剪区域左上角 X (全屏系), 非裁剪模式为 0"""
        return self._crop_x

    @property
    def crop_offset_y(self) -> int:
        """裁剪区域左上角 Y (全屏系), 非裁剪模式为 0"""
        return self._crop_y

    @property
    def is_ffmpeg(self) -> bool:
        return self._ffmpeg_proc is not None

    @property
    def capture_fps(self) -> float:
        """采集卡实际输出帧率 (每秒实际读到的帧数)"""
        return self._fps

    @property
    def frame_seq(self) -> int:
        """当前最新帧序号 (采集线程每帧自增, 供主循环去重)"""
        return self._frame_seq

    @property
    def frame_ts(self) -> float:
        """当前最新帧的采集时刻 (time.time)"""
        return self._frame_ts

    @property
    def actual_width(self) -> int:
        return self._actual_width

    @property
    def actual_height(self) -> int:
        return self._actual_height

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def capture_format(self) -> str:
        return self._format

    def set_format(self, fmt: str):
        """设置采集格式: 'mjpeg' / 'yuv' / 'nv12'"""
        if fmt.lower() in ('mjpeg', 'yuv', 'yuyv', 'yuv422', 'nv12'):
            self._format = fmt.lower()
            # 统一为 fourcc 格式
            if self._format == 'yuyv':
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

        # ffmpeg 硬解模式: NVIDIA cuvid 解码 + 中心裁剪, 失败自动回退 OpenCV
        if self._use_ffmpeg:
            ok = await asyncio.to_thread(self._start_ffmpeg)
            if ok:
                return True
            logger.warning("ffmpeg capture failed, falling back to OpenCV")

        logger.info(f"Opening capture card at camera index {self._camera_index}...")
        # 先用 ffmpeg 探测设备是否被占用: OpenCV DSHOW 在设备被其他进程
        # (如残留后端/采集软件) 占用时会原生崩溃 (0xC0000005), 而不是返回失败
        if not self._probe_device_available(self._camera_index):
            logger.error(f"Capture device {self._camera_index} is busy (another process is using it). "
                         f"Close other apps using the capture card and retry.")
            self._cap = None
            return False
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

        # 先设置采集格式占位 (后续分辨率设置会重置流, 需在分辨率后再设一次)
        # 注意: DSHOW 下设置 FRAME_WIDTH/HEIGHT/FPS 会重置流, 冲掉之前设的 FOURCC,
        # 导致 NV12 协商失败回退到默认格式 (帧率被限到 120fps)。
        # 因此先设一次 (部分驱动此顺序可生效), 分辨率设置完成后必须重新设 FOURCC。

        # 再读几帧让摄像头稳定 (AVerMedia 需要稳定后才能切分辨率)
        for _ in range(5):
            self._cap.read()
            time.sleep(0.03)

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

        # 分辨率设置完成后重新设 FOURCC (否则被上面的 set 冲掉, NV12 失效)
        if self._format == 'yuv':
            for fourcc_str in [('Y','U','Y','2'), ('Y','U','Y','V'), ('U','Y','V','Y')]:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                if self._cap.set(cv2.CAP_PROP_FOURCC, fourcc):
                    logger.info(f"  YUV FOURCC set to {''.join(fourcc_str)}")
                    break
            else:
                logger.warning("  Could not set YUV format, falling back to default")
        elif self._format == 'nv12':
            fourcc = cv2.VideoWriter_fourcc('N', 'V', '1', '2')
            if self._cap.set(cv2.CAP_PROP_FOURCC, fourcc):
                logger.info("  Format: NV12 (uncompressed, no decode)")
            else:
                logger.warning("  Could not set NV12 format, falling back to default")
        else:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            logger.info("  Format: MJPEG (compressed, lower bandwidth)")

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Capture card started: {actual_width}x{actual_height} @ {actual_fps:.1f}fps (target {self._target_fps})")
        self._actual_width = actual_width
        self._actual_height = actual_height
        self._running = True

        # 启动帧捕获循环 (专用线程, 与推理线程池隔离)
        # 注意: 不能 run_in_executor(None) 共用默认线程池 — 推理/编码任务
        # 会抢占线程, 把采集线程饿到 ~120fps (独立测试可达 230fps)
        threading.Thread(target=self._capture_loop, daemon=True,
                         name='capture-loop').start()

        return True

    async def stop(self):
        """停止采集卡捕获"""
        self._running = False
        # 停止 ffmpeg 子进程
        if self._ffmpeg_proc:
            try:
                self._ffmpeg_proc.kill()
            except Exception:
                pass
            self._ffmpeg_proc = None
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Capture card stopped")

    def _probe_device_available(self, camera_index: int, timeout: float = 6.0) -> bool:
        """用 ffmpeg 探测采集设备是否可用 (未被其他进程占用)

        OpenCV DSHOW 在设备被占用时会原生崩溃 (0xC0000005), 而非返回失败。
        这里先用 ffmpeg 打开设备抓帧验证可用性; 抓不到 = 设备忙, 返回 False,
        由调用方优雅报错, 避免 Python 进程崩溃。

        注意: 探测命令必须用采集卡枚举声明的分辨率/帧率组合 (1080p), 否则
        dshow 会因 "Could not set video options" 误报 busy (720p 枚举仅 60Hz)。
        """
        try:
            devices = list_dshow_video_devices()
            if camera_index >= len(devices):
                logger.warning(f"probe: device index {camera_index} not found")
                return True  # 让后续 OpenCV 打开流程自行报错 (索引不存在不崩溃)
            dev_name = devices[camera_index]
            r = subprocess.run(
                ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                 '-fflags', 'nobuffer', '-flags', 'low_delay',
                 '-probesize', '32', '-analyzeduration', '0', '-rtbufsize', '512k',
                 '-f', 'dshow', '-framerate', '60', '-video_size', '1920x1080',
                 '-i', f'video={dev_name}', '-frames:v', '1',
                 '-f', 'null', '-'],
                capture_output=True, timeout=timeout)
            ok = r.returncode == 0
            if not ok:
                # 一次失败可能是 dshow 冷启动抖动, 重试一次再判定 busy
                r2 = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                     '-fflags', 'nobuffer', '-flags', 'low_delay',
                     '-probesize', '32', '-analyzeduration', '0', '-rtbufsize', '512k',
                     '-f', 'dshow', '-framerate', '60', '-video_size', '1920x1080',
                     '-i', f'video={dev_name}', '-frames:v', '1',
                     '-f', 'null', '-'],
                    capture_output=True, timeout=timeout)
                ok = r2.returncode == 0
                if not ok:
                    logger.warning(f"probe: device {dev_name} busy (ffmpeg rc={r.returncode}/{r2.returncode})")
            return ok
        except subprocess.TimeoutExpired:
            logger.warning(f"probe: timeout probing device {camera_index}")
            return False
        except Exception as e:
            logger.warning(f"probe: error probing device {camera_index}: {e}")
            return True  # 探测本身异常时放行, 交给 OpenCV 处理

    def _start_ffmpeg(self) -> bool:
        """启动 ffmpeg 硬解采集 (NVIDIA cuvid 解码 + 解码器级中心裁剪, nv12 管道输出)

        mjpeg_cuvid 的 -crop 选项在硬件解码器内部直接只解中心区域
        (NVDEC 只输出 ROI 的 DCT 块, 不经过完整 1080p 解码 + filter 裁剪),
        输出 nv12 原始帧到管道, Python 读帧转 BGR。相比 OpenCV 软解 (101fps)
        提升明显 (实测 ~290fps), 且输出帧与模型输入同尺寸, 检测预处理更快。
        """
        try:
            devices = list_dshow_video_devices()
            if self._camera_index >= len(devices):
                logger.error(f"ffmpeg: device index {self._camera_index} out of range ({len(devices)})")
                return False
            dev_name = devices[self._camera_index]

            # 中心裁剪参数: 解码器级 -crop 格式 topxbottomxleftxright
            cx = cy = 0
            crop_args = []
            vf = None
            if self._crop_size and self._crop_size < self._target_width and self._crop_size < self._target_height:
                cx = (self._target_width - self._crop_size) // 2
                cy = (self._target_height - self._crop_size) // 2
                crop_args = ['-crop', f'{cy}x{cy}x{cx}x{cx}']

            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
                # 低延迟: 禁用缓冲/分析, 减小 dshow 缓冲, 避免管道积压导致帧变老
                '-fflags', 'nobuffer', '-flags', 'low_delay',
                '-probesize', '32', '-analyzeduration', '0', '-rtbufsize', '512k',
                '-f', 'dshow', '-framerate', str(self._target_fps),
                '-video_size', f'{self._target_width}x{self._target_height}',
                '-vcodec', 'mjpeg_cuvid',
            ]
            cmd += crop_args
            # 减小解码器内部帧队列 (默认较大), 降低硬件解码环节延迟 (必须是输入侧选项)
            cmd += ['-surfaces', '3']
            cmd += ['-i', f'video={dev_name}']
            if vf:
                cmd += ['-vf', vf]
            cmd += ['-flush_packets', '1']
            cmd += ['-pix_fmt', 'nv12', '-f', 'rawvideo', '-']

            logger.info(f"ffmpeg capture: {dev_name} crop={vf or 'none'}")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            time.sleep(1.0)  # 等设备打开
            if proc.poll() is not None:
                logger.error(f"ffmpeg exited early (code {proc.returncode})")
                return False

            self._ffmpeg_proc = proc
            self._crop_x = cx
            self._crop_y = cy
            fw = self._crop_size if (vf or crop_args) else self._target_width
            fh = self._crop_size if (vf or crop_args) else self._target_height
            self._actual_width = fw
            self._actual_height = fh
            self._running = True

            # 读帧线程 (纯线程工作, 不依赖事件循环)
            threading.Thread(target=self._ffmpeg_capture_loop, daemon=True).start()
            logger.info(f"ffmpeg capture started: {fw}x{fh} (crop offset {cx},{cy})")
            return True
        except Exception as e:
            # 清理可能残留的 ffmpeg 进程 (否则设备被占用, OpenCV 回退也打不开)
            if self._ffmpeg_proc:
                try:
                    self._ffmpeg_proc.kill()
                except Exception:
                    pass
                self._ffmpeg_proc = None
            self._running = False
            logger.error(f"ffmpeg start error: {e}")
            return False

    def _ffmpeg_capture_loop(self):
        """ffmpeg 管道读帧线程: nv12 -> BGR -> 双缓冲 (与 OpenCV 模式同一套接口)

        用精确帧对齐读 (每次只读一帧的剩余字节): 若用大块 read(1MB) 一次
        可能拿到 1~2 帧 (nv12 帧 614KB), 导致帧突发积压 (实测最大 52ms 抖动,
        表现为自瞄跟不住人)。对齐读实测帧间隔 p95 抖动 <4ms。
        """
        proc = self._ffmpeg_proc
        if not proc:
            return
        fw = self._actual_width or self._target_width
        fh = self._actual_height or self._target_height
        frame_size = fw * fh * 3 // 2  # nv12 = 1.5 bytes/pixel
        try:
            while self._running and proc.poll() is None:
                # 对齐读: 拼满一整帧
                frame_buf = bytearray()
                while len(frame_buf) < frame_size:
                    chunk = proc.stdout.read(frame_size - len(frame_buf))
                    if not chunk:
                        break
                    frame_buf += chunk
                if len(frame_buf) < frame_size:
                    break
                nv12 = np.frombuffer(frame_buf, np.uint8).reshape((fh * 3 // 2, fw))
                bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

                self._frame_count += 1
                self._frame_seq += 1
                self._frame_ts = time.time()
                # 采集实际帧率统计
                self._fps_count += 1
                fps_now = time.time()
                if self._fps_timer == 0:
                    self._fps_timer = fps_now
                fps_elapsed = fps_now - self._fps_timer
                if fps_elapsed >= 1.0:
                    self._fps = self._fps_count / fps_elapsed
                    self._fps_count = 0
                    self._fps_timer = fps_now

                # 双缓冲: 写当前缓冲区, 读指针切到刚写完的
                self._buffers[self._write_idx] = bgr
                self._write_idx ^= 1
                self._read_idx = self._write_idx ^ 1
        except Exception as e:
            logger.error(f"ffmpeg capture loop error: {e}")
        logger.info("ffmpeg capture loop ended")

    async def read_frame(self) -> Optional[np.ndarray]:
        """读取最新帧 (非阻塞, 双缓冲免拷贝)

        返回上一次写入的缓冲区引用 (0ms 开销)。
        主循环处理期间, 捕获线程写入另一个缓冲区,
        处理速度跟不上时才会覆盖。
        """
        # 记录消费进度 (背压控制用: 采集端据此判断是否需要放慢)
        self._last_consumed_seq = self._frame_seq
        return self._buffers[self._read_idx]

    def _capture_loop(self):
        """帧捕获循环 (在独立线程中运行)"""
        # 降采集线程优先级: 让推理线程优先拿 CPU (采集是生产者, 慢一点可接受,
        # 推理是消费者, 慢了直接拖管线帧率)
        try:
            import ctypes
            THREAD_PRIORITY_LOWEST = -2
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, THREAD_PRIORITY_LOWEST)
        except Exception:
            pass
        while self._running and self._cap:
            # 背压: 消费端未跟上时让出 CPU, 避免无谓的 NV12->BGR 转换
            # 采集 236fps 但管线只消费 ~150fps, 每秒 ~80 帧白转换抢 CPU 导致偶发卡顿
            # 若消费端落后 >=2 帧 (还没来取), 让出 CPU 等消费端追上
            if self._frame_seq - self._last_consumed_seq >= 2:
                time.sleep(0.002)
                continue  # 不抓新帧, 先让消费端处理已有帧
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Failed to read frame from capture card")
                continue

            self._frame_count += 1
            self._frame_seq += 1
            self._frame_ts = time.time()

            # 采集实际帧率统计 (每秒一次)
            self._fps_count += 1
            fps_now = time.time()
            if self._fps_timer == 0:
                self._fps_timer = fps_now
            fps_elapsed = fps_now - self._fps_timer
            if fps_elapsed >= 1.0:
                self._fps = self._fps_count / fps_elapsed
                self._fps_count = 0
                self._fps_timer = fps_now

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

        只用 ffmpeg 枚举设备名 (不实际打开设备), 避免 DSHOW 抖动
        干扰后续采集卡打开, 也避免 "can't be used to capture by index" 警告。

        Returns:
            [(index, name, is_capture_card), ...] 列表
        """
        devices = list_dshow_video_devices()
        available = []
        for i, dev_name in enumerate(devices):
            if i >= max_index:
                break
            # 按设备名识别采集卡 (AVerMedia/MS2130) — 不打开设备
            is_capture = any(
                any(k in dev_name.lower() for k in kw)
                for kw in CAPTURE_CARD_KEYWORDS
            )
            suffix = " - 采集卡" if is_capture else f" - 摄像头"
            name = f"[{i}] {dev_name}" + suffix
            available.append((i, name, is_capture))
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