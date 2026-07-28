#!/usr/bin/env python3
"""
main.py - Mouse Forwarder 后端入口

基于 asyncio WebSocket 服务器, 参考 Multi3DViz 架构:
- 启动 WebSocket 服务器供前端连接
- 捕获鼠标事件 (pynput)
- 通过串口转发到 CH32V305
- 目标检测 + 轨迹计算 (Valorant AI 瞄准)

启动方式:
  python main.py [--port PORT] [--baud BAUDRATE] [--model MODEL_PATH]
  
当 Electron 启动时, 会打印 "READY ws://127.0.0.1:PORT"
供 Electron 主进程检测。
"""

import argparse
import asyncio
import json
import logging
import os
import glob
import signal
import sys
import time
import threading
from typing import Any, Optional, List, Tuple

import numpy as np
import websockets

from mouse_monitor import MouseMonitor, MouseEvent
from serial_forwarder import SerialForwarder
from protocol import encode_packet
from flasher import flash_firmware, detect_bootloader, erase_flash
from trajectory_calculator import TrajectoryCalculator, TrajectoryConfig, Detection
from capture_inference import draw_detections_on_frame, encode_frame_jpeg, update_class_names
from config import Config

# 锁模式需要键盘监听
try:
    from pynput import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('main')

# 默认端口
DEFAULT_WS_PORT = 8765
DEFAULT_SERIAL_BAUD = 115200


class MouseForwarderBackend:
    """
    鼠标转发器后端 (参考 Multi3DViz 架构)
    
    管理 WebSocket 连接, 鼠标监听, 串口转发,
    以及目标检测与 AI 瞄准轨迹。
    """
    
    def __init__(self, ws_port: int = DEFAULT_WS_PORT,
                 serial_baud: int = DEFAULT_SERIAL_BAUD,
                 model_path: Optional[str] = None):
        self.ws_port = ws_port
        self.serial_baud = serial_baud
        self.model_path = model_path
        
        # 组件 - pynput 鼠标捕获 (经过系统处理, 手感自然)
        self.mouse = MouseMonitor(on_event=self._on_mouse_event)
        self.serial = SerialForwarder(baudrate=serial_baud)
        self.serial.set_connection_callback(self._on_serial_connection)
        
        # 目标检测 & 轨迹组件 (延迟初始化)
        self.capture_card = None
        self.detector = None
        self.trajectory = TrajectoryCalculator()

        # 持久化配置 (必须在其他配置之前加载)
        self.config = Config()
        
        # 从持久化配置恢复轨迹参数
        traj_cfg = self.config.get('trajectory', default={})
        if traj_cfg:
            for k, v in traj_cfg.items():
                if hasattr(self.trajectory.config, k):
                    setattr(self.trajectory.config, k, v)
            # 从持久化配置恢复轨迹参数 (但不恢复 enabled 状态, 需手动开启)
            saved_enabled = traj_cfg.get('enabled', False)
            if 'enabled' in traj_cfg:
                del traj_cfg['enabled']
            for k, v in traj_cfg.items():
                if hasattr(self.trajectory.config, k):
                    setattr(self.trajectory.config, k, v)
            self._trajectory_enabled = False
            self.trajectory.config.enabled = False
            logger.info(f"Loaded trajectory config from saved settings")

        # WebSocket 客户端
        self._ws_client: Optional[Any] = None
        
        # 保存事件循环引用 (用于线程安全调用)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 统计信息
        self.start_time = time.time()
        self.stats = {
            'mouse_events': 0,
            'packets_sent': 0,
            'bytes_sent': 0,
            'trajectory_events': 0,
            'detections': 0,
        }
        
        # 状态
        self._mouse_active = False
        self._capture_active = False
        self._detection_active = False
        
        self._trajectory_enabled = False
        self._show_video = True  # 默认显示画面
        self._lock_mode = False  # 锁定模式: 全屏黑幕
        self._keyboard_listener = None  # Escape 键监听器
        self._trigger_enabled = False  # 自动扳机
        self._trigger_threshold = 5  # 自动扳机触发阈值 (像素)
        self._trigger_armed = False  # 扳机状态 (避免重复触发)
        self._trigger_last_fire = 0  # 上次触发时间
        
        # 管线帧率统计
        self._pipeline_fps = 0.0
        self._pipeline_fps_count = 0
        self._pipeline_fps_timer = 0.0
        
        # 检测管道任务
        self._pipeline_task: Optional[asyncio.Task] = None
        
        # 屏幕中心 (自动检测, 默认 1920x1080)
        self._screen_w = 1920
        self._screen_h = 1080
        try:
            import ctypes
            user32 = ctypes.windll.user32
            self._screen_w = user32.GetSystemMetrics(0)
            self._screen_h = user32.GetSystemMetrics(1)
            logger.info(f"Detected screen: {self._screen_w}x{self._screen_h}")
        except Exception:
            logger.info(f"Using default screen: {self._screen_w}x{self._screen_h}")
    
    async def start(self):
        """启动后端服务"""
        # 保存事件循环引用 (用于线程安全调用)
        self._loop = asyncio.get_running_loop()
        self.trajectory.set_screen_center(self._screen_w / 2, self._screen_h / 2)
        
        # 自动查找 CH32V305 串口
        ch32_port = SerialForwarder.find_ch32v305_port()
        if ch32_port:
            logger.info(f"Auto-detected CH32V305 on {ch32_port}")
            await self.serial.connect(ch32_port)
        else:
            logger.info("CH32V305 not found, will wait for connection")
        
        # 启动鼠标监听
        self.mouse.start()
        self._mouse_active = True
        
        # 自动查找模型: 优先用启动参数, 其次自动检测, 最后用最后加载的
        if not self.model_path or not os.path.exists(self.model_path):
            # 尝试从持久化配置恢复最后加载的模型
            saved_model = self.config.get('model', 'last_path', default='')
            if saved_model and os.path.exists(saved_model):
                self.model_path = saved_model
                logger.info(f"Restored last model from config: {self.model_path}")
        
        # 初始化检测组件 (如果提供了模型路径)
        if self.model_path and os.path.exists(self.model_path):
            await self._init_detection_pipeline(self.model_path)
            # 自动启动采集卡 (后台启动, 不阻塞)
            if self.detector and self.detector.is_loaded:
                logger.info("Model loaded, auto-starting capture...")
                # 延迟启动, 等 WebSocket 服务就绪后客户端连接
                asyncio.create_task(self._delayed_auto_start())
        
        # 启动 WebSocket 服务器 (带端口重试)
        async def handler(ws):
            await self._handle_client(ws)
        
        max_retries = 5
        for attempt in range(max_retries):
            port = self.ws_port + attempt
            try:
                logger.info(f"Starting WS server on port {port}")
                server = await websockets.serve(handler, "127.0.0.1", port)
                self.ws_port = port
                print(f"READY ws://127.0.0.1:{port}", flush=True)
                logger.info(f"Server listening on 127.0.0.1:{port}")
                # 服务器已成功启动
                await asyncio.Future()  # 永远运行
                return
            except OSError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Port {port} busy, trying {port + 1}...")
                    continue
                else:
                    logger.error(f"All ports {self.ws_port}-{self.ws_port + max_retries - 1} busy")
                    raise
    
    async def stop(self):
        """停止后端服务"""
        # 退出锁定模式 (恢复鼠标监听)
        if self._lock_mode:
            if self._keyboard_listener:
                try:
                    self._keyboard_listener.stop()
                except Exception:
                    pass
                self._keyboard_listener = None
            self.mouse.set_suppress(False)
            self._lock_mode = False
        
        # 停止检测管道
        if self._pipeline_task:
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass
        
        # 停止采集卡
        if self.capture_card:
            await self.capture_card.stop()
        
        self.mouse.stop()
        await self.serial.disconnect()
        logger.info("Backend stopped")
    
    # ================================================================
    # 检测管道初始化
    # ================================================================
    
    async def _init_detection_pipeline(self, model_path: str):
        """初始化目标检测管道"""
        try:
            from capture_card import CaptureCard
            from object_detector import ObjectDetector
            
            self.capture_card = CaptureCard(camera_index=0)
            self.detector = ObjectDetector()
            
            # 加载模型 (在后台线程中执行, 避免阻塞)
            await asyncio.to_thread(self.detector.load_model, model_path)
            # 更新类别名称 (与实际模型匹配)
            await asyncio.to_thread(update_class_names, model_path)
            # 保存最后加载的模型路径
            self.config.set('model', 'last_path', model_path)
            
            logger.info("Detection pipeline initialized")
            
        except ImportError as e:
            logger.warning(f"Detection pipeline not available: {e}")
            logger.warning("Install: pip install onnxruntime opencv-python-headless numpy")
        except Exception as e:
            logger.error(f"Failed to init detection pipeline: {e}")

    async def _delayed_auto_start(self):
        """延迟自动启动采集卡 (等前端连接后再启动)"""
        try:
            # 等待前端连接 (最多等 10 秒)
            for _ in range(100):
                if self._ws_client is not None:
                    break
                await asyncio.sleep(0.1)

            await asyncio.sleep(1)  # 额外等前端就绪
            logger.info("Auto-starting capture card...")
            # 自动检测 MS2130 采集卡索引
            try:
                from capture_card import CaptureCard
                ms2130_idx = await asyncio.to_thread(CaptureCard.find_ms2130)
                logger.info(f"Auto-detected capture card at index {ms2130_idx}")
            except Exception as e:
                logger.warning(f"Auto-detect failed: {e}, using index 0")
                ms2130_idx = 0
            await self._start_capture(ms2130_idx)
        except Exception as e:
            logger.error(f"Auto-start capture failed: {e}")

    async def _start_capture(self, camera_index: int = 0):
        """启动采集卡捕获"""
        try:
            if not self.capture_card:
                from capture_card import CaptureCard
                self.capture_card = CaptureCard(camera_index=camera_index)
            
            self.capture_card.set_camera_index(camera_index)
            logger.info(f"Starting capture on camera {camera_index}...")
            success = await self.capture_card.start()
            
            if success:
                self._capture_active = True
                # 启动检测管道
                self._pipeline_task = asyncio.create_task(self._detection_loop())
                await self._send({
                    'type': 'capture_status',
                    'running': True,
                    'camera_index': camera_index,
                })
                logger.info(f"Capture started on camera {camera_index}")
            else:
                await self._send({
                    'type': 'capture_status',
                    'running': False,
                    'error': f'无法打开摄像头 {camera_index}',
                })
                logger.warning(f"Failed to start capture on camera {camera_index}")
            
            return success
        except Exception as e:
            logger.error(f"Capture start error: {e}")
            import traceback
            traceback.print_exc()
            await self._send({
                'type': 'capture_status',
                'running': False,
                'error': str(e),
            })
            return False
    
    async def _stop_capture(self):
        """停止采集卡捕获"""
        if self._pipeline_task:
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass
            self._pipeline_task = None
        
        if self.capture_card:
            await self.capture_card.stop()
        
        self._capture_active = False
        self._detection_active = False
        
        await self._send({
            'type': 'capture_status',
            'running': False,
        })
        logger.info("Capture stopped")

    # ================================================================
    # 锁定模式: 鼠标不控制本机, 只转发到目标 PC
    # ================================================================

    def _recenter_loop(self):
        """
        光标回中线程 (独立于 pynput 钩子线程)

        在锁定模式下, 每隔 5ms 检查光标位置。
        超出中心 950×500 范围时调用 SetCursorPos 回中。
        """
        while self._lock_mode:
            try:
                import win32api
                x, y = win32api.GetCursorPos()
                sw = win32api.GetSystemMetrics(0)
                sh = win32api.GetSystemMetrics(1)
                cx, cy = sw // 2, sh // 2
                if abs(x - cx) > 950 or abs(y - cy) > 500:
                    self.mouse.skip_next_move()
                    win32api.SetCursorPos((cx, cy))
            except Exception:
                pass
            time.sleep(0.005)

    async def _lock_mode_on(self):
        """进入锁定模式: 弹出全屏黑幕, 鼠标不控制本机, 按 Escape 退出"""
        if self._lock_mode:
            return

        self._lock_mode = True
        logger.info("Lock mode ON - showing black overlay")

        # 启动独立回中线程 (超出中心 ±480×270 范围时回中)
        threading.Thread(target=self._recenter_loop, daemon=True).start()

        # 隐藏系统光标
        self._overlay_thread = None
        self._overlay_root = None
        self._overlay_stop = threading.Event()

        def overlay_thread():
            try:
                import tkinter as tk
                root = tk.Tk()
                self._overlay_root = root
                root.overrideredirect(True)  # 无边框
                root.attributes('-topmost', True)  # 置顶
                root.configure(bg='black')
                # 手动设置全屏尺寸 (比 attributes('-fullscreen') 更兼容)
                screen_w = root.winfo_screenwidth()
                screen_h = root.winfo_screenheight()
                root.geometry(f'{screen_w}x{screen_h}+0+0')
                root.focus_force()
                root.grab_set()  # 捕获所有输入

                # 阻止 Alt+F4 关闭
                root.protocol("WM_DELETE_WINDOW", lambda: None)

                # 轮询检查是否需要关闭
                def check_stop():
                    if self._overlay_stop.is_set():
                        try:
                            root.destroy()
                        except Exception:
                            pass
                        return
                    root.after(100, check_stop)
                root.after(100, check_stop)

                root.mainloop()
            except ImportError:
                logger.warning("tkinter not available, using suppress mode only")
            except Exception as e:
                logger.error(f"Overlay error: {e}")

        self._overlay_thread = threading.Thread(target=overlay_thread, daemon=True)
        self._overlay_thread.start()

        # 启动键盘监听 (检测 Escape 键退出)
        if HAS_KEYBOARD and self._loop:
            def on_press(key):
                try:
                    if key == keyboard.Key.esc:
                        logger.info("Escape pressed, exiting lock mode")
                        asyncio.run_coroutine_threadsafe(
                            self._lock_mode_off(),
                            self._loop
                        )
                except Exception:
                    pass

            self._keyboard_listener = keyboard.Listener(on_press=on_press)
            self._keyboard_listener.start()
            logger.info("Keyboard listener started (Escape to exit lock mode)")

        # 通知前端
        await self._send({
            'type': 'lock_mode_status',
            'enabled': True,
        })

    async def _lock_mode_off(self):
        """退出锁定模式"""
        if not self._lock_mode:
            return

        self._lock_mode = False
        logger.info("Lock mode OFF")

        # 关闭全屏黑幕
        if self._overlay_stop:
            self._overlay_stop.set()
        if self._overlay_root:
            try:
                self._overlay_root.quit()
            except Exception:
                pass
        self._overlay_root = None
        self._overlay_thread = None

        # 停止键盘监听
        if self._keyboard_listener:
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None
            logger.info("Keyboard listener stopped")

        # 通知前端
        await self._send({
            'type': 'lock_mode_status',
            'enabled': False,
        })

    async def _detection_loop(self):
        """检测管道主循环 (无帧率限制, 尽可能快)"""
        logger.info("Detection pipeline loop started")
        
        # 帧发送限频 (每秒最多 20 帧)
        last_frame_time = 0
        frame_interval = 0.05
        
        while self._capture_active and self.capture_card:
            try:
                loop_start = time.perf_counter()
                
                # 读取最新帧
                frame = await self.capture_card.read_frame()
                if frame is None:
                    await asyncio.sleep(0.001)
                    continue
                
                # 管线帧率统计 (只统计实际处理的帧)
                self._pipeline_fps_count += 1
                fps_now = time.time()
                if self._pipeline_fps_timer == 0:
                    self._pipeline_fps_timer = fps_now
                fps_elapsed = fps_now - self._pipeline_fps_timer
                if fps_elapsed >= 1.0:
                    self._pipeline_fps = self._pipeline_fps_count / fps_elapsed
                    self._pipeline_fps_count = 0
                    self._pipeline_fps_timer = fps_now
                
                # 执行检测 (在后台线程运行, 不阻塞)
                if self.detector and self.detector.is_loaded:
                    detections = await asyncio.to_thread(
                        self.detector.detect, frame, frame.shape[:2]
                    )
                else:
                    detections = []
                
                self.stats['detections'] = len(detections)
                
                # 计算轨迹
                ai_steps = self.trajectory.calculate(detections) if self._trajectory_enabled else []

                # 自动扳机检测: 如果目标在屏幕中心阈值范围内, 自动按下鼠标左键
                await self._check_auto_trigger(detections)

                # 发送 AI 轨迹到串口 (仅当轨迹启用时)
                if ai_steps and self.serial.is_connected:
                    for dx, dy in ai_steps:
                        packet = encode_packet(0, dx, dy, 0)
                        success = await self.serial.send(packet)
                        if success:
                            self.stats['trajectory_events'] += 1
                            self.stats['packets_sent'] += 1
                            self.stats['bytes_sent'] += len(packet)
                
                # 发送检测结果到前端 (限频, 避免带宽过高)
                now = time.time()
                send_frame = (now - last_frame_time) >= frame_interval
                
                if self._ws_client and send_frame:
                    last_frame_time = now
                    await self._send_detection_frame(detections, ai_steps, frame)
                
                # 不 sleep, 让循环全速运行
                # 如果需要限制 CPU 占用, 可以加一个极短的 sleep
                elapsed = time.perf_counter() - loop_start
                if elapsed < 0.005:
                    await asyncio.sleep(0.005 - elapsed)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("Detection pipeline loop ended")

    # ================================================================
    # 自动扳机: 检测到目标在屏幕中心附近时自动按下左键
    # ================================================================

    async def _check_auto_trigger(self, detections: List[Detection]):
        """
        自动扳机检测

        当目标离屏幕中心的距离小于阈值时, 自动发送鼠标左键按下/释放事件。
        使用 armed 状态防止重复触发。
        """
        if not self._trigger_enabled:
            # 如果扳机关闭但之前按下了, 释放
            if self._trigger_armed:
                await self._fire_trigger(False)
                self._trigger_armed = False
            return

        # 查找离屏幕中心最近的目标
        target_dist = float('inf')
        for d in detections:
            if d.confidence < self.trajectory.config.min_confidence:
                continue
            # FOV 范围检查
            if self.trajectory.config.fov_radius > 0:
                fov_sq = self.trajectory.config.fov_radius ** 2
                dist_sq = (d.cx - self._screen_w / 2) ** 2 + (d.cy - self._screen_h / 2) ** 2
                if dist_sq > fov_sq:
                    continue
                dist = dist_sq ** 0.5
            else:
                dist = ((d.cx - self._screen_w / 2) ** 2 + (d.cy - self._screen_h / 2) ** 2) ** 0.5
            if dist < target_dist:
                target_dist = dist

        # 检查是否在阈值范围内
        in_range = target_dist <= self._trigger_threshold

        if in_range and not self._trigger_armed:
            # 进入阈值范围 - 按下左键
            await self._fire_trigger(True, target_dist)
            self._trigger_armed = True
        elif not in_range and self._trigger_armed:
            # 离开阈值范围 - 释放左键
            await self._fire_trigger(False, target_dist)
            self._trigger_armed = False

    async def _fire_trigger(self, pressed: bool, target_dist: float = -1):
        """
        发送扳机事件 (鼠标左键按下/释放)

        通过串口发送给目标 PC, 同时通过本机 pynput 模拟 (如果未锁定模式)
        """
        # 计算按钮状态
        if pressed:
            self._buttons_state_for_trigger = 1  # bit0 = 左键
        else:
            self._buttons_state_for_trigger = 0

        # 通过串口发送到目标 PC
        if self.serial.is_connected:
            packet = encode_packet(
                buttons=self._buttons_state_for_trigger,
                dx=0, dy=0, wheel=0
            )
            await self.serial.send(packet)
            self.stats['packets_sent'] += 1
            self.stats['bytes_sent'] += len(packet)

        # 通知前端扳机触发
        if self._ws_client:
            await self._send({
                'type': 'trigger_event',
                'pressed': pressed,
                'target_distance': target_dist,
            })

    # ================================================================
    # WebSocket 客户端处理
    # ================================================================
    
    async def _handle_client(self, ws):
        """
        处理 WebSocket 客户端连接
        
        参考 Multi3DViz: 单客户端模式, 拒绝多余连接
        """
        if self._ws_client is not None:
            logger.warning("Rejecting duplicate client")
            await ws.close(1008, "Only one client allowed")
            return
        
        self._ws_client = ws
        remote = ws.remote_address
        logger.info(f"Client connected: {remote}")
        
        try:
            # 发送就绪事件
            await self._send({
                'type': 'ready',
                'port': self.ws_port,
            })
            
            # 发送初始状态
            await self._send_state()
            
            # 发送串口连接状态
            await self._send({
                'type': 'serial_status',
                'connected': self.serial.is_connected,
                'port': self.serial.port_name,
            })
            
            # 发送检测状态
            await self._send({
                'type': 'detection_status',
                'model_loaded': self.detector.is_loaded if self.detector else False,
                'capture_active': self._capture_active,
                'trajectory_enabled': self._trajectory_enabled,
                'model_path': self.model_path or '',
            })
            
            # 处理客户端消息
            async for message in ws:
                await self._handle_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {remote}")
        finally:
            self._ws_client = None
    
    async def _handle_message(self, message: str):
        """处理 WebSocket 消息"""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            # --- 串口管理 ---
            if msg_type == 'connect_serial':
                port = data.get('port')
                success = await self.serial.connect(port)
                await self._send({
                    'type': 'serial_status',
                    'connected': self.serial.is_connected,
                    'port': port,
                })

            elif msg_type == 'disconnect_serial':
                await self.serial.disconnect()
                await self._send({
                    'type': 'serial_status',
                    'connected': False,
                    'port': None,
                })
            
            elif msg_type == 'list_ports':
                ports = SerialForwarder.list_ports()
                await self._send({
                    'type': 'port_list',
                    'ports': ports,
                })
            
            # --- 烧录 ---
            elif msg_type == 'detect_bootloader':
                dev = detect_bootloader()
                await self._send({
                    'type': 'bootloader_detect',
                    'found': dev is not None,
                    'device': dev,
                })
            
            elif msg_type == 'flash_firmware':
                path = data.get('path', '')
                chip = data.get('chip', 'CH32V305')
                result = await asyncio.to_thread(flash_firmware, path, chip)
                await self._send({
                    'type': 'flash_result',
                    'success': result['success'],
                    'message': result['message'],
                })
            
            # --- 状态 ---
            elif msg_type == 'get_state':
                await self._send_state()
            
            # --- 采集卡控制 ---
            elif msg_type == 'capture_start':
                camera_index = data.get('camera_index', 0)
                await self._start_capture(camera_index)
            
            elif msg_type == 'capture_stop':
                await self._stop_capture()
            
            elif msg_type == 'list_cameras':
                try:
                    from capture_card import CaptureCard
                    # 使用 run_in_executor 避免阻塞事件循环
                    cameras = await asyncio.to_thread(CaptureCard.list_cameras)
                    cam_list = [{'index': c[0], 'name': c[1], 'is_ms2130': c[2]} for c in cameras]
                except Exception as e:
                    logger.error(f"List cameras error: {e}")
                    cam_list = []
                await self._send({
                    'type': 'camera_list',
                    'cameras': cam_list,
                })

            # --- 模型管理 ---
            elif msg_type == 'list_models':
                try:
                    # 在项目根目录和 pc_app 目录下查找 .onnx 文件
                    search_dirs = [
                        os.path.join(os.path.dirname(__file__), '..'),
                        os.path.join(os.path.dirname(__file__), '..', '..'),
                    ]
                    models = []
                    for d in search_dirs:
                        abs_d = os.path.abspath(d)
                        for f in glob.glob(os.path.join(abs_d, '*.onnx')):
                            models.append({
                                'path': f,
                                'name': os.path.basename(f),
                                'size': os.path.getsize(f),
                            })
                    # 去重 (按文件名)
                    seen = set()
                    unique_models = []
                    for m in models:
                        if m['name'] not in seen:
                            seen.add(m['name'])
                            unique_models.append(m)
                    await self._send({
                        'type': 'model_list',
                        'models': unique_models,
                        'current': self.model_path or '',
                    })
                except Exception as e:
                    logger.error(f"List models error: {e}")
                    await self._send({
                        'type': 'model_list',
                        'models': [],
                        'current': '',
                    })

            elif msg_type == 'load_model':
                model_path = data.get('path', '')
                if not model_path or not os.path.exists(model_path):
                    await self._send({
                        'type': 'model_status',
                        'loaded': False,
                        'error': f'Model not found: {model_path}',
                    })
                    return
                try:
                    # 如果正在采集, 先停止检测管道
                    if self._pipeline_task:
                        self._pipeline_task.cancel()
                        try:
                            await self._pipeline_task
                        except asyncio.CancelledError:
                            pass
                        self._pipeline_task = None

                    # 加载新模型
                    logger.info(f"Loading model: {model_path}")
                    if self.detector:
                        await asyncio.to_thread(self.detector.load_model, model_path)
                    else:
                        from object_detector import ObjectDetector
                        self.detector = ObjectDetector()
                        await asyncio.to_thread(self.detector.load_model, model_path)
                    # 更新类别名称 (与实际模型匹配)
                    await asyncio.to_thread(update_class_names, model_path)

                    self.model_path = model_path
                    # 保存最后加载的模型路径
                    self.config.set('model', 'last_path', model_path)
                    logger.info(f"Model loaded: {model_path}")

                    # 如果采集卡正在运行, 重新启动检测管道
                    if self._capture_active:
                        self._pipeline_task = asyncio.create_task(self._detection_loop())

                    await self._send({
                        'type': 'model_status',
                        'loaded': True,
                        'path': model_path,
                        'name': os.path.basename(model_path),
                    })
                except Exception as e:
                    logger.error(f"Load model error: {e}")
                    await self._send({
                        'type': 'model_status',
                        'loaded': False,
                        'error': str(e),
                    })

            # --- 轨迹控制 ---
            elif msg_type == 'trajectory_enable':
                self._trajectory_enabled = True
                self.trajectory.config.enabled = True
                self.config.set('trajectory', 'enabled', True)
                logger.info("Trajectory enabled")
                await self._send({
                    'type': 'trajectory_status',
                    'enabled': True,
                })

            elif msg_type == 'trajectory_disable':
                self._trajectory_enabled = False
                self.trajectory.config.enabled = False
                self.config.set('trajectory', 'enabled', False)
                logger.info("Trajectory disabled")
                await self._send({
                    'type': 'trajectory_status',
                    'enabled': False,
                })
            
            elif msg_type == 'trajectory_config':
                config = self.trajectory.config
                if 'smooth_factor' in data:
                    config.smooth_factor = float(data['smooth_factor'])
                if 'max_step_px' in data:
                    config.max_step_px = int(data['max_step_px'])
                if 'min_confidence' in data:
                    config.min_confidence = float(data['min_confidence'])
                    # 同步到目标检测器 (推理时也用这个阈值)
                    if self.detector:
                        self.detector.CONFIDENCE_THRESHOLD = config.min_confidence
                if 'target_offset_x' in data:
                    config.target_offset_x = int(data['target_offset_x'])
                if 'target_offset_y' in data:
                    config.target_offset_y = int(data['target_offset_y'])
                if 'jitter_amount' in data:
                    config.jitter_amount = float(data['jitter_amount'])
                if 'target_priority' in data:
                    config.target_priority = int(data['target_priority'])
                if 'prediction_ticks' in data:
                    config.prediction_ticks = int(data['prediction_ticks'])
                if 'fov_radius' in data:
                    config.fov_radius = int(data['fov_radius'])
                if 'trigger_enabled' in data:
                    self._trigger_enabled = bool(data['trigger_enabled'])
                if 'trigger_threshold' in data:
                    self._trigger_threshold = int(data['trigger_threshold'])
                if 'invert_ai_x' in data:
                    config.invert_ai_x = bool(data['invert_ai_x'])
                if 'invert_ai_y' in data:
                    config.invert_ai_y = bool(data['invert_ai_y'])
                if 'kp' in data:
                    config.kp = float(data['kp'])
                if 'ki' in data:
                    config.ki = float(data['ki'])
                if 'kd' in data:
                    config.kd = float(data['kd'])
                if 'integral_limit' in data:
                    config.integral_limit = float(data['integral_limit'])
                if 'max_steps_per_frame' in data:
                    config.max_steps_per_frame = int(data['max_steps_per_frame'])
                if 'settle_deadzone' in data:
                    config.settle_deadzone = float(data['settle_deadzone'])
                if 'unsettle_hysteresis' in data:
                    config.unsettle_hysteresis = float(data['unsettle_hysteresis'])
                if 'y_scale' in data:
                    config.y_scale = float(data['y_scale'])
                self.trajectory.set_config(config)
                # 持久化保存所有轨迹配置
                self.config.set('trajectory', 'smooth_factor', config.smooth_factor)
                self.config.set('trajectory', 'max_step_px', config.max_step_px)
                self.config.set('trajectory', 'min_confidence', config.min_confidence)
                self.config.set('trajectory', 'target_offset_x', config.target_offset_x)
                self.config.set('trajectory', 'target_offset_y', config.target_offset_y)
                self.config.set('trajectory', 'jitter_amount', config.jitter_amount)
                self.config.set('trajectory', 'target_priority', config.target_priority)
                self.config.set('trajectory', 'prediction_ticks', config.prediction_ticks)
                self.config.set('trajectory', 'fov_radius', config.fov_radius)
                self.config.set('trajectory', 'trigger_enabled', self._trigger_enabled)
                self.config.set('trajectory', 'trigger_threshold', self._trigger_threshold)
                self.config.set('trajectory', 'invert_ai_x', config.invert_ai_x)
                self.config.set('trajectory', 'invert_ai_y', config.invert_ai_y)
                self.config.set('trajectory', 'kp', config.kp)
                self.config.set('trajectory', 'ki', config.ki)
                self.config.set('trajectory', 'kd', config.kd)
                self.config.set('trajectory', 'integral_limit', config.integral_limit)
                self.config.set('trajectory', 'max_steps_per_frame', config.max_steps_per_frame)
                self.config.set('trajectory', 'settle_deadzone', config.settle_deadzone)
                self.config.set('trajectory', 'unsettle_hysteresis', config.unsettle_hysteresis)
                self.config.set('trajectory', 'y_scale', config.y_scale)
                await self._send({
                    'type': 'trajectory_config_ack',
                    'config': {
                        'smooth_factor': config.smooth_factor,
                        'max_step_px': config.max_step_px,
                        'min_confidence': config.min_confidence,
                        'target_offset_x': config.target_offset_x,
                        'target_offset_y': config.target_offset_y,
                        'jitter_amount': config.jitter_amount,
                        'target_priority': config.target_priority,
                        'prediction_ticks': config.prediction_ticks,
                        'fov_radius': config.fov_radius,
                        'trigger_enabled': self._trigger_enabled,
                        'trigger_threshold': self._trigger_threshold,
                        'invert_ai_x': config.invert_ai_x,
                        'invert_ai_y': config.invert_ai_y,
                        'kp': config.kp,
                        'ki': config.ki,
                        'kd': config.kd,
                        'integral_limit': config.integral_limit,
                        'max_steps_per_frame': config.max_steps_per_frame,
                        'settle_deadzone': config.settle_deadzone,
                        'unsettle_hysteresis': config.unsettle_hysteresis,
                        'y_scale': config.y_scale,
                    }
                })
            
            elif msg_type == 'trajectory_clear':
                self.trajectory.clear_trail()
                await self._send({
                    'type': 'trajectory_cleared',
                })
            
            elif msg_type == 'trajectory_get_points':
                points = self.trajectory.get_trail_points()
                await self._send({
                    'type': 'trajectory_points',
                    'points': [
                        {'x': p.x, 'y': p.y, 'is_ai': p.is_ai}
                        for p in points
                    ],
                })
            
            # --- 画面显示开关 ---
            elif msg_type == 'toggle_video':
                self._show_video = data.get('show', not self._show_video)
                logger.info(f"Video display {'enabled' if self._show_video else 'disabled'}")
                await self._send({
                    'type': 'video_status',
                    'show_video': self._show_video,
                })

            # --- 锁定模式 ---
            elif msg_type == 'lock_mode_enable':
                await self._lock_mode_on()

            elif msg_type == 'lock_mode_disable':
                await self._lock_mode_off()

            # --- 屏幕尺寸 ---
            elif msg_type == 'set_screen_size':
                w = data.get('width', self._screen_w)
                h = data.get('height', self._screen_h)
                self._screen_w = w
                self._screen_h = h
                self.trajectory.set_screen_center(w / 2, h / 2)
                logger.info(f"Screen size set to {w}x{h}")
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON: {message}")
    
    # ================================================================
    # 事件回调
    # ================================================================
    
    def _on_mouse_event(self, event: MouseEvent):
        """鼠标事件回调 (从 pynput 线程直接发送串口)"""
        self.stats['mouse_events'] += 1

        # 过滤 SetCursorPos 回弹事件 (单次位移超过 200px 的直接丢弃)
        if abs(event.dx) > 200 or abs(event.dy) > 200:
            return

        # 编码并直接发送到串口 (不走 asyncio, 减少延迟)
        packet = encode_packet(event.buttons, event.dx, event.dy, event.wheel)
        if self.serial.is_connected:
            try:
                self.serial._serial.write(packet)
                self.stats['packets_sent'] += 1
                self.stats['bytes_sent'] += len(packet)
            except Exception:
                pass

        # 异步通知前端 (不阻塞串口发送)
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._notify_mouse_event(event),
                self._loop
            )

    async def _notify_mouse_event(self, event: MouseEvent):
        """异步通知前端鼠标事件"""
        if self._ws_client:
            try:
                await self._send({
                    'type': 'mouse_event',
                    'buttons': event.buttons,
                    'left': event.left,
                    'right': event.right,
                    'middle': event.middle,
                    'dx': event.dx,
                    'dy': event.dy,
                    'wheel': event.wheel,
                    'serial_connected': self.serial.is_connected,
                })
            except Exception:
                pass
    
    def _on_serial_connection(self, connected: bool, port: Optional[str]):
        """串口连接状态变化回调"""
        logger.info(f"Serial connection changed: connected={connected}, port={port}")
        
        if self._ws_client:
            asyncio.run_coroutine_threadsafe(
                self._send({
                    'type': 'serial_status',
                    'connected': connected,
                    'port': port,
                }),
                self._loop
            )
    
    async def _send_detection_frame(self, detections: List[Detection],
                                    ai_steps: List[Tuple[int, int]],
                                    frame: Optional[np.ndarray] = None):
        """发送检测帧数据到前端"""
        if not self._ws_client:
            return
        
        try:
            data = {
                'type': 'detection_frame',
                'detections': [
                    {
                        'x': d.x, 'y': d.y,
                        'w': d.w, 'h': d.h,
                        'cx': d.cx, 'cy': d.cy,
                        'confidence': d.confidence,
                        'class_id': d.class_id,
                    }
                    for d in detections
                ],
                'ai_steps': [{'dx': s[0], 'dy': s[1]} for s in ai_steps],
                'trajectory_stats': self.trajectory.get_stats(),
                'inference_ms': round(self.detector.avg_inference_time_ms, 1) if self.detector and self.detector.is_loaded else 0,
                'pipeline_fps': round(self._pipeline_fps, 1),
                'inf_fps': round(1000 / self.detector.avg_inference_time_ms, 1) if self.detector and self.detector.avg_inference_time_ms > 0 else 0,
                # 目标与屏幕中心的偏移量 (用于前端显示)
                'target_dx': self.trajectory.aim_x - self._screen_w / 2 if self.trajectory.selected_target else 0,
                'target_dy': self.trajectory.aim_y - self._screen_h / 2 if self.trajectory.selected_target else 0,
                # AI 步数总计
                'ai_step_count': len(ai_steps),
                'ai_step_total_dx': sum(s[0] for s in ai_steps) if ai_steps else 0,
                'ai_step_total_dy': sum(s[1] for s in ai_steps) if ai_steps else 0,
                # 对准状态
                'is_settled': self.trajectory.is_settled,
                'selected_target': self.trajectory.selected_target is not None,
                'target_info': {
                    'class_id': self.trajectory.selected_target.class_id,
                    'confidence': round(self.trajectory.selected_target.confidence, 3),
                } if self.trajectory.selected_target else None,
            }
            
            # 如果有帧且画面显示开启, 添加 JPEG 编码的标注帧
            if frame is not None and self._show_video:
                # 在后台线程中执行绘图和编码
                inference_ms = self.detector.avg_inference_time_ms if self.detector else 0

                # 先缩小到 1080p 再编码, 减少传输延迟
                import cv2
                h, w = frame.shape[:2]
                scale = 1.0
                if w > 1920:
                    scale = 1920 / w
                    new_w, new_h = int(w * scale), int(h * scale)
                    small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    # 检测框坐标也要按比例缩放
                    scaled_detections = []
                    for d in detections:
                        scaled_detections.append(Detection(
                            x=d.x * scale, y=d.y * scale,
                            w=d.w * scale, h=d.h * scale,
                            confidence=d.confidence, class_id=d.class_id,
                            cx=d.cx * scale, cy=d.cy * scale,
                        ))
                else:
                    small_frame = frame
                    scaled_detections = detections
                    scale = 1.0

                # 缩放瞄准点坐标
                sel = self.trajectory.selected_target
                scaled_target_cx = sel.cx * scale if sel else None
                scaled_target_cy = sel.cy * scale if sel else None
                scaled_aim_x = self.trajectory.aim_x * scale
                scaled_aim_y = self.trajectory.aim_y * scale

                annotated = await asyncio.to_thread(
                    draw_detections_on_frame, small_frame.copy(), scaled_detections, self._pipeline_fps, inference_ms,
                    scaled_target_cx, scaled_target_cy, scaled_aim_x, scaled_aim_y,
                    self.trajectory.is_settled
                )
                # 降低 JPEG 质量到 50 以加快编码速度
                jpeg_bytes = await asyncio.to_thread(encode_frame_jpeg, annotated, 50)
                # 将 JPEG 字节用 base64 编码发送
                import base64
                data['frame_jpeg'] = base64.b64encode(jpeg_bytes).decode('ascii')
                data['frame_width'] = int(small_frame.shape[1])
                data['frame_height'] = int(small_frame.shape[0])
            
            await self._send(data)
        except Exception as e:
            logger.error(f"Send detection frame error: {e}")
    
    # ================================================================
    # 状态管理
    # ================================================================
    
    async def _send_state(self):
        """发送完整状态到前端 (包含所有配置, 用于同步UI)"""
        uptime = time.time() - self.start_time
        c = self.trajectory.config
        await self._send({
            'type': 'state',
            'mouse_active': self._mouse_active,
            'serial_connected': self.serial.is_connected,
            'serial_port': self.serial.port_name,
            'stats': self.stats,
            'uptime': uptime,
            'trajectory': {
                'enabled': self._trajectory_enabled,
                'capture_active': self._capture_active,
                'model_loaded': self.detector.is_loaded if self.detector else False,
            },
            'lock_mode': self._lock_mode,
            # 当前配置 (前端用于同步滑块)
            'config': {
                'smooth_factor': c.smooth_factor,
                'max_step_px': c.max_step_px,
                'min_confidence': c.min_confidence,
                'target_offset_x': c.target_offset_x,
                'target_offset_y': c.target_offset_y,
                'jitter_amount': c.jitter_amount,
                'target_priority': c.target_priority,
                'prediction_ticks': c.prediction_ticks,
                'fov_radius': c.fov_radius,
                'trigger_enabled': self._trigger_enabled,
                'trigger_threshold': self._trigger_threshold,
                'invert_ai_x': c.invert_ai_x,
                'invert_ai_y': c.invert_ai_y,
                'kp': c.kp,
                'ki': c.ki,
                'kd': c.kd,
                'integral_limit': c.integral_limit,
                'max_steps_per_frame': c.max_steps_per_frame,
                'settle_deadzone': c.settle_deadzone,
                'unsettle_hysteresis': c.unsettle_hysteresis,
                'y_scale': c.y_scale,
            },
        })
    
    async def _send(self, data: dict):
        """发送 JSON 消息到 WebSocket 客户端"""
        if self._ws_client:
            try:
                await self._ws_client.send(json.dumps(data))
            except Exception as e:
                logger.error(f"Send error: {e}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='Mouse Forwarder Backend')
    parser.add_argument('--port', type=int, default=DEFAULT_WS_PORT,
                       help=f'WebSocket port (default: {DEFAULT_WS_PORT})')
    parser.add_argument('--baud', type=int, default=DEFAULT_SERIAL_BAUD,
                       help=f'Serial baud rate (default: {DEFAULT_SERIAL_BAUD})')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to ONNX model file (default: auto-detect valorant.onnx)')
    args = parser.parse_args()
    
    # 自动检测模型路径
    model_path = args.model
    if model_path is None:
        # 从项目根目录查找
        import os
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'best.onnx'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'best.pt'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'valorant.onnx'),
            os.path.join(os.path.dirname(__file__), '..', 'best.onnx'),
            os.path.join(os.path.dirname(__file__), '..', 'valorant.onnx'),
            os.path.join(os.path.dirname(__file__), 'best.onnx'),
            os.path.join(os.path.dirname(__file__), 'valorant.onnx'),
        ]
        for p in possible_paths:
            abs_p = os.path.abspath(p)
            if os.path.exists(abs_p):
                model_path = abs_p
                logger.info(f"Auto-detected model: {model_path}")
                break
        if model_path is None:
            logger.info("No model found, object detection will be disabled")
    
    backend = MouseForwarderBackend(
        ws_port=args.port,
        serial_baud=args.baud,
        model_path=model_path,
    )
    
    try:
        asyncio.run(backend.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        asyncio.run(backend.stop())


if __name__ == '__main__':
    main()