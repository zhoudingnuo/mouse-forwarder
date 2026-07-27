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
import signal
import sys
import time
from typing import Any, Optional, List, Tuple

import numpy as np
import websockets

from mouse_monitor import MouseMonitor, MouseEvent
from serial_forwarder import SerialForwarder
from protocol import encode_packet
from flasher import flash_firmware, detect_bootloader, erase_flash
from trajectory_calculator import TrajectoryCalculator, TrajectoryConfig, Detection
from capture_inference import draw_detections_on_frame, encode_frame_jpeg

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
        
        # 组件
        self.mouse = MouseMonitor(on_event=self._on_mouse_event)
        self.serial = SerialForwarder(baudrate=serial_baud)
        self.serial.set_connection_callback(self._on_serial_connection)
        
        # 目标检测 & 轨迹组件 (延迟初始化)
        self.capture_card = None
        self.detector = None
        self.trajectory = TrajectoryCalculator()
        
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
        self._lock_mode = False  # 锁定模式: 鼠标不控制本机
        self._keyboard_listener = None  # Escape 键监听器
        
        # 检测管道任务
        self._pipeline_task: Optional[asyncio.Task] = None
        
        # 屏幕中心 (默认 1920x1080)
        self._screen_w = 1920
        self._screen_h = 1080
    
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
        
        # 初始化检测组件 (如果提供了模型路径)
        if self.model_path:
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
            await self._start_capture(0)  # camera_index=0 会自动查找 MS2130
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

    async def _lock_mode_on(self):
        """进入锁定模式: 鼠标不控制本机, 按 Escape 退出"""
        if self._lock_mode:
            return

        self._lock_mode = True
        logger.info("Lock mode ON - mouse will not control local machine")

        # 切换鼠标监听器为 suppress 模式 (阻止事件传播到本机)
        self.mouse.set_suppress(True)

        # 启动键盘监听 (检测 Escape 键退出)
        if HAS_KEYBOARD and self._loop:
            def on_press(key):
                try:
                    if key == keyboard.Key.esc:
                        logger.info("Escape pressed, exiting lock mode")
                        # 从线程安全地调用协程
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

        # 恢复鼠标监听器正常模式
        self.mouse.set_suppress(False)

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
                
                # 执行检测 (在后台线程运行, 不阻塞)
                if self.detector and self.detector.is_loaded:
                    detections = await asyncio.to_thread(
                        self.detector.detect, frame, frame.shape[:2]
                    )
                else:
                    detections = []
                
                self.stats['detections'] = len(detections)
                
                # 计算轨迹
                ai_steps = self.trajectory.calculate(detections)
                
                # 发送 AI 轨迹到串口
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
                await self.serial.connect(port)
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
            
            # --- 轨迹控制 ---
            elif msg_type == 'trajectory_enable':
                self._trajectory_enabled = True
                self.trajectory.config.enabled = True
                logger.info("Trajectory enabled")
                await self._send({
                    'type': 'trajectory_status',
                    'enabled': True,
                })
            
            elif msg_type == 'trajectory_disable':
                self._trajectory_enabled = False
                self.trajectory.config.enabled = False
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
                if 'target_offset_x' in data:
                    config.target_offset_x = int(data['target_offset_x'])
                if 'target_offset_y' in data:
                    config.target_offset_y = int(data['target_offset_y'])
                if 'jitter_amount' in data:
                    config.jitter_amount = float(data['jitter_amount'])
                self.trajectory.set_config(config)
                await self._send({
                    'type': 'trajectory_config_ack',
                    'config': {
                        'smooth_factor': config.smooth_factor,
                        'max_step_px': config.max_step_px,
                        'min_confidence': config.min_confidence,
                        'target_offset_x': config.target_offset_x,
                        'target_offset_y': config.target_offset_y,
                        'jitter_amount': config.jitter_amount,
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
        """鼠标事件回调 (从 pynput 线程调用)"""
        self.stats['mouse_events'] += 1
        
        # 更新轨迹计算器的鼠标位置
        # 近似: 从屏幕中心 + 累积位移
        # 注: 更精确的鼠标位置需要额外跟踪
        self.trajectory.update_mouse_position(
            self._screen_w / 2 + event.dx,
            self._screen_h / 2 + event.dy,
        )
        
        # 添加真实鼠标轨迹点
        self.trajectory.add_real_mouse_point(event.dx, event.dy)
        
        # 编码并发送到串口
        packet = encode_packet(event.buttons, event.dx, event.dy, event.wheel)
        
        # 使用 asyncio.run_coroutine_threadsafe 从线程安全地调用
        asyncio.run_coroutine_threadsafe(
            self._forward_mouse_event(event, packet),
            self._loop
        )
    
    async def _forward_mouse_event(self, event: MouseEvent, packet: bytes):
        """转发鼠标事件到串口和前端"""
        # 发送到串口
        if self.serial.is_connected:
            success = await self.serial.send(packet)
            if success:
                self.stats['packets_sent'] += 1
                self.stats['bytes_sent'] += len(packet)
        
        # 发送到前端
        if self._ws_client:
            try:
                # 获取轨迹点 (限频发送, 避免数据过大)
                trail_points = self.trajectory.get_trail_points()
                recent_trail = trail_points[-50:] if len(trail_points) > 50 else trail_points
                
                await self._send({
                    'type': 'mouse_event',
                    'buttons': event.buttons,
                    'left': event.left,
                    'right': event.right,
                    'middle': event.middle,
                    'back': getattr(event, 'back', False),
                    'forward': getattr(event, 'forward', False),
                    'dx': event.dx,
                    'dy': event.dy,
                    'wheel': event.wheel,
                    'serial_connected': self.serial.is_connected,
                    'trajectory': {
                        'enabled': self._trajectory_enabled,
                        'trail_points': [
                            {'x': p.x, 'y': p.y, 'is_ai': p.is_ai}
                            for p in recent_trail
                        ],
                    },
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
            }
            
            # 如果有帧且画面显示开启, 添加 JPEG 编码的标注帧
            if frame is not None and self._show_video:
                # 在后台线程中执行绘图和编码
                inference_ms = self.detector.avg_inference_time_ms if self.detector else 0
                
                # 先缩小到 720p 再编码, 减少传输延迟
                import cv2
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    new_w, new_h = int(w * scale), int(h * scale)
                    small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    small_frame = frame
                
                annotated = await asyncio.to_thread(
                    draw_detections_on_frame, small_frame.copy(), detections, inference_ms
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
        """发送完整状态到前端"""
        uptime = time.time() - self.start_time
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
            os.path.join(os.path.dirname(__file__), '..', '..', 'valorant.onnx'),
            os.path.join(os.path.dirname(__file__), '..', 'valorant.onnx'),
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