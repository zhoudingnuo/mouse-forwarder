"""
mouse_monitor.py - 鼠标捕获模块

使用 pynput 监听全局鼠标事件, 计算相对位移并回调。
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from pynput import mouse

logger = logging.getLogger(__name__)


@dataclass
class MouseEvent:
    """鼠标事件数据"""
    buttons: int = 0       # 按钮状态 (bit0=左, bit1=右, bit2=中, bit4=后退, bit5=前进)
    dx: int = 0            # X 相对位移
    dy: int = 0            # Y 相对位移
    wheel: int = 0         # 滚轮增量
    left: bool = False
    right: bool = False
    middle: bool = False
    back: bool = False     # 侧键后退 (X1)
    forward: bool = False  # 侧键前进 (X2)


# 按钮位映射
BUTTON_BITS = {
    mouse.Button.left:   1,
    mouse.Button.right:  2,
    mouse.Button.middle: 4,
    mouse.Button.x1:     16,  # 后退键 (bit4)
    mouse.Button.x2:     32,  # 前进键 (bit5)
}


class MouseMonitor:
    """
    鼠标监控器
    
    使用 pynput 的全局鼠标监听器捕获鼠标事件,
    计算相对位移并通过回调函数通知。
    支持 suppress 模式: 阻止鼠标事件传播到本机 OS。
    """
    
    def __init__(self, on_event: Optional[Callable[[MouseEvent], None]] = None):
        self._on_event = on_event
        self._listener: Optional[mouse.Listener] = None
        self._running = False
        self._suppressed = False  # 是否阻止鼠标事件传播到本机

        # 鼠标状态
        self._prev_x: Optional[int] = None
        self._prev_y: Optional[int] = None
        self._buttons_state = 0

        # 事件队列 (用于异步处理)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def is_suppressed(self) -> bool:
        return self._suppressed
    
    def set_event_callback(self, callback: Callable[[MouseEvent], None]):
        """设置事件回调"""
        self._on_event = callback
    
    def start(self, suppress: bool = False):
        """启动鼠标监听"""
        if self._running:
            logger.warning("MouseMonitor already running")
            return
        
        self._running = True
        self._suppressed = suppress
        self._prev_x = None
        self._prev_y = None
        self._buttons_state = 0
        
        # 创建监听器, suppress=True 阻止事件传播到本机
        self._listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
            suppress=suppress,
        )
        self._listener.start()
        logger.info(f"Mouse monitor started (suppress={suppress})")
    
    def stop(self):
        """停止鼠标监听"""
        self._running = False
        if self._listener and self._listener.running:
            self._listener.stop()
            self._listener = None
        logger.info("Mouse monitor stopped")
    
    def set_suppress(self, suppress: bool) -> bool:
        """
        切换锁定模式

        锁定模式只屏蔽鼠标按钮事件 (左/右/中键), 不影响移动和滚轮。
        这样本机光标位置正常, 不会闪回, 但点击不会作用到本机窗口。
        移动和点击数据照常转发到目标 PC。

        Args:
            suppress: True=锁定 (屏蔽按钮), False=正常

        Returns:
            是否成功切换
        """
        if not self._running:
            logger.warning("MouseMonitor not running, cannot set suppress")
            return False

        if self._suppressed == suppress:
            return True

        logger.info(f"Switching lock mode: {self._suppressed} -> {suppress}")
        self._suppressed = suppress

        # 只在按钮事件上 suppress, 不拦截移动
        # 重新创建监听器, suppress=True 会拦截按钮, 但我们让 on_move 正常工作
        old_listener = self._listener
        if old_listener:
            old_listener.stop()
            # 等待监听器线程退出
            if hasattr(old_listener, '_thread') and old_listener._thread:
                for _ in range(50):
                    if not old_listener.running:
                        break
                    time.sleep(0.1)
            time.sleep(0.1)

        # 重置位置跟踪 (避免重启后产生跳变)
        self._prev_x = None
        self._prev_y = None

        # 创建新监听器
        # suppress=True 在 Windows 上: 拦截按钮事件, 但移动事件仍会触发回调
        # 这样本机不会响应点击, 但光标位置正常更新
        self._listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
            suppress=suppress,
        )
        self._listener.start()
        self._listener.wait()

        logger.info(f"Lock mode changed to {suppress}")
        return True

    def _emit_event(self, event: MouseEvent):
        """发送事件到回调"""
        if self._on_event:
            try:
                self._on_event(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def _on_move(self, x: int, y: int):
        """鼠标移动回调"""
        if not self._running:
            return

        if self._prev_x is not None and self._prev_y is not None:
            dx = x - self._prev_x
            dy = y - self._prev_y

            # 只发送有实际位移的事件
            if dx != 0 or dy != 0:
                event = MouseEvent(
                    buttons=self._buttons_state,
                    dx=dx,
                    dy=dy,
                    left=bool(self._buttons_state & 1),
                    right=bool(self._buttons_state & 2),
                    middle=bool(self._buttons_state & 4),
                    back=bool(self._buttons_state & 16),
                    forward=bool(self._buttons_state & 32),
                )
                self._emit_event(event)
        
        self._prev_x = x
        self._prev_y = y
    
    def _on_click(self, x: int, y: int, button: mouse.Button, pressed: bool):
        """鼠标点击回调 (支持左、右、中、后退X1、前进X2)"""
        if not self._running:
            return

        # 更新按钮状态
        bit = BUTTON_BITS.get(button, 0)
        if bit == 0:
            return  # 不支持的按钮

        if pressed:
            self._buttons_state |= bit
        else:
            self._buttons_state &= ~bit

        # 发送点击事件 (带 0 位移)
        event = MouseEvent(
            buttons=self._buttons_state,
            dx=0,
            dy=0,
            left=bool(self._buttons_state & 1),
            right=bool(self._buttons_state & 2),
            middle=bool(self._buttons_state & 4),
            back=bool(self._buttons_state & 16),
            forward=bool(self._buttons_state & 32),
        )
        self._emit_event(event)
    
    def _on_scroll(self, x: int, y: int, dx: int, dy: int):
        """滚轮滚动回调"""
        if not self._running:
            return
        
        # dy 是垂直滚动, dx 是水平滚动 (通常为 0)
        event = MouseEvent(
            buttons=self._buttons_state,
            dx=0,
            dy=0,
            wheel=int(dy),
            left=bool(self._buttons_state & 1),
            right=bool(self._buttons_state & 2),
            middle=bool(self._buttons_state & 4),
            back=bool(self._buttons_state & 16),
            forward=bool(self._buttons_state & 32),
        )
        self._emit_event(event)