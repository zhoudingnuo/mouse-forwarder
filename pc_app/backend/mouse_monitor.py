"""
mouse_monitor.py - 鼠标捕获模块

使用 pynput 监听全局鼠标事件, 计算相对位移并回调。
支持动态抑制 (suppress) 模式。
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
    mouse.Button.x1:     8,   # 后退键 (bit3 = Button 4)
    mouse.Button.x2:     16,  # 前进键 (bit4 = Button 5)
}


class MouseMonitor:
    """
    鼠标监控器 (基于 pynput)

    使用 pynput 的全局鼠标监听器捕获鼠标事件,
    计算相对位移并通过回调函数通知。
    支持动态切换 suppress 模式 (重启监听器)。
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
        self._skip_move = False  # 跳过下一个移动事件 (SetCursorPos 回弹)
        # 上一步有效移动 (大位移/回中时沿用, 保持自瞄连续)
        self._last_dx = 0
        self._last_dy = 0

        # 屏幕尺寸 (用于检测光标环绕跳边)
        self._screen_w, self._screen_h = self._get_screen_size()
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def is_suppressed(self) -> bool:
        return self._suppressed
    
    def set_event_callback(self, callback: Callable[[MouseEvent], None]):
        """设置事件回调"""
        self._on_event = callback

    def skip_next_move(self):
        """跳过下一个移动事件 (用于 SetCursorPos 后防止回弹)"""
        self._skip_move = True

    def suppress_current_event(self):
        """
        抑制当前 WH_MOUSE_LL 事件 (PowerToys 风格: return 1)

        在 pynput 回调中调用此方法, 设置 listener._suppress = True。
        pynput 的 C 代码会在回调返回后检查此标志, 并返回 1 抑制原始事件。
        这样 SetCursorPos 后的回弹事件被抑制, 光标不会回到原位。
        """
        if self._listener:
            self._listener._suppress = True
    
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
        
        # 创建监听器
        self._listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
            suppress=suppress,
        )
        self._listener.start()
        self._listener.wait()
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
        切换抑制模式 (重启监听器)

        切换 suppress 后需要重启监听器, 会短暂丢失事件。
        但这是唯一可靠的方式 (pynput 不支持动态切换 suppress)。

        Args:
            suppress: True=抑制 (阻止事件传递到本机)

        Returns:
            是否成功切换
        """
        if not self._running:
            logger.warning("MouseMonitor not running, cannot set suppress")
            return False

        if self._suppressed == suppress:
            return True

        logger.info(f"Switching suppress: {self._suppressed} -> {suppress}")

        # 停止旧监听器
        old_listener = self._listener
        if old_listener:
            old_listener.stop()
            # 等待线程退出
            if hasattr(old_listener, '_thread') and old_listener._thread:
                for _ in range(50):
                    if not old_listener.running:
                        break
                    time.sleep(0.02)
            time.sleep(0.05)
        
        self._suppressed = suppress
        self._prev_x = None
        self._prev_y = None
        
        # 创建新监听器
        self._listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
            suppress=suppress,
        )
        self._listener.start()
        self._listener.wait()
        
        logger.info(f"Suppress changed to {suppress}")
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

        # 重置抑制标志 (上次回调可能设置了 suppress_current_event)
        if self._listener and self._listener._suppress:
            self._listener._suppress = False

        # SetCursorPos 回中: 不转发回中位移, 但沿用上一步操作保持自瞄连续
        if self._skip_move:
            self._skip_move = False
            # 沿用上一步有效移动 (AI 自瞄方向连续)
            if self._last_dx != 0 or self._last_dy != 0:
                event = MouseEvent(
                    buttons=self._buttons_state,
                    dx=self._last_dx,
                    dy=self._last_dy,
                    left=bool(self._buttons_state & 1),
                    right=bool(self._buttons_state & 2),
                    middle=bool(self._buttons_state & 4),
                    back=bool(self._buttons_state & 8),
                    forward=bool(self._buttons_state & 16),
                )
                self._emit_event(event)
            # 更新基准位置 (避免后续产生虚假大位移)
            self._prev_x = x
            self._prev_y = y
            return

        if self._prev_x is not None and self._prev_y is not None:
            dx = x - self._prev_x
            dy = y - self._prev_y

            # 大位移 (光标环绕/回中跳变): 不丢弃, 沿用上一步操作
            if abs(dx) > self._screen_w // 2 or abs(dy) > self._screen_h // 2:
                logger.debug(f"Cursor wrap detected: dx={dx}, dy={dy}, reusing last move")
                if self._last_dx != 0 or self._last_dy != 0:
                    event = MouseEvent(
                        buttons=self._buttons_state,
                        dx=self._last_dx,
                        dy=self._last_dy,
                        left=bool(self._buttons_state & 1),
                        right=bool(self._buttons_state & 2),
                        middle=bool(self._buttons_state & 4),
                        back=bool(self._buttons_state & 8),
                        forward=bool(self._buttons_state & 16),
                    )
                    self._emit_event(event)
                self._prev_x = x
                self._prev_y = y
                return

            # 只发送有实际位移的事件
            if dx != 0 or dy != 0:
                # 记录上一步有效移动 (供大位移时沿用)
                self._last_dx = dx
                self._last_dy = dy
                event = MouseEvent(
                    buttons=self._buttons_state,
                    dx=dx,
                    dy=dy,
                    left=bool(self._buttons_state & 1),
                    right=bool(self._buttons_state & 2),
                    middle=bool(self._buttons_state & 4),
                    back=bool(self._buttons_state & 8),
                    forward=bool(self._buttons_state & 16),
                )
                self._emit_event(event)
        
        self._prev_x = x
        self._prev_y = y
    
    def _on_click(self, x: int, y: int, button: mouse.Button, pressed: bool):
        """鼠标点击回调 (支持左、右、中、后退X1、前进X2)"""
        if not self._running:
            return

        bit = BUTTON_BITS.get(button, 0)
        if bit == 0:
            return  # 不支持的按钮

        if pressed:
            self._buttons_state |= bit
        else:
            self._buttons_state &= ~bit

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
    
    def _get_screen_size(self):
        """获取屏幕尺寸 (用于检测光标环绕跳边)"""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            return 1920, 1080