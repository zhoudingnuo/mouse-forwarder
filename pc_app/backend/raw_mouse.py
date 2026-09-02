"""
raw_mouse.py - Raw Input 鼠标捕获 (直接发送, 按钮状态保持)
"""

import ctypes
import threading
import logging
from ctypes import wintypes
from typing import Callable, Optional

logger = logging.getLogger(__name__)

WM_INPUT = 0x00FF
RIM_TYPEMOUSE = 0
RID_INPUT = 0x10000003
RIDEV_INPUTSINK = 0x100        # 后台窗口也接收输入
RIDEV_EXINPUTSINK = 0x10       # 接收合成 (SendInput 注入) 事件

RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
RI_MOUSE_LEFT_BUTTON_UP = 0x0002
RI_MOUSE_RIGHT_BUTTON_DOWN = 0x0004
RI_MOUSE_RIGHT_BUTTON_UP = 0x0008
RI_MOUSE_MIDDLE_BUTTON_DOWN = 0x0010
RI_MOUSE_MIDDLE_BUTTON_UP = 0x0020
RI_MOUSE_BUTTON_4_DOWN = 0x0040   # 侧键 X1 (后退)
RI_MOUSE_BUTTON_4_UP = 0x0080
RI_MOUSE_BUTTON_5_DOWN = 0x0100   # 侧键 X2 (前进)
RI_MOUSE_BUTTON_5_UP = 0x0200
RI_MOUSE_WHEEL = 0x0400


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HANDLE),
    ]

class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", wintypes.USHORT),
        ("usButtonFlags", wintypes.USHORT),
        ("usButtonData", wintypes.USHORT),
        ("ulRawButtons", wintypes.ULONG),
        ("lLastX", wintypes.LONG),
        ("lLastY", wintypes.LONG),
        ("ulExtraInformation", wintypes.ULONG),
    ]

class RAWINPUT(ctypes.Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("mouse", RAWMOUSE),
    ]


class RawMouseMonitor:
    def __init__(self, on_event: Optional[Callable] = None):
        self._on_event = on_event
        self._running = False
        self._thread = None
        self._hwnd = None
        self._buttons = 0
        self._wnd_proc_ref = None

    @property
    def is_running(self) -> bool:
        return self._running

    def set_event_callback(self, callback: Callable):
        self._on_event = callback

    def start(self):
        if self._running:
            return
        self._running = True
        self._buttons = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name='raw-mouse')
        self._thread.start()
        logger.info("Raw mouse monitor started")

    def stop(self):
        self._running = False
        if self._hwnd:
            try:
                ctypes.windll.user32.PostMessageW(self._hwnd, 0x0010, 0, 0)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Raw mouse monitor stopped")

    def _run(self):
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        # 所有 Win32 调用必须设置 argtypes, 否则 64 位句柄会被截断为 32 位导致溢出
        user32.GetRawInputData.restype = wintypes.UINT
        user32.GetRawInputData.argtypes = [
            wintypes.LPARAM, wintypes.UINT, ctypes.c_void_p,
            ctypes.POINTER(wintypes.UINT), wintypes.UINT
        ]
        user32.DefWindowProcW.argtypes = [
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        ]
        user32.DefWindowProcW.restype = wintypes.LONG
        user32.RegisterClassW.argtypes = [ctypes.c_void_p]
        user32.RegisterClassW.restype = wintypes.ATOM
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
            wintypes.DWORD, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
        ]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.RegisterRawInputDevices.argtypes = [
            ctypes.c_void_p, wintypes.UINT, wintypes.UINT
        ]
        user32.RegisterRawInputDevices.restype = wintypes.BOOL
        user32.GetMessageW.argtypes = [
            ctypes.POINTER(wintypes.MSG), wintypes.HWND,
            wintypes.UINT, wintypes.UINT
        ]
        user32.GetMessageW.restype = wintypes.BOOL
        user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.PostQuitMessage.argtypes = [ctypes.c_int]
        user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
        user32.UnregisterClassW.restype = wintypes.BOOL

        WNDPROC = ctypes.WINFUNCTYPE(
            wintypes.LONG, wintypes.HWND, wintypes.UINT,
            wintypes.WPARAM, wintypes.LPARAM
        )

        class WNDCLASS(ctypes.Structure):
            _fields_ = [
                ('style', wintypes.UINT), ('lpfnWndProc', WNDPROC),
                ('cbClsExtra', wintypes.INT), ('cbWndExtra', wintypes.INT),
                ('hInstance', wintypes.HMODULE), ('hIcon', wintypes.HICON),
                ('hCursor', wintypes.HANDLE), ('hbrBackground', wintypes.HBRUSH),
                ('lpszMenuName', wintypes.LPCWSTR), ('lpszClassName', wintypes.LPCWSTR),
            ]

        def _wnd_proc_local(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                self._handle_raw_input(lparam)
                return 0
            if msg == 2:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wnd_proc_ref = WNDPROC(_wnd_proc_local)
        wc = WNDCLASS()
        wc.lpfnWndProc = self._wnd_proc_ref
        wc.lpszClassName = "RawMouseWnd"
        wc.hInstance = kernel32.GetModuleHandleW(None)
        user32.RegisterClassW(ctypes.byref(wc))
        self._hwnd = user32.CreateWindowExW(0, "RawMouseWnd", "RawMouse", 0, 0, 0, 0, 0, 0, 0, wc.hInstance, 0)

        rid = RAWINPUTDEVICE()
        rid.usUsagePage = 0x01
        rid.usUsage = 0x02
        rid.dwFlags = RIDEV_INPUTSINK
        rid.hwndTarget = self._hwnd
        user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid))
        logger.info("Raw input registered")

        msg = wintypes.MSG()
        while self._running:
            ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if ret <= 0:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        try:
            user32.DestroyWindow(self._hwnd)
        except Exception:
            pass
        self._hwnd = None
        try:
            user32.UnregisterClassW("RawMouseWnd", wc.hInstance)
        except Exception:
            pass

    def _handle_raw_input(self, hrawinput):
        user32 = ctypes.windll.user32
        size = wintypes.UINT(0)
        ret = user32.GetRawInputData(hrawinput, RID_INPUT, None, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
        if ret == 0xFFFFFFFF or size.value == 0:
            return

        buf = (ctypes.c_ubyte * size.value)()
        ret = user32.GetRawInputData(hrawinput, RID_INPUT, buf, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
        if ret == 0 or ret == 0xFFFFFFFF:
            return

        raw = RAWINPUT.from_buffer(buf)
        if raw.header.dwType != RIM_TYPEMOUSE:
            return

        mouse = raw.mouse
        btn_flags = mouse.usButtonFlags

        # 按钮状态 (与协议一致: bit0=左, bit1=右, bit2=中, bit4=后退, bit5=前进)
        if btn_flags & RI_MOUSE_LEFT_BUTTON_DOWN: self._buttons |= 0x01
        if btn_flags & RI_MOUSE_LEFT_BUTTON_UP: self._buttons &= ~0x01
        if btn_flags & RI_MOUSE_RIGHT_BUTTON_DOWN: self._buttons |= 0x02
        if btn_flags & RI_MOUSE_RIGHT_BUTTON_UP: self._buttons &= ~0x02
        if btn_flags & RI_MOUSE_MIDDLE_BUTTON_DOWN: self._buttons |= 0x04
        if btn_flags & RI_MOUSE_MIDDLE_BUTTON_UP: self._buttons &= ~0x04
        if btn_flags & RI_MOUSE_BUTTON_4_DOWN: self._buttons |= 0x10   # X1 后退 (FLAG_BACK)
        if btn_flags & RI_MOUSE_BUTTON_4_UP: self._buttons &= ~0x10
        if btn_flags & RI_MOUSE_BUTTON_5_DOWN: self._buttons |= 0x20   # X2 前进 (FLAG_FORWARD)
        if btn_flags & RI_MOUSE_BUTTON_5_UP: self._buttons &= ~0x20

        dx = mouse.lLastX
        dy = mouse.lLastY

        wheel = 0
        if btn_flags & RI_MOUSE_WHEEL:
            wheel_data = mouse.usButtonData
            if wheel_data >= 0x8000:
                wheel_data -= 0x10000
            wheel = wheel_data // 120

        # 直接发送: 有位移/滚轮/按钮按下时都发 (防止目标 PC 自动松开)
        if dx != 0 or dy != 0 or wheel != 0 or btn_flags != 0 or self._buttons != 0:
            if self._on_event:
                try:
                    self._on_event({
                        'buttons': self._buttons,
                        'dx': dx,
                        'dy': dy,
                        'wheel': wheel,
                        'left': bool(self._buttons & 0x01),
                        'right': bool(self._buttons & 0x02),
                        'middle': bool(self._buttons & 0x04),
                    })
                except Exception as e:
                    logger.error(f"Raw mouse callback error: {e}")