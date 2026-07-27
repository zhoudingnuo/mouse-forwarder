"""
test_mouse.py - 测试 CH32V305 鼠标转发器
通过 COM 口发送鼠标数据, 验证 HID 鼠标是否移动
"""

import serial
import time
import struct
import sys

import serial
import time
import struct
import sys
import serial.tools.list_ports

def find_ch32v305_port():
    """自动查找 CH32V305 的 CDC 串口"""
    for p in serial.tools.list_ports.comports():
        if p.vid == 0x1A86:
            return p.device
    return None

PORT = find_ch32v305_port() or "COM6"
BAUD = 115200

def calc_checksum(data):
    """异或校验"""
    s = 0
    for b in data:
        s ^= b
    return s

def encode_packet(buttons=0, dx=0, dy=0, wheel=0):
    """编码鼠标数据包 (6字节)"""
    flags = buttons & 0x07
    if wheel != 0:
        flags |= 0x08
    dx = max(-128, min(127, dx))
    dy = max(-128, min(127, dy))
    wheel = max(-128, min(127, wheel))
    data = struct.pack('<BBbbb', 0xAA, flags, dx, dy, wheel)
    return data + bytes([calc_checksum(data)])

def main():
    print(f"=== CH32V305 鼠标转发器测试 ===")
    print(f"端口: {PORT}, 波特率: {BAUD}")
    print()

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
        print(f"[OK] 已连接 {PORT}")
    except Exception as e:
        print(f"[错误] 无法打开 {PORT}: {e}")
        print("请检查 COM 端口号, 在设备管理器中查看")
        sys.exit(1)

    # 发送测试数据
    tests = [
        ("鼠标向右移动 10px", 0, 10, 0, 0),
        ("鼠标向右移动 10px", 0, 10, 0, 0),
        ("鼠标向右移动 10px", 0, 10, 0, 0),
        ("鼠标向右移动 10px", 0, 10, 0, 0),
        ("鼠标向右移动 10px", 0, 10, 0, 0),
        ("鼠标向下移动 10px", 0, 0, 10, 0),
        ("鼠标向下移动 10px", 0, 0, 10, 0),
        ("鼠标向下移动 10px", 0, 0, 10, 0),
        ("鼠标向下移动 10px", 0, 0, 10, 0),
        ("鼠标向下移动 10px", 0, 0, 10, 0),
        ("鼠标向左移动 10px", 0, -10, 0, 0),
        ("鼠标向左移动 10px", 0, -10, 0, 0),
        ("鼠标向左移动 10px", 0, -10, 0, 0),
        ("鼠标向左移动 10px", 0, -10, 0, 0),
        ("鼠标向左移动 10px", 0, -10, 0, 0),
        ("鼠标向上移动 10px", 0, 0, -10, 0),
        ("鼠标向上移动 10px", 0, 0, -10, 0),
        ("鼠标向上移动 10px", 0, 0, -10, 0),
        ("鼠标向上移动 10px", 0, 0, -10, 0),
        ("鼠标向上移动 10px", 0, 0, -10, 0),
        ("左键点击", 0x01, 0, 0, 0),
        ("左键释放", 0x00, 0, 0, 0),
        ("滚轮向上", 0, 0, 0, 1),
        ("滚轮向上", 0, 0, 0, 1),
        ("滚轮向下", 0, 0, 0, -1),
        ("滚轮向下", 0, 0, 0, -1),
    ]

    print(f"\n开始发送 {len(tests)} 个测试命令...")
    print("请观察鼠标光标是否移动!\n")

    for i, (desc, btn, dx, dy, wheel) in enumerate(tests, 1):
        pkt = encode_packet(btn, dx, dy, wheel)
        ser.write(pkt)
        print(f"[{i:2d}/{len(tests)}] {desc}: {pkt.hex()}")
        time.sleep(0.2)  # 200ms 间隔

    print("\n[完成] 测试结束")
    print("如果鼠标光标动了, 说明转发器工作正常!")
    ser.close()

if __name__ == "__main__":
    main()