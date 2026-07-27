"""
protocol.py - PC ↔ CH32V305 通信协议 (Python 端)

数据包格式 (6 字节):
  Byte 0: Header      0xAA
  Byte 1: Flags       bit0=左键, bit1=右键, bit2=中键, bit3=滚轮有效, bit4=后退键, bit5=前进键
  Byte 2: Delta X     int8, 相对位移 (-128~127)
  Byte 3: Delta Y     int8, 相对位移 (-128~127)
  Byte 4: Wheel       int8, 滚轮增量 (-128~127)
  Byte 5: Checksum    字节 0~4 的异或和
"""

import struct
import logging

logger = logging.getLogger(__name__)

# 协议常量
PKG_HEADER = 0xAA
PKG_SIZE = 6

# Flags 位定义
FLAG_LEFT_BUTTON = 1 << 0
FLAG_RIGHT_BUTTON = 1 << 1
FLAG_MIDDLE_BUTTON = 1 << 2
FLAG_WHEEL_VALID = 1 << 3
FLAG_BACK_BUTTON = 1 << 4    # 鼠标侧键 - 后退 (X1)
FLAG_FORWARD_BUTTON = 1 << 5 # 鼠标侧键 - 前进 (X2)


def calc_checksum(data: bytes) -> int:
    """计算异或校验和"""
    checksum = 0
    for b in data:
        checksum ^= b
    return checksum


def encode_packet(buttons: int, dx: int, dy: int, wheel: int = 0) -> bytes:
    """
    编码鼠标数据为串口数据包
    
    Args:
        buttons: 按钮状态 (bit0=左键, bit1=右键, bit2=中键, bit4=后退, bit5=前进)
        dx: X 轴相对位移 (-128~127)
        dy: Y 轴相对位移 (-128~127)
        wheel: 滚轮增量 (-128~127)
    
    Returns:
        6 字节的串口数据包
    """
    flags = buttons & 0x3F  # 保留低 6 位
    if wheel != 0:
        flags |= FLAG_WHEEL_VALID
    
    # 限制范围
    dx = max(-128, min(127, dx))
    dy = max(-128, min(127, dy))
    wheel = max(-128, min(127, wheel))
    
    # 打包
    data = struct.pack('<BBbbb', PKG_HEADER, flags, dx, dy, wheel)
    checksum = calc_checksum(data)
    data += struct.pack('<B', checksum)
    
    return data


def decode_packet(data: bytes) -> dict:
    """
    解码串口数据包
    
    Args:
        data: 6 字节的串口数据包
    
    Returns:
        包含按钮、坐标、滚轮的字典, 或 None (无效)
    """
    if len(data) < PKG_SIZE:
        logger.warning(f"Packet too short: {len(data)} bytes")
        return None
    
    if data[0] != PKG_HEADER:
        return None
    
    # 校验
    checksum = calc_checksum(data[:5])
    if checksum != data[5]:
        logger.warning(f"Checksum mismatch: {checksum} != {data[5]}")
        return None
    
    flags = data[1]
    dx = struct.unpack('<b', data[2:3])[0]
    dy = struct.unpack('<b', data[3:4])[0]
    wheel = struct.unpack('<b', data[4:5])[0] if (flags & FLAG_WHEEL_VALID) else 0
    
    return {
        'buttons': flags & 0x3F,
        'left': bool(flags & FLAG_LEFT_BUTTON),
        'right': bool(flags & FLAG_RIGHT_BUTTON),
        'middle': bool(flags & FLAG_MIDDLE_BUTTON),
        'back': bool(flags & FLAG_BACK_BUTTON),
        'forward': bool(flags & FLAG_FORWARD_BUTTON),
        'dx': dx,
        'dy': dy,
        'wheel': wheel,
    }