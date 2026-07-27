"""
flasher.py - CH32V305 固件烧录模块
直接通过 USB 与 CH32V305 的 bootloader 通信烧录固件
"""

import struct
import logging
import os
import subprocess
import tempfile
import json
from typing import Optional

logger = logging.getLogger(__name__)

# WCH ISPTool CLI 路径
WCHISP_CLI = r"C:\WCH.CN\App\WCHISPTool\WCHISPTool_CH32Vxxx\WCHISPTool_CH32Vxxx.exe"
WCHISP_DIR = r"C:\WCH.CN\App\WCHISPTool\WCHISPTool_CH32Vxxx"

# CH32V305 bootloader 的 VID/PID
WCH_VID = 0x4348
WCH_PID_BOOT = 0x55E0


def is_bootloader_mode() -> bool:
    """检查 CH32V305 是否处于烧录模式"""
    try:
        import usb.core
        import usb.backend.libusb1
        dev = usb.core.find(idVendor=WCH_VID, idProduct=WCH_PID_BOOT)
        return dev is not None
    except (ImportError, usb.core.USBError):
        # 如果 pyusb 不可用, 尝试通过 WCHISPTool 检测
        return _check_via_wchisp()
    except Exception as e:
        logger.warning(f"Error checking bootloader: {e}")
        return False


def _check_via_wchisp() -> bool:
    """通过 WCHISPTool CLI 检测设备"""
    if not os.path.exists(WCHISP_CLI):
        return False
    try:
        result = subprocess.run(
            [WCHISP_CLI, "--Device", "USB", "--Chip", "CH32V305"],
            capture_output=True, text=True, timeout=5,
            cwd=WCHISP_DIR
        )
        # 如果返回 JSON 且有 Device 字段, 说明检测到了
        if result.stdout:
            data = json.loads(result.stdout)
            return data.get("Device") == "USB"
        return False
    except Exception:
        return False


def flash_firmware(hex_path: str, chip: str = "CH32V305") -> dict:
    """
    烧录固件到 CH32V305
    
    Args:
        hex_path: .hex 或 .bin 文件路径
        chip: 芯片型号
    
    Returns:
        {"success": bool, "message": str}
    """
    if not os.path.exists(hex_path):
        return {"success": False, "message": f"文件不存在: {hex_path}"}
    
    if not os.path.exists(WCHISP_CLI):
        return {"success": False, "message": "未找到 WCHISPTool, 请先安装"}
    
    logger.info(f"Flashing {hex_path} to {chip}...")
    
    try:
        result = subprocess.run(
            [WCHISP_CLI, "--Device", "USB", "--Chip", chip, "--Program", hex_path],
            capture_output=True, text=True, timeout=30,
            cwd=WCHISP_DIR
        )
        
        output = result.stdout
        logger.info(f"Flash result: {output}")
        
        if result.returncode == 0:
            return {"success": True, "message": "烧录成功!"}
        else:
            # 解析 JSON 错误
            try:
                err = json.loads(output)
                return {"success": False, "message": err.get("Message", "烧录失败")}
            except json.JSONDecodeError:
                return {"success": False, "message": f"烧录失败 (code={result.returncode})"}
    
    except subprocess.TimeoutExpired:
        return {"success": False, "message": "烧录超时"}
    except Exception as e:
        return {"success": False, "message": f"烧录错误: {e}"}


def erase_flash(chip: str = "CH32V305") -> dict:
    """擦除 CH32V305 Flash"""
    if not os.path.exists(WCHISP_CLI):
        return {"success": False, "message": "未找到 WCHISPTool"}
    
    try:
        result = subprocess.run(
            [WCHISP_CLI, "--Device", "USB", "--Chip", chip, "--Erase"],
            capture_output=True, text=True, timeout=20,
            cwd=WCHISP_DIR
        )
        return {"success": result.returncode == 0, "message": result.stdout}
    except Exception as e:
        return {"success": False, "message": f"擦除错误: {e}"}


def get_device_info() -> dict:
    """获取设备信息"""
    if not os.path.exists(WCHISP_CLI):
        return {"success": False, "message": "未找到 WCHISPTool"}
    
    try:
        result = subprocess.run(
            [WCHISP_CLI, "--Device", "USB", "--Chip", "CH32V305"],
            capture_output=True, text=True, timeout=5,
            cwd=WCHISP_DIR
        )
        try:
            data = json.loads(result.stdout)
            return {"success": True, "info": data}
        except json.JSONDecodeError:
            return {"success": False, "message": result.stdout}
    except Exception as e:
        return {"success": False, "message": str(e)}


def detect_bootloader() -> Optional[dict]:
    """
    检测 CH32V305 烧录模式设备
    
    Returns:
        {"vid": 0x4348, "pid": 0x55E0, "description": "CH32V305 Bootloader"} 或 None
    """
    try:
        import usb.core
        dev = usb.core.find(idVendor=WCH_VID, idProduct=WCH_PID_BOOT)
        if dev:
            return {
                "vid": dev.idVendor,
                "pid": dev.idProduct,
                "description": "CH32V305 Bootloader",
                "manufacturer": "WCH",
            }
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"USB detect error: {e}")
    
    return None