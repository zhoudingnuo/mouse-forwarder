"""
serial_forwarder.py - 串口转发模块

通过 pyserial 将鼠标数据发送到 CH32V305 的 CDC 虚拟串口。
"""

import asyncio
import logging
from typing import Optional

import serial
import serial.tools.list_ports

logger = logging.getLogger(__name__)


class SerialForwarder:
    """
    串口转发器
    
    管理到 CH32V305 CDC 串口的连接,
    提供数据发送和连接管理功能。
    """
    
    # CH32V305 CDC 的 VID/PID
    WCH_VID = 0x4348
    WCH_PID_CDC = 0x55E1
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self._port: Optional[str] = port
        self._baudrate: int = baudrate
        self._serial: Optional[serial.Serial] = None
        self._connected: bool = False
        self._lock = asyncio.Lock()
        
        # 统计信息
        self.bytes_sent: int = 0
        self.packets_sent: int = 0
        
        # 状态回调
        self._on_connection_change = None
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def port_name(self) -> Optional[str]:
        return self._port
    
    def set_connection_callback(self, callback):
        """设置连接状态变化回调"""
        self._on_connection_change = callback
    
    @staticmethod
    def list_ports() -> list[dict]:
        """
        列出所有可用串口
        
        Returns:
            串口信息列表: [{port, description, vid, pid, hwid}, ...]
        """
        ports = []
        for p in serial.tools.list_ports.comports():
            ports.append({
                'port': p.device,
                'description': p.description,
                'vid': p.vid,
                'pid': p.pid,
                'hwid': p.hwid,
            })
        return ports
    
    @staticmethod
    def find_ch32v305_port() -> Optional[str]:
        """
        自动查找 CH32V305 的 CDC 串口
        
        Returns:
            串口名, 如 "COM3", 未找到则返回 None
        """
        for p in serial.tools.list_ports.comports():
            if p.vid == SerialForwarder.WCH_VID:
                logger.info(f"Found CH32V305 on {p.device}")
                return p.device
        return None
    
    async def connect(self, port: Optional[str] = None, baudrate: Optional[int] = None) -> bool:
        """
        连接到串口
        
        Args:
            port: 串口名, None 则自动查找
            baudrate: 波特率, None 则使用默认值
        
        Returns:
            是否成功连接
        """
        async with self._lock:
            # 断开现有连接
            await self._disconnect()
            
            if port:
                self._port = port
            elif not self._port:
                self._port = self.find_ch32v305_port()
            
            if baudrate:
                self._baudrate = baudrate
            
            if not self._port:
                logger.error("No port specified and CH32V305 not found")
                return False
            
            try:
                self._serial = serial.Serial(
                    port=self._port,
                    baudrate=self._baudrate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=0.1,
                    write_timeout=0.1,
                )
                self._connected = self._serial.is_open
                logger.info(f"Connected to {self._port} at {self._baudrate} baud")
                
                if self._on_connection_change:
                    self._on_connection_change(True, self._port)
                
                return True
                
            except serial.SerialException as e:
                logger.error(f"Failed to connect to {self._port}: {e}")
                self._connected = False
                return False
    
    async def disconnect(self):
        """断开串口连接"""
        async with self._lock:
            await self._disconnect()
    
    async def _disconnect(self):
        """内部断开方法 (需持有锁)"""
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception as e:
                logger.error(f"Error closing serial: {e}")
        
        self._serial = None
        self._connected = False
        
        if self._on_connection_change:
            self._on_connection_change(False, None)
        
        logger.info("Disconnected")
    
    async def send(self, data: bytes) -> bool:
        """
        发送数据到串口
        
        Args:
            data: 要发送的数据
        
        Returns:
            是否发送成功
        """
        if not self._connected or not self._serial:
            logger.warning("Not connected, cannot send")
            return False
        
        async with self._lock:
            try:
                written = self._serial.write(data)
                self.bytes_sent += written
                self.packets_sent += 1
                return True
            except serial.SerialTimeoutException:
                logger.warning("Serial write timeout")
                return False
            except serial.SerialException as e:
                logger.error(f"Serial write error: {e}")
                await self._disconnect()
                return False