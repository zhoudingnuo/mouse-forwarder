@echo off
chcp 65001 >nul
cd /d "C:\WCH.CN\App\WCHISPTool\WCHISPTool_CH32Vxxx"
echo [*] 检测设备...
WCHISPTool_CH32Vxxx.exe --Device USB --Chip CH32V305 > "C:\Users\AnlangZ\ZCodeProject\mouse-forwarder\tools\flash_log.txt" 2>&1
type "C:\Users\AnlangZ\ZCodeProject\mouse-forwarder\tools\flash_log.txt"
echo.
echo [*] 开始烧录...
WCHISPTool_CH32Vxxx.exe --Device USB --Chip CH32V305 --Program "C:\Users\AnlangZ\ZCodeProject\mouse-forwarder\tools\dual_usb_v2.hex" >> "C:\Users\AnlangZ\ZCodeProject\mouse-forwarder\tools\flash_log.txt" 2>&1
type "C:\Users\AnlangZ\ZCodeProject\mouse-forwarder\tools\flash_log.txt"
echo.
echo [*] 完成, 按任意键退出
pause >nul