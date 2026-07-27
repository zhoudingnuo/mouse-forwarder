@echo off
cd /d "C:\WCH.CN\App\WCHISPTool\WCHISPTool_CH32Vxxx"
WCHISPTool_CH32Vxxx.exe --Device USB --Chip CH32V305 --Program %1 2> C:\Users\AnlangZ\ZCodeProject\mouse-forwarder\flash_result.txt
type C:\Users\AnlangZ\ZCodeProject\mouse-forwarder\flash_result.txt
echo.
echo Exit code: %ERRORLEVEL%
pause