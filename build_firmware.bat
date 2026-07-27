@echo off
title Mouse Forwarder - 编译固件
chcp 65001 >nul

set APP_DIR=%~dp0
set TOOLS_DIR=%APP_DIR%tools
set FIRMWARE_DIR=%APP_DIR%firmware
set GCC_DIR=%TOOLS_DIR%\riscv-gcc\xpack-riscv-none-elf-gcc-14.2.0-2

echo ============================================
echo   编译 CH32V305 鼠标转发器固件
echo ============================================
echo.

:: 设置工具链路径
set PATH=%GCC_DIR%\bin;%PATH%

:: 检查编译器
riscv-none-elf-gcc --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!!] 未找到 RISC-V GCC 编译器!
    echo     正在解压工具链...
    if not exist "%GCC_DIR%" (
        mkdir "%TOOLS_DIR%\riscv-gcc"
        powershell -Command "Expand-Archive -Path '%TOOLS_DIR%\gcc.zip' -DestinationPath '%TOOLS_DIR%\riscv-gcc' -Force"
    )
    if not exist "%GCC_DIR%\bin\riscv-none-elf-gcc.exe" (
        echo [!!] 工具链解压失败, 请手动解压 %TOOLS_DIR%\gcc.zip
        pause
        exit /b 1
    )
    set PATH=%GCC_DIR%\bin;%PATH%
)

echo [OK] 编译器: 
riscv-none-elf-gcc --version | findstr "gcc"

echo.
echo [*] 编译固件...
echo.

:: 编译参数
set CPU=-march=rv32imafc -mabi=ilp32f -msmall-data-limit=8 -mno-save-restore
set OPT=-Os -ffunction-sections -fdata-sections
set WARN=-Wall -Wno-unused-parameter
set INC=-I"%FIRMWARE_DIR%" -I"%FIRMWARE_DIR%\User"
set SRC=%FIRMWARE_DIR%\User\main.c
set SRC=%SRC% %FIRMWARE_DIR%\User\mouse_forwarder.c
set SRC=%SRC% %FIRMWARE_DIR%\User\ch32v30x_usbfs_device.c
set SRC=%SRC% %FIRMWARE_DIR%\User\ch32v30x_usbhs_device.c
set SRC=%SRC% %FIRMWARE_DIR%\User\ch32v30x_it.c
set LINK=-T"%FIRMWARE_DIR%\Link.ld" -nostartfiles -Wl,--gc-sections

riscv-none-elf-gcc %CPU% %OPT% %WARN% %INC% %SRC% %LINK% -o "%TOOLS_DIR%\firmware.elf"

if %ERRORLEVEL% NEQ 0 (
    echo [!!] 编译失败!
    pause
    exit /b 1
)

echo [OK] 编译成功, 生成 .elf 文件

:: 生成 .hex 文件
riscv-none-elf-objcopy -O ihex "%TOOLS_DIR%\firmware.elf" "%TOOLS_DIR%\firmware.hex"
riscv-none-elf-objcopy -O binary "%TOOLS_DIR%\firmware.elf" "%TOOLS_DIR%\firmware.bin"

echo [OK] 固件文件已生成:
echo      %TOOLS_DIR%\firmware.hex
echo      %TOOLS_DIR%\firmware.bin
echo.
echo [*] 现在可以用烧录功能刷入固件了!
echo.
pause