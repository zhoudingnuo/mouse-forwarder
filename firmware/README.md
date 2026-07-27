# CH32V305 鼠标转发器固件

## 功能

将 CH32V305 配置为 USB 鼠标转发器：

- **USB1 (FS)**: CDC 虚拟串口，接收控制 PC 发来的鼠标数据
- **USB2 (HS)**: HID 鼠标设备，输出到目标 PC

## 硬件要求

- nanoCH32V305 开发板 (CH32V305RBT6)
- 2 根 USB Type-C 数据线
- WCH-Link 调试器 (可选，用于首次烧录)

## 开发环境

### 方式一：MounRiver Studio (推荐)

1. 下载安装 [MounRiver Studio](http://www.mounriver.com)
2. 下载 [CH32V30x EVT SDK](https://www.wch.cn/downloads/CH32V30xEVT_ZIP.html)
3. 在 MRS 中创建新工程，选择 CH32V305
4. 将 `User/` 目录下的文件复制到工程中
5. 将 `protocol.h` 复制到工程根目录
6. 编译并下载

### 方式二：命令行 (GCC + Makefile)

使用 RISC-V GCC 工具链:

```bash
# 安装工具链 (xpack)
xpm install @xpack-dev-tools/riscv-none-embed-gcc@latest

# 编译 (需配合 WCH SDK 的 Makefile)
cd EVT/EXAM/USB/
# 复制文件到对应目录
cp /path/to/mouse-forwarder/firmware/User/* ./
make
```

## 烧录方法

### 首次烧录 (使用 WCH-Link)

1. 连接 WCH-Link 到开发板的 SWD 接口
2. 在 MounRiver Studio 中点击 `Flash → Download`
3. 或使用命令行: `wchisp flash build/ch32v30x.bin`

### 后续更新 (通过 USB 烧录)

1. 按住 BOOT 键，按一下 RST 键，松开 RST，再松开 BOOT
2. 打开 WCHISPTool
3. 选择芯片: CH32V305
4. 选择 HEX/BIN 文件
5. 点击下载

## 文件说明

| 文件 | 说明 |
|------|------|
| `User/main.c` | 主程序入口 |
| `User/mouse_forwarder.c` | 鼠标转发核心逻辑 |
| `User/mouse_forwarder.h` | 转发器头文件 |
| `User/ch32v30x_usbfs_device.c` | USB1 FS CDC 虚拟串口 |
| `User/ch32v30x_usbhs_device.c` | USB2 HS HID 鼠标 |
| `User/ch32v30x_it.c` | 中断处理 |
| `protocol.h` | 通信协议定义 |

## 通信协议

PC → CH32V305 数据包 (6 bytes):

| Byte | 字段 | 说明 |
|------|------|------|
| 0 | Header | 0xAA |
| 1 | Flags | bit0=左键, bit1=右键, bit2=中键, bit3=滚轮有效 |
| 2 | Delta X | int8, 相对位移 (-128~127) |
| 3 | Delta Y | int8, 相对位移 (-128~127) |
| 4 | Wheel | int8, 滚轮增量 (-128~127) |
| 5 | Checksum | 字节 0~4 的异或和 |

## 指示灯说明

- 板载 LED 常亮: 系统运行正常
- LED 闪烁: 正在接收数据
- LED 熄灭: 系统异常