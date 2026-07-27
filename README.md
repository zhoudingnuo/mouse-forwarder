# Mouse Forwarder - CH32V305 USB 鼠标转发器

将控制 PC 的鼠标输入通过 CH32V305 转发到目标电脑，让 CH32V305 作为 USB HID 鼠标设备输出。

## 系统架构

```
┌──────────────────────┐     串口(CDC)     ┌──────────────────────┐     USB HID      ┌──────────────┐
│  控制 PC              │ ◄──────────────► │  CH32V305            │ ◄─────────────► │  目标 PC      │
│  (运行本软件)          │    USB1 (FS)      │  USB1←CDC接收数据     │    USB2 (HS)     │  (接收鼠标)    │
│                       │                   │  USB2→HID鼠标输出     │                  │              │
└──────────────────────┘                   └──────────────────────┘                  └──────────────┘
```

## 项目结构

```
mouse-forwarder/
├── pc_app/                      # PC 端软件
│   ├── electron/                # Electron 壳
│   │   ├── main.js             # 主进程 (启动 Python 后端)
│   │   ├── preload.js          # 安全桥接
│   │   └── package.json        # Electron 配置
│   ├── frontend/                # 前端 UI (参考 Multi3DViz 设计)
│   │   ├── index.html          # 主界面
│   │   ├── css/theme.css       # 暗色主题 (ZCode 风格)
│   │   └── js/
│   │       ├── app.js          # 应用控制器
│   │       ├── ws_client.js    # WebSocket 客户端
│   │       ├── mouse_panel.js  # 鼠标实时数据面板
│   │       └── settings.js     # 串口设置面板
│   └── backend/                 # Python 后端
│       ├── main.py             # asyncio WS 服务器入口
│       ├── mouse_monitor.py    # 鼠标捕获 (pynput)
│       ├── serial_forwarder.py # 串口转发 (pyserial)
│       ├── protocol.py         # 通信协议编解码
│       └── requirements.txt    # Python 依赖
├── firmware/                    # CH32V305 固件
│   ├── protocol.h             # 通信协议定义
│   ├── User/
│   │   ├── main.c             # 主程序
│   │   ├── mouse_forwarder.c/h# 转发核心逻辑
│   │   ├── ch32v30x_usbfs_device.c # USB1 FS CDC 串口
│   │   ├── ch32v30x_usbhs_device.c # USB2 HS HID 鼠标
│   │   └── ch32v30x_it.c     # 中断处理
│   └── README.md              # 固件编译说明
└── README.md                   # 本文件
```

## 快速开始

### 1. 烧录固件到 CH32V305

**使用 MounRiver Studio (推荐):**
1. 下载 [MounRiver Studio](http://www.mounriver.com)
2. 下载 [CH32V30x EVT SDK](https://www.wch.cn/downloads/CH32V30xEVT_ZIP.html)
3. 创建新工程 → 选择 CH32V305
4. 将 `firmware/` 下的文件添加到工程
5. 编译并下载

**或者使用 WCHISPTool:**
1. 按住 BOOT 键，按 RST，松 RST，松 BOOT → 进入烧录模式
2. 打开 WCHISPTool，选择芯片 CH32V305
3. 选择编译好的 .hex 文件，点击下载

### 2. 安装 PC 端软件

```bash
# 安装 Python 依赖
cd pc_app/backend
pip install -r requirements.txt

# 安装 Electron 依赖
cd pc_app/electron
npm install
```

### 3. 连接硬件

1. CH32V305 的 **USB1** (FS 口) → 连接到 **控制 PC** (运行本软件)
2. CH32V305 的 **USB2** (HS 口) → 连接到 **目标 PC** (接收鼠标)
3. 控制 PC 上会出现一个新的 COM 口

### 4. 运行

**开发模式 (Python 直接运行):**
```bash
cd pc_app/backend
python main.py
# 浏览器打开: http://127.0.0.1:8765  (实际是 WebSocket)
```

**Electron 模式:**
```bash
cd pc_app/electron
npm start
```

### 5. 使用

1. 软件打开后会自动检测 CH32V305 的串口
2. 点击"连接"建立串口通信
3. 鼠标开始被捕获并转发到目标 PC
4. 目标 PC 上会识别出名为 "CH32V305 HID Mouse" 的 USB 鼠标

## 通信协议

| 字节 | 字段 | 说明 |
|------|------|------|
| 0 | Header | 0xAA |
| 1 | Flags | bit0=左键, bit1=右键, bit2=中键, bit3=滚轮有效 |
| 2 | Delta X | int8, -128~127 |
| 3 | Delta Y | int8, -128~127 |
| 4 | Wheel | int8, -128~127 |
| 5 | Checksum | 异或校验 |

## 注意事项

- **管理员权限**: Windows 下鼠标捕获需要管理员权限
- **USB 数据线**: 必须使用支持数据传输的 USB 线 (非充电线)
- **双 USB 口**: 需要两根 USB 线分别连接两台电脑
- **首次使用**: 目标 PC 可能需要等待驱动安装完成

## 技术栈

- **PC 前端**: Electron + HTML/CSS/JS (暗色主题)
- **PC 后端**: Python asyncio + WebSocket + pynput + pyserial
- **MCU 固件**: C (WCH SDK), USB FS CDC + USB HS HID
- **通信**: WebSocket (前后端) + USB CDC (PC↔MCU) + USB HID (MCU↔目标PC)

## 参考

- [Multi3DViz](https://github.com/zhoudingnuo/Multi3DViz) - UI/前后端架构参考
- [nanoCH32V305](https://github.com/wuxx/nanoCH32V305) - 硬件参考
- [WCH CH32V30xEVT SDK](https://www.wch.cn/downloads/CH32V30xEVT_ZIP.html)