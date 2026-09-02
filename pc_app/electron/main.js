/**
 * main.js - Electron 主进程
 * 本地桌面应用, 启动 Python 后端子进程 + 原生窗口
 */

const { app, BrowserWindow, ipcMain, Tray, Menu, nativeImage } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let backendProcess = null;
let mainWindow = null;
let tray = null;
let actualPort = 8765;  // 会被 READY 消息更新

// 单实例锁: 防止重复启动导致两个后端同时抢采集卡 (DSHOW 崩溃 0xC0000005)
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
    console.log('Mouse Forwarder already running, exiting this instance');
    app.quit();
} else {
    app.on('second-instance', () => {
        // 已有实例运行时, 再次启动 -> 聚焦已有窗口
        if (mainWindow) {
            mainWindow.show();
            mainWindow.focus();
        }
    });
}

/**
 * 启动 Python 后端
 */
function startBackend() {
    return new Promise((resolve, reject) => {
        const backendDir = path.join(__dirname, '..', 'backend');
        const backendScript = path.join(backendDir, 'main.py');

        const pythonCandidates = ['python', 'python3'];

        function tryStart(index) {
            if (index >= pythonCandidates.length) {
                reject(new Error('找不到 Python, 请确认已安装并添加到 PATH'));
                return;
            }

            const pythonExe = pythonCandidates[index];
            console.log(`Trying: ${pythonExe}`);

            const proc = spawn(pythonExe, [backendScript], {
                cwd: backendDir,
                stdio: ['ignore', 'pipe', 'pipe'],
                // 隐藏 Python 后端的控制台窗口 (与无终端启动配合, 不弹黑窗)
                windowsHide: true,
                env: { ...process.env, PYTHONUNBUFFERED: '1' },
            });

            let started = false;

            proc.stdout.on('data', (data) => {
                const text = data.toString();
                console.log(`[backend] ${text.trim()}`);
                if (!started) {
                    const match = text.match(/READY ws:\/\/127\.0\.0\.1:(\d+)/);
                    if (match) {
                        actualPort = parseInt(match[1]);
                        started = true;
                        backendProcess = proc;
                        resolve(actualPort);
                    }
                }
            });

            proc.stderr.on('data', (data) => {
                const text = data.toString();
                if (!text.includes('DEPRECATION') && !text.includes('WARNING')) {
                    if (!text.includes('INFO') && !text.includes('WARNING')) {
                        console.log(`[backend-err] ${text}`);
                    }
                }
            });

            // 进程启动失败 (exe 不存在等)
            proc.on('error', () => {
                if (!started) tryStart(index + 1);
            });

            // 进程退出 (被超时杀死或自行退出)
            proc.on('exit', (code) => {
                console.log(`Backend exited: ${code}`);
                if (!started) tryStart(index + 1);
            });

            // 超时: 模型加载可能较慢, 给够时间
            setTimeout(() => {
                if (!started) {
                    proc.kill();
                    // exit 事件会触发 tryStart, 无需重复调用
                }
            }, 20000);  // 20 秒超时 (YOLO 模型加载 + CUDA 初始化)
        }

        tryStart(0);
    });
}

/**
 * 停止后端
 */
function stopBackend() {
    if (backendProcess) {
        try { backendProcess.kill(); } catch(e) {}
        backendProcess = null;
    }
}

/**
 * 创建主窗口
 */
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        minWidth: 900,
        minHeight: 600,
        title: 'Mouse Forwarder',
        backgroundColor: '#1e1e1e',
        frame: false,              // 无边框, 用 HTML 标题栏
        show: false,               // 准备好后再显示
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });

    mainWindow.loadFile(path.join(__dirname, '..', 'frontend', 'index.html'));

    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
    });

    mainWindow.on('close', (event) => {
        // 最小化到托盘, 而不是退出
        if (!app.isQuitting) {
            event.preventDefault();
            mainWindow.hide();
        }
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

/**
 * 创建系统托盘
 */
function createTray() {
    // 创建一个 16x16 的图标
    const icon = nativeImage.createEmpty();
    tray = new Tray(icon);
    
    const contextMenu = Menu.buildFromTemplate([
        {
            label: '显示窗口',
            click: () => {
                if (mainWindow) {
                    mainWindow.show();
                    mainWindow.focus();
                }
            }
        },
        { type: 'separator' },
        {
            label: '退出',
            click: () => {
                app.isQuitting = true;
                stopBackend();
                app.quit();
            }
        }
    ]);

    tray.setToolTip('Mouse Forwarder - CH32V305');
    tray.setContextMenu(contextMenu);

    tray.on('double-click', () => {
        if (mainWindow) {
            mainWindow.show();
            mainWindow.focus();
        }
    });
}

// ============================================================
// 应用生命周期
// ============================================================

app.isQuitting = false;

if (gotLock) {
    app.whenReady().then(async () => {
        createTray();
        createWindow();

        // 启动后端
        try {
            const port = await startBackend();
            actualPort = port;
            console.log(`Backend started on port ${port}`);
            if (mainWindow) {
                mainWindow.webContents.send('backend-ready', { port });
            }
        } catch (err) {
            console.error('Backend error:', err.message);
            if (mainWindow) {
                mainWindow.webContents.send('backend-error', { message: err.message });
            }
        }

        app.on('activate', () => {
            if (BrowserWindow.getAllWindows().length === 0) {
                createWindow();
            } else if (mainWindow) {
                mainWindow.show();
            }
        });
    });
}

app.on('before-quit', () => {
    app.isQuitting = true;
    stopBackend();
});

app.on('window-all-closed', () => {
    // Windows 下不退出, 保留托盘
});

// IPC
ipcMain.handle('get-ws-port', () => actualPort);
ipcMain.handle('minimize-window', () => { if (mainWindow) mainWindow.minimize(); });
ipcMain.handle('maximize-window', () => {
    if (mainWindow) {
        mainWindow.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize();
    }
});
ipcMain.handle('close-window', () => { if (mainWindow) mainWindow.close(); });
ipcMain.handle('quit-app', () => {
    app.isQuitting = true;
    stopBackend();
    app.quit();
});