/**
 * app.js - 应用主控制器
 * 本地桌面应用, 通过 Electron IPC 或直接 WS 连接后端
 * 
 * 集成: 鼠标转发 + 目标检测 + AI 轨迹 + 采集卡画面
 */

(function () {
    'use strict';

    let ws = null;
    let mousePanel = null;
    let settings = null;
    let flashPanel = null;
    let capturePanel = null;
    let state = {
        mouseActive: false,
        serialConnected: false,
        stats: { mouse_events: 0, packets_sent: 0, bytes_sent: 0, uptime: 0 },
        trajectory: { enabled: false },
        lockMode: false,
    };

    /**
     * 获取 WS 端口
     */
    async function getWsPort() {
        if (window.electronAPI) {
            return await window.electronAPI.getWsPort();
        }
        return 8765;
    }

    // ================================================================
    // 内容标签切换
    // ================================================================

    function bindTabSwitching() {
        const tabs = document.querySelectorAll('.tab-btn');
        const panels = {
            'mouse': document.getElementById('mouse-panel'),
            'capture': document.getElementById('capture-panel'),
        };

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // 更新标签状态
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                // 切换面板
                const target = tab.dataset.tab;
                Object.entries(panels).forEach(([key, panel]) => {
                    if (panel) {
                        panel.classList.toggle('active', key === target);
                    }
                });
            });
        });
    }

    // ================================================================
    // 轨迹控制 UI 绑定
    // ================================================================

    function bindTrajectoryControls() {
        const btnToggle = document.getElementById('btn-trajectory');
        const btnClear = document.getElementById('btn-trajectory-clear');

        if (btnToggle) {
            btnToggle.addEventListener('click', () => {
                const enabled = state.trajectory && state.trajectory.enabled;
                if (enabled) {
                    ws.send({ type: 'trajectory_disable' });
                } else {
                    ws.send({ type: 'trajectory_enable' });
                }
            });
        }

        if (btnClear) {
            btnClear.addEventListener('click', () => {
                ws.send({ type: 'trajectory_clear' });
                if (mousePanel) mousePanel.clearTrail();
                if (settings) settings.addLog('轨迹已清空', 'info');
            });
        }

        // 平滑度滑块
        const sliderSmooth = document.getElementById('slider-smooth');
        const lblSmooth = document.getElementById('lbl-smooth');
        if (sliderSmooth) {
            sliderSmooth.addEventListener('input', () => {
                const val = parseFloat(sliderSmooth.value);
                if (lblSmooth) lblSmooth.textContent = val.toFixed(2);
                sendTrajectoryConfig();
            });
        }

        // 最大步长
        const sliderMaxStep = document.getElementById('slider-maxstep');
        const lblMaxStep = document.getElementById('lbl-maxstep');
        if (sliderMaxStep) {
            sliderMaxStep.addEventListener('input', () => {
                if (lblMaxStep) lblMaxStep.textContent = sliderMaxStep.value;
                sendTrajectoryConfig();
            });
        }

        // 置信度
        const sliderConf = document.getElementById('slider-confidence');
        const lblConf = document.getElementById('lbl-confidence');
        if (sliderConf) {
            sliderConf.addEventListener('input', () => {
                if (lblConf) lblConf.textContent = parseFloat(sliderConf.value).toFixed(2);
                sendTrajectoryConfig();
            });
        }

        // 偏移量
        const offsetX = document.getElementById('offset-x');
        const offsetY = document.getElementById('offset-y');
        if (offsetX) offsetX.addEventListener('change', sendTrajectoryConfig);
        if (offsetY) offsetY.addEventListener('change', sendTrajectoryConfig);

        // 抖动
        const sliderJitter = document.getElementById('slider-jitter');
        const lblJitter = document.getElementById('lbl-jitter');
        if (sliderJitter) {
            sliderJitter.addEventListener('input', () => {
                if (lblJitter) lblJitter.textContent = parseFloat(sliderJitter.value).toFixed(2);
                sendTrajectoryConfig();
            });
        }
    }

    function sendTrajectoryConfig() {
        const smooth = parseFloat(document.getElementById('slider-smooth')?.value) || 0.35;
        const maxStep = parseInt(document.getElementById('slider-maxstep')?.value) || 10;
        const confidence = parseFloat(document.getElementById('slider-confidence')?.value) || 0.45;
        const offsetX = parseInt(document.getElementById('offset-x')?.value) || 0;
        const offsetY = parseInt(document.getElementById('offset-y')?.value) || 0;
        const jitter = parseFloat(document.getElementById('slider-jitter')?.value) || 0.15;

        ws.send({
            type: 'trajectory_config',
            smooth_factor: smooth,
            max_step_px: maxStep,
            min_confidence: confidence,
            target_offset_x: offsetX,
            target_offset_y: offsetY,
            jitter_amount: jitter,
        });
    }

    // ================================================================
    // 初始化
    // ================================================================

    async function init() {
        mousePanel = new MousePanel();
        settings = new SettingsPanel();
        flashPanel = new FlashPanel();
        capturePanel = new CapturePanel();

        settings.setCallbacks({
            onConnect: (port) => {
                settings.addLog('正在连接串口...', 'info');
                ws.send({ type: 'connect_serial', port });
            },
            onDisconnect: () => {
                settings.addLog('正在断开串口...', 'info');
                ws.send({ type: 'disconnect_serial' });
            },
            onRefreshPorts: () => {
                settings.addLog('刷新端口列表...', 'info');
                ws.send({ type: 'list_ports' });
            },
        });

        // Electron 环境: 监听后端事件
        if (window.electronAPI) {
            window.electronAPI.onBackendReady((data) => {
                settings.addLog(`后端就绪, 端口: ${data.port}`, 'ok');
            });
            window.electronAPI.onBackendError((data) => {
                settings.addLog(`后端错误: ${data.message}`, 'err');
            });
        }

        // 获取端口并连接
        const port = await getWsPort();
        const url = `ws://127.0.0.1:${port}`;

        // 绑定窗口控制按钮
        if (window.electronAPI) {
            document.getElementById('btn-min').onclick = () => window.electronAPI.minimizeWindow();
            document.getElementById('btn-max').onclick = () => window.electronAPI.maximizeWindow();
            document.getElementById('btn-close').onclick = () => window.electronAPI.closeWindow();
        } else {
            document.getElementById('window-controls').style.display = 'none';
        }

        ws = new WSClient(url);

        // 连接面板
        flashPanel.setWs(ws);
        capturePanel.setWs(ws);

        // 绑定控制
        bindTabSwitching();
        bindTrajectoryControls();

        // ---- 事件监听 ----

        ws.on('connected', () => {
            settings.addLog('已连接到后端服务', 'ok');
            ws.send({ type: 'list_ports' });
            ws.send({ type: 'get_state' });
        });

        ws.on('disconnected', () => {
            settings.addLog('后端连接断开', 'warn');
        });

        ws.on('ready', () => {
            settings.addLog('后端就绪', 'ok');
        });

        ws.on('state', onState);
        ws.on('mouse_event', onMouseEvent);
        ws.on('serial_status', onSerialStatus);

        ws.on('port_list', (data) => {
            if (data.ports) {
                settings.updatePortList(data.ports);
                if (data.ports.some(p => p.vid === 0x4348) && !state.serialConnected) {
                    settings.addLog('检测到 CH32V305, 点击"连接"开始转发', 'ok');
                }
            }
        });

        // 烧录事件
        ws.on('flash_result', (data) => flashPanel.onFlashResult(data));
        ws.on('bootloader_detect', (data) => flashPanel.onBootloaderDetect(data));
        ws.on('file_selected', (data) => flashPanel.onFileSelected(data));

        // ---- 采集卡 & 检测事件 ----

        ws.on('capture_status', (data) => {
            capturePanel.onCaptureStatus(data);
            // 更新侧边栏按钮状态
            const btnStart = document.getElementById('btn-capture-start');
            const btnStop = document.getElementById('btn-capture-stop');
            if (btnStart) btnStart.disabled = data.running;
            if (btnStop) btnStop.disabled = !data.running;

            if (data.running) {
                settings.addLog(`采集卡已启动 (Camera ${data.camera_index})`, 'ok');
            } else {
                if (data.error) {
                    settings.addLog(`采集卡错误: ${data.error}`, 'err');
                } else {
                    settings.addLog('采集卡已停止', 'info');
                }
            }
        });

        ws.on('camera_list', (data) => {
            capturePanel.updateCameraList(data.cameras || []);
            const select = document.getElementById('camera-select');
            if (select && data.cameras) {
                select.innerHTML = '';
                if (data.cameras.length > 0) {
                    for (const cam of data.cameras) {
                        const opt = document.createElement('option');
                        opt.value = cam.index;
                        // 优先使用后端返回的 is_ms2130 标记, 其次按名称匹配
                        const isMS2130 = cam.is_ms2130 || (cam.name && cam.name.toLowerCase().includes('ms2130'));
                        opt.textContent = isMS2130 ? `🔴 ${cam.index}: ${cam.name}` : `${cam.index}: ${cam.name}`;
                        if (isMS2130) opt.selected = true;
                        select.appendChild(opt);
                    }
                    settings.addLog(`找到 ${data.cameras.length} 个摄像头`, 'ok');
                } else {
                    select.innerHTML = '<option value="0">未发现摄像头</option>';
                    settings.addLog('未发现摄像头', 'warn');
                }
            }
        });

        ws.on('detection_status', (data) => {
            capturePanel.onDetectionStatus(data);
            const statusEl = document.getElementById('det-model-status');
            if (statusEl) {
                if (data.model_loaded) {
                    statusEl.innerHTML = '<span class="dot ok"></span> 模型已加载';
                    settings.addLog('目标检测模型已加载', 'ok');
                } else {
                    statusEl.innerHTML = '<span class="dot"></span> 未加载';
                }
            }
        });

        ws.on('trajectory_status', (data) => {
            capturePanel.onTrajectoryStatus(data);
            if (state.trajectory) state.trajectory.enabled = data.enabled;
            // 更新轨迹按钮
            const btn = document.getElementById('btn-trajectory');
            if (btn) {
                btn.textContent = data.enabled ? '禁用轨迹' : '启用轨迹';
                btn.className = data.enabled ? 'btn primary' : 'btn';
            }
            settings.addLog(
                data.enabled ? 'AI 轨迹已启用' : 'AI 轨迹已禁用',
                data.enabled ? 'ok' : 'info'
            );
        });

        ws.on('trajectory_config_ack', (data) => {
            settings.addLog('轨迹参数已更新', 'ok');
        });

        ws.on('trajectory_cleared', () => {
            capturePanel.onTrajectoryCleared();
            if (mousePanel) mousePanel.clearTrail();
            settings.addLog('后端轨迹已清空', 'info');
        });

        // 检测帧数据 (包含 JPEG 帧和检测结果)
        ws.on('detection_frame', (data) => {
            capturePanel.onDetectionFrame(data);
            if (mousePanel) mousePanel.updateDetectionFrame(data);
        });

        // 画面显示开关
        ws.on('video_status', (data) => {
            capturePanel.onVideoStatus(data);
        });

        // 锁定模式
        ws.on('lock_mode_status', (data) => {
            state.lockMode = data.enabled;
            updateLockUI(data.enabled);
            settings.addLog(data.enabled ? '🔒 锁定模式已开启 - 鼠标不控制本机' : '锁定模式已关闭', data.enabled ? 'warn' : 'info');
        });

        // 绑定锁定模式按钮
        const btnLock = document.getElementById('btn-lock-mode');
        if (btnLock) {
            btnLock.addEventListener('click', () => {
                if (state.lockMode) {
                    ws.send({ type: 'lock_mode_disable' });
                } else {
                    ws.send({ type: 'lock_mode_enable' });
                }
            });
        }

        ws.connect();
        settings.addLog('Mouse Forwarder 启动中...', 'info');
        settings.addLog(`连接至 ${url}`, 'info');
    }

    // ================================================================
    // 状态更新
    // ================================================================

function onState(data) {
        if (data.mouse_active !== undefined) state.mouseActive = data.mouse_active;
        if (data.serial_connected !== undefined) state.serialConnected = data.serial_connected;
        settings.updateMouseStatus(state.mouseActive);
        settings.updateSerialStatus(data.serial_connected, data.serial_port);
        mousePanel.updateStatus(state.mouseActive, state.serialConnected);
        if (data.stats) {
            state.stats = data.stats;
            mousePanel.updateStats(data.stats);
        }
        if (data.trajectory) {
            state.trajectory = data.trajectory;
            const btn = document.getElementById('btn-trajectory');
            if (btn) {
                btn.textContent = data.trajectory.enabled ? '禁用轨迹' : '启用轨迹';
                btn.className = data.trajectory.enabled ? 'btn primary' : 'btn';
            }
        }
        // 锁定模式状态
        if (data.lock_mode !== undefined) {
            state.lockMode = data.lock_mode;
            updateLockUI(data.lock_mode);
        }
    }

    function onMouseEvent(event) {
        if (!state.mouseActive) {
            state.mouseActive = true;
            settings.updateMouseStatus(true);
            settings.addLog('鼠标捕获已激活', 'ok');
        }
        mousePanel.updateEvent(event);
        state.stats.mouse_events++;
        if (event.serial_connected) state.stats.packets_sent++;
        mousePanel.updateStats(state.stats);
    }

    function onSerialStatus(data) {
        state.serialConnected = data.connected;
        settings.updateSerialStatus(data.connected, data.port);
        mousePanel.updateStatus(state.mouseActive, data.connected);
        settings.addLog(
            data.connected ? `串口已连接: ${data.port}` : '串口已断开',
            data.connected ? 'ok' : 'warn'
        );
    }

    /**
     * 更新锁定模式 UI
     */
    function updateLockUI(enabled) {
        const btnLock = document.getElementById('btn-lock-mode');
        const dotLock = document.getElementById('dot-lock');
        const labelLock = document.getElementById('label-lock');
        const dotLockStatus = document.getElementById('dot-lock-status');
        const lockStatusText = document.getElementById('lock-status-text');

        if (btnLock) {
            btnLock.textContent = enabled ? '🔓 锁定模式 (开) - 按 Esc 退出' : '🔒 锁定模式 (关)';
            btnLock.className = enabled ? 'btn danger' : 'btn';
        }

        // 顶部栏指示器
        if (dotLock && labelLock) {
            dotLock.className = enabled ? 'dot warn' : 'dot';
            labelLock.textContent = enabled ? '锁定: 开' : '锁定: 关';
        }

        // 侧边栏状态
        if (dotLockStatus) {
            dotLockStatus.className = enabled ? 'dot warn' : 'dot';
        }
        if (lockStatusText) {
            lockStatusText.textContent = enabled ? '已锁定 - 鼠标只转发到目标 PC' : '未锁定';
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();