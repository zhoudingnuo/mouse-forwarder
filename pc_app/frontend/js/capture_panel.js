/**
 * capture_panel.js - 采集卡推理画面展示面板
 * 
 * 显示采集卡实时画面, 叠加目标检测框和 AI 轨迹。
 * 通过 WebSocket 接收 JPEG 帧和检测结果。
 */

class CapturePanel {
    constructor() {
        // DOM 元素
        this.container = document.getElementById('capture-panel');
        this.videoImg = document.getElementById('capture-video');
        this.canvas = document.getElementById('capture-overlay');
        this.ctx = null;
        this.btnStart = document.getElementById('btn-capture-start');
        this.btnStop = document.getElementById('btn-capture-stop');
        this.btnTrajectory = document.getElementById('btn-trajectory');
        this.btnToggleVideo = document.getElementById('btn-toggle-video');
        this.cameraSelect = document.getElementById('camera-select');
        this.btnRefreshCams = document.getElementById('btn-refresh-cameras');
        this.btnFormatToggle = document.getElementById('btn-format-toggle');
        this.captureStatus = document.getElementById('capture-status');
        this.captureDetections = document.getElementById('capture-detections');
        this.captureInference = document.getElementById('capture-inference');
        this.captureResolution = document.getElementById('capture-resolution');
		this.captureTargetOffset = document.getElementById('capture-target-offset');
	this.captureAiSteps = document.getElementById('capture-ai-steps');
	this.captureSettled = document.getElementById('capture-settled');
	this.capturePipelineFps = document.getElementById('capture-pipeline-fps');
	this.captureInfFps = document.getElementById('capture-inf-fps');
        this.trailCanvas = document.getElementById('trail-canvas');
        this.trailCtx = null;
        this.emptyState = document.getElementById('capture-empty');

        // 状态
        this._ws = null;
        this._captureActive = false;
        this._trajectoryEnabled = false;
        this._showVideo = true;
        this._captureFormat = 'yuv';
        this._frameWidth = 0;
        this._frameHeight = 0;
        this._lastDetections = [];
        this._lastAiSteps = [];

        // 轨迹点缓存
        this._trailPoints = [];
        this._maxTrailPoints = 200;

        // 初始化 Canvas
        this._initCanvas();

        // 绑定事件
        this._bindEvents();

        // 设置初始按钮状态
        this._updateVideoBtn();
    }

    /**
     * 设置 WebSocket 客户端
     */
    setWs(ws) {
        this._ws = ws;
    }

    /**
     * 初始化 Canvas
     */
    _initCanvas() {
        if (this.canvas) {
            this.ctx = this.canvas.getContext('2d');
        }
        if (this.trailCanvas) {
            this.trailCtx = this.trailCanvas.getContext('2d');
        }
    }

    /**
     * 绑定 DOM 事件
     */
    _bindEvents() {
        if (this.btnStart) {
            this.btnStart.addEventListener('click', () => {
                const idx = this.cameraSelect ? parseInt(this.cameraSelect.value) : 0;
                this._sendCommand('capture_start', { camera_index: idx });
                this.setStatus('正在启动采集...', 'info');
            });
        }

        if (this.btnStop) {
            this.btnStop.addEventListener('click', () => {
                this._sendCommand('capture_stop');
            });
        }

        if (this.btnTrajectory) {
            this.btnTrajectory.addEventListener('click', () => {
                if (this._trajectoryEnabled) {
                    this._sendCommand('trajectory_disable');
                } else {
                    this._sendCommand('trajectory_enable');
                }
            });
        }

        if (this.btnRefreshCams) {
            this.btnRefreshCams.addEventListener('click', () => {
                this._sendCommand('list_cameras');
            });
        }

        // 采集格式切换 (MJPEG → YUV422 → NV12)
        if (this.btnFormatToggle) {
            this.btnFormatToggle.addEventListener('click', () => {
                const order = ['mjpeg', 'yuv', 'nv12'];
                const cur = order.indexOf(this._captureFormat);
                const newFmt = order[(cur + 1) % order.length];
                this._captureFormat = newFmt;
                this.btnFormatToggle.textContent = `格式: ${newFmt.toUpperCase()}`;
                this._sendCommand('set_capture_format', { format: newFmt });
            });
        }

        if (this.btnToggleVideo) {
            this.btnToggleVideo.addEventListener('click', () => {
                this._showVideo = !this._showVideo;
                this._sendCommand('toggle_video', { show: this._showVideo });
                this._updateVideoBtn();
            });
        }

        // 清空瞄准日志
        const btnClearAim = document.getElementById('btn-clear-aim-log');
        if (btnClearAim) {
            btnClearAim.addEventListener('click', () => {
                const container = document.getElementById('aim-log-entries');
                if (container) container.innerHTML = '';
            });
        }
    }

    /**
     * 发送 WebSocket 命令
     */
    _sendCommand(type, data = {}) {
        if (this._ws && this._ws.connected) {
            this._ws.send({ type, ...data });
        }
    }

    /**
     * 更新摄像头列表
     */
    updateCameraList(cameras) {
        if (!this.cameraSelect) return;

        this.cameraSelect.innerHTML = '';
        for (const cam of cameras) {
            const opt = document.createElement('option');
            opt.value = cam.index;
            // 检测是否为 MS2130 (优先用后端标记, 其次按名称匹配)
            const isMS2130 = cam.is_ms2130 || (cam.name && cam.name.toLowerCase().includes('ms2130'));
            opt.textContent = isMS2130
                ? `🔴 Camera ${cam.index} - MS2130 采集卡`
                : `Camera ${cam.index} - 内置摄像头`;
            if (isMS2130) {
                opt.style.color = 'var(--accent)';
                opt.selected = true;
            }
            this.cameraSelect.appendChild(opt);
        }
    }

    /**
     * 处理检测帧数据
     */
    onDetectionFrame(data) {
        // 更新检测数量
        if (data.detections) {
            this._lastDetections = data.detections;
            if (this.captureDetections) {
                this.captureDetections.textContent = data.detections.length;
            }
        }

        if (data.ai_steps) {
            this._lastAiSteps = data.ai_steps;
        }

	// 更新推理信息 (显示推理耗时)
			if (this.captureInference) {
			    if (data.inference_ms !== undefined) {
			        this.captureInference.textContent = `${data.inference_ms}ms`;
			    } else if (data.trajectory_stats) {
			        this.captureInference.textContent = `${data.trajectory_stats.targets_acquired || 0}`;
			    }
			}

			// 更新管线帧率
			if (data.pipeline_fps !== undefined && this.capturePipelineFps) {
			    this.capturePipelineFps.textContent = `${data.pipeline_fps} FPS`;
			}

			// 更新推理帧率
			if (data.inf_fps !== undefined && this.captureInfFps) {
			    this.captureInfFps.textContent = `${data.inf_fps} FPS`;
			}

// 更新分辨率
		if (data.frame_width && data.frame_height && this.captureResolution) {
		    this.captureResolution.textContent = `${data.frame_width}x${data.frame_height}`;
		}

		// 更新目标偏移
		if (this.captureTargetOffset) {
		    if (data.target_dx !== undefined && data.target_dy !== undefined) {
		        const dx = Math.round(data.target_dx);
		        const dy = Math.round(data.target_dy);
		        this.captureTargetOffset.textContent = `(${dx >= 0 ? '+' : ''}${dx}, ${dy >= 0 ? '+' : ''}${dy})`;
		        this.captureTargetOffset.style.color = (Math.abs(dx) < 5 && Math.abs(dy) < 5) ? 'var(--status-ok)' : 'var(--accent)';
		    } else {
		        this.captureTargetOffset.textContent = '-';
		        this.captureTargetOffset.style.color = '';
		    }
		}

		// 更新 AI 步数
		if (this.captureAiSteps) {
		    if (data.ai_step_count && data.ai_step_count > 0) {
		        const totalDx = data.ai_step_total_dx || 0;
		        const totalDy = data.ai_step_total_dy || 0;
		        this.captureAiSteps.textContent = `${data.ai_step_count}步 (${totalDx >= 0 ? '+' : ''}${totalDx}, ${totalDy >= 0 ? '+' : ''}${totalDy})`;
		        this.captureAiSteps.style.color = 'var(--status-ok)';
		    } else {
		        this.captureAiSteps.textContent = '待命中';
		        this.captureAiSteps.style.color = 'var(--text-muted)';
		    }
		}

		// 更新对准状态
		if (this.captureSettled) {
		    if (data.is_settled) {
		        this.captureSettled.textContent = '✅ 已对准';
		        this.captureSettled.style.color = 'var(--status-ok)';
		    } else if (data.selected_target) {
		        this.captureSettled.textContent = '🎯 瞄准中';
		        this.captureSettled.style.color = 'var(--accent)';
		    } else {
		        this.captureSettled.textContent = '⏳ 搜索中';
		        this.captureSettled.style.color = 'var(--text-muted)';
		    }
		}

        // 如果有 JPEG 帧, 显示
        if (data.frame_jpeg) {
            this._displayFrame(data.frame_jpeg, data.frame_width, data.frame_height);
            // 隐藏空状态
            if (this.emptyState) this.emptyState.style.display = 'none';
        }

// 在 Canvas 上绘制检测框和轨迹
		this._drawOverlay();

		// 处理瞄准日志事件
		if (data.aim_events && data.aim_events.length > 0) {
		    this._addAimLog(data.aim_events);
		}
	    }

	    /**
	     * 添加瞄准日志条目
	     */
	    _addAimLog(events) {
		const container = document.getElementById('aim-log-entries');
		if (!container) return;

		for (const ev of events) {
		    const entry = document.createElement('div');
		    entry.className = `aim-log-entry al-${ev.type}`;

		    const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });

		    const iconMap = {
		        detection: '🔍',
		        target: '🎯',
		        switch: '🔄',
		        settle: '✅',
		        step: '📤',
		        search: '⏳',
		    };

		    entry.innerHTML = `
		        <span class="al-time">[${time}]</span>
		        <span class="al-icon">${iconMap[ev.type] || '•'}</span>
		        <span class="al-text">${this._escapeHtml(ev.text)}</span>
		    `;
		    container.appendChild(entry);
		}

		// 限制最多 200 条
		while (container.children.length > 200) {
		    container.removeChild(container.firstChild);
		}

		// 自动滚动到底部
		container.scrollTop = container.scrollHeight;
	    }

    /**
     * 显示 JPEG 帧
     */
    _displayFrame(base64Data, width, height) {
        if (!this.videoImg) return;

        this._frameWidth = width || this._frameWidth;
        this._frameHeight = height || this._frameHeight;

        this.videoImg.src = `data:image/jpeg;base64,${base64Data}`;
        this.videoImg.style.display = 'block';

        // 调整 Canvas 尺寸匹配
        if (this.canvas) {
            this.canvas.style.width = this.videoImg.style.width || '100%';
            this.canvas.style.height = this.videoImg.style.height || 'auto';
        }
        if (this.trailCanvas) {
            this.trailCanvas.style.width = this.canvas.style.width;
            this.trailCanvas.style.height = this.canvas.style.height;
        }
    }

    /**
     * 在 Canvas 上绘制检测框和轨迹
     */
    _drawOverlay() {
        const canvas = this.canvas;
        if (!canvas || !this.ctx) return;

        const rect = canvas.getBoundingClientRect();
        const displayW = rect.width;
        const displayH = rect.height;

        // 设置 Canvas 实际分辨率
        canvas.width = displayW;
        canvas.height = displayH;

        const ctx = this.ctx;
        ctx.clearRect(0, 0, displayW, displayH);

        // 绘制检测框 (已由后端画在 JPEG 上, 跳过避免双框)
        // 轨迹绘制在下方 _drawTrail() 中

        // 绘制 AI 轨迹连线
        this._drawTrail();
    }

    /**
     * 绘制轨迹连线
     */
    _drawTrail() {
        const canvas = this.trailCanvas;
        if (!canvas || !this.trailCtx) return;

        const rect = canvas.getBoundingClientRect();
        const displayW = rect.width;
        const displayH = rect.height;

        canvas.width = displayW;
        canvas.height = displayH;

        const ctx = this.trailCtx;
        ctx.clearRect(0, 0, displayW, displayH);

        if (this._trailPoints.length < 2) return;

        const scaleX = displayW / (this._frameWidth || 1920);
        const scaleY = displayH / (this._frameHeight || 1080);

        // 绘制轨迹线
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(78, 201, 176, 0.6)';
        ctx.lineWidth = 2;

        const points = this._trailPoints.slice(-100);
        for (let i = 0; i < points.length; i++) {
            const px = points[i].x * scaleX;
            const py = points[i].y * scaleY;
            if (i === 0) {
                ctx.moveTo(px, py);
            } else {
                ctx.lineTo(px, py);
            }
        }
        ctx.stroke();

        // 绘制轨迹点
        const lastPoint = points[points.length - 1];
        if (lastPoint) {
            ctx.beginPath();
            ctx.arc(lastPoint.x * scaleX, lastPoint.y * scaleY, 4, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(78, 201, 176, 0.9)';
            ctx.fill();
        }
    }

    /**
     * 更新采集卡状态
     */
    onCaptureStatus(data) {
        this._captureActive = data.running;
        this._updateUI();
        
        if (data.running) {
            this.setStatus('采集卡运行中', 'ok');
            this.addLog('采集卡已启动', 'ok');
            if (this.emptyState) this.emptyState.style.display = 'none';
        } else {
            this.setStatus(data.error || '已停止', data.error ? 'err' : 'warn');
            this.addLog(data.error ? `采集卡错误: ${data.error}` : '采集卡已停止', data.error ? 'err' : 'warn');
            // 清空画面
            if (this.videoImg) this.videoImg.src = '';
            if (this.ctx && this.canvas) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            if (this.trailCtx && this.trailCanvas) this.trailCtx.clearRect(0, 0, this.trailCanvas.width, this.trailCanvas.height);
            if (this.emptyState) this.emptyState.style.display = '';
        }
    }

    /**
     * 更新轨迹状态
     */
    onTrajectoryStatus(data) {
        this._trajectoryEnabled = data.enabled;
        this._updateUI();
        this.addLog(data.enabled ? 'AI 轨迹已启用' : 'AI 轨迹已禁用', data.enabled ? 'ok' : 'warn');
    }

    /**
     * 更新轨迹配置确认
     */
    onTrajectoryConfig(data) {
        // 配置已确认
    }

    /**
     * 更新轨迹点
     */
    onTrajectoryPoints(data) {
        if (data.points) {
            this._trailPoints = data.points;
        }
    }

    /**
     * 轨迹已清空
     */
    onTrajectoryCleared() {
        this._trailPoints = [];
        if (this.trailCtx) {
            this.trailCtx.clearRect(0, 0, this.trailCanvas.width, this.trailCanvas.height);
        }
        this.addLog('轨迹已清空', 'info');
    }

    /**
     * 更新检测状态
     */
    onDetectionStatus(data) {
        if (this.captureStatus) {
            if (data.model_loaded) {
                this.setStatus('模型已加载', 'ok');
            } else {
                this.setStatus('模型未加载', 'warn');
            }
        }
        this._trajectoryEnabled = data.trajectory_enabled || false;
        this._captureActive = data.capture_active || false;
        this._updateUI();
    }

    /**
     * 更新 UI 状态
     */
    _updateUI() {
        if (this.btnStart) this.btnStart.disabled = this._captureActive;
        if (this.btnStop) this.btnStop.disabled = !this._captureActive;
        if (this.cameraSelect) this.cameraSelect.disabled = this._captureActive;

        if (this.btnTrajectory) {
            this.btnTrajectory.textContent = this._trajectoryEnabled ? '禁用轨迹' : '启用轨迹';
            this.btnTrajectory.className = this._trajectoryEnabled ? 'btn primary' : 'btn';
        }

        // 更新顶部栏指示器
        const dotCapture = document.getElementById('dot-capture');
        const labelCapture = document.getElementById('label-capture');
        if (dotCapture && labelCapture) {
            dotCapture.className = this._captureActive ? 'dot ok' : 'dot';
            labelCapture.textContent = this._captureActive ? '采集: 活跃' : '采集: 待命';
        }
    }

    /**
     * 设置状态文本
     */
    setStatus(msg, level = 'info') {
        if (!this.captureStatus) return;
        const colors = {
            info: 'var(--text-muted)',
            ok: 'var(--status-ok)',
            warn: 'var(--status-warn)',
            err: 'var(--status-err)',
        };
        this.captureStatus.textContent = msg;
        this.captureStatus.style.color = colors[level] || colors.info;
    }

    /**
     * 更新画面显示按钮状态
     */
    _updateVideoBtn() {
        if (!this.btnToggleVideo) return;
        if (this._showVideo) {
            this.btnToggleVideo.textContent = '📹 显示画面: 开';
            this.btnToggleVideo.className = 'btn primary';
        } else {
            this.btnToggleVideo.textContent = '📹 显示画面: 关';
            this.btnToggleVideo.className = 'btn';
        }
    }

    /**
     * 处理视频状态回执
     */
    onVideoStatus(data) {
        this._showVideo = data.show_video;
        this._updateVideoBtn();
        this.addLog(data.show_video ? '画面显示已开启' : '画面显示已关闭', 'info');
        // 关闭时清理画面
        if (!data.show_video) {
            if (this.videoImg) this.videoImg.src = '';
            if (this.ctx && this.canvas) this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            if (this.trailCtx && this.trailCanvas) this.trailCtx.clearRect(0, 0, this.trailCanvas.width, this.trailCanvas.height);
        }
    }

    /**
     * 添加日志
     */
    addLog(message, level = 'info') {
        const logEntries = document.getElementById('log-entries');
        if (!logEntries) return;

        const entry = document.createElement('div');
        entry.className = `log-entry log-${level}`;
        const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
        const levelMap = { info: 'INFO', ok: ' OK ', warn: 'WARN', err: 'ERR ' };
        entry.innerHTML = `
            <span class="log-time">[${time}]</span>
            <span class="log-level">${levelMap[level] || 'INFO'}</span>
            <span class="log-msg">${this._escapeHtml(message)}</span>
        `;
        logEntries.appendChild(entry);
        logEntries.scrollTop = logEntries.scrollHeight;
        while (logEntries.children.length > 500) {
            logEntries.removeChild(logEntries.firstChild);
        }
    }

    _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
}

// 全局导出
window.CapturePanel = CapturePanel;