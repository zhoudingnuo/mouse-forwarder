/**
 * mouse_panel.js - 鼠标实时数据显示面板
 * 参考 Multi3DViz 的实时数据展示模式
 * 
 * 功能:
 * - 鼠标事件实时显示 (坐标、按钮、滚轮)
 * - 轨迹状态统计
 * - 检测统计 (AI 轨迹计数)
 */

class MousePanel {
    constructor() {
        // DOM 元素
        this.display = document.getElementById('mouse-display');
        this.coordDx = document.getElementById('coord-dx');
        this.coordDy = document.getElementById('coord-dy');
        this.btnLeft = document.getElementById('btn-left');
        this.btnRight = document.getElementById('btn-right');
        this.btnMiddle = document.getElementById('btn-middle');
        this.wheelValue = document.getElementById('wheel-value');
        this.streamBar = document.getElementById('stream-bar');
        this.streamRate = document.getElementById('stream-rate');
        this.statTrajectory = document.getElementById('stat-trajectory');

        // 状态
        this._lastEventTime = 0;
        this._packetCount = 0;
        this._rateTimer = null;
        this._currentRate = 0;
        this._barTimeout = null;

        // 启动速率计算
        this._startRateCalc();
    }

    /**
     * 更新鼠标事件显示
     */
    updateEvent(event) {
        this._lastEventTime = Date.now();
        this._packetCount++;

        // 显示活动状态
        this.display.classList.add('active');

        // 更新坐标
        this._updateCoord(this.coordDx, event.dx || 0);
        this._updateCoord(this.coordDy, event.dy || 0);

        // 更新按钮状态
        this._updateButton(this.btnLeft, event.left || false);
        this._updateButton(this.btnRight, event.right || false, 'right');
        this._updateButton(this.btnMiddle, event.middle || false);

        // 更新滚轮
        if (event.wheel !== undefined && event.wheel !== 0) {
            this.wheelValue.textContent = event.wheel > 0 ? '+' + event.wheel : event.wheel;
            this.wheelValue.style.color = 'var(--accent)';
            clearTimeout(this._wheelTimeout);
            this._wheelTimeout = setTimeout(() => {
                this.wheelValue.textContent = '0';
                this.wheelValue.style.color = '';
            }, 500);
        }

        // 数据流条
        this._updateStreamBar();
    }

    /**
     * 更新 AI 检测帧数据 (统计信息)
     */
    updateDetectionFrame(data) {
        if (data.trajectory_stats && this.statTrajectory) {
            this.statTrajectory.textContent = data.trajectory_stats.trajectories_computed || 0;
        }
    }

    /**
     * 更新状态栏信息
     */
    updateStatus(mouseActive, serialConnected) {
        const sbMouse = document.getElementById('sb-mouse');
        const sbSerial = document.getElementById('sb-serial');
        const sbDotMouse = document.getElementById('sb-dot-mouse');
        const sbDotSerial = document.getElementById('sb-dot-serial');

        if (mouseActive) {
            sbMouse.innerHTML = '<span class="dot ok"></span>鼠标: 活跃';
            sbDotMouse.className = 'dot ok';
        } else {
            sbMouse.innerHTML = '<span class="dot"></span>鼠标: 等待中';
            sbDotMouse.className = 'dot';
        }

        if (serialConnected) {
            sbSerial.innerHTML = '<span class="dot ok"></span>串口: 已连接';
            sbDotSerial.className = 'dot ok';
        } else {
            sbSerial.innerHTML = '<span class="dot"></span>串口: 未连接';
            sbDotSerial.className = 'dot';
        }
    }

    /**
     * 更新统计数据
     */
    updateStats(stats) {
        const elEvents = document.getElementById('stat-events');
        const elPackets = document.getElementById('stat-packets');
        const elBytes = document.getElementById('stat-bytes');
        const elUptime = document.getElementById('stat-uptime');
        const sbPackets = document.getElementById('sb-packets');

        if (elEvents) elEvents.textContent = this._formatNum(stats.mouse_events || 0);
        if (elPackets) elPackets.textContent = this._formatNum(stats.packets_sent || 0);
        if (elBytes) elBytes.textContent = this._formatBytes(stats.bytes_sent || 0);
        if (sbPackets) sbPackets.textContent = `已发送: ${this._formatNum(stats.packets_sent || 0)}`;

        if (elUptime && stats.uptime !== undefined) {
            elUptime.textContent = this._formatDuration(stats.uptime);
        }
    }

    /**
     * 启用/禁用轨迹 (仅更新 UI 状态, 无 Canvas)
     */
    setTrailEnabled(enabled) {
        // 由 app.js 控制按钮文字, 这里不做视觉绘制
    }

    /**
     * 清空所有轨迹 (仅更新状态)
     */
    clearTrail() {
        // 无 Canvas, 无需操作
    }

    /* ---- 内部方法 ---- */

    _updateCoord(el, value) {
        el.textContent = value;
        el.className = 'coord-value' + (value < 0 ? ' negative' : '');
    }

    _updateButton(el, active, cls = '') {
        const indicator = el;
        if (active) {
            indicator.className = 'btn-indicator active' + (cls ? ' ' + cls : '');
        } else {
            indicator.className = 'btn-indicator';
        }
    }

    _updateStreamBar() {
        const bar = this.streamBar;
        bar.style.width = '100%';
        bar.style.transition = 'width 0.15s linear';
        
        clearTimeout(this._barTimeout);
        this._barTimeout = setTimeout(() => {
            bar.style.transition = 'width 0.5s ease-out';
            bar.style.width = '20%';
        }, 150);
    }

    _startRateCalc() {
        let lastCount = 0;
        this._rateTimer = setInterval(() => {
            const rate = this._packetCount - lastCount;
            lastCount = this._packetCount;
            this._currentRate = rate;
            this.streamRate.textContent = `${rate} pkt/s`;

            if (Date.now() - this._lastEventTime > 2000) {
                this.display.classList.remove('active');
            }
        }, 1000);
    }

    _formatNum(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return n.toString();
    }

    _formatBytes(n) {
        if (n >= 1048576) return (n / 1048576).toFixed(1) + 'MB';
        if (n >= 1024) return (n / 1024).toFixed(1) + 'KB';
        return n + 'B';
    }

    _formatDuration(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        const h = Math.floor(m / 60);
        if (h > 0) {
            return `${h}:${String(m % 60).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
        }
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }
}

// 全局导出
window.MousePanel = MousePanel;