/**
 * settings.js - 设置面板 (串口管理)
 * 参考 Multi3DViz 的 plugin panel 设计模式
 */

class SettingsPanel {
    constructor() {
        // DOM 元素
        this.portSelect = document.getElementById('port-select');
        this.btnConnect = document.getElementById('btn-connect');
        this.btnDisconnect = document.getElementById('btn-disconnect');
        this.btnRefresh = document.getElementById('btn-refresh-ports');
        this.btnClearLog = document.getElementById('btn-clear-log');
        this.logEntries = document.getElementById('log-entries');
        this.aimLogEntries = document.getElementById('aim-log-entries');

        // 顶部栏指示器
        this.dotMouse = document.getElementById('dot-mouse');
        this.dotSerial = document.getElementById('dot-serial');
        this.labelMouse = document.getElementById('label-mouse');
        this.labelSerial = document.getElementById('label-serial');

        // 连接状态回调
        this._onConnect = null;
        this._onDisconnect = null;
        this._onRefreshPorts = null;

        // 绑定事件
        this._bindEvents();
    }

    /**
     * 设置事件回调
     */
    setCallbacks(callbacks) {
        this._onConnect = callbacks.onConnect || null;
        this._onDisconnect = callbacks.onDisconnect || null;
        this._onRefreshPorts = callbacks.onRefreshPorts || null;
    }

    /**
     * 更新端口列表
     */
    updatePortList(ports) {
        // 保留当前选中的端口
        const currentPort = this.portSelect.value;

        this.portSelect.innerHTML = '<option value="">-- 自动检测 --</option>';

        for (const p of ports) {
            const opt = document.createElement('option');
            opt.value = p.port;
            
            // 标记 WCH 设备 (VID 0x1A86 = 运行模式 CDC, 0x4348 = 烧录模式)
            if (p.vid === 0x1A86 || p.vid === 0x4348) {
                opt.textContent = `🔗 ${p.port} - ${p.description} (CH32V305)`;
                opt.style.color = 'var(--accent)';
            } else {
                opt.textContent = `${p.port} - ${p.description}`;
            }
            
            this.portSelect.appendChild(opt);
        }

        // 恢复选中
        if (currentPort) {
            for (const opt of this.portSelect.options) {
                if (opt.value === currentPort) {
                    opt.selected = true;
                    break;
                }
            }
        }
    }

    /**
     * 更新串口连接状态
     */
    updateSerialStatus(connected, port) {
        if (connected) {
            this.btnConnect.disabled = true;
            this.btnDisconnect.disabled = false;
            this.dotSerial.className = 'dot ok';
            this.labelSerial.textContent = `串口: ${port || '已连接'}`;
        } else {
            this.btnConnect.disabled = false;
            this.btnDisconnect.disabled = true;
            this.dotSerial.className = 'dot';
            this.labelSerial.textContent = '串口: 未连接';
        }
    }

    /**
     * 更新鼠标状态
     */
    updateMouseStatus(active) {
        if (active) {
            this.dotMouse.className = 'dot ok';
            this.labelMouse.textContent = '鼠标: 活跃';
        } else {
            this.dotMouse.className = 'dot';
            this.labelMouse.textContent = '鼠标: 待命';
        }
    }

    /**
     * 添加日志
     */
    addLog(message, level = 'info') {
        const entry = document.createElement('div');
        entry.className = `log-entry log-${level}`;

        const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
        const levelMap = {
            info: 'INFO',
            ok: ' OK ',
            warn: 'WARN',
            err: 'ERR ',
        };

        entry.innerHTML = `
            <span class="log-time">[${time}]</span>
            <span class="log-level">${levelMap[level] || 'INFO'}</span>
            <span class="log-msg">${this._escapeHtml(message)}</span>
        `;

        this.logEntries.appendChild(entry);
        this.logEntries.scrollTop = this.logEntries.scrollHeight;

        // 限制日志行数
        while (this.logEntries.children.length > 500) {
            this.logEntries.removeChild(this.logEntries.firstChild);
        }
    }

	    /**
	     * 清空日志
	     */
	    clearLog() {
	        this.logEntries.innerHTML = '';
	        this.addLog('日志已清空', 'info');
	    }

	    /**
	     * 添加瞄准日志 (到专用面板)
	     */
	    addAimLog(message) {
	        if (!this.aimLogEntries) return;
	        const entry = document.createElement('div');
	        entry.className = 'log-entry';
	        const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
	        entry.innerHTML = `<span class="log-time">[${time}]</span><span class="log-msg">${this._escapeHtml(message)}</span>`;
	        this.aimLogEntries.appendChild(entry);
	        this.aimLogEntries.scrollTop = this.aimLogEntries.scrollHeight;
	        while (this.aimLogEntries.children.length > 200) {
	            this.aimLogEntries.removeChild(this.aimLogEntries.firstChild);
	        }
	    }

	    /* ---- 内部方法 ---- */

    _bindEvents() {
        this.btnConnect.addEventListener('click', () => {
            const port = this.portSelect.value || null;
            if (this._onConnect) this._onConnect(port);
        });

        this.btnDisconnect.addEventListener('click', () => {
            if (this._onDisconnect) this._onDisconnect();
        });

        this.btnRefresh.addEventListener('click', () => {
            if (this._onRefreshPorts) this._onRefreshPorts();
        });

        this.btnClearLog.addEventListener('click', () => {
            this.clearLog();
        });
    }

    _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
}

// 全局导出
window.SettingsPanel = SettingsPanel;