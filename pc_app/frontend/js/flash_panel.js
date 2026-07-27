/**
 * flash_panel.js - 固件烧录面板
 */

class FlashPanel {
    constructor() {
        this.chipSelect = document.getElementById('chip-select');
        this.firmwarePath = document.getElementById('firmware-path');
        this.btnBrowse = document.getElementById('btn-browse');
        this.btnFlash = document.getElementById('btn-flash');
        this.btnDetect = document.getElementById('btn-detect-bootloader');
        this.flashStatus = document.getElementById('flash-status');

        this._ws = null;
        this._onFlash = null;
        this._bindEvents();
    }

    setWs(ws) {
        this._ws = ws;
    }

    _bindEvents() {
        this.btnBrowse.addEventListener('click', () => {
            // Electron 环境: 用文件对话框
            if (window.electronAPI) {
                // 需要额外的 IPC 来打开文件对话框
                this._selectFileElectron();
            } else {
                // 浏览器环境: 用 input 元素
                this._selectFileBrowser();
            }
        });

        this.btnFlash.addEventListener('click', () => {
            const path = this.firmwarePath.value;
            const chip = this.chipSelect.value;
            if (!path) {
                this.setStatus('请先选择固件文件', 'warn');
                return;
            }
            this._startFlash(path, chip);
        });

        this.btnDetect.addEventListener('click', () => {
            this._detectBootloader();
        });
    }

    _selectFileBrowser() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.hex,.bin';
        input.onchange = () => {
            if (input.files[0]) {
                this.firmwarePath.value = input.files[0].name;
                this.setStatus(`已选择: ${input.files[0].name}`, 'ok');
            }
        };
        input.click();
    }

    _selectFileElectron() {
        // 通过 IPC 打开文件对话框
        if (this._ws) {
            this._ws.send({ type: 'select_file', filter: 'hex;bin' });
        }
    }

    _startFlash(path, chip) {
        this.btnFlash.disabled = true;
        this.btnFlash.textContent = '烧录中...';
        this.setStatus('正在烧录...', 'info');

        if (this._ws) {
            this._ws.send({
                type: 'flash_firmware',
                path: path,
                chip: chip,
            });
        }
    }

    _detectBootloader() {
        this.setStatus('正在检测...', 'info');
        if (this._ws) {
            this._ws.send({ type: 'detect_bootloader' });
        }
    }

    setStatus(msg, level = 'info') {
        const colors = {
            info: 'var(--text-muted)',
            ok: 'var(--status-ok)',
            warn: 'var(--status-warn)',
            err: 'var(--status-err)',
        };
        this.flashStatus.textContent = msg;
        this.flashStatus.style.color = colors[level] || colors.info;
    }

    onFlashResult(data) {
        this.btnFlash.disabled = false;
        this.btnFlash.textContent = '烧录';

        if (data.success) {
            this.setStatus(data.message || '烧录成功!', 'ok');
        } else {
            this.setStatus(data.message || '烧录失败', 'err');
        }
    }

    onBootloaderDetect(data) {
        if (data.found) {
            this.setStatus('检测到烧录模式设备!', 'ok');
        } else {
            this.setStatus('未检测到烧录模式设备, 请按住BOOT+按RST', 'warn');
        }
    }

    onFileSelected(data) {
        if (data.path) {
            this.firmwarePath.value = data.path;
            this.setStatus(`已选择: ${data.path.split('\\').pop() || data.path.split('/').pop()}`, 'ok');
        }
    }
}

window.FlashPanel = FlashPanel;