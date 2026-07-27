/**
 * ws_client.js - WebSocket 客户端
 * 参考 Multi3DViz 的 ws_client.js 架构
 */

class WSClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this._listeners = {};
        this._reconnectTimer = null;
        this._reconnectDelay = 1000;
        this._maxReconnectDelay = 5000;
        this._pending = {};
        this._reqId = 0;
        this.connected = false;
    }

    /**
     * 连接 WebSocket 服务器
     */
    connect() {
        if (this.ws) {
            this.close();
        }

        try {
            this.ws = new WebSocket(this.url);
        } catch (e) {
            console.error('WebSocket creation failed:', e);
            this._scheduleReconnect();
            return;
        }

        this.ws.onopen = () => {
            console.log('WS connected:', this.url);
            this.connected = true;
            this._reconnectDelay = 1000;
            this._emit('connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this._handleMessage(data);
            } catch (e) {
                console.warn('WS parse error:', e);
            }
        };

        this.ws.onclose = () => {
            console.log('WS disconnected');
            this.connected = false;
            this.ws = null;
            this._emit('disconnected');
            this._scheduleReconnect();
        };

        this.ws.onerror = (err) => {
            console.error('WS error:', err);
            this._emit('error', err);
        };
    }

    /**
     * 关闭连接
     */
    close() {
        if (this.ws) {
            this.ws.onclose = null;
            this.ws.close();
            this.ws = null;
        }
        this.connected = false;
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
        }
    }

    /**
     * 发送 JSON 消息
     */
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
            return true;
        }
        return false;
    }

    /**
     * 发送请求并等待响应 (Promise)
     */
    request(data) {
        return new Promise((resolve, reject) => {
            const id = ++this._reqId;
            data._req_id = id;
            this._pending[id] = { resolve, reject };

            if (!this.send(data)) {
                delete this._pending[id];
                reject(new Error('Not connected'));
            }

            // 超时
            setTimeout(() => {
                if (this._pending[id]) {
                    delete this._pending[id];
                    reject(new Error('Request timeout'));
                }
            }, 5000);
        });
    }

    /**
     * 注册事件监听
     */
    on(event, callback) {
        if (!this._listeners[event]) {
            this._listeners[event] = [];
        }
        this._listeners[event].push(callback);
    }

    /**
     * 移除事件监听
     */
    off(event, callback) {
        if (!this._listeners[event]) return;
        this._listeners[event] = this._listeners[event]
            .filter(cb => cb !== callback);
    }

    /* ---- 内部方法 ---- */

    _handleMessage(data) {
        // 处理请求响应
        if (data._req_id !== undefined && this._pending[data._req_id]) {
            const { resolve } = this._pending[data._req_id];
            delete this._pending[data._req_id];
            resolve(data);
            return;
        }

        // 按类型分发
        const type = data.type;
        if (type) {
            this._emit(type, data);
        }
        // 通用消息事件
        this._emit('message', data);
    }

    _emit(event, data) {
        const handlers = this._listeners[event] || [];
        for (const cb of handlers) {
            try {
                cb(data);
            } catch (e) {
                console.error(`WS handler error (${event}):`, e);
            }
        }
    }

    _scheduleReconnect() {
        if (this._reconnectTimer) return;

        this._reconnectTimer = setTimeout(() => {
            this._reconnectTimer = null;
            console.log(`Reconnecting (delay=${this._reconnectDelay}ms)...`);
            this.connect();
            this._reconnectDelay = Math.min(
                this._reconnectDelay * 1.5,
                this._maxReconnectDelay
            );
        }, this._reconnectDelay);
    }
}

// 全局导出
window.WSClient = WSClient;