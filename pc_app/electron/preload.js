/**
 * preload.js - 安全桥接 Electron API 到渲染进程
 */

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    getWsPort: () => ipcRenderer.invoke('get-ws-port'),
    minimizeWindow: () => ipcRenderer.invoke('minimize-window'),
    maximizeWindow: () => ipcRenderer.invoke('maximize-window'),
    closeWindow: () => ipcRenderer.invoke('close-window'),
    quitApp: () => ipcRenderer.invoke('quit-app'),
    onBackendReady: (callback) => {
        ipcRenderer.on('backend-ready', (_event, data) => callback(data));
    },
    onBackendError: (callback) => {
        ipcRenderer.on('backend-error', (_event, data) => callback(data));
    },
});