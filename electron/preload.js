'use strict';

const { contextBridge, ipcRenderer } = require('electron');

/**
 * Secure bridge between the sandboxed renderer and the main process.
 * Only explicitly whitelisted values are exposed here; the renderer reaches
 * the FastAPI sidecar through plain HTTP (fetch), not through Node APIs.
 */
contextBridge.exposeInMainWorld('aerotwin', {
  platform: process.platform,
  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
    node: process.versions.node,
  },
  pickVideoFile: () => ipcRenderer.invoke('aerotwin:pick-video-file'),
  getBackendUrl: () => ipcRenderer.invoke('aerotwin:get-backend-url'),
});

