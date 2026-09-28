'use strict';

const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

const BACKEND_URL = process.env.AEROTWIN_BACKEND_URL || 'http://127.0.0.1:8000';
const DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';

let backendProcess = null;

function checkBackendHealth(timeoutMs = 1500) {
  return new Promise((resolve) => {
    try {
      const parsed = new URL(BACKEND_URL);
      const req = http.request(
        {
          hostname: parsed.hostname,
          port: parsed.port || 8000,
          path: '/health',
          method: 'GET',
          timeout: timeoutMs,
        },
        (res) => {
          resolve(res.statusCode === 200);
        }
      );
      req.on('error', () => resolve(false));
      req.on('timeout', () => {
        req.destroy();
        resolve(false);
      });
      req.end();
    } catch {
      resolve(false);
    }
  });
}

async function waitForBackend(maxAttempts = 30, intervalMs = 500) {
  for (let i = 0; i < maxAttempts; i++) {
    const alive = await checkBackendHealth();
    if (alive) return true;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return false;
}

function startBackendSidecar() {
  const backendDir = path.join(__dirname, '..', 'backend');
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

  console.log(`[AeroTwin] Launching FastAPI backend sidecar in ${backendDir}...`);
  backendProcess = spawn(pythonCmd, ['run_backend.py'], {
    cwd: backendDir,
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: true,
  });

  backendProcess.stdout?.on('data', (chunk) => {
    process.stdout.write(`[sidecar] ${chunk}`);
  });

  backendProcess.stderr?.on('data', (chunk) => {
    process.stderr.write(`[sidecar err] ${chunk}`);
  });

  backendProcess.on('exit', (code, signal) => {
    console.log(`[AeroTwin] Backend sidecar exited with code ${code}, signal ${signal}`);
    backendProcess = null;
  });
}

function stopBackendSidecar() {
  if (backendProcess) {
    console.log('[AeroTwin] Stopping backend sidecar...');
    try {
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', String(backendProcess.pid), '/f', '/t']);
      } else {
        backendProcess.kill('SIGTERM');
      }
    } catch (err) {
      console.error('[AeroTwin] Error killing sidecar:', err);
    }
    backendProcess = null;
  }
}

/**
 * Register IPC handlers exposed via preload.js.
 */
function registerIpcHandlers() {
  ipcMain.handle('aerotwin:pick-video-file', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Select UAV Video for Inspection',
      properties: ['openFile'],
      filters: [
        { name: 'Video Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'm4v'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    });
    if (result.canceled || !result.filePaths || result.filePaths.length === 0) {
      return null;
    }
    return result.filePaths[0];
  });

  ipcMain.handle('aerotwin:get-backend-url', () => BACKEND_URL);
}

/**
 * Create the main application window with secure defaults:
 * context isolation on, Node integration off, sandbox on.
 */
function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.once('ready-to-show', () => mainWindow.show());

  if (!app.isPackaged) {
    mainWindow.loadURL(DEV_SERVER_URL).catch((error) => {
      console.error(
        `[AeroTwin] Could not reach the Vite dev server at ${DEV_SERVER_URL}. ` +
          'Start it with "npm run dev" inside frontend/.',
        error
      );
    });
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'frontend', 'dist', 'index.html'));
  }
}

app.whenReady().then(async () => {
  registerIpcHandlers();

  // Check if sidecar is already running; if not, spawn it
  const alreadyRunning = await checkBackendHealth(500);
  if (!alreadyRunning) {
    startBackendSidecar();
    const up = await waitForBackend();
    if (!up) {
      console.warn('[AeroTwin] Warning: Backend sidecar did not report healthy within timeout.');
    }
  }

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  stopBackendSidecar();
});

app.on('will-quit', () => {
  stopBackendSidecar();
});

