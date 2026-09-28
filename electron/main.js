'use strict';

const { app, BrowserWindow } = require('electron');
const path = require('path');

const DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';

/**
 * Create the main application window with secure defaults:
 * context isolation on, Node integration off, sandbox on.
 * The renderer talks to the FastAPI sidecar over HTTP, not via Node.
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
    // Development: load the Vite dev server.
    mainWindow.loadURL(DEV_SERVER_URL).catch((error) => {
      console.error(
        `[AeroTwin] Could not reach the Vite dev server at ${DEV_SERVER_URL}. ` +
          'Start it with "npm run dev" inside frontend/.',
        error
      );
    });
  } else {
    // Production: load the built frontend bundle.
    mainWindow.loadFile(path.join(__dirname, '..', 'frontend', 'dist', 'index.html'));
  }
}

app.whenReady().then(() => {
  createWindow();

  // macOS: re-create the window when the dock icon is clicked.
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
