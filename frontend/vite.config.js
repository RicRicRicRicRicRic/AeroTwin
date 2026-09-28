import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

/**
 * AeroTwin AI renderer build.
 *
 * `base: './'` keeps every asset reference relative so the packaged Electron
 * window can load the bundle straight from `file://`.
 */
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: './',
  server: {
    port: 5173,
    strictPort: true,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
});
