import React from 'react';
import { createRoot } from 'react-dom/client';

import App from './App.jsx';
import ErrorBoundary from './components/ErrorBoundary.jsx';
import './index.css';

const container = document.getElementById('root');
if (!container) {
  throw new Error('AeroTwin renderer bootstrap failed: #root element is missing.');
}

createRoot(container).render(
  <React.StrictMode>
    {/* Top-level safety net: any error outside the per-page boundary (e.g. in
        the shell itself) shows a friendly card instead of a blank white screen. */}
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
