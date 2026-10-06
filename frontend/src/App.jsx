import { useState } from 'react';

import AssessmentView from './pages/AssessmentView.jsx';
import Dashboard from './pages/Dashboard.jsx';
import InspectionView from './pages/InspectionView.jsx';
import ReportView from './pages/ReportView.jsx';
import ErrorBoundary from './components/ErrorBoundary.jsx';

export default function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: '⚡' },
    { id: 'inspection', label: 'Inspection Pipeline', icon: '🚁' },
    { id: 'assessment', label: 'Seismic Assessment', icon: '📊' },
    { id: 'reports', label: 'Reports', icon: '📑' },
  ];

  return (
    <div className="flex min-h-screen bg-slate-50 text-slate-800 antialiased">
      {/* Sidebar Navigation */}
      <aside className="w-64 border-r border-slate-200 bg-white">
        <div className="flex h-16 items-center border-b border-slate-200 px-6">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-sky-600 font-bold text-white shadow-sm">
              A
            </span>
            <div>
              <span className="text-sm font-bold tracking-tight text-slate-900">AeroTwin AI</span>
              <span className="block text-[10px] uppercase font-semibold text-sky-600">
                Desktop Suite
              </span>
            </div>
          </div>
        </div>

        <nav className="p-4 space-y-1">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id)}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-xs font-semibold transition ${
                currentPage === item.id
                  ? 'bg-sky-50 text-sky-700 shadow-sm'
                  : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
              }`}
            >
              <span className="text-base">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </nav>


      </aside>

      {/* Main Content Area */}
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl p-8">
          {/* Per-page error boundary: a render crash shows a friendly card
              (with Reset) while the sidebar stays navigable. The `key` remounts
              the boundary on navigation, clearing any captured error. */}
          <ErrorBoundary key={currentPage}>
            {currentPage === 'dashboard' && <Dashboard onNavigate={(page) => setCurrentPage(page)} />}
            {currentPage === 'inspection' && <InspectionView />}
            {currentPage === 'assessment' && <AssessmentView />}
            {currentPage === 'reports' && <ReportView />}
          </ErrorBoundary>
        </div>
      </main>
    </div>
  );
}
