import { useEffect, useState } from 'react';

import ScoreCard from '../components/ScoreCard.jsx';
import api from '../services/api.js';

export default function Dashboard({ onNavigate }) {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [assessments, setAssessments] = useState([]);
  const [reports, setReports] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    // Fault-isolate every call: one failing endpoint (or a missing method)
    // must never abort the batch — otherwise `setHealth` would never run and
    // the badge would stay on "Backend Offline" even with a healthy sidecar.
    const safe = (fn, fallback) => Promise.resolve().then(fn).catch(() => fallback);

    async function loadData() {
      // Every response is coerced before it reaches state, so a malformed or
      // error-shaped payload can never break .map()/.filter()/.slice()/.length
      // during render.
      const toArray = (value) => (Array.isArray(value) ? value : []);
      try {
        setLoading(true);
        const [h, a, r, j, p] = await Promise.all([
          safe(() => api.health(), null),
          safe(() => api.listAssessments(), []),
          safe(() => api.listReports(), []),
          safe(() => api.listJobs(), []),
          safe(() => api.listProfiles(), []),
        ]);
        if (cancelled) return;
        const assessmentList = toArray(a);
        const reportList = toArray(r);
        const jobList = toArray(j);
        setHealth(h && typeof h === 'object' ? h : null);
        // No dedicated stats endpoint exists; derive the metric cards from the
        // lists already fetched for the recent-activity panels.
        setStats({
          jobs_count: jobList.length,
          assessments_count: assessmentList.length,
          reports_count: reportList.length,
          profiles_count: toArray(p).length,
        });
        setAssessments(assessmentList);
        setReports(reportList);
        setJobs(jobList);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadData();
    return () => {
      cancelled = true;
    };
  }, []);

  const latestAssessment = assessments[0] || null;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
          <p className="text-xs text-slate-500">
            AeroTwin AI Structural Health Monitoring & Post-Disaster Inspection
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-block h-2.5 w-2.5 rounded-full ${
              health?.status === 'healthy' ? 'bg-emerald-500' : 'bg-red-500'
            }`}
          />
          <span className="text-xs font-medium text-slate-600">
            Backend {health?.status === 'healthy' ? 'Online' : 'Offline'}
          </span>
        </div>
      </header>

      {error ? (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-xs text-red-700">
          {error}
        </div>
      ) : null}

      {/* Quick Metrics */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
            Processed Jobs
          </div>
          <div className="mt-2 text-2xl font-bold text-slate-800">
            {stats?.jobs_count ?? jobs.length}
          </div>
          <div className="mt-1 text-xs text-slate-500">Video & CV inference runs</div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
            Assessments
          </div>
          <div className="mt-2 text-2xl font-bold text-slate-800">
            {stats?.assessments_count ?? assessments.length}
          </div>
          <div className="mt-1 text-xs text-slate-500">Seismic calculations</div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
            Reports
          </div>
          <div className="mt-2 text-2xl font-bold text-slate-800">
            {stats?.reports_count ?? reports.length}
          </div>
          <div className="mt-1 text-xs text-slate-500">PDF & JSON dossiers</div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
            Profiles
          </div>
          <div className="mt-2 text-2xl font-bold text-slate-800">
            {stats?.profiles_count ?? '—'}
          </div>
          <div className="mt-1 text-xs text-slate-500">Managed structural assets</div>
        </div>
      </div>

      {/* Workflow Navigation Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <div
          onClick={() => onNavigate('inspection')}
          className="group cursor-pointer rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:border-sky-400 hover:shadow"
        >
          <div className="flex items-center justify-between">
            <span className="text-xl">🚁</span>
            <span className="text-xs font-semibold text-sky-600 group-hover:underline">
              Inspect →
            </span>
          </div>
          <h3 className="mt-3 text-sm font-semibold text-slate-800">1. UAV Inspection Pipeline</h3>
          <p className="mt-1 text-xs text-slate-500">
            Import drone video, extract & preprocess frames, run material segmentation and crack
            quantification.
          </p>
        </div>

        <div
          onClick={() => onNavigate('assessment')}
          className="group cursor-pointer rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:border-sky-400 hover:shadow"
        >
          <div className="flex items-center justify-between">
            <span className="text-xl">📊</span>
            <span className="text-xs font-semibold text-sky-600 group-hover:underline">
              Assess →
            </span>
          </div>
          <h3 className="mt-3 text-sm font-semibold text-slate-800">
            2. Seismic Vulnerability Assessment
          </h3>
          <p className="mt-1 text-xs text-slate-500">
            Combine structural building parameters with visual defect quantification to calculate
            seismic vulnerability.
          </p>
        </div>

        <div
          onClick={() => onNavigate('reports')}
          className="group cursor-pointer rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:border-sky-400 hover:shadow"
        >
          <div className="flex items-center justify-between">
            <span className="text-xl">📑</span>
            <span className="text-xs font-semibold text-sky-600 group-hover:underline">
              Reports →
            </span>
          </div>
          <h3 className="mt-3 text-sm font-semibold text-slate-800">3. Inspection Reports</h3>
          <p className="mt-1 text-xs text-slate-500">
            Generate and download formal engineering PDF reports with factor breakdowns and
            recommendations.
          </p>
        </div>
      </div>

      {/* Latest Assessment Preview & Recent Activities */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h2 className="mb-3 text-sm font-semibold text-slate-800">Latest Seismic Assessment</h2>
          {latestAssessment ? (
            <ScoreCard assessment={latestAssessment} />
          ) : (
            <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center text-xs text-slate-400">
              No assessments recorded yet. Run one from the Assessment page.
            </div>
          )}
        </div>

        <div className="space-y-6">
          {/* Recent Jobs */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="mb-3 text-sm font-semibold text-slate-800">Recent Processing Jobs</h2>
            {jobs.length === 0 ? (
              <p className="text-xs text-slate-400">No processing jobs run yet.</p>
            ) : (
              <div className="divide-y divide-slate-100 overflow-hidden text-xs">
                {jobs.slice(0, 5).map((job) => (
                  <div key={job?.job_id ?? Math.random()} className="flex items-center justify-between py-2">
                    <div>
                      <div className="font-medium capitalize text-slate-700">
                        {String(job?.job_type ?? 'unknown').replace(/_/g, ' ')}
                      </div>
                      <div className="font-mono text-[10px] text-slate-400">
                        {String(job?.job_id ?? '').slice(0, 8)}
                      </div>
                    </div>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                        job?.status === 'completed'
                          ? 'bg-emerald-50 text-emerald-700'
                          : job?.status === 'failed'
                          ? 'bg-red-50 text-red-700'
                          : 'bg-amber-50 text-amber-700'
                      }`}
                    >
                      {job?.status ?? 'unknown'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Recent Reports */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="mb-3 text-sm font-semibold text-slate-800">Recent Reports</h2>
            {reports.length === 0 ? (
              <p className="text-xs text-slate-400">No reports generated yet.</p>
            ) : (
              <div className="divide-y divide-slate-100 overflow-hidden text-xs">
                {reports.slice(0, 5).map((rep) => (
                  <div key={rep?.report_id ?? Math.random()} className="flex items-center justify-between py-2">
                    <div>
                      <div className="font-medium text-slate-700">
                        {rep?.building_name ?? 'Inspection report'}
                      </div>
                      <div className="font-mono text-[10px] text-slate-400">
                        {String(rep?.report_id ?? '').slice(0, 8)} ·{' '}
                        {String(rep?.report_format ?? 'unknown')}
                      </div>
                    </div>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                        rep?.status === 'completed'
                          ? 'bg-emerald-50 text-emerald-700'
                          : rep?.status === 'failed'
                          ? 'bg-red-50 text-red-700'
                          : 'bg-amber-50 text-amber-700'
                      }`}
                    >
                      {rep?.status ?? 'unknown'}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}