import { useEffect, useState } from 'react';

import api, { artifactUrl, waitForReport } from '../services/api.js';

export default function ReportView() {
  const [assessments, setAssessments] = useState([]);
  const [selectedAssessmentId, setSelectedAssessmentId] = useState('');
  const [reports, setReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);

  // Form options
  const [format, setFormat] = useState('pdf');
  const [inspectorName, setInspectorName] = useState('Dr. Alex Mercer, PE');
  const [organization, setOrganization] = useState('Structural Dynamics Group');
  const [notes, setNotes] = useState(
    'Comprehensive post-disaster UAV facade inspection and seismic vulnerability assessment.'
  );

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      const [aList, rList] = await Promise.all([
        api.listAssessments().catch(() => []),
        api.listReports().catch(() => []),
      ]);
      const completedAssessments = aList.filter((a) => a.status === 'completed');
      setAssessments(completedAssessments);
      setReports(rList);
      if (completedAssessments.length > 0) {
        setSelectedAssessmentId(completedAssessments[0].assessment_id);
      }
      if (rList.length > 0) {
        setSelectedReport(rList[0]);
      }
    } catch (err) {
      console.error(err);
    }
  }

  async function handleGenerateReport(e) {
    e.preventDefault();
    if (!selectedAssessmentId) {
      setError('Please select a completed assessment first.');
      return;
    }
    setBusy(true);
    setError(null);
    setStatusMessage(`Compiling ${format.toUpperCase()} engineering report...`);
    try {
      const req = {
        assessment_id: selectedAssessmentId,
        format,
        inspector_name: inspectorName,
        organization,
        notes,
      };
      const created = await api.generateReport(req);
      const finished = await waitForReport(created.report_id);
      setSelectedReport(finished);
      setReports((prev) => [finished, ...prev.filter((r) => r.report_id !== finished.report_id)]);
      setStatusMessage(`Report ${finished.report_id.slice(0, 8)} generated successfully.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">Inspection Reports</h1>
        <p className="text-xs text-slate-500">
          Generate formal engineering dossiers (PDF / JSON) documenting visual defects, factor
          breakdowns, and seismic recommendations.
        </p>
      </header>

      {statusMessage && (
        <div className="rounded-lg bg-emerald-50 px-4 py-2 text-xs font-medium text-emerald-800">
          {statusMessage}
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-xs text-red-700">
          {error}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Left Column: Generate New Report Form */}
        <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-slate-800">Generate Engineering Report</h2>
          <form onSubmit={handleGenerateReport} className="mt-4 space-y-3">
            <div>
              <label className="block text-xs font-medium text-slate-600">
                Completed Assessment
              </label>
              <select
                value={selectedAssessmentId}
                onChange={(e) => setSelectedAssessmentId(e.target.value)}
                disabled={busy || assessments.length === 0}
                className="mt-1 w-full rounded border border-slate-300 p-2 text-xs"
              >
                {assessments.length === 0 ? (
                  <option value="">No completed assessments found</option>
                ) : (
                  assessments.map((a) => (
                    <option key={a.assessment_id} value={a.assessment_id}>
                      {a.building_name} (Score: {a.result?.vulnerability_score?.toFixed(1)})
                    </option>
                  ))
                )}
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-600">Output Format</label>
              <div className="mt-1 flex gap-4 text-xs text-slate-700">
                <label className="flex items-center gap-1.5">
                  <input
                    type="radio"
                    name="format"
                    value="pdf"
                    checked={format === 'pdf'}
                    onChange={() => setFormat('pdf')}
                  />
                  PDF Document (ReportLab)
                </label>
                <label className="flex items-center gap-1.5">
                  <input
                    type="radio"
                    name="format"
                    value="json"
                    checked={format === 'json'}
                    onChange={() => setFormat('json')}
                  />
                  JSON Data Dossier
                </label>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-slate-600">Inspector Name</label>
                <input
                  type="text"
                  value={inspectorName}
                  onChange={(e) => setInspectorName(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-300 p-2 text-xs"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600">Organization</label>
                <input
                  type="text"
                  value={organization}
                  onChange={(e) => setOrganization(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-300 p-2 text-xs"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-600">
                Engineering Observations & Notes
              </label>
              <textarea
                rows={3}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                className="mt-1 w-full rounded border border-slate-300 p-2 text-xs"
              />
            </div>

            <button
              type="submit"
              disabled={busy || !selectedAssessmentId}
              className="w-full rounded-lg bg-sky-600 py-2.5 text-xs font-semibold text-white hover:bg-sky-700 disabled:opacity-50"
            >
              {busy ? 'Compiling Report...' : `Compile ${format.toUpperCase()} Report`}
            </button>
          </form>
        </section>

        {/* Right Column: Selected Report Details & List */}
        <div className="space-y-6">
          <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-800">Report Details</h2>
            {selectedReport ? (
              <div className="mt-3 space-y-3 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-800">
                    {selectedReport.building_name}
                  </span>
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 font-mono text-[10px] font-bold uppercase text-slate-600">
                    {selectedReport.format}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-600">
                  <div>
                    <span className="text-slate-400">Inspector:</span>{' '}
                    {selectedReport.inspector_name || '—'}
                  </div>
                  <div>
                    <span className="text-slate-400">Organization:</span>{' '}
                    {selectedReport.organization || '—'}
                  </div>
                  <div>
                    <span className="text-slate-400">File size:</span>{' '}
                    {selectedReport.file_size_bytes
                      ? `${(selectedReport.file_size_bytes / 1024).toFixed(1)} KB`
                      : '—'}
                  </div>
                  <div>
                    <span className="text-slate-400">Date:</span> {selectedReport.created_at}
                  </div>
                </div>

                {selectedReport.file_path ? (
                  <div className="pt-2">
                    <a
                      href={artifactUrl(
                        'reports',
                        selectedReport.file_path.replace(/^.*[\\/]reports[\\/]/i, '')
                      )}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center justify-center rounded-lg bg-slate-800 px-4 py-2 font-medium text-white hover:bg-slate-900"
                    >
                      Download / Open {selectedReport.format.toUpperCase()}
                    </a>
                  </div>
                ) : null}
              </div>
            ) : (
              <p className="mt-2 text-xs text-slate-400">
                Select a report below to view details and download.
              </p>
            )}
          </section>

          {/* Report History */}
          <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Generated Reports
            </h3>
            {reports.length === 0 ? (
              <p className="mt-2 text-xs text-slate-400">No reports generated yet.</p>
            ) : (
              <div className="mt-2 divide-y divide-slate-100">
                {reports.map((r) => (
                  <div
                    key={r.report_id}
                    onClick={() => setSelectedReport(r)}
                    className={`flex cursor-pointer items-center justify-between py-2 text-xs hover:bg-slate-50 ${
                      selectedReport?.report_id === r.report_id
                        ? 'font-semibold text-sky-700'
                        : 'text-slate-700'
                    }`}
                  >
                    <div>
                      <div>{r.building_name}</div>
                      <div className="font-mono text-[10px] text-slate-400">
                        {String(r.report_id).slice(0, 8)} · {r.created_at}
                      </div>
                    </div>
                    <span className="rounded bg-slate-100 px-2 py-0.5 font-mono text-[10px] font-bold uppercase text-slate-600">
                      {r.format}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

