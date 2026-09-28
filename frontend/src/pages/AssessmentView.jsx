import { useEffect, useState } from 'react';

import ScoreCard from '../components/ScoreCard.jsx';
import api, { waitForAssessment } from '../services/api.js';

export default function AssessmentView() {
  const [profiles, setProfiles] = useState([]);
  const [selectedProfileId, setSelectedProfileId] = useState('');
  const [assessments, setAssessments] = useState([]);
  const [selectedAssessment, setSelectedAssessment] = useState(null);

  // New profile form
  const [profileForm, setProfileForm] = useState({
    name: 'Metropolitan Tower Block B',
    address: '450 Seismic Way, Sector 4',
    construction_type: 'reinforced_concrete',
    year_built: 1985,
    stories: 12,
    code_compliance: 'moderate',
  });

  // Defect summary input
  const [defectSummary, setDefectSummary] = useState({
    frames_analyzed: 50,
    frames_with_cracks: 18,
    mean_area_ratio: 0.042,
    mean_width_px: 3.8,
    longest_crack_px: 124.5,
  });

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');

  useEffect(() => {
    loadInitialData();
  }, []);

  async function loadInitialData() {
    try {
      const [pList, aList] = await Promise.all([
        api.listBuildingProfiles().catch(() => []),
        api.listAssessments().catch(() => []),
      ]);
      setProfiles(pList);
      setAssessments(aList);
      if (pList.length > 0) setSelectedProfileId(pList[0].profile_id);
      if (aList.length > 0) setSelectedAssessment(aList[0]);
    } catch (err) {
      console.error(err);
    }
  }

  async function handleCreateProfile(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const created = await api.createBuildingProfile(profileForm);
      setProfiles((prev) => [created, ...prev]);
      setSelectedProfileId(created.profile_id);
      setStatusMessage(`Created profile: ${created.name}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleRunAssessment() {
    if (!selectedProfileId) {
      setError('Select or create a building profile first.');
      return;
    }
    setBusy(true);
    setError(null);
    setStatusMessage('Evaluating seismic vulnerability...');
    try {
      const req = {
        building_profile_id: selectedProfileId,
        defect_summary: defectSummary,
      };
      const created = await api.createAssessment(req);
      const finished = await waitForAssessment(created.assessment_id);
      setSelectedAssessment(finished);
      setAssessments((prev) => [finished, ...prev.filter((a) => a.assessment_id !== finished.assessment_id)]);
      setStatusMessage('Seismic assessment completed successfully.');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">Seismic Vulnerability Assessment</h1>
        <p className="text-xs text-slate-500">
          Combine structural building characteristics with AI-extracted visual defect metrics.
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
        <div className="space-y-6">
          <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-800">1 · Building Profile</h2>
            <div className="mt-3">
              <label className="text-xs font-medium text-slate-600">Select Existing Asset</label>
              <select
                value={selectedProfileId}
                onChange={(e) => setSelectedProfileId(e.target.value)}
                className="mt-1 w-full rounded-lg border border-slate-300 p-2 text-xs"
              >
                <option value="">-- Choose Profile --</option>
                {profiles.map((p) => (
                  <option key={p.profile_id} value={p.profile_id}>
                    {p.name} ({p.construction_type}, built {p.year_built})
                  </option>
                ))}
              </select>
            </div>

            <details className="mt-4 rounded-lg bg-slate-50 p-3 text-xs text-slate-700">
              <summary className="cursor-pointer font-medium text-slate-600">
                + Add New Building Profile
              </summary>
              <form onSubmit={handleCreateProfile} className="mt-3 space-y-2">
                <div>
                  <label className="block text-[11px] text-slate-500">Building Name</label>
                  <input
                    type="text"
                    required
                    value={profileForm.name}
                    onChange={(e) => setProfileForm({ ...profileForm, name: e.target.value })}
                    className="w-full rounded border border-slate-300 p-1.5 text-xs"
                  />
                </div>
                <div>
                  <label className="block text-[11px] text-slate-500">Address</label>
                  <input
                    type="text"
                    value={profileForm.address}
                    onChange={(e) => setProfileForm({ ...profileForm, address: e.target.value })}
                    className="w-full rounded border border-slate-300 p-1.5 text-xs"
                  />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-[11px] text-slate-500">Construction Type</label>
                    <select
                      value={profileForm.construction_type}
                      onChange={(e) =>
                        setProfileForm({ ...profileForm, construction_type: e.target.value })
                      }
                      className="w-full rounded border border-slate-300 p-1.5 text-xs"
                    >
                      <option value="reinforced_concrete">Reinforced Concrete</option>
                      <option value="unreinforced_masonry">Unreinforced Masonry</option>
                      <option value="confined_masonry">Confined Masonry</option>
                      <option value="steel_frame">Steel Frame</option>
                      <option value="timber">Timber</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[11px] text-slate-500">Compliance</label>
                    <select
                      value={profileForm.code_compliance}
                      onChange={(e) =>
                        setProfileForm({ ...profileForm, code_compliance: e.target.value })
                      }
                      className="w-full rounded border border-slate-300 p-1.5 text-xs"
                    >
                      <option value="high">High (Modern Seismic)</option>
                      <option value="moderate">Moderate</option>
                      <option value="low">Low (Pre-Code)</option>
                    </select>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-[11px] text-slate-500">Year Built</label>
                    <input
                      type="number"
                      value={profileForm.year_built}
                      onChange={(e) =>
                        setProfileForm({
                          ...profileForm,
                          year_built: parseInt(e.target.value) || 1990,
                        })
                      }
                      className="w-full rounded border border-slate-300 p-1.5 text-xs"
                    />
                  </div>
                  <div>
                    <label className="block text-[11px] text-slate-500">Stories</label>
                    <input
                      type="number"
                      value={profileForm.stories}
                      onChange={(e) =>
                        setProfileForm({ ...profileForm, stories: parseInt(e.target.value) || 1 })
                      }
                      className="w-full rounded border border-slate-300 p-1.5 text-xs"
                    />
                  </div>
                </div>
                <button
                  type="submit"
                  disabled={busy}
                  className="mt-2 w-full rounded bg-sky-600 py-1.5 font-medium text-white hover:bg-sky-700 disabled:opacity-50"
                >
                  Save Building Profile
                </button>
              </form>
            </details>
          </section>
          <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-slate-800">2 · Defect Quantification Input</h2>
            <p className="mt-1 text-xs text-slate-500">
              Visual damage metrics aggregated from UAV crack mapping.
            </p>
            <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="block text-[11px] text-slate-500">Frames Analyzed</label>
                <input
                  type="number"
                  value={defectSummary.frames_analyzed}
                  onChange={(e) =>
                    setDefectSummary({
                      ...defectSummary,
                      frames_analyzed: parseInt(e.target.value) || 0,
                    })
                  }
                  className="mt-1 w-full rounded border border-slate-300 p-1.5"
                />
              </div>
              <div>
                <label className="block text-[11px] text-slate-500">Frames With Cracks</label>
                <input
                  type="number"
                  value={defectSummary.frames_with_cracks}
                  onChange={(e) =>
                    setDefectSummary({
                      ...defectSummary,
                      frames_with_cracks: parseInt(e.target.value) || 0,
                    })
                  }
                  className="mt-1 w-full rounded border border-slate-300 p-1.5"
                />
              </div>
              <div>
                <label className="block text-[11px] text-slate-500">Mean Area Ratio (%)</label>
                <input
                  type="number"
                  step="0.001"
                  value={defectSummary.mean_area_ratio}
                  onChange={(e) =>
                    setDefectSummary({
                      ...defectSummary,
                      mean_area_ratio: parseFloat(e.target.value) || 0,
                    })
                  }
                  className="mt-1 w-full rounded border border-slate-300 p-1.5"
                />
              </div>
              <div>
                <label className="block text-[11px] text-slate-500">Mean Crack Width (px)</label>
                <input
                  type="number"
                  step="0.1"
                  value={defectSummary.mean_width_px}
                  onChange={(e) =>
                    setDefectSummary({
                      ...defectSummary,
                      mean_width_px: parseFloat(e.target.value) || 0,
                    })
                  }
                  className="mt-1 w-full rounded border border-slate-300 p-1.5"
                />
              </div>
            </div>

            <button
              onClick={handleRunAssessment}
              disabled={busy || !selectedProfileId}
              className="mt-4 w-full rounded-lg bg-sky-600 py-2.5 text-xs font-semibold text-white hover:bg-sky-700 disabled:opacity-50"
            >
              {busy ? 'Evaluating...' : 'Run Seismic Vulnerability Engine'}
            </button>
          </section>
        </div>
        <div className="space-y-6">
          <ScoreCard assessment={selectedAssessment} />

          {assessments.length > 0 && (
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Assessment History
              </h3>
              <div className="mt-2 divide-y divide-slate-100">
                {assessments.map((a) => (
                  <div
                    key={a.assessment_id}
                    onClick={() => setSelectedAssessment(a)}
                    className={`flex cursor-pointer items-center justify-between py-2 text-xs hover:bg-slate-50 ${
                      selectedAssessment?.assessment_id === a.assessment_id
                        ? 'font-semibold text-sky-700'
                        : 'text-slate-700'
                    }`}
                  >
                    <div>
                      <div>{a.building_name}</div>
                      <div className="font-mono text-[10px] text-slate-400">
                        {String(a.assessment_id).slice(0, 8)} · {a.created_at}
                      </div>
                    </div>
                    <span className="font-mono font-bold">
                      {a.result?.vulnerability_score?.toFixed(1) ?? '—'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

