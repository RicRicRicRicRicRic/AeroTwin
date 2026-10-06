import { useEffect, useState } from 'react';

import CrackViewer from '../components/CrackViewer.jsx';
import ImageUploader from '../components/ImageUploader.jsx';
import MaterialMap from '../components/MaterialMap.jsx';
import StructuralElementViewer from '../components/StructuralElementViewer.jsx';
import api, { waitForJob } from '../services/api.js';

export default function InspectionView() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState('');
  const [extractFps, setExtractFps] = useState(1);
  const [preprocessingMethod, setPreprocessingMethod] = useState('clahe');

  // Job states
  const [extractJob, setExtractJob] = useState(null);
  const [preprocJob, setPreprocJob] = useState(null);
  const [materialJob, setMaterialJob] = useState(null);
  const [crackJob, setCrackJob] = useState(null);
  const [elementJob, setElementJob] = useState(null);

  const [activeTab, setActiveTab] = useState('cracks');
  const [busy, setBusy] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    loadVideos();
  }, []);

  async function loadVideos() {
    try {
      const list = await api.listVideos();
      // Guard the payload before it reaches state so downstream .map()/.length
      // in the uploader can never crash the view on a malformed response.
      setVideos(Array.isArray(list) ? list : []);
      if (Array.isArray(list) && list.length > 0 && !selectedVideo) {
        setSelectedVideo(list[0].filename);
      }
    } catch (err) {
      console.error(err);
    }
  }

  const framesRun = preprocJob?.output_dir || extractJob?.output_dir || '';

  async function runExtract() {
    if (!selectedVideo) return;
    setBusy(true);
    setError(null);
    setStatusMessage('Extracting frames from video…');
    try {
      const resp = await api.extractFrames(selectedVideo, extractFps);
      const done = await waitForJob(resp.job_id);
      setExtractJob(done);
      setStatusMessage(`Extracted ${done.frames_written} frames.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function runPreprocessing() {
    const inputDir = extractJob?.output_dir;
    if (!inputDir) {
      setError('Extract frames first before running preprocessing.');
      return;
    }
    setBusy(true);
    setError(null);
    setStatusMessage(`Running ${preprocessingMethod.toUpperCase()} preprocessing…`);
    try {
      const resp = await api.preprocessFrames(inputDir, preprocessingMethod);
      const done = await waitForJob(resp.job_id);
      setPreprocJob(done);
      setStatusMessage(`Preprocessed ${done.frames_written} frames.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function runMaterialSeg() {
    const inputDir = preprocJob?.output_dir || extractJob?.output_dir;
    if (!inputDir) {
      setError('Frames required for segmentation.');
      return;
    }
    setBusy(true);
    setError(null);
    setStatusMessage('Running Material Segmentation model…');
    try {
      const resp = await api.segmentMaterials(inputDir);
      const done = await waitForJob(resp.job_id);
      setMaterialJob(done);
      setStatusMessage(`Material segmentation completed for ${done.frames_written} frames.`);
      setActiveTab('material');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function runCrackMap() {
    const inputDir = preprocJob?.output_dir || extractJob?.output_dir;
    if (!inputDir) {
      setError('Frames required for crack detection.');
      return;
    }
    setBusy(true);
    setError(null);
    setStatusMessage('Running Crack Detection & Mapping model…');
    try {
      const resp = await api.mapCracks(inputDir);
      const done = await waitForJob(resp.job_id);
      setCrackJob(done);
      setStatusMessage(`Crack mapping completed for ${done.frames_written} frames.`);
      setActiveTab('cracks');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function runElementDetect() {
    const inputDir = preprocJob?.output_dir || extractJob?.output_dir;
    if (!inputDir) {
      setError('Frames required for structural element detection.');
      return;
    }
    setBusy(true);
    setError(null);
    setStatusMessage('Running Structural Element Detection model…');
    try {
      const resp = await api.detectElements(inputDir);
      const done = await waitForJob(resp.job_id);
      setElementJob(done);
      setStatusMessage(`Structural element detection completed for ${done.frames_written} frames.`);
      setActiveTab('elements');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-slate-900">UAV Inspection Pipeline</h1>
        <p className="text-xs text-slate-500">
          Step-by-step UAV inspection: video import → frame extraction → preprocessing → AI model
          inference.
        </p>
      </header>

      {/* Step 1: Video import */}
      <ImageUploader
        videos={videos}
        selectedFilename={selectedVideo}
        onSelectVideo={(filename) => setSelectedVideo(filename)}
        onIngested={(newVid) => {
          setVideos((prev) => [newVid, ...prev]);
          setSelectedVideo(newVid.filename);
        }}
        disabled={busy}
      />

      {/* Pipeline Controls: Steps 2, 3, 4 */}
      <div className="grid gap-4 md:grid-cols-3">
        {/* Step 2: Frame Extraction */}
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            2 · Extract Frames
          </div>
          <p className="mt-1 text-xs text-slate-500">Sample frames at specified target rate.</p>
          <div className="mt-3 flex items-center gap-2">
            <label className="text-xs text-slate-600">
              FPS:
              <input
                type="number"
                min="0.1"
                max="30"
                step="0.5"
                value={extractFps}
                onChange={(e) => setExtractFps(parseFloat(e.target.value) || 1)}
                disabled={busy}
                className="ml-2 w-16 rounded border border-slate-300 px-2 py-1 text-xs"
              />
            </label>
            <button
              onClick={runExtract}
              disabled={busy || !selectedVideo}
              className="flex-1 rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-900 disabled:opacity-50"
            >
              Extract
            </button>
          </div>
          {extractJob && (
            <div className="mt-2 text-[11px] text-emerald-600">
              ✓ {extractJob.frames_written} frames ready
            </div>
          )}
        </div>

        {/* Step 3: Frame Preprocessing */}
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            3 · Preprocess Frames
          </div>
          <p className="mt-1 text-xs text-slate-500">Enhance contrast & normalize lighting.</p>
          <div className="mt-3 flex items-center gap-2">
            <select
              value={preprocessingMethod}
              onChange={(e) => setPreprocessingMethod(e.target.value)}
              disabled={busy}
              className="w-28 rounded border border-slate-300 px-2 py-1 text-xs"
            >
              <option value="clahe">CLAHE</option>
              <option value="histogram_eq">Hist Eq</option>
              <option value="bilateral">Bilateral</option>
              <option value="gaussian_blur">Gaussian</option>
            </select>
            <button
              onClick={runPreprocessing}
              disabled={busy || !extractJob}
              className="flex-1 rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-900 disabled:opacity-50"
            >
              Preprocess
            </button>
          </div>
          {preprocJob && (
            <div className="mt-2 text-[11px] text-emerald-600">
              ✓ {preprocJob.frames_written} enhanced frames
            </div>
          )}
        </div>

        {/* Step 4: Run AI Models */}
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            4 · Run AI Models
          </div>
          <p className="mt-1 text-xs text-slate-500">Execute CV models on preprocessed frames.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              onClick={runCrackMap}
              disabled={busy || (!preprocJob && !extractJob)}
              className="rounded bg-rose-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-rose-700 disabled:opacity-50"
            >
              Cracks
            </button>
            <button
              onClick={runMaterialSeg}
              disabled={busy || (!preprocJob && !extractJob)}
              className="rounded bg-indigo-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
            >
              Materials
            </button>
            <button
              onClick={runElementDetect}
              disabled={busy || (!preprocJob && !extractJob)}
              className="rounded bg-emerald-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
            >
              Elements
            </button>
          </div>
          {elementJob && (
            <div className="mt-2 text-[11px] text-emerald-600">
              ✓ {elementJob.frames_written} elements detected
            </div>
          )}
        </div>
      </div>

      {statusMessage && (
        <div className="rounded-lg bg-sky-50 px-4 py-2 text-xs font-medium text-sky-800">
          {statusMessage}
        </div>
      )}

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-xs text-red-700">
          {error}
        </div>
      )}

      {/* Step 5: Visual Inspection Viewers */}
      <div className="space-y-4">
        <div className="flex border-b border-slate-200">
          <button
            onClick={() => setActiveTab('cracks')}
            className={`border-b-2 px-4 py-2 text-xs font-semibold ${
              activeTab === 'cracks'
                ? 'border-rose-600 text-rose-600'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            Crack Mapping ({crackJob ? `${crackJob.frames_written} frames` : 'idle'})
          </button>
          <button
            onClick={() => setActiveTab('material')}
            className={`border-b-2 px-4 py-2 text-xs font-semibold ${
              activeTab === 'material'
                ? 'border-indigo-600 text-indigo-600'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            Material Masks ({materialJob ? `${materialJob.frames_written} frames` : 'idle'})
          </button>
          <button
            onClick={() => setActiveTab('elements')}
            className={`border-b-2 px-4 py-2 text-xs font-semibold ${
              activeTab === 'elements'
                ? 'border-emerald-600 text-emerald-600'
                : 'border-transparent text-slate-500 hover:text-slate-800'
            }`}
          >
            Structural Elements ({elementJob ? `${elementJob.frames_written} frames` : 'idle'})
          </button>
        </div>

        {activeTab === 'cracks' && (
          <CrackViewer
            framesRun={framesRun ? framesRun.replace(/^.*[\\/]/, '') : ''}
            job={crackJob}
          />
        )}
        {activeTab === 'material' && (
          <MaterialMap
            framesRun={framesRun ? framesRun.replace(/^.*[\\/]/, '') : ''}
            job={materialJob}
          />
        )}
        {activeTab === 'elements' && (
          <StructuralElementViewer
            framesRun={framesRun ? framesRun.replace(/^.*[\\/]/, '') : ''}
            job={elementJob}
          />
        )}
      </div>
    </div>
  );
}