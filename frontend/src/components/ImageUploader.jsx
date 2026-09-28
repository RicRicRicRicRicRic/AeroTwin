import { useState } from 'react';

import api from '../services/api.js';

/**
 * ImageUploader — imports raw UAV videos into the backend raw store.
 *
 * In Electron the native file dialog supplies a real absolute path (which the
 * sidecar needs); in a plain browser the operator can paste the path. Already
 * ingested videos can be re-selected without re-importing.
 */
export default function ImageUploader({
  videos = [],
  selectedFilename = '',
  onSelectVideo,
  onIngested,
  disabled = false,
}) {
  const [sourcePath, setSourcePath] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [record, setRecord] = useState(null);

  const canPickNative =
    typeof window !== 'undefined' && typeof window.aerotwin?.pickVideoFile === 'function';

  const selectedVideo =
    record || videos.find((video) => video.filename === selectedFilename) || null;

  async function browseForFile() {
    setError(null);
    try {
      const picked = await window.aerotwin.pickVideoFile();
      if (picked) setSourcePath(picked);
    } catch (err) {
      setError(err?.message || 'The file picker could not be opened.');
    }
  }

  async function importVideo() {
    const target = sourcePath.trim();
    if (!target) {
      setError('Choose an MP4/MOV file or paste its absolute path first.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const ingested = await api.ingestVideo(target);
      setRecord(ingested);
      if (onIngested) onIngested(ingested);
    } catch (err) {
      setError(err?.message || 'Video import failed.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <header className="mb-3">
        <h2 className="text-sm font-semibold text-slate-800">1 · Import UAV video</h2>
        <p className="text-xs text-slate-500">
          Raw MP4/MOV files are copied into <code>data/inputs/raw_videos/</code> and never
          overwritten.
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-2">
        {canPickNative ? (
          <button
            type="button"
            onClick={browseForFile}
            disabled={disabled || busy}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          >
            Browse…
          </button>
        ) : null}
        <input
          type="text"
          value={sourcePath}
          onChange={(event) => setSourcePath(event.target.value)}
          placeholder="D:\\flights\\facade_survey.mp4"
          disabled={disabled || busy}
          className="min-w-[240px] flex-1 rounded-lg border border-slate-300 px-3 py-2 font-mono text-xs text-slate-700 focus:border-sky-500 focus:outline-none disabled:opacity-50"
        />
        <button
          type="button"
          onClick={importVideo}
          disabled={disabled || busy}
          className="rounded-lg bg-sky-600 px-4 py-2 text-xs font-semibold text-white hover:bg-sky-700 disabled:opacity-50"
        >
          {busy ? 'Importing…' : 'Import video'}
        </button>
      </div>

      {!canPickNative ? (
        <p className="mt-2 text-[11px] text-slate-400">
          Running outside Electron: paste the absolute path of the video on this machine.
        </p>
      ) : null}

      {error ? (
        <p className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          {error}
        </p>
      ) : null}

      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <label className="text-xs text-slate-600">
          <span className="mb-1 block font-medium">Already imported</span>
          <select
            value={selectedVideo?.filename || ''}
            onChange={(event) => onSelectVideo && onSelectVideo(event.target.value)}
            disabled={disabled}
            className="w-full rounded-lg border border-slate-300 px-3 py-2 text-xs text-slate-700 focus:border-sky-500 focus:outline-none disabled:opacity-50"
          >
            <option value="">— select a video —</option>
            {videos.map((video) => (
              <option key={video.video_id} value={video.filename}>
                {video.filename} ({video.duration_seconds?.toFixed(1)} s)
              </option>
            ))}
          </select>
        </label>

        {selectedVideo ? (
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 self-end rounded-lg bg-slate-50 p-3 text-[11px] text-slate-600">
            <dt className="text-slate-400">Frames</dt>
            <dd className="font-mono">{selectedVideo.frame_count}</dd>
            <dt className="text-slate-400">FPS</dt>
            <dd className="font-mono">{selectedVideo.fps}</dd>
            <dt className="text-slate-400">Resolution</dt>
            <dd className="font-mono">
              {selectedVideo.width}×{selectedVideo.height}
            </dd>
            <dt className="text-slate-400">Duration</dt>
            <dd className="font-mono">{selectedVideo.duration_seconds?.toFixed(1)} s</dd>
          </dl>
        ) : (
          <div className="self-end text-[11px] text-slate-400">
            No video selected yet — import one to start the pipeline.
          </div>
        )}
      </div>
    </section>
  );
}
