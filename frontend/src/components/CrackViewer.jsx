import { useEffect, useMemo, useState } from 'react';

import api, { artifactRelativePath, artifactUrl } from '../services/api.js';

/**
 * CrackViewer — mapped crack defects for one crack-mapping job.
 *
 * The backend already renders the visual maps: `crack_mask_<id>.png` (binary)
 * and `crack_overlay_<id>.jpg` (frame + red overlay). This viewer toggles
 * between them, walks the frames, and shows the per-frame defect metrics that
 * were logged to `data/processed/crack_maps/<stem>/<run>/metrics.json`.
 */
export default function CrackViewer({ framesRun = '', job = null, className = '' }) {
  const relativeDir = useMemo(
    () => artifactRelativePath(job?.output_dir, 'crack_maps'),
    [job?.output_dir]
  );
  const [frames, setFrames] = useState([]);
  const [index, setIndex] = useState(0);
  const [viewMode, setViewMode] = useState('overlay');
  const [perFrameMetrics, setPerFrameMetrics] = useState([]);
  const [summary, setSummary] = useState(job?.metrics || null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const currentFrame = frames[index] || null;

  useEffect(() => {
    let cancelled = false;
    if (!relativeDir) {
      setFrames([]);
      return () => {
        cancelled = true;
      };
    }
    setLoading(true);
    setError(null);
    api
      .listArtifacts('crack_maps', relativeDir)
      .then((listing) => {
        if (cancelled) return;
        const fileNames = listing.items.filter((item) => !item.is_dir).map((item) => item.name);
        const frameIds = fileNames
          .filter((name) => name.startsWith('crack_mask_'))
          .map((name) => name.replace(/^crack_mask_/, '').replace(/\.png$/i, ''))
          .sort();
        setFrames(
          frameIds.map((frameId) => ({
            frameId,
            maskPath: `${relativeDir}/crack_mask_${frameId}.png`,
            overlayPath: `${relativeDir}/crack_overlay_${frameId}.jpg`,
            framePath: framesRun ? `${framesRun}/frame_${frameId}.jpg` : '',
          }))
        );
        setIndex(0);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    api
      .fetchArtifactJson('crack_maps', `${relativeDir}/metrics.json`)
      .then((payload) => {
        if (cancelled || !payload) return;
        setSummary(payload.summary || job?.metrics || null);
        setPerFrameMetrics(Array.isArray(payload.frames) ? payload.frames : []);
      })
      .catch(() => {
        /* metrics.json is optional; the job-level summary still renders */
      });
    return () => {
      cancelled = true;
    };
  }, [relativeDir, framesRun, job?.metrics]);

  const currentMetrics = currentFrame
    ? perFrameMetrics.find((row) =>
        String(row.frame_filename || '').endsWith(`_${currentFrame.frameId}.jpg`)
      ) || null
    : null;

  let displayUrl = '';
  if (currentFrame) {
    if (viewMode === 'mask') {
      displayUrl = artifactUrl('crack_maps', currentFrame.maskPath);
    } else if (viewMode === 'overlay') {
      displayUrl = artifactUrl('crack_maps', currentFrame.overlayPath);
    } else if (viewMode === 'raw' && currentFrame.framePath) {
      displayUrl = artifactUrl('frames', currentFrame.framePath);
    } else {
      displayUrl = artifactUrl('crack_maps', currentFrame.overlayPath);
    }
  }

  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-4 shadow-sm ${className}`}>
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-800">Crack mapping & quantification</h3>
          <p className="text-xs text-slate-500">
            {job
              ? `job ${String(job.job_id).slice(0, 8)} · ${frames.length} frames mapped`
              : 'No crack-mapping job selected'}
          </p>
        </div>
        <div className="flex rounded-lg border border-slate-200 bg-slate-50 p-0.5 text-xs">
          {['overlay', 'mask', 'raw'].map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setViewMode(mode)}
              className={`rounded-md px-2.5 py-1 font-medium capitalize ${
                viewMode === mode
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-800'
              }`}
            >
              {mode}
            </button>
          ))}
        </div>
      </header>

      {error ? (
        <p className="mb-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          {error}
        </p>
      ) : null}

      {!relativeDir ? (
        <p className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-xs text-slate-500">
          Run crack mapping on preprocessed frames to view defect overlays.
        </p>
      ) : frames.length === 0 ? (
        <p className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-xs text-slate-500">
          {loading ? 'Loading crack maps…' : 'No crack map files found.'}
        </p>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
          <div>
            <div className="relative aspect-video w-full overflow-hidden rounded-lg border border-slate-200 bg-slate-900">
              {displayUrl ? (
                <img
                  src={displayUrl}
                  alt={`Frame ${currentFrame?.frameId}`}
                  className="h-full w-full object-contain"
                />
              ) : null}
            </div>
            <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
              <button
                type="button"
                onClick={() => setIndex((value) => Math.max(value - 1, 0))}
                disabled={index === 0}
                className="rounded border border-slate-300 px-2 py-1 hover:bg-slate-50 disabled:opacity-40"
              >
                ‹ Prev
              </button>
              <span className="font-mono">
                frame {currentFrame?.frameId} ({index + 1}/{frames.length})
              </span>
              <button
                type="button"
                onClick={() => setIndex((value) => Math.min(value + 1, frames.length - 1))}
                disabled={index >= frames.length - 1}
                className="rounded border border-slate-300 px-2 py-1 hover:bg-slate-50 disabled:opacity-40"
              >
                Next ›
              </button>
            </div>
          </div>
          <div className="space-y-3">
            <div>
              <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                Frame metrics
              </div>
              {currentMetrics ? (
                <dl className="grid grid-cols-2 gap-2 rounded-lg bg-slate-50 p-3 text-[11px] text-slate-700">
                  <dt className="text-slate-400">Has crack</dt>
                  <dd className="font-semibold">{currentMetrics.has_crack ? 'Yes' : 'No'}</dd>
                  <dt className="text-slate-400">Crack count</dt>
                  <dd className="font-mono font-semibold">{currentMetrics.crack_count}</dd>
                  <dt className="text-slate-400">Mean width</dt>
                  <dd className="font-mono">{currentMetrics.mean_width_px?.toFixed(2)} px</dd>
                  <dt className="text-slate-400">Max width</dt>
                  <dd className="font-mono">{currentMetrics.max_width_px?.toFixed(2)} px</dd>
                  <dt className="text-slate-400">Total length</dt>
                  <dd className="font-mono">{currentMetrics.total_length_px?.toFixed(1)} px</dd>
                  <dt className="text-slate-400">Longest</dt>
                  <dd className="font-mono">{currentMetrics.max_length_px?.toFixed(1)} px</dd>
                  <dt className="text-slate-400">Area ratio</dt>
                  <dd className="font-mono">{currentMetrics.area_ratio_pct?.toFixed(4)} %</dd>
                </dl>
              ) : (
                <p className="rounded-lg bg-slate-50 p-3 text-[11px] text-slate-400">
                  Per-frame quantification metrics not available.
                </p>
              )}
            </div>

            {summary ? (
              <div>
                <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Run summary
                </div>
                <dl className="grid grid-cols-2 gap-2 rounded-lg border border-slate-200 bg-white p-3 text-[11px] text-slate-700">
                  <dt className="text-slate-400">Cracked frames</dt>
                  <dd className="font-mono">
                    {summary.frames_with_cracks}/{summary.frames_analyzed}
                  </dd>
                  <dt className="text-slate-400">Mean area</dt>
                  <dd className="font-mono">{summary.mean_area_ratio?.toFixed(4)} %</dd>
                  <dt className="text-slate-400">Mean width</dt>
                  <dd className="font-mono">{summary.mean_width_px?.toFixed(2)} px</dd>
                  <dt className="text-slate-400">Longest crack</dt>
                  <dd className="font-mono">{summary.longest_crack_px?.toFixed(1)} px</dd>
                </dl>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </section>
  );
}
