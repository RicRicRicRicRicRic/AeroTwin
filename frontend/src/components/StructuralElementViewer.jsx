import { useEffect, useMemo, useRef, useState } from 'react';

import api, { artifactRelativePath, artifactUrl } from '../services/api.js';
import { loadImage } from './MaterialMap.jsx';

export const ELEMENT_CLASSES = [
  'background',
  'window',
  'door',
  'column',
  'wall',
  'beam',
];

const ELEMENT_COLOURS = [
  [15, 23, 42],
  [59, 130, 246],
  [239, 68, 68],
  [249, 115, 22],
  [168, 85, 247],
  [234, 179, 8],
];

const ELEMENT_HEX = ['#0f172a', '#3b82f6', '#ef4444', '#f97316', '#a855f7', '#eab308'];

export default function StructuralElementViewer({
  framesRun = '',
  job = null,
  className = '',
}) {
  const relativeDir = useMemo(
    () => artifactRelativePath(job?.output_dir, 'structural_elements'),
    [job?.output_dir]
  );
  const [masks, setMasks] = useState([]);
  const [index, setIndex] = useState(0);
  const [blendOnFrame, setBlendOnFrame] = useState(true);
  const [distribution, setDistribution] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activePanel, setActivePanel] = useState('overlay');
  const canvasRef = useRef(null);

  const currentMask = masks[index] || null;
  const frameId = currentMask
    ? currentMask.name.replace(/^mask_/, '').replace(/\.(png|jpe?g)$/i, '')
    : '';
  const framePath = framesRun && frameId ? `${framesRun}/frame_${frameId}.jpg` : '';

  useEffect(() => {
    let cancelled = false;
    if (!relativeDir) {
      setMasks([]);
      return () => {
        cancelled = true;
      };
    }
    setLoading(true);
    setError(null);
    api
      .listArtifacts('structural_elements', relativeDir)
      .then((listing) => {
        if (cancelled) return;
        setMasks(
          listing.items.filter((item) => !item.is_dir && item.name.startsWith('mask_'))
        );
        setIndex(0);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [relativeDir]);

  useEffect(() => {
    let cancelled = false;
    if (!currentMask) {
      return () => {
        cancelled = true;
      };
    }
    setError(null);
    (async () => {
      try {
        const maskImage = await loadImage(
          artifactUrl('structural_elements', currentMask.relative_path)
        );
        const frameImage = await loadImage(artifactUrl('frames', framePath)).catch(() => null);
        if (cancelled) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        const width = maskImage.naturalWidth;
        const height = maskImage.naturalHeight;
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d');

        // Draw raw mask into scratch canvas to read class ID per pixel
        const scratch = document.createElement('canvas');
        scratch.width = width;
        scratch.height = height;
        const scratchContext = scratch.getContext('2d');
        scratchContext.drawImage(maskImage, 0, 0);
        const maskData = scratchContext.getImageData(0, 0, width, height).data;

        // Draw background: original frame if available for overlay/original panel
        if (frameImage && blendOnFrame && activePanel !== 'segmentation') {
          context.drawImage(frameImage, 0, 0, width, height);
        } else {
          context.fillStyle = '#0f172a';
          context.fillRect(0, 0, width, height);
        }

        const imageData = context.getImageData(0, 0, width, height);
        const counts = new Array(ELEMENT_CLASSES.length).fill(0);
        const fgAlpha = activePanel === 'segmentation' ? 1.0 : 0.48;

        for (let p = 0; p < maskData.length; p += 4) {
          const classId = maskData[p];
          counts[classId] += 1;

          if (fgAlpha > 0 && classId < ELEMENT_COLOURS.length && classId > 0) {
            const colour = ELEMENT_COLOURS[classId];
            imageData.data[p] = Math.round(imageData.data[p] * (1 - fgAlpha) + colour[0] * fgAlpha);
            imageData.data[p + 1] = Math.round(imageData.data[p + 1] * (1 - fgAlpha) + colour[1] * fgAlpha);
            imageData.data[p + 2] = Math.round(imageData.data[p + 2] * (1 - fgAlpha) + colour[2] * fgAlpha);
            imageData.data[p + 3] = 255;
          }
        }

        if (cancelled) return;
        context.putImageData(imageData, 0, 0);

        const total = counts.reduce((sum, value) => sum + value, 0) || 1;
        setDistribution(
          ELEMENT_CLASSES.map((name, classId) => ({
            name,
            classId,
            count: counts[classId],
            ratio: (counts[classId] / total) * 100,
          }))
        );
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [currentMask, framePath, blendOnFrame, activePanel]);

  // Build a map from class id → color for the distribution bars
  const classColorMap = useMemo(
    () =>
      ELEMENT_COLOURS.reduce(
        (acc, [r, g, b], id) => ({ ...acc, [id]: `rgb(${r},${g},${b})` }),
        {}
      ),
    []
  );

  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-4 shadow-sm ${className}`}>
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-800">Structural element detection</h3>
          <p className="text-xs text-slate-500">
            {job
              ? `job ${String(job.job_id).slice(0, 8)} · ${job.frames_written ?? 0} element masks`
              : 'No structural element job selected'}
          </p>
        </div>
        <label className="flex items-center gap-2 text-[11px] text-slate-600">
          <input
            type="checkbox"
            checked={blendOnFrame}
            onChange={(event) => setBlendOnFrame(event.target.checked)}
            className="h-3.5 w-3.5 rounded border-slate-300"
          />
          Blend over frame
        </label>
      </header>

      {error ? (
        <p className="mb-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          {error}
        </p>
      ) : null}

      {!relativeDir ? (
        <p className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-xs text-slate-500">
          Run structural element detection to populate this viewer.
        </p>
      ) : masks.length === 0 ? (
        <p className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-xs text-slate-500">
          {loading ? 'Loading element masks…' : 'No element mask files found.'}
        </p>
      ) : (
        <div className="space-y-3">
          {/* Panel toggle */}
          <div className="flex gap-1 text-[10px]">
            <button
              onClick={() => setActivePanel('original')}
              className={`px-2 py-1 rounded border ${
                activePanel === 'original' ? 'bg-emerald-100 text-emerald-700 border-emerald-300' : 'border-slate-200 text-slate-500 hover:bg-slate-50'
              }`}
            >
              Original
            </button>
            <button
              onClick={() => setActivePanel('segmentation')}
              className={`px-2 py-1 rounded border ${
                activePanel === 'segmentation' ? 'bg-emerald-100 text-emerald-700 border-emerald-300' : 'border-slate-200 text-slate-500 hover:bg-slate-50'
              }`}
            >
              Segmentation
            </button>
            <button
              onClick={() => setActivePanel('overlay')}
              className={`px-2 py-1 rounded border ${
                activePanel === 'overlay' ? 'bg-emerald-100 text-emerald-700 border-emerald-300' : 'border-slate-200 text-slate-500 hover:bg-slate-50'
              }`}
            >
              Overlay
            </button>
          </div>

          <div className="viewer-container">

            <div className="image-viewport relative w-full aspect-video rounded-lg border border-slate-200 bg-slate-900 overflow-hidden">
              <canvas
                ref={canvasRef}
                className="w-full h-full block"
              />
            </div>

            <div className="frame-navigation-toolbar bg-slate-900 border-t border-slate-200 pt-2">
              <button
                onClick={() => setIndex((value) => Math.max(value - 1, 0))}
                disabled={index === 0}
                className="px-2 py-0.5 rounded border border-white/30 bg-white/10 text-white hover:bg-white/20 disabled:opacity-30"
              >
                ‹
              </button>

              <div className="frame-information flex-1 text-center text-[10px] font-medium text-slate-200">
                {currentMask?.name} ({index + 1}/{masks.length})
              </div>

              <button
                onClick={() => setIndex((value) => Math.min(value + 1, masks.length - 1))}
                disabled={index >= masks.length - 1}
                className="px-2 py-0.5 rounded border border-white/30 bg-white/10 text-white hover:bg-white/20 disabled:opacity-30"
              >
                ›
              </button>
            </div>

          </div>

          {/* Distribution legend */}
          <div className="border border-slate-200 rounded-lg p-2 bg-slate-50">
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Element distribution
            </div>
            {distribution ? (
              <ul className="space-y-1.5 text-[10px]">
                {distribution.map((entry) => (
                  <li key={entry.classId}>
                    <div className="flex items-center justify-between text-slate-600">
                      <span className="flex items-center gap-2">
                        <span
                          className="h-3 w-3 rounded border border-slate-300/50"
                          style={{ backgroundColor: classColorMap[entry.classId] }}
                        />
                        {entry.name}
                      </span>
                      <span className="font-mono text-slate-400">{entry.ratio.toFixed(2)}%</span>
                    </div>
                    <div className="mt-0.5 h-1 w-full overflow-hidden rounded-full bg-slate-200">
                      <div
                        className="h-full"
                        style={{
                          width: `${Math.min(entry.ratio, 100)}%`,
                          backgroundColor: classColorMap[entry.classId],
                        }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-[10px] text-slate-400">
                Breakdown appears once an element mask is rendered.
              </p>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
