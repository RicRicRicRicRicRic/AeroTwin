import { useEffect, useMemo, useRef, useState } from 'react';

import api, { artifactRelativePath, artifactUrl } from '../services/api.js';

/**
 * MaterialMap — colourised material segmentation viewer.
 *
 * Masks are stored by the backend as class-index PNGs; this component recolours
 * them on a canvas (optionally blended over the source frame) and reports the
 * class distribution of the displayed frame. Images are fetched from the
 * sidecar artifact API and read with `crossOrigin="anonymous"` so the canvas
 * stays untainted.
 */

// Must mirror MATERIAL_CLASSES in backend/app/models/material_segmenter.py
export const MATERIAL_CLASSES = [
  'background',
  'concrete',
  'metal',
  'glass',
  'brick',
  'roofing',
];

const CLASS_COLOURS = [
  [15, 23, 42],
  [148, 163, 184],
  [56, 189, 248],
  [167, 139, 250],
  [245, 158, 11],
  [16, 185, 129],
];

const CLASS_HEX = ['#0f172a', '#94a3b8', '#38bdf8', '#a78bfa', '#f59e0b', '#10b981'];

export function loadImage(url, crossOrigin = 'anonymous') {
  return new Promise((resolve, reject) => {
    const image = new Image();
    if (crossOrigin) image.crossOrigin = crossOrigin;
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load image: ${url}`));
    image.src = url;
  });
}

export default function MaterialMap({ framesRun = '', job = null, className = '' }) {
  const relativeDir = useMemo(
    () => artifactRelativePath(job?.output_dir, 'material_masks'),
    [job?.output_dir]
  );
  const [masks, setMasks] = useState([]);
  const [index, setIndex] = useState(0);
  const [blendOnFrame, setBlendOnFrame] = useState(true);
  const [distribution, setDistribution] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
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
      .listArtifacts('material_masks', relativeDir)
      .then((listing) => {
        if (cancelled) return;
        setMasks(listing.items.filter((item) => !item.is_dir && item.name.startsWith('mask_')));
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
    if (!currentMask) return () => {
      cancelled = true;
    };
    setError(null);
    (async () => {
      try {
        const maskImage = await loadImage(
          artifactUrl('material_masks', currentMask.relative_path)
        );
        let frameImage = null;
        if (blendOnFrame && framePath) {
          try {
            frameImage = await loadImage(artifactUrl('frames', framePath));
          } catch {
            frameImage = null;
          }
        }
        if (cancelled) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        const width = maskImage.naturalWidth;
        const height = maskImage.naturalHeight;
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d');
        if (frameImage) {
          context.drawImage(frameImage, 0, 0, width, height);
        } else {
          context.fillStyle = '#0f172a';
          context.fillRect(0, 0, width, height);
        }

        const scratch = document.createElement('canvas');
        scratch.width = width;
        scratch.height = height;
        const scratchContext = scratch.getContext('2d');
        scratchContext.drawImage(maskImage, 0, 0);
        const maskData = scratchContext.getImageData(0, 0, width, height).data;

        const imageData = context.getImageData(0, 0, width, height);
        const alpha = frameImage && blendOnFrame ? 0.45 : 1;
        const counts = new Array(MATERIAL_CLASSES.length).fill(0);
        for (let p = 0; p < maskData.length; p += 4) {
          const classIndex = Math.min(maskData[p], MATERIAL_CLASSES.length - 1);
          counts[classIndex] += 1;
          const colour = CLASS_COLOURS[classIndex];
          imageData.data[p] = Math.round(imageData.data[p] * (1 - alpha) + colour[0] * alpha);
          imageData.data[p + 1] = Math.round(imageData.data[p + 1] * (1 - alpha) + colour[1] * alpha);
          imageData.data[p + 2] = Math.round(imageData.data[p + 2] * (1 - alpha) + colour[2] * alpha);
          imageData.data[p + 3] = 255;
        }
        if (cancelled) return;
        context.putImageData(imageData, 0, 0);

        const total = counts.reduce((sum, value) => sum + value, 0) || 1;
        setDistribution(
          counts.map((count, classIndex) => ({
            name: MATERIAL_CLASSES[classIndex],
            count,
            ratio: (count / total) * 100,
          }))
        );
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [currentMask, framePath, blendOnFrame]);

  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-4 shadow-sm ${className}`}>
      <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-800">Material segmentation</h3>
          <p className="text-xs text-slate-500">
            {job
              ? `job ${String(job.job_id).slice(0, 8)} · ${job.frames_written ?? 0} masks`
              : 'No segmentation job selected'}
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
          Run material segmentation to populate this viewer.
        </p>
      ) : masks.length === 0 ? (
        <p className="rounded-lg border border-dashed border-slate-300 p-6 text-center text-xs text-slate-500">
          {loading ? 'Loading masks…' : 'No mask files found in this job output.'}
        </p>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
          <div>
            <canvas
              ref={canvasRef}
              className="w-full rounded-lg border border-slate-200 bg-slate-900"
            />
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
                {currentMask?.name} ({index + 1}/{masks.length})
              </span>
              <button
                type="button"
                onClick={() => setIndex((value) => Math.min(value + 1, masks.length - 1))}
                disabled={index >= masks.length - 1}
                className="rounded border border-slate-300 px-2 py-1 hover:bg-slate-50 disabled:opacity-40"
              >
                Next ›
              </button>
            </div>
          </div>

          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Class distribution
            </div>
            {distribution ? (
              <ul className="space-y-1.5 text-[11px]">
                {distribution.map((entry, classIndex) => (
                  <li key={entry.name}>
                    <div className="flex items-center justify-between text-slate-600">
                      <span className="flex items-center gap-2">
                        <span
                          className="h-3 w-3 rounded"
                          style={{ backgroundColor: CLASS_HEX[classIndex] }}
                        />
                        {entry.name}
                      </span>
                      <span className="font-mono">{entry.ratio.toFixed(2)}%</span>
                    </div>
                    <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-slate-100">
                      <div
                        className="h-full"
                        style={{
                          width: `${entry.ratio}%`,
                          backgroundColor: CLASS_HEX[classIndex],
                        }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-[11px] text-slate-400">
                Distribution appears once a mask is rendered.
              </p>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

