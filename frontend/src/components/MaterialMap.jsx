import { useEffect, useMemo, useRef, useState } from 'react';

import api, { artifactRelativePath, artifactUrl } from '../services/api.js';

/**
 * MaterialMap — 46-class DMS46 material segmentation viewer.
 *
 * Masks are stored by the backend as raw class-index PNGs (uint8, values 0-45).
 * This component:
 * 1. Loads the raw categorical mask and the original frame
 * 2. Colorizes the mask using the exact DMS46 46-color palette
 * 3. Renders three panels: original frame, colorized segmentation, overlay
 * 4. Calculates class distribution from the raw class-ID mask (not from RGB)
 *
 * Images are fetched from the sidecar artifact API and read with
 * `crossOrigin="anonymous"` so the canvas stays untainted.
 */

// Must mirror DMS46_CLASS_DISPLAY_NAMES / DMS46_CLASS_COLOURS in
// backend/app/models/material_segmenter.py
export const DMS46_CLASS_DISPLAY_NAMES = [
  'Animal skin',
  'Bone/teeth/horn',
  'Brickwork',
  'Cardboard',
  'Carpet/rug',
  'Ceiling tile',
  'Ceramic',
  'Chalkboard/blackboard',
  'Clutter',
  'Concrete',
  'Cork/corkboard',
  'Engineered stone',
  'Fabric/cloth',
  'Fire',
  'Foliage',
  'Food',
  'Fur',
  'Gemstone/quartz',
  'Glass',
  'Hair',
  'Ice',
  'Leather',
  'Metal',
  'Mirror',
  'Paint/plaster/enamel',
  'Paper',
  'Photograph/painting',
  'Plastic, clear',
  'Plastic, non-clear',
  'Rubber/latex',
  'Sand',
  'Skin/lips',
  'Sky',
  'Snow',
  'Soil/mud',
  'Stone, natural',
  'Stone, polished',
  'Tile',
  'Wallpaper',
  'Water',
  'Wax',
  'Whiteboard',
  'Wicker',
  'Wood',
  'Wood, tree',
  'Asphalt',
];

// Exact HEX colors matching the user's requested palette (0-45).
// Format: (R, G, B) as uint8 values.
export const DMS46_CLASS_COLOURS = [
  [188, 188, 137],  // 0 Animal skin
  [0, 188, 0],      // 1 Bone/teeth/horn
  [188, 188, 0],    // 2 Brickwork
  [0, 0, 188],      // 3 Cardboard
  [188, 0, 188],    // 4 Carpet/rug
  [0, 188, 188],    // 5 Ceiling tile
  [241, 241, 241],  // 6 Ceramic
  [0, 137, 137],    // 7 Chalkboard/blackboard
  [225, 0, 0],      // 8 Clutter
  [137, 188, 0],    // 9 Concrete
  [225, 188, 0],    // 10 Cork/corkboard
  [137, 0, 188],    // 11 Engineered stone
  [137, 188, 188],  // 12 Fabric/cloth
  [225, 188, 188],  // 13 Fire
  [0, 137, 0],      // 14 Foliage
  [188, 137, 0],    // 15 Food
  [137, 225, 188],  // 16 Fur
  [188, 137, 188],  // 17 Gemstone/quartz
  [0, 137, 188],    // 18 Glass
  [188, 225, 0],    // 19 Hair
  [188, 225, 188],  // 20 Ice
  [137, 137, 0],    // 21 Leather
  [137, 225, 0],    // 22 Metal
  [225, 225, 0],    // 23 Mirror
  [225, 137, 188],  // 24 Paint/plaster/enamel
  [0, 225, 0],      // 25 Paper
  [0, 0, 137],      // 26 Photograph/painting
  [188, 0, 0],      // 27 Plastic, clear
  [0, 188, 137],    // 28 Plastic, non-clear
  [188, 0, 137],    // 29 Rubber/latex
  [0, 0, 225],      // 30 Sand
  [225, 188, 137],  // 31 Skin/lips
  [0, 188, 225],    // 32 Sky
  [188, 188, 225],  // 33 Snow
  [225, 0, 137],    // 34 Soil/mud
  [225, 225, 188],  // 35 Stone, natural
  [137, 0, 225],    // 36 Stone, polished
  [137, 188, 225],  // 37 Tile
  [225, 188, 225],  // 38 Wallpaper
  [0, 137, 225],    // 39 Water
  [188, 137, 137],  // 40 Wax
  [188, 188, 188],  // 41 Whiteboard
  [188, 225, 137],  // 42 Wicker
  [137, 0, 0],      // 43 Wood
  [188, 137, 225],  // 44 Wood, tree
  [137, 137, 137],  // 45 Asphalt
];

const DMS46_CLASS_HEX = DMS46_CLASS_COLOURS.map(
  ([r, g, b]) => `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
);

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
  const [imageError, setImageError] = useState(null);
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
        // Load the raw categorical mask (class IDs 0-45 as uint8 PNG)
        const maskImage = await loadImage(
          artifactUrl('material_masks', currentMask.relative_path)
        );
        // Always load the original frame for alignment with the mask
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

        // Determine active panel
        const showOverlay = activePanel === 'overlay' && frameImage && blendOnFrame;
        const showSegmentation = activePanel === 'segmentation';
        const showOriginal = activePanel === 'original';

        // Draw background: original frame if available, else dark
        if (frameImage && (showOverlay || showOriginal)) {
          context.drawImage(frameImage, 0, 0, width, height);
        } else {
          context.fillStyle = '#0f172a';
          context.fillRect(0, 0, width, height);
        }

        // If showing overlay or segmentation-only, colorize the class-index mask
        const imageData = context.getImageData(0, 0, width, height);
        const counts = new Array(DMS46_CLASS_DISPLAY_NAMES.length).fill(0);
        let fgAlpha = activePanel === 'segmentation' ? 1.0 : 0.48;
        if (showOriginal) fgAlpha = 0; // no color overlay for original panel

        for (let p = 0; p < maskData.length; p += 4) {
          // Read the raw class ID from the red channel of the mask PNG
          const classId = maskData[p];
          counts[classId] += 1;

          if (fgAlpha > 0 && classId < DMS46_CLASS_COLOURS.length) {
            const colour = DMS46_CLASS_COLOURS[classId];
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
          DMS46_CLASS_DISPLAY_NAMES.map((name, classId) => ({
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
      DMS46_CLASS_COLOURS.reduce(
        (acc, [r, g, b], id) => ({ ...acc, [id]: `rgb(${r},${g},${b})` }),
        {}
      ),
    []
  );

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
        <div className="space-y-3">
          {/* Panel toggle */}
          <div className="flex gap-1 text-[10px]">
            <button
              onClick={() => setActivePanel('original')}
              className={`px-2 py-1 rounded border ${
                activePanel === 'original' ? 'bg-indigo-100 text-indigo-700 border-indigo-300' : 'border-slate-200 text-slate-500 hover:bg-slate-50'
              }`}
            >
              Original
            </button>
            <button
              onClick={() => setActivePanel('segmentation')}
              className={`px-2 py-1 rounded border ${
                activePanel === 'segmentation' ? 'bg-indigo-100 text-indigo-700 border-indigo-300' : 'border-slate-200 text-slate-500 hover:bg-slate-50'
              }`}
            >
              Segmentation
            </button>
            <button
              onClick={() => setActivePanel('overlay')}
              className={`px-2 py-1 rounded border ${
                activePanel === 'overlay' ? 'bg-indigo-100 text-indigo-700 border-indigo-300' : 'border-slate-200 text-slate-500 hover:bg-slate-50'
              }`}
            >
              Overlay
            </button>
          </div>

          {/* Image viewport */}
          <div className="viewer-container">

            <div className="image-viewport relative w-full aspect-video rounded-lg border border-slate-200 bg-slate-900 overflow-hidden">
              <canvas
                ref={canvasRef}
                className="w-full h-full block"
              />
            </div>

            <div className="frame-navigation-toolbar bg-slate-900 border-t border-slate-200 pt-2">
              <button
                type="button"
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
                type="button"
                onClick={() => setIndex((value) => Math.min(value + 1, masks.length - 1))}
                disabled={index >= masks.length - 1}
                className="px-2 py-0.5 rounded border border-white/30 bg-white/10 text-white hover:bg-white/20 disabled:opacity-30"
              >
                ›
              </button>
            </div>

          </div>

          {/* Distribution legend - scrollable */}
          <div className="max-h-[400px] overflow-y-auto space-y-2 border border-slate-200 rounded-lg p-2 bg-slate-50">
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Material distribution (46 classes)
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
                <li className="pt-2 border-t border-slate-200 text-[10px] text-slate-400">
                  Total: {distribution.reduce((sum, e) => sum + e.count, 0)} px
                </li>
              </ul>
            ) : (
              <p className="text-[10px] text-slate-400">
                Distribution appears once a mask is rendered.
              </p>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
