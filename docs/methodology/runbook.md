# AeroTwin AI — Build, Release & Evaluation Runbook (Methodology)

**Scope.** This document covers exactly two things: (1) the **release build pipeline**
(PyInstaller sidecar → Vite bundle → electron-builder → desktop app) and (2) the
**automated system-evaluation telemetry** that feeds the thesis results chapter.
Development mode, the test suite, and troubleshooting live in the root
[`HowToRun.md`](../../HowToRun.md).

Reference commit for every number quoted below: `96625df` (*Phase 7*), Windows 10
`10.0.26200`, CPU-only (`device: "cpu"`), Python 3.11.9, torch 2.14.0,
opencv-python 5.0.0.93, PyInstaller 6.22.3.

---

## 1. Reproducibility contract

Domain rule 3 ("reproducibility, metrics & determinism") is enforced by construction, not by
convention. Before quoting any result, confirm all five properties:

| Property | How it is guaranteed |
|---|---|
| **Fixed seeds** | `tests/evaluate_system.py` seeds `random`, `numpy.random`, and `torch.manual_seed(--seed)` (default **42**); services read `settings.random_seed` (`AEROTWIN_RANDOM_SEED`, default 42). |
| **Input isolation** | The evaluator sets `AEROTWIN_DATA_DIR` to a fresh temp directory, so the repository's `data/` is never read or written. `--data-dir PATH` (or a pre-set `AEROTWIN_DATA_DIR`) overrides the location. |
| **Hermetic weights** | Real `.pt` files are used when all three are present; otherwise deterministic random-init fixtures are generated into a temp dir exposed via `AEROTWIN_WEIGHTS_DIR`, and the run is flagged `weights.source: "synthetic_random_init"`. |
| **Provenance** | The telemetry JSON records the source clip probe, per-weight file size + SHA-256, every package version (incl. `pyinstaller`), platform, Python version, and inference device. |
| **Audit trail** | Aggregation persists per-frame observations (`include_observations`, default `true`); every `processing_jobs` row carries `processing_time_seconds` and `metrics_json`. |

Record this block beside every table/figure that reaches the thesis:

```powershell
git rev-parse HEAD          # commit under test
cd backend
python tests/evaluate_system.py --seed 42     # seeded, isolated, exit 0 = all 9 stages completed
```

---

## 2. Release build pipeline

```powershell
# 0. One-time tooling: PyInstaller 6.22.3 (pinned) must exist in the ACTIVE venv
pip install -r backend\requirements.txt

# 1. Freeze the FastAPI backend into a standalone sidecar (onedir)
npm run build:backend
#    -> backend\dist\aerotwin_backend\aerotwin_backend.exe + _internal\  (~580 MB, torch-dominated)

# 2. Vite production bundle (base: './' so file:// loading works inside Electron)
npm run build:frontend
#    -> frontend\dist\index.html + hashed CSS/JS

# 3. Unpacked, immediately launchable desktop app
npm run pack
#    -> assets\installers\win-unpacked\AeroTwin AI.exe

# 4. Installer for the current OS (NSIS .exe / dmg / AppImage)
npm run dist:win
#    -> assets\installers\AeroTwin AI-Setup-1.0.0.exe
```

`npm run build:all` runs steps 1–2 (`build:frontend` **then** `build:backend`); `npm run pack`
and every `dist:*` script already invoke it, so a bare `npm run pack` rebuilds both artifacts.

**Ordering rule (weights).** Drop real weights into `backend\app\models\weights\`
**before** `build:backend`: the spec collects that directory into the bundle at the same
relative path as in the repository, so they end up in
`assets\installers\win-unpacked\resources\backend\_internal\app\models\weights\`. Adding weights
after a build has no effect until the sidecar is re-frozen and re-packed.

### Artifact inventory

| Artifact | Path | Produced by | Consumed by |
|---|---|---|---|
| Frozen sidecar | `backend\dist\aerotwin_backend\aerotwin_backend.exe` + `_internal\` | `npm run build:backend` | `extraResources` copy |
| Renderer bundle | `frontend\dist\` (`index.html`, `assets/*`) | `npm run build:frontend` | `app.asar` (`files`) |
| Unpacked app | `assets\installers\win-unpacked\AeroTwin AI.exe` | `npm run pack` | local validation / manual testing |
| Installer | `assets\installers\AeroTwin AI-Setup-1.0.0.exe` | `npm run dist:win` | distribution |
| Release metadata | `assets\installers\builder-debug.yml` | any `electron-builder` run | build troubleshooting |

### Frozen-path contract (why production "just works")

| Concern | Development | Frozen / packaged |
|---|---|---|
| Backend entry | `backend\run_backend.py` (uvicorn `reload=True`) | `backend\run_frozen.py` (ASGI object, **no** reload) |
| Startup command | `python run_backend.py` (spawned with `shell: true`) | `resources\backend\aerotwin_backend.exe` (no shell) |
| Weights root | `backend\app\models\weights\` | `sys._MEIPASS` → `_internal\app\models\weights\` |
| Data root | repository `data\` | `%APPDATA%\AeroTwinAI\data` (writable, per-user) |
| Console | visible | `console=False`; `run_frozen.py` guards `sys.stdout`/`sys.stderr` |
| Overrides | `AEROTWIN_*` env vars | `AEROTWIN_*` env vars (still win) |

Versioning: `productName`/`version` come from the root `package.json`
(`AeroTwin AI` / `1.0.0`), and the NSIS artifact name is
`${productName}-Setup-${version}.${ext}`. Bump `version` before producing a release installer.

---

## 3. Weight provenance — what the numbers mean

The evaluator decides once, at import time, and records the decision in the telemetry:

| Condition | `weights.source` | Claims the run supports |
|---|---|---|
| **All three** of `material_model.pt`, `element_model.pt`, `crack_model.pt` exist in `backend\app\models\weights\` | `"real"` — per-file `size_bytes` + `sha256` recorded | Inference latency **and** task-level accuracy (IoU / F1 / score calibration) |
| **Any** file missing | `"synthetic_random_init"` — deterministic seed-derived fixtures in a temp dir, exposed through `AEROTWIN_WEIGHTS_DIR` | Inference latency, dedup mechanics, scoring machinery, report generation — but **not** accuracy or score calibration |

When the run is synthetic, the generated `system_evaluation.md` ends with this disclaimer (quote it
verbatim in the thesis rather than paraphrasing):

> Model weights were NOT present in `backend/app/models/weights/`; deterministic random-init
> fixtures were used. Inference runtimes, dedup ratios, and scoring mechanics are valid, but
> task-level accuracy (IoU/F1/score calibration) would NOT be.

The three consumers, for reference: `material_model.pt` → `POST /api/processing/segment-materials`,
`element_model.pt` → `POST /api/processing/detect-elements`, `crack_model.pt` →
`POST /api/processing/map-cracks`. Each endpoint returns a descriptive **HTTP 503** while its file is
absent — the documented fallback, also visible in the packaged app.

---

## 4. Running the evaluation

```powershell
cd backend
python tests/evaluate_system.py                            # seeded 42, isolated, cleans up after itself
python tests/evaluate_system.py --keep                     # keep temp data + weights dirs for inspection
python tests/evaluate_system.py --seed 7                   # alternate seed (re-run to confirm stability)
python tests/evaluate_system.py --data-dir ..\tmp_eval     # explicit data dir (you own cleanup)
```

| Flag | Default | Effect |
|---|---|---|
| `--seed N` | `42` | Seeds `random`, NumPy, and PyTorch; forwarded to the app through `settings.random_seed`. |
| `--data-dir PATH` | fresh OS temp dir | Runs the pipeline in *PATH*; the script no longer owns the directory, so nothing is deleted. |
| `--keep` | off | Keeps the temp data/weights directories after a successful run and prints their location. |

**Environment hygiene.** Run it from a clean shell. If `AEROTWIN_DATA_DIR` or
`AEROTWIN_WEIGHTS_DIR` is already exported, the script honours them, treats the data dir as
pre-existing (no cleanup, `--keep` unnecessary) and — because `AEROTWIN_WEIGHTS_DIR` wins over the
"are real weights missing?" probe — may silently evaluate against fixtures left behind by an earlier
test run.

**Exit codes.** `0` = all nine stages reached `completed`; `1` = a stage failed. Job runners never
raise (errors land on the `processing_jobs` row), so the evaluator converts a failed row into a hard
script failure — telemetry is still written for failed runs, with `outcome: "failed"` plus the error.

**Expected console output** (≈2–3 s total on CPU with synthetic weights):

```
[eval] AeroTwin system evaluation | seed=42 | data=C:\Users\...\Temp\aerotwin_eval_<stamp>
[eval] real weights absent → synthetic fixtures in C:\Users\...\Temp\aerotwin_eval_weights_xxxx (source=synthetic_random_init)
[eval] video_import: Synthesize UAV clip and ingest into data/inputs/raw_videos/
[eval] frame_extraction: Extract frames from the ingested UAV video
[eval] weights_calibration: Score a deterministic head-bias grid on decoded frames (real detectors)
[eval] material_segmentation: Tiled MiniUNet inference: masks per frame
[eval] element_detection: Detect structural elements (column/beam/wall)
[eval] crack_mapping: Detect + skeletonize cracks; log frame defect metrics
[eval] cross_frame_aggregation: Deduplicate frame-local detections into the global defect registry
[eval] seismic_assessment: Deterministic weighted seismic vulnerability score
[eval] report_generation: Compile JSON + PDF inspection report
[eval] telemetry: C:\...\AeroTwin\docs\sample_outputs\system_evaluation.json
[eval] summary:   C:\...\AeroTwin\docs\sample_outputs\system_evaluation.md
[eval] outcome=completed | stages=9 | total=3.0s | score=55.13
```

**Outputs.** `docs\sample_outputs\system_evaluation.json` (machine-readable telemetry) and
`docs\sample_outputs\system_evaluation.md` (chapter-ready summary). Both carry a UTC timestamp —
commit them alongside the thesis results so every figure has a diffable source.

---

## 5. Telemetry schema — `aerotwin.system_evaluation.v1`

```jsonc
{
  "schema": "aerotwin.system_evaluation.v1",
  "outcome": "completed",                        // or "failed"
  "error": null,
  "generated_at": "2026-09-29T12:17:19.728811+00:00",
  "started_at":   "2026-09-29T12:17:13.782295+00:00",
  "run_stamp": "20260929T121713Z",
  "seed": 42,
  "environment": { "platform": "Windows-10-10.0.26200-SP0", "python": "3.11.9",
                   "machine": "AMD64", "device": "cpu",
                   "packages": { "torch": "2.14.0", "opencv-python": "5.0.0.93",
                                 "numpy": "2.4.6", "fastapi": "0.141.1", "uvicorn": "0.54.0",
                                 "pydantic": "2.13.5", "reportlab": "5.0.1",
                                 "pytest": "9.1.1", "pyinstaller": "6.22.3" } },
  "weights": { "source": "synthetic_random_init", "directory": "...",
               "files": { "material_model.pt": { "size_bytes": 463701, "sha256": "72ba..." },
                          "element_model.pt":  { "size_bytes": 463501, "sha256": "d195..." },
                          "crack_model.pt":    { "size_bytes": 463229, "sha256": "1a4f..." } } },
  "pipeline": {
    "stages": [ { "stage": "frame_extraction", "description": "...", "status": "completed",
                  "wall_time_seconds": 0.092, "job_id": "...", "job_status": "completed",
                  "processing_time_seconds": 0.022, "frames_written": 8,
                  "output_dir": "...", "metrics": { } } ],
    "stage_wall_time_total_seconds": 2.2152,
    "outcomes": { "video_import": {}, "aggregation": {}, "assessment": {}, "report": {} },
    "aggregation": { "raw_crack_observations": 8, "unique_crack_defects": 1,
                     "crack_dedup_ratio": 0.875, "crack_dedup_reduction": 7,
                     "mean_observations_per_crack": 8.0,
                     "raw_element_observations": 16, "unique_element_instances": 2,
                     "frames_analyzed": 8, "iou_threshold": 0.3, "max_frame_gap": 10 },
    "assessment": { "vulnerability_score": 55.13, "risk_level": "Substantial",
                    "factors": [ { "name": "crack_intensity", "value": 0.6,
                                   "weight": 0.3, "contribution": 0.18 } ] },
    "report": { "json_path": "...", "pdf_path": "...", "size_bytes": 23979 }
  },
  "overall_wall_time_seconds": 3.02,
  "notes": [ "...seeded run...", "...isolated data dir...", "...weights disclaimer when synthetic..." ]
}
```

Field semantics worth knowing before charting:

- `wall_time_seconds` = evaluator-side stage latency (includes HTTP/background-task overhead);
  `processing_time_seconds` = the backend's own timer from the `processing_jobs` row. Report both —
  the gap *is* the orchestration overhead.
- `metrics.mean_frame_inference_seconds` (material / element / crack stages) is the per-frame figure;
  use it instead of stage totals whenever frame counts differ between runs.
- `crack_dedup_ratio` = `raw_crack_observations → unique_crack_defects` reduction under greedy
  bbox-IoU ≥ `iou_threshold` within `max_frame_gap` frames; `include_observations` keeps the
  per-frame provenance for auditing.
- `assessment.factors[]` is exactly the weighted sum behind `vulnerability_score`
  (`weight × value = contribution`), so every score is arithmetically checkable.

---

## 6. Mapping telemetry → thesis tables

| Chapter artefact | Telemetry path | Notes |
|---|---|---|
| Per-stage runtime table | `pipeline.stages[].wall_time_seconds` + `.processing_time_seconds` | Add `stage_wall_time_total_seconds` as the footer row. |
| Per-frame inference latency | `pipeline.stages[].metrics.mean_frame_inference_seconds` | Convert to ms/frame; state the device (`environment.device`). |
| Cross-frame dedup figure | `pipeline.aggregation.*` | Report raw → unique plus `crack_dedup_ratio`. |
| Vulnerability score & factor decomposition | `pipeline.assessment.*` | Table of factor / value / weight / contribution. |
| Report artefact sizes | `pipeline.report.*` | `size_bytes` of JSON + PDF. |
| Reproducibility appendix | `environment.*`, `weights.*`, `seed`, `run_stamp`, `git rev-parse HEAD` | Enables a reader to rebuild the exact environment. |

**Ground-truth metrics (IoU / F1).** The evaluator measures *system* behaviour (latency, dedup,
scoring, artifacts) on synthetic media; it does **not** compute segmentation IoU or crack-detection
F1, because no annotated ground-truth set is shipped with the repository. Those figures require a
labelled evaluation set fed through the same endpoints, compared offline with the recorded per-frame
masks in `data\processed\material_masks\` / `crack_maps\` (each result directory carries a
`manifest.json` + `metrics.json`). Do not substitute the synthetic-fixture run for those numbers.

---

## 7. Release checklist

1. `git rev-parse HEAD` → record the commit.
2. Real weights in `backend\app\models\weights\`? → determines whether accuracy claims are allowed
   ([§3](#3-weight-provenance--what-the-numbers-mean)).
3. `npm run build:frontend` clean; `npm run build:backend` clean (PyInstaller installed).
4. Bump `version` in the root `package.json` before a distribution build.
5. `npm run pack` and launch `assets\installers\win-unpacked\AeroTwin AI.exe`; confirm
   `/api/health` → 200 and that `%APPDATA%\AeroTwinAI\data\aerotwin.db` (WAL) is created.
6. `npm run dist:win` → `assets\installers\AeroTwin AI-Setup-1.0.0.exe`.
7. `cd backend; python -m pytest` → 103 passed; `python tests/evaluate_system.py` → exit 0.
8. Archive `docs\sample_outputs\system_evaluation.{json,md}` with the release.


