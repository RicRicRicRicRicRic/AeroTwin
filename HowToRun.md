# AeroTwin AI — Runbook

How to **execute, test, and ship** AeroTwin AI: a UAV-imagery structural health monitoring
framework (Electron desktop wrapper + React/Vite renderer + FastAPI/PyTorch sidecar).

Verified on this machine (2026-09-29, repository commit `96625df` *Phase 7*):
Node **v26.2.0**, npm **11.0.0**, Python **3.11.9** in `backend\.venv`, torch **2.14.0**.

| § | Path |
|---|---|
| [0](#0-architecture--data-paths) | Architecture & data paths |
| [1](#1-prerequisites--one-time-setup) | Prerequisites & one-time setup (incl. weights, env vars) |
| [2](#2-path-a--development-mode-in-the-browser) | **A** — Browser development (FastAPI + Vite) |
| [3](#3-path-b--electron-desktop-app-development) | **B** — Electron desktop app (development) |
| [4](#4-testing-suite-103-tests) | **Testing** — `pytest` (103 tests) |
| [5](#5-system-evaluation-script-thesis-telemetry) | **Evaluation** — `tests/evaluate_system.py` |
| [6](#6-production-desktop-app-sidecar--electron) | **Production** — sidecar + Electron package + executable |
| [7](#7-verification-checklist) | Verification checklist |
| [8](#8-troubleshooting) | Troubleshooting |
| [9](#9-corrections-to-earlier-instructions) | Corrections to earlier instructions |

---

## 0. Architecture & data paths

**Development** (source tree, two processes plus the renderer):

```
┌────────────┐   http://localhost:5173   ┌──────────────┐   http://127.0.0.1:8000  ┌──────────────────┐
│  Electron  │ ────────────────────────► │ Vite (React) │ ───────────────────────► │ FastAPI sidecar  │
│  (dev win) │                           │  frontend/   │                          │ python run_…py   │
└────────────┘                           └──────────────┘                          └────────┬─────────┘
                                                                                           │
                                                          repo `data/` (db + images) ◄──────┘
```

**Packaged** (installer / unpacked app, one user-facing process):

```
AeroTwin AI.exe ──spawns──► resources\backend\aerotwin_backend.exe   (frozen, no Python install needed)
        │                              │
        └── loads frontend\dist        └── data → %APPDATA%\AeroTwinAI\data
            from app.asar                  weights → resources\backend\_internal\app\models\weights
```

| Thing | Development | Packaged (`npm run pack`) |
|---|---|---|
| Data dir | `data\` in the repo | `%APPDATA%\AeroTwinAI\data` |
| SQLite | `data\aerotwin.db` (WAL) | `%APPDATA%\AeroTwinAI\data\aerotwin.db` |
| Weights | `backend\app\models\weights\` | `resources\backend\_internal\app\models\weights\` |
| Backend launch | `python run_backend.py` (uvicorn auto-reload) | `aerotwin_backend.exe` (frozen, no reload) |
| Frontend | Vite dev server :5173 | static bundle inside `app.asar` |
| Overrides | `AEROTWIN_*` env vars | `AEROTWIN_*` env vars (same behaviour) |

**"I want to …" → run this:**

| Goal | Command(s) | Section |
|---|---|---|
| Use the UI in a browser | `cd backend; python run_backend.py` **+** `cd frontend; npm run dev` | [2](#2-path-a--development-mode-in-the-browser) |
| Use the desktop window with hot-reload | `cd frontend; npm run dev` **+** `npm start` | [3](#3-path-b--electron-desktop-app-development) |
| Verify nothing regressed | `cd backend; python -m pytest` | [4](#4-testing-suite-103-tests) |
| Produce thesis telemetry | `cd backend; python tests/evaluate_system.py` | [5](#5-system-evaluation-script-thesis-telemetry) |
| Build the shippable desktop app | `npm run pack` → launch `assets\installers\win-unpacked\AeroTwin AI.exe` | [6](#6-production-desktop-app-sidecar--electron) |
| Ship an installer | `npm run dist:win` | [6](#6-production-desktop-app-sidecar--electron) |

---

## 1. Prerequisites & one-time setup

### 1.1 Tooling

| Tool | Verified version | Notes |
|---|---|---|
| Node.js | v26.2.0 (≥ 20 required by Vite 7) | `node --version` |
| npm | 11.0.0 | installs root + `frontend/` deps |
| Python | 3.11.9 — `backend\.venv\Scripts\python.exe` | **all** Python commands below assume this venv |
| PyInstaller | 6.22.3 (pinned; **not yet installed** here) | needed only for [§6](#6-production-desktop-app-sidecar--electron) |
| Disk | ~1.5 GB free to build | onedir bundle ≈ 580 MB + NSIS installer ≈ 850 MB |

### 1.2 Activate the virtual environment (do this first, in every new terminal)

```powershell
cd C:\Users\ricmi\Documents\GitHub\AeroTwin
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # only if Activate.ps1 is blocked
.\backend\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
# expected: C:\Users\ricmi\Documents\GitHub\AeroTwin\backend\.venv\Scripts\python.exe
```

> If the printed path is *not* inside `backend\.venv`, nothing about the backend will work as
> documented (missing pytest/torch, or Electron spawning the wrong interpreter — see
> [§3](#3-path-b--electron-desktop-app-development)). VS Code activates this venv automatically
> (`.vscode\settings.json` → `python.terminal.activateEnvironment: true`).

### 1.3 Install dependencies (once per machine / after pulling)

```powershell
pip install -r backend\requirements.txt    # backend runtime + tests + PyInstaller
npm install                                # root deps; "postinstall" also runs `cd frontend && npm install`
```

Pinned by `backend\requirements.txt`: `fastapi 0.141.1`, `uvicorn[standard] 0.54.0`,
`pydantic 2.13.5`, `opencv-python 5.0.0.93`, `numpy 2.4.6`, `torch 2.14.0`, `reportlab 5.0.1`,
`pillow 12.3.0`, `pytest 9.1.1`, `httpx 0.28.1`, `pyinstaller 6.22.3`.

- The PyPI Windows torch wheel is **CPU-only**; the code auto-detects the device. Install the CUDA
  build from `https://download.pytorch.org/whl/cu126` only if you have the disk/GPU for it.
- Without PyInstaller, `npm run build:backend` fails immediately with
  `pyinstaller : The term 'pyinstaller' is not recognized...` — that is the expected state on a
  fresh clone, not a broken repo.

### 1.4 Model weights — placement expectations

Drop the trained checkpoints into `backend\app\models\weights\` (create nothing else; the
directory is tracked via `.gitkeep` and its contents are git-ignored):

| File | Consumed by | Endpoint | If the file is missing |
|---|---|---|---|
| `material_model.pt` | `app\models\material_segmenter.py` | `POST /api/processing/segment-materials` | **503** — `Model weights 'material_model.pt' not found in <dir>...` |
| `element_model.pt` | `app\models\structural_element_detector.py` | `POST /api/processing/detect-elements` | **503** (element detection consumes material masks) |
| `crack_model.pt` | `app\models\crack_detector.py` | `POST /api/processing/map-cracks` | **503** |

```powershell
Get-ChildItem backend\app\models\weights   # expected today: only .gitkeep → all three endpoints answer 503
```

- Checkpoints may be a plain state dict or a dict containing `state_dict`; a shape/architecture
  mismatch is reported as a 503-mapped `ModelLoadError` rather than a traceback.
- Override the directory with `AEROTWIN_WEIGHTS_DIR` (used by the tests and the evaluation script).
- **Everything except the three AI endpoints works without weights**: video ingest, frame
  extraction, cross-frame aggregation, seismic assessment, and JSON/PDF report generation.
- The packaged app resolves weights from
  `resources\backend\_internal\app\models\weights\` — copy the `.pt` files there **before**
  `npm run build:backend` if you want them inside the installer ([§6](#6-production-desktop-app-sidecar--electron)).

### 1.5 Environment variables

All backend tunables are read once at import (`app\core\config.py`), so set them **before** starting
a process:

| Variable | Default | Purpose |
|---|---|---|
| `AEROTWIN_DATA_DIR` | repo `data\` (dev) / `%APPDATA%\AeroTwinAI\data` (frozen) | Root for inputs, processed artifacts, outputs, and `aerotwin.db`. |
| `AEROTWIN_WEIGHTS_DIR` | `backend\app\models\weights\` | Model checkpoint directory (tests and the evaluator point this at temp dirs). |
| `AEROTWIN_HOST` | `127.0.0.1` | Sidecar bind address (Electron sets it when spawning the sidecar). |
| `AEROTWIN_PORT` | `8000` | Sidecar port; also drive it from `AEROTWIN_BACKEND_URL` for Electron. |
| `AEROTWIN_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173,null` | Allowed browser origins — the trailing `null` is what lets the packaged `file://` window call the API. |
| `AEROTWIN_RANDOM_SEED` | `42` | Seed used by services for reproducible inference/scoring. |
| `AEROTWIN_BACKEND_URL` (Electron) | `http://127.0.0.1:8000` | URL Electron probes (`/api/health`) and passes to the spawned sidecar's port. |
| `VITE_DEV_SERVER_URL` (Electron) | `http://localhost:5173` | Where the *unpackaged* window points. |

### 1.6 Fixing a broken Electron binary install

`npm start` can fail instantly, before any window appears:

```
node_modules\electron\index.js:17
    throw new Error('Electron failed to install correctly, please delete node_modules/electron and try installing again');
```

**Cause.** Electron's `postinstall` (`node_modules\electron\install.js`) downloads
`electron-v<version>-win32-x64.zip` and unzips it with `extract-zip` → `yauzl` (both last published ~2020).
On Node 26 that unzip **never resolves and never rejects**: `install.js` exits `0` having written nothing, so
`node_modules\electron\dist\` is left truncated (hundreds of KB instead of ~286 MB) and `path.txt` is never
created. `index.js` reads the executable *name* out of `path.txt` and throws when it is absent. A plain
`npm install` does not repair this, because the postinstall script *is* the broken step.

**Confirm it** (the first two are the tell-tale signs; the third is the healthy size):

```powershell
Test-Path node_modules\electron\path.txt                                   # False = broken
Test-Path node_modules\electron\dist\electron.exe                          # False = broken
(Get-ChildItem node_modules\electron\dist -Recurse -File | Measure-Object Length -Sum).Sum   # ≈ 285,648,122 when healthy
```

**Repair — offline, from the zip `@electron/get` already cached.** It was SHA-verified against
`node_modules\electron\checksums.json` when downloaded, so this is the identical artifact npm wanted:

```powershell
$zip = Get-ChildItem "$env:LOCALAPPDATA\electron\Cache" -Recurse -Filter 'electron-v*-win32-x64.zip' |
       Sort-Object Length -Descending | Select-Object -First 1
Add-Type -AssemblyName System.IO.Compression.FileSystem
Remove-Item -Recurse -Force node_modules\electron\dist -ErrorAction SilentlyContinue
[System.IO.Compression.ZipFile]::ExtractToDirectory($zip.FullName, "$PWD\node_modules\electron\dist")   # ~5 s
[System.IO.File]::WriteAllText("$PWD\node_modules\electron\path.txt", 'electron.exe', (New-Object System.Text.UTF8Encoding($false)))
```

`path.txt` must contain **exactly** `electron.exe` with no trailing newline — `Set-Content` / `echo >` append
CRLF, which leaves Electron permanently "not installed". Verify both links of the chain:

```powershell
node -e "console.log(require('electron'))"        # -> ...\node_modules\electron\dist\electron.exe
.\node_modules\electron\dist\electron.exe --version   # -> v34.5.8
```

Repeating `npm install` on Node 26 reproduces the silent failure; either re-run this subsection or perform
the install under an LTS Node (20/22) where `extract-zip` completes normally.

---

## 2. Path A — development mode in the browser

Two terminals, both from `C:\Users\ricmi\Documents\GitHub\AeroTwin`, venv activated ([§1.2](#12-activate-the-virtual-environment-do-this-first-in-every-new-terminal)).

### 2.1 Terminal 1 — FastAPI backend

```powershell
cd backend
python run_backend.py
```

Expected startup output (uvicorn with `reload=True`, host/port from `AEROTWIN_HOST`/`AEROTWIN_PORT`):

```
INFO:     Will watch for changes in these directories: ['C:\\Users\\ricmi\\Documents\\GitHub\\AeroTwin\\backend']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
AeroTwin backend ready | database=C:\\...\\AeroTwin\\data\\aerotwin.db (WAL enabled)
```

The FastAPI lifespan creates every managed directory (`data\inputs\...`, `data\processed\...`,
`data\outputs\...`) and the SQLite schema idempotently — existing files are never overwritten.

### 2.2 Verify the backend

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health | ConvertTo-Json -Depth 4
# { "status": "healthy", "service": "AeroTwin AI Backend", "version": "0.1.0",
#   "timestamp": "2026-09-29T...Z",
#   "database": { "path": "...\\data\\aerotwin.db", "reachable": true, "journal_mode": "wal", "error": null } }
```

| Check | Where | Expected |
|---|---|---|
| Health | <http://127.0.0.1:8000/api/health> | HTTP 200, `status: healthy`, `journal_mode: wal` (HTTP 503 + `status: degraded` if the DB is unreachable — never a traceback) |
| Swagger UI | <http://127.0.0.1:8000/docs> | Processing / assessment / reports routes |
| OpenAPI JSON | <http://127.0.0.1:8000/openapi.json> | Full schema |

### 2.3 Terminal 2 — React/Vite frontend

```powershell
cd frontend
npm run dev
```

```
  VITE v7.x  ready in ~300 ms
  ➜  Local:   http://localhost:5173/
```

Vite is configured with `strictPort: true` on **5173** (the backend's CORS allow-list is pinned to
that origin), so a conflicting process makes `npm run dev` fail loudly instead of silently moving to
5174. Open <http://localhost:5173>. The renderer resolves its API base URL from
`?apiBaseUrl=` → `localStorage` → `http://127.0.0.1:8000` (`frontend\src\services\api.js`).

From the repository root, `npm run dev:frontend` is a shortcut for `cd frontend && npm run dev`.

### 2.4 End-to-end smoke test over HTTP

Everything the UI does is plain REST under `http://127.0.0.1:8000/api` — usable from a third terminal:

```powershell
$base = 'http://127.0.0.1:8000'

# 1. Register a raw UAV clip (201 Created). MP4/MOV/MKV/AVI/M4V; the file is COPIED, never moved.
$video = Invoke-RestMethod -Method Post "$base/api/processing/ingest-video" -ContentType 'application/json' `
  -Body (@{ source_path = 'C:\path\to\uav_clip.mp4' } | ConvertTo-Json)

# 2. Queue frame extraction (202 Accepted), then poll the job row
$job = Invoke-RestMethod -Method Post "$base/api/processing/extract-frames" -ContentType 'application/json' `
  -Body (@{ filename = $video.filename; frame_step = 10; image_format = 'jpg' } | ConvertTo-Json)
do { Start-Sleep -Milliseconds 500
     $row = Invoke-RestMethod "$base/api/processing/jobs/$($job.job_id)" } while ($row.status -notin 'completed','failed')
$row | Select-Object status, frames_written, processing_time_seconds, output_dir

# 3. AI analysis stage — `<stem>/<run_id>` relative to data\processed\frames\
$framesPath = ($row.output_dir -split 'frames\\')[-1] -replace '\\','/'
```

| # | Endpoint | Purpose | Expected |
|---|---|---|---|
| 1 | `POST /api/processing/ingest-video` | Copy a clip into `data\inputs\raw_videos\` + register it | **201**; 404 unknown source, 415 unsupported format, 422 unprobeable |
| 2 | `POST /api/processing/extract-frames` | Background frame extraction job | **202** + `job_id` |
| 3 | `GET /api/processing/jobs/{job_id}` | Job status/metrics (`processing_time_seconds`, `frames_written`) | **200**; **404** unknown id |
| 4 | `POST /api/processing/segment-materials` | Material segmentation over a frames run | **202**, or **503** while `material_model.pt` is missing |
| 5 | `POST /api/processing/detect-elements` | Structural element detection (needs material masks) | **202**, or **503** while `element_model.pt` is missing |
| 6 | `POST /api/processing/map-cracks` | Crack mapping over a frames run | **202**, or **503** while `crack_model.pt` is missing |
| 7 | `POST /api/processing/aggregate-results` | Dedup per-frame defects → global entities | **202**; **404** unknown job, **409** if the source job is not yet completed |
| 8 | `GET /api/processing/aggregations/{job_id}` | Aggregation summary + global entities | **200**; **404** unknown id |
| 9 | `GET /api/processing/videos` \| `/jobs` | Registered videos / recent jobs | **200** |
| 10 | `GET /api/processing/artifacts/listing` \| `/file` | Browse/stream masks, maps, reports (`?category=…&path=…`) | **200**; **400** on traversal attempts |
| 11 | `POST /api/assessment/profiles` | Create a building profile | **201**; **409** duplicate name |
| 12 | `GET /api/assessment/profiles` \| `/profiles/{id}` | List / fetch profiles | **200**; **404** unknown id |
| 13 | `POST /api/assessment/calculate` | Seismic vulnerability scoring job | **202**; **404** unknown profile/job |
| 14 | `GET /api/assessment/assessments` \| `/{id}` | Assessment list / one assessment | **200**; **404** |
| 15 | `POST /api/reports/generate` | JSON + PDF report (`report_format`: `json`\|`pdf`\|`both`) | **202**; **404** unknown assessment |
| 16 | `GET /api/reports` \| `/{id}` | Report list / one report | **200**; **404** |

Example payloads for the analysis and assessment stages:

```powershell
# Material segmentation (tile_size optional, 64–2048; omit for the model default)
@{ frames_path = $framesPath } | ConvertTo-Json    # POST /api/processing/segment-materials

# Crack mapping (crack_threshold default 0.5)
@{ frames_path = $framesPath; crack_threshold = 0.5 } | ConvertTo-Json    # POST /api/processing/map-cracks

# Cross-frame aggregation (dedup controls; keep per-frame provenance for the audit trail)
@{ crack_job_id = $crackJobId; element_job_id = $elementJobId; iou_threshold = 0.3;
   max_frame_gap = 10; include_observations = $true } | ConvertTo-Json     # POST /api/processing/aggregate-results

# Building profile, then assessment
@{ name = 'Block A'; structure_age_years = 30; construction_type = 'rc_frame';
   num_stories = 4; code_compliance = 'partially_compliant' } | ConvertTo-Json   # POST /api/assessment/profiles
@{ profile_id = $profileId; crack_job_id = $crackJobId } | ConvertTo-Json        # POST /api/assessment/calculate
@{ assessment_id = $assessmentId; report_format = 'both'; include_visual_maps = $true } | ConvertTo-Json  # POST /api/reports/generate
```

**Without weights**, stages 4–6 stop at 503, so aggregation/assessment/reporting can only be exercised
through [§5](#5-system-evaluation-script-thesis-telemetry) (which generates deterministic fixtures) or by
placing real `.pt` files first. Frame extraction, job bookkeeping, artifact browsing, health, and docs all
work regardless. Interactive equivalents for every payload: <http://127.0.0.1:8000/docs>.

---

## 3. Path B — Electron desktop app (development)

**Terminal 1** (renderer):

```powershell
cd frontend
npm run dev
```

**Terminal 2** (desktop shell — venv activated, repo root):

```powershell
npm start
```

`npm run dev` is an alias of `npm start` (both execute `electron .`) — neither one starts Vite, so
Terminal 1 must already be running.

What Electron does at startup, step by step:

1. Probes `http://127.0.0.1:8000/api/health` (500 ms timeout).
2. **If the backend is not healthy**, it spawns the sidecar itself: `python run_backend.py` with
   `cwd = backend\`, `shell: true`, and `AEROTWIN_HOST=127.0.0.1` / `AEROTWIN_PORT=<port from AEROTWIN_BACKEND_URL>`.
   Because it goes through the shell, the `python` on `PATH` must be `backend\.venv` — misleading errors
   here are almost always "wrong interpreter" problems. If the probe *succeeds* (you started the backend
   yourself in [§2](#2-path-a--development-mode-in-the-browser)), no second sidecar is spawned.
3. Waits up to 15 s (30 × 500 ms) for `/api/health`.
4. Opens the window: **1440×900** (min 1024×700), `contextIsolation: true`, `nodeIntegration: false`,
   `sandbox: true`; `preload.js` exposes only `aerotwin.pickVideoFile()` and `aerotwin.getBackendUrl()`.
   The video picker is a native dialog filtered to `mp4, mov, avi, mkv, m4v`.
5. Multiplexes sidecar output into its own console as `[sidecar] …` / `[sidecar err] …`.
6. On window close → `before-quit` / `will-quit` → `taskkill /pid <pid> /f /t` (Windows) or `SIGTERM`.

```powershell
# verify while the window is open
Invoke-RestMethod http://127.0.0.1:8000/api/health           # 200 healthy
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object OwningProcess
# cleanup if Electron was force-killed (orphaned sidecar)
Get-Process aerotwin_backend -ErrorAction SilentlyContinue | Stop-Process -Force
```

### 3.1 Testing the production bundle without packaging

The **unpackaged** window always loads `VITE_DEV_SERVER_URL` (default `http://localhost:5173`); the built
`frontend\dist\index.html` is only used when `app.isPackaged`. So `npm run build` + `npm start` still
needs the Vite dev server running. To point a dev window at the built bundle instead:

```powershell
cd frontend; npm run build; cd ..
$env:VITE_DEV_SERVER_URL = 'file:///' + ((Resolve-Path .\frontend\dist\index.html).Path -replace '\\','/')
npm start
```

That exercises the real renderer bundle over `file://` (the backend's CORS list already allows the `null`
origin). For the *packaged* behaviour — bundled sidecar, per-user data dir, `app.asar` — use
[§6](#6-production-desktop-app-sidecar--electron).

---

## 4. Testing suite (103 tests)

```powershell
cd backend
python -m pytest                       # whole suite
python -m pytest -v                    # per-test names
python -m pytest -q                    # compact
python -m pytest -p no:cacheprovider   # leave no .pytest_cache behind
python -m pytest tests/test_crack_mapping.py -v
python -m pytest -k "dedup or seismic"
```

Expected result: **`103 passed`** (measured here: 14.6 s end-to-end; collection alone is ≈6.6 s).

| Test file | Tests | Covers |
|---|---|---|
| `tests\test_database.py` | 5 | WAL enablement, idempotent re-init, commit/rollback, foreign keys, row factory |
| `tests\test_preprocessing.py` | 21 | video probing/ingest, sampling maths, frame extraction, path-traversal safety |
| `tests\test_material_segmentation.py` | 12 | tiled inference, mask rendering, weight loading, 503 mapping |
| `tests\test_crack_detection.py` | 6 | crack thresholds, overlay rendering |
| `tests\test_crack_mapping.py` | 8 | crack-mapping job, `manifest.json` / `metrics.json` artifacts |
| `tests\test_cross_frame_aggregation.py` | 27 | IoU/frame-gap dedup, entity construction, registry persistence, API status codes |
| `tests\test_seismic_calculator.py` | 24 | vulnerability factors, profiles, assessments, JSON/PDF reports, full API workflows |

### 4.1 Environment isolation (why your real data is safe)

`backend\tests\conftest.py` executes **before any `app.*` import** and unconditionally overwrites:

```python
os.environ["AEROTWIN_DATA_DIR"]    = tempfile.mkdtemp(prefix="aerotwin_test_data_")
os.environ["AEROTWIN_WEIGHTS_DIR"] = tempfile.mkdtemp(prefix="aerotwin_test_weights_")
```

Consequences: the suite never opens `data\aerotwin.db`, never writes into `data\inputs\raw_images\` or
`data\inputs\raw_videos\`, and never reads your real `.pt` files (fixtures write dummy checkpoints into the
temp weights dir). Any `AEROTWIN_*` values exported in your shell are ignored for the duration of the run,
and cleanup is handled by the OS temp policy.

### 4.2 Interpreter & rootdir rules

`backend\pytest.ini` declares `testpaths = tests` and `pythonpath = .`, so run pytest **with `backend\` as
the working directory**. If `python` is not the venv interpreter:

```powershell
python -c "import sys; print(sys.executable)"             # must be ...\backend\.venv\Scripts\python.exe
cd backend; .\.venv\Scripts\python.exe -m pytest -q        # works even with no activation
```

The suite emits one informational warning (`StarletteDeprecationWarning: Using httpx with
starlette.testclient is deprecated`) driven by fastapi 0.141.1 / httpx 0.28.1 — it does not fail the run.

---

## 5. System evaluation script (thesis telemetry)

Runs the **entire production pipeline** on synthetic UAV footage, times every stage, and writes
telemetry plus a chapter-ready summary. It calls the same service functions the API uses, so it is a true
end-to-end exercise rather than a unit-test shortcut.

```powershell
cd backend
python tests/evaluate_system.py                 # seed 42, fully isolated, self-cleaning
python tests/evaluate_system.py --keep          # keep the temp data + weights dirs afterwards
python tests/evaluate_system.py --seed 7        # alternate seed
python tests/evaluate_system.py --data-dir ..\tmp_eval
```

| Flag | Default | Effect |
|---|---|---|
| `--seed N` | `42` | Fixes `random`, NumPy, and PyTorch seeds; forwarded to services via `settings.random_seed`. |
| `--data-dir PATH` | fresh OS temp dir | Runs in *PATH*; the script no longer owns the directory, so nothing is deleted. |
| `--keep` | off | Keeps the temp directories after a successful run and prints their location. |

| # | Stage | What is measured |
|---|---|---|
| 1 | `video_import` | Synthetic clip synthesis + ingest (frames, fps, resolution, stored path) |
| 2 | `frame_extraction` | Frames written, backend `processing_time_seconds` |
| 3 | `weights_calibration` | Deterministic head-bias grid scored on decoded frames (crack head scale / background bias) |
| 4 | `material_segmentation` | `mean_frame_inference_seconds` → ms/frame |
| 5 | `element_detection` | Elements detected, ms/frame |
| 6 | `crack_mapping` | Frames with cracks, crack pixels, mean crack length |
| 7 | `cross_frame_aggregation` | Dedup ratio (raw → unique cracks/elements), IoU threshold, frame gap |
| 8 | `seismic_assessment` | Vulnerability score, risk level, factor contributions |
| 9 | `report_generation` | JSON + PDF byte sizes |

**Behaviour you should expect**

- **Exit code** `0` = all nine stages completed; `1` = a stage failed (telemetry is still written, with
  `outcome: "failed"` and the error message).
- **Isolation** — `AEROTWIN_DATA_DIR` points at a temp directory, so the repository's `data\` is never
  read or written. Run it from a clean shell: pre-exported `AEROTWIN_DATA_DIR` / `AEROTWIN_WEIGHTS_DIR`
  are honoured instead (and then nothing is cleaned up).
- **Weights** — real `.pt` files are used when all three exist; otherwise deterministic random-init
  fixtures are generated and the run is flagged `weights.source: "synthetic_random_init"`. The summary
  then carries an explicit disclaimer: runtimes, dedup mechanics, and scoring are valid; task-level
  accuracy is **not**. See [§1.4](#14-model-weights--placement-expectations).
- **Outputs** — `docs\sample_outputs\system_evaluation.json` and `system_evaluation.md`.
- **Runtime** — ≈3 s on CPU with synthetic fixtures; wall times vary with machine load even when the
  seeded pipeline decisions do not.

Baseline from the committed sample (`seed 42`, CPU, synthetic weights) — useful for spotting a regression:

| Metric | Value |
|---|---|
| Stage wall-time total | 2.2152 s (overall ≈3.0 s) |
| Material / element / crack inference | ≈3.7 / 4.0 ms per frame |
| Cross-frame dedup | 8 raw → 1 unique crack (**0.875**); 16 → 2 elements |
| Vulnerability score | **55.13 / 100** — *Substantial* |
| Report artifacts | 23 979 bytes (JSON + PDF) |

Full telemetry schema, field semantics, and the mapping from fields to thesis tables/figures:
[`docs/methodology/runbook.md`](docs/methodology/runbook.md).

---

## 6. Production desktop app (sidecar + Electron)

**Prerequisites**

```powershell
pip install -r backend\requirements.txt    # brings in the pinned pyinstaller==6.22.3
python -m PyInstaller --version            # -> 6.22.3  (fails today: not installed yet)
```

Put real `.pt` weights into `backend\app\models\weights\` **before** step 1 if you want them inside the
bundle ([§1.4](#14-model-weights--placement-expectations)).

### Step 1 — freeze the FastAPI backend

```powershell
npm run build:backend
# expands to: pyinstaller --noconfirm --clean --distpath backend/dist --workpath backend/build backend/aerotwin_backend.spec
```

Produces an **onedir** bundle:

```
backend\dist\aerotwin_backend\
├── aerotwin_backend.exe          # entry point: run_frozen.py (no uvicorn reload)
└── _internal\                    # sys._MEIPASS at runtime
    ├── app\models\weights\       # weights collected from the repo path
    └── torch\, cv2\, uvicorn\, …
```

```powershell
Test-Path backend\dist\aerotwin_backend\aerotwin_backend.exe                # True
Get-ChildItem backend\dist\aerotwin_backend\_internal\app\models\weights    # .gitkeep only until you add weights
```

Design choices baked into `aerotwin_backend.spec`: **onedir** (no ~2 GB temp extraction on every start),
**`console=False`** (no stray console window beside the GUI), **no UPX** (UPX-decompressed torch/opencv
DLLs crash at import), and the stale MSVC/UCRT DLLs plus torch's `protoc.exe` are excluded (the former
cause torch's `WinError 1114`, the latter trips electron-builder's signing pass). Expect ≈580 MB,
dominated by `torch_cpu.dll`; the build takes a few minutes.

### Step 2 — build the renderer

```powershell
npm run build:frontend     # -> frontend\dist\index.html + hashed assets (Vite base: './' for file://)
```

`npm run build:all` runs step 2 then step 1. Every packaging script below already calls it.

### Step 3 — pack the unpacked desktop app

```powershell
npm run pack               # = npm run build:all && electron-builder --dir
```

```
assets\installers\win-unpacked\
├── AeroTwin AI.exe                    # the executable you launch
├── resources\
│   ├── app.asar                       # electron\** + frontend\dist\** + package.json
│   └── backend\aerotwin_backend.exe   # frozen sidecar (build.extraResources → "backend")
└── *.dll, *.pak, *.bin …              # Electron/Chromium runtime
```

```powershell
Test-Path 'assets\installers\win-unpacked\AeroTwin AI.exe'
Test-Path 'assets\installers\win-unpacked\resources\backend\aerotwin_backend.exe'
```

Icons are required by electron-builder and already exist: `assets\icons\icon.ico` (Windows target pins
the `.ico`; app-builder panics on a multi-size ICO it can't convert) and `assets\icons\icon.png`. Both are
regenerated deterministically by `python assets\icons\generate_icons.py`.

### Step 4 — build the installer

```powershell
npm run dist:win      # -> assets\installers\AeroTwin AI-Setup-1.0.0.exe   (NSIS, x64)
npm run dist:mac      # -> .dmg (x64 + arm64, unsigned: identity = null)
npm run dist:linux    # -> AppImage
```

The NSIS target is non-one-click, **per-user** (no admin prompt), lets the user choose the install
directory, and creates desktop + Start-menu shortcuts. The artifact name comes from
`${productName}-Setup-${version}.${ext}` — bump `version` in the root `package.json` before cutting a
release.

### Step 5 — launch and verify the standalone executable

```powershell
& ".\assets\installers\win-unpacked\AeroTwin AI.exe"      # note the space in the name → quote it
```

| Observation | Expected |
|---|---|
| Window | 1440×900 AeroTwin AI window; React UI from `app.asar` |
| Sidecar | `resources\backend\aerotwin_backend.exe` spawned by the main process (no Python install required) |
| Health | `GET http://127.0.0.1:8000/api/health` → **200** within ≈2 s of the window opening |
| Data | `%APPDATA%\AeroTwinAI\data` created with `inputs\`, `processed\`, `outputs\`, `aerotwin.db` (WAL) |
| Weights | read from `<install>\resources\backend\_internal\app\models\weights`; the three AI endpoints answer **503** until real `.pt` files are bundled |
| Shutdown | closing the window runs `taskkill /pid <sidecar> /f /t`; no orphan process |

Verify independently from a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health | ConvertTo-Json -Depth 3
Get-ChildItem "$env:APPDATA\AeroTwinAI\data"
Get-Process -Name 'AeroTwin AI', aerotwin_backend -ErrorAction SilentlyContinue | Select-Object Id, ProcessName
```

Frozen-sidecar notes:

- The exe is built `console=False`, so running it directly prints nothing. Its stdout is piped into
  Electron, so `[AeroTwin] Frozen backend starting on http://127.0.0.1:8000 | data=… | weights=…` plus the
  `[sidecar] …` uvicorn lines appear in the terminal that launched Electron.
- Standalone smoke test of the frozen binary (independent of Electron):

  ```powershell
  $env:AEROTWIN_PORT = '8010'
  $env:AEROTWIN_DATA_DIR = "$env:TEMP\aerotwin_sidecar_smoke"
  & .\backend\dist\aerotwin_backend\aerotwin_backend.exe        # runs in this terminal, no output
  # second terminal:
  Invoke-RestMethod http://127.0.0.1:8010/api/health
  Get-Process aerotwin_backend | Stop-Process -Force             # stop it (Ctrl+C is unreliable in a GUI subsystem exe)
  ```

- **Rebuild rule:** the packaged app ships whatever is inside `backend\dist\aerotwin_backend\`, so any
  backend change (or newly added weight file) requires re-running step 1 before step 3. `npm run pack` and
  the `dist:*` scripts do that automatically via `build:all`.

---

## 7. Verification checklist

| Service / artifact | Command or URL | Expected |
|---|---|---|
| Backend health | <http://127.0.0.1:8000/api/health> | `{"status":"healthy", "database":{"reachable":true,"journal_mode":"wal"}}` |
| Swagger docs | <http://127.0.0.1:8000/docs> | Processing (`ingest-video`, `extract-frames`, `segment-materials`, `detect-elements`, `map-cracks`, `aggregate-results`, `artifacts/*`), assessment, reports |
| Frontend UI | <http://localhost:5173> | AeroTwin AI dashboard with sidebar navigation |
| Electron (dev) | `npm start` with Vite running | 1440×900 window; `[sidecar]` logs; health 200; no duplicate backend |
| Test suite | `cd backend; python -m pytest` | `103 passed` |
| Evaluation | `cd backend; python tests/evaluate_system.py` | Exit `0`, `outcome=completed`, 9 stages, telemetry in `docs\sample_outputs\` |
| Frozen sidecar | `backend\dist\aerotwin_backend\aerotwin_backend.exe` | Serves `/api/health` standalone (no Python install needed) |
| Packed app | `npm run pack` → `assets\installers\win-unpacked\AeroTwin AI.exe` | Window + bundled sidecar; data in `%APPDATA%\AeroTwinAI\data` |
| Installer | `npm run dist:win` | `assets\installers\AeroTwin AI-Setup-1.0.0.exe` |
| AI endpoints | `segment-materials` / `detect-elements` / `map-cracks` | **503 today** (no `.pt` weights in `backend\app\models\weights\`) — 202 once real weights are placed |

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `python : The term 'python' is not recognized…` | venv not on `PATH`: activate it ([§1.2](#12-activate-the-virtual-environment-do-this-first-in-every-new-terminal)) or call `.\backend\.venv\Scripts\python.exe` explicitly. |
| `pyinstaller : The term 'pyinstaller' is not recognized…` | PyInstaller is not installed in the active venv: `pip install -r backend\requirements.txt` ([§6](#6-production-desktop-app-sidecar--electron) prerequisite). |
| Backend change has no effect in the packaged app | Packaging ships `backend\dist\aerotwin_backend\` verbatim — re-run step 1 (or `npm run pack`, which does) after any backend or weights change. |
| Port 8000 busy / sidecar exits immediately | A previous sidecar is alive: `Get-NetTCPConnection -LocalPort 8000 \| Select-Object OwningProcess`, then `Stop-Process -Force`. Or run on another port: `$env:AEROTWIN_PORT='8010'` **and** open the UI with `?apiBaseUrl=http://127.0.0.1:8010`. |
| `npm run dev` fails on port 5173 | Vite uses `strictPort: true`; free the port (`Get-NetTCPConnection -LocalPort 5173`) — do **not** move to 5174, the backend's CORS allow-list targets 5173. |
| `Error: Electron failed to install correctly…` on `npm start` (no window at all) | The postinstall unzip silently no-oped (`extract-zip`/`yauzl` never settle on Node 26): `dist\` truncated **and** `path.txt` missing → repair offline from the `@electron/get` cache ([§1.6](#16-fixing-a-broken-electron-binary-install)). |
| Blank Electron window / "Could not reach the Vite dev server" | Start `npm run dev` first (the unpackaged window always loads `http://localhost:5173`), then reload the window. |
| Two backends / no `[sidecar]` output | Electron reuses an already-healthy backend. Start the backend yourself ([§2.1](#21-terminal-1--fastapi-backend)) *or* let Electron own it — not both. |
| Sidecar starts, but not from your venv | Electron spawns plain `python` from `PATH`, so a machine-wide Python holding the same packages can silently serve the API (verified here: `...\Programs\Python\Python311\python.exe run_backend.py`). Audit with `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object CommandLine -match 'run_backend' | Select-Object ProcessId, CommandLine` and expect `backend\.venv\Scripts\python.exe`. |
| Orphaned sidecar left after force-killing Electron | The sidecar is spawned through a shell + uvicorn reload, so its `cmd → python → python` chain can be re-parented instead of dying. Kill it by port owner: `Get-NetTCPConnection -LocalPort 8000 -State Listen` → `taskkill /pid <OwningProcess> /t /f`. |
| Frontend shows "Cannot reach the AeroTwin backend" | Backend down, health 503 (DB unreachable), or an origin mismatch. Check `/api/health`, then open `http://localhost:5173/?apiBaseUrl=http://127.0.0.1:8000`. |
| `WinError 1114: a dynamic link library (DLL) initialization routine failed` from the frozen sidecar | Stale MSVC/UCRT DLLs bundled from the Python install; the spec excludes them — rebuild clean (`npm run build:backend` passes `--clean`). |
| AI endpoints return **503** | Expected until `material_model.pt` / `element_model.pt` / `crack_model.pt` exist in the weights dir ([§1.4](#14-model-weights--placement-expectations)); check `AEROTWIN_WEIGHTS_DIR` as well. |
| `400`/`404` from `segment-materials`, `map-cracks` | `frames_path` must be `<stem>/<run_id>` **relative to `data\processed\frames\`** (copy it from the extraction job's `output_dir`); anything outside that root is rejected. |
| `409` from `aggregate-results` | The source crack/element job has not reached `completed` yet — poll the job row first. |
| Test suite collects 0 tests | Wrong rootdir: run `python -m pytest` **from `backend\`**, where `pytest.ini` lives. |
| `sqlite3.OperationalError: database is locked` | WAL is enabled but only one writer at a time: stop stray backend/eval processes and retry. |

| Evaluation exits `1` | Read `[eval] stage '<name>' failed: <error>`; `processing_jobs.error` holds the detail and telemetry is still written with `outcome: "failed"`. |
| Evaluation picks up unexpected weights | A pre-exported `AEROTWIN_WEIGHTS_DIR` won over the "real weights missing?" probe — run it in a clean shell ([§5](#5-system-evaluation-script-thesis-telemetry)). |
| Packaged app takes many seconds to show the window on first launch | Antivirus scanning the ≈580 MB torch/opencv DLL tree; later launches are fast. |
| `electron-builder` fails while converting icons | Regenerate real icon files: `python assets\icons\generate_icons.py` (electron-builder needs a genuine multi-size `.ico` plus a PNG). |
| Packaged app cannot write its data | `%APPDATA%\AeroTwinAI\data` is the writable per-user default; override with `AEROTWIN_DATA_DIR` if policy blocks it. |
| `backend\dist\…\aerotwin_backend.exe` prints nothing | Built `console=False` by design — verify over HTTP instead of stdout ([§6](#6-production-desktop-app-sidecar--electron) step 5). |

---

## 9. Corrections to earlier instructions

If you were following a previous revision of this runbook, these three points changed:

1. **`npm run build` + `npm start` does not exercise the production bundle.** An unpackaged Electron
   window always loads `VITE_DEV_SERVER_URL`; `frontend\dist\index.html` is only loaded when
   `app.isPackaged`. For a real production test, use `npm run pack` and launch
   `assets\installers\win-unpacked\AeroTwin AI.exe`; to preview the built bundle without packaging, point
   `VITE_DEV_SERVER_URL` at the `file://…\frontend\dist\index.html` path
   ([§3.1](#31-testing-the-production-bundle-without-packaging)).
2. **PyInstaller is a prerequisite that may be missing.** Install it from `backend\requirements.txt`
   before `npm run build:backend` ([§6](#6-production-desktop-app-sidecar--electron)).
3. **Model weights are absent in this repository** (only `.gitkeep` in `backend\app\models\weights\`), so
   the three AI endpoints answer **503** in both dev and packaged modes. Only the evaluation script
   substitutes deterministic fixtures, and it flags them as synthetic in the telemetry
   ([§5](#5-system-evaluation-script-thesis-telemetry)).

Related documents: [`docs/methodology/runbook.md`](docs/methodology/runbook.md) (release pipeline +
evaluation telemetry in depth), `all_phases.md` (implementation history per phase),
`.clinerules` (coding and domain conventions).
