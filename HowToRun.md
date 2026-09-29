Here is how you can run **AeroTwin AI** on your machine, depending on how you want to work with it:

---

### Option 1: Full Desktop App (Electron + React + Python Backend) — Recommended

In this mode, you launch the Vite development server and the Electron desktop window. Electron automatically detects if the Python backend is running; if not, it will start it as a background sidecar.

Open **two** terminal windows at the root of the project (`c:\Users\ricmi\Documents\GitHub\AeroTwin`):

#### Terminal 1: Start the Frontend Dev Server
```powershell
cd frontend
npm run dev
```
*(Leave this running — it hosts the React interface on `http://localhost:5173`)*

#### Terminal 2: Launch the Electron Desktop App
```powershell
npm start
```
*Electron will open a 1440×900 desktop window, connect to the frontend, and automatically manage the Python backend.*

---

### Option 2: Browser Mode (FastAPI Backend + React Frontend)

If you just want to test in Google Chrome or Microsoft Edge without opening Electron:

#### Terminal 1: Start the Python Backend
```powershell
cd backend
python run_backend.py
```
*The FastAPI backend will start at `http://127.0.0.1:8000`.*
- **Interactive API Documentation:** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.
- **Health Check:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

#### Terminal 2: Start the React Frontend
```powershell
cd frontend
npm run dev
```
- Open [http://localhost:5173](http://localhost:5173) in your browser.

---

### Option 3: Production Build (Standalone Distributable)

To compile the frontend into optimized static assets that load directly inside Electron from `file://`:

```powershell
# 1. Build the frontend production bundle
cd frontend
npm run build
cd ..

# 2. Launch Electron pointing to the built bundle
npm start
```

---

### Option 4: Production Installers (Packaged Backend + Electron)

To compile the FastAPI backend into a standalone executable (PyInstaller), bundle it with the
built frontend and Electron main process, and produce a platform installer:

```powershell
# One-time tooling: PyInstaller is pinned in backend/requirements.txt
pip install -r backend/requirements.txt

# 1. Compile the backend sidecar -> backend/dist/aerotwin_backend/aerotwin_backend.exe
npm run build:backend

# 2. Vite production bundle -> frontend/dist
npm run build:frontend

# 3. Unpacked app for local testing -> assets/installers/win-unpacked/
npm run pack

# 4. Installer for the current OS (Windows NSIS .exe / macOS dmg / Linux AppImage)
npm run dist:win        # -> assets/installers/AeroTwin AI-Setup-1.0.0.exe
npm run dist:mac
npm run dist:linux
```

Notes:

- `npm run build:all` runs steps 1 and 2 together.
- Drop real model weights into `backend/app/models/weights/{material,element,crack}_model.pt`
  **before** step 1 so they are embedded in the sidecar bundle; without them the packaged app
  still runs and the analysis endpoints answer with a descriptive HTTP 503.
- The packaged app stores its data in a per-user writable directory
  (`%APPDATA%\AeroTwinAI\data` on Windows) unless `AEROTWIN_DATA_DIR` is set.

---

### System Evaluation Script (Thesis Telemetry)

Runs the whole pipeline on synthetic UAV data and writes performance telemetry
(inference runtimes, dedup ratios, seismic factors, report sizes) to `docs/sample_outputs/`:

```powershell
cd backend
python tests/evaluate_system.py            # add --keep to inspect the temporary data dir
```

Outputs: `docs/sample_outputs/system_evaluation.json` and `system_evaluation.md`.
The run is seeded (42), fully isolated from `data/`, and exits non-zero if any stage fails.

---

### Running the Test Suite

To verify that all 103 unit and integration tests are passing across the preprocessing, computer vision, aggregation, and seismic calculation pipelines:

```powershell
cd backend
python -m pytest -v
```

---

### Quick Verification Checklist

| Service / View | URL / Command | Expected Behavior |
| :--- | :--- | :--- |
| **Backend Health** | `http://127.0.0.1:8000/api/health` | Returns `{"status":"healthy","database":{"journal_mode":"wal"}}` |
| **Swagger Docs** | `http://127.0.0.1:8000/docs` | Lists all processing, assessment, and report endpoints |
| **Frontend UI** | `http://localhost:5173` | AeroTwin AI Dashboard with sidebar navigation |
| **Test Suite** | `pytest` in `backend/` | `103 passed` |
| **Packaged Sidecar** | `backend\dist\aerotwin_backend\aerotwin_backend.exe` | Serves the API stand-alone (no Python install required) |
| **System Telemetry** | `docs/sample_outputs/system_evaluation.md` | Per-stage runtimes, dedup ratio, seismic score, report artifacts |