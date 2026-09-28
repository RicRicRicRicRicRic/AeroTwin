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

### Running the Test Suite

To verify that all 76 unit and integration tests are passing across the preprocessing, computer vision, and seismic calculation pipelines:

```powershell
cd backend
python -m pytest -v
```

---

### Quick Verification Checklist

| Service / View | URL / Command | Expected Behavior |
| :--- | :--- | :--- |
| **Backend Health** | `http://127.0.0.1:8000/api/health` | Returns `{"status":"healthy","database":{"journal_mode":"wal"}}` |
| **Swagger Docs** | `http://127.0.0.1:8000/docs` | Lists all 18 processing, assessment, and report endpoints |
| **Frontend UI** | `http://localhost:5173` | AeroTwin AI Dashboard with sidebar navigation |
| **Test Suite** | `pytest` in `backend/` | `76 passed` |