# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the AeroTwin AI FastAPI backend sidecar.

Build (from the repository root; PyInstaller 6.x, see backend/requirements.txt)::

    pyinstaller --noconfirm --clean \\
        --distpath backend/dist --workpath backend/build \\
        backend/aerotwin_backend.spec

Produces an **onedir** bundle at ``backend/dist/aerotwin_backend/``::

    backend/dist/aerotwin_backend/
    ├── aerotwin_backend[.exe]        <- entry: run_frozen.py (no reload)
    └── _internal/                    <- sys._MEIPASS at runtime
        ├── app/models/weights/*.pt   <- bundled weights (collected below)
        ├── torch/, cv2/, uvicorn/... <- collected third-party packages
        └── ...

electron-builder copies that folder verbatim into the installer
(``package.json`` → ``build.extraResources`` → ``resources/backend/``) and
``electron/main.js`` spawns ``resources/backend/aerotwin_backend[.exe]``.

Path resolution: ``app/core/config.py`` detects ``sys.frozen`` and resolves the
weights directory through ``sys._MEIPASS`` (``BUNDLE_DIR / "app/models/weights"``)
and the data directory to a per-user writable location — no path errors in
production builds.

Design notes:
* **onedir** (not onefile): faster start (no temp extraction of ~2 GB torch),
  simpler antivirus story, and electron-builder can copy it as resources.
* **console=False**: no stray console window next to the Electron GUI; uvicorn's
  log stream still reaches Electron because it spawns the sidecar with pipes
  (and ``run_frozen.py`` guards ``sys.stdout`` for the no-pipes case).
* **no UPX**: UPX-decompressed torch/opencv DLLs crash at import time.
"""

import os

from PyInstaller.utils.hooks import collect_all, copy_metadata

# SPECPATH is a PyInstaller-provided global: absolute directory of this spec.
SPECPATH  # noqa: B018  (documented global, intentionally referenced)

# ---------------------------------------------------------------------------
# Dynamically imported third-party code that static analysis cannot see:
# uvicorn imports loops/protocols/lifespan by string, so collect its modules
# and metadata explicitly. ``torch`` and ``cv2`` are handled by the hooks that
# ship with PyInstaller / pyinstaller-hooks-contrib (hook-torch, hook-cv2) —
# calling collect_all('torch') here would add ~12k non-runtime files (C++
# headers, tests, samples) and bloat the bundle for no benefit.
# ---------------------------------------------------------------------------
extra_datas, extra_binaries, extra_hiddenimports = [], [], []
for _package in ("uvicorn",):
    _pkg_datas, _pkg_binaries, _pkg_hidden = collect_all(_package)
    extra_datas += _pkg_datas
    extra_binaries += _pkg_binaries
    extra_hiddenimports += _pkg_hidden

# importlib.metadata lookups (pydantic/typing-inspection report versions at
# import time); harmless when a distribution is absent.
for _distribution in (
    "torch",
    "uvicorn",
    "fastapi",
    "starlette",
    "pydantic",
    "pydantic_core",
    "pydantic-core",
    "anyio",
    "click",
    "h11",
    "typing_extensions",
    "opencv-python",
):
    try:
        extra_datas += copy_metadata(_distribution)
    except Exception:  # noqa: BLE001 - optional metadata, best effort
        pass

#: Model weights are bundled at the same relative path as in the repository so
#: config.py resolves them identically in dev and frozen mode.
WEIGHTS_DIR = os.path.join(SPECPATH, "app", "models", "weights")

a = Analysis(  # noqa: N806 (PyInstaller spec convention)
    [os.path.join(SPECPATH, "run_frozen.py")],
    pathex=[SPECPATH],
    binaries=extra_binaries,
    datas=[(WEIGHTS_DIR, os.path.join("app", "models", "weights"))] + extra_datas,
    hiddenimports=[
        # anyio/Starlette pick the asyncio backend dynamically.
        "anyio._backends._asyncio",
        # uvicorn resolves these by module string at startup.
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.http.httptools_impl",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
        "uvicorn.lifespan.off",
    ]
    + extra_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pytest", "_pytest", "matplotlib", "IPython", "setuptools"],
    noarchive=False,
)

#: MSVC/UCRT runtime DLLs that PyInstaller copies out of the *Python install*
#: (v14.29 here). Inside the bundle they shadow the newer system CRT in
#: System32, and torch >= 2.9 refuses to initialise c10.dll against the stale
#: runtime ("WinError 1114: a dynamic link library (DLL) initialization
#: routine failed"). Dropping the bundled copies makes the sidecar use the
#: same OS-provided CRT that dev-mode torch already loads successfully
#: (see pytorch/pytorch#169429). Newer bundled copies are kept.
STALE_CRT_DLLS = {
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
    "vccorlib140.dll",
    "ucrtbase.dll",
}

#: Development-only executables that the torch wheel ships. ``protoc.exe`` is a
#: protobuf/ONNX build tool that inference never calls; keeping it out saves
#: ~2.7 MB and stops electron-builder's code-signing pass from tripping over a
#: non-runtime binary inside ``resources/backend``.
DEV_TOOL_EXES = {"protoc.exe"}

EXCLUDED_BINARIES = STALE_CRT_DLLS | DEV_TOOL_EXES
a.binaries = [  # noqa: N806 (PyInstaller spec convention)
    entry
    for entry in a.binaries
    if os.path.basename(entry[0]).lower() not in EXCLUDED_BINARIES
]

pyz = PYZ(a.pure)  # noqa: N806 (PyInstaller spec convention)

exe = EXE(  # noqa: N806 (PyInstaller spec convention)
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="aerotwin_backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(  # noqa: N806 (PyInstaller spec convention)
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="aerotwin_backend",
)
