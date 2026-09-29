"""Frozen (PyInstaller) launcher for the AeroTwin AI backend sidecar.

Development uses ``run_backend.py`` (auto-reload). A PyInstaller bundle cannot
use uvicorn's ``reload`` mode — it re-spawns the interpreter and watches files
that do not exist inside a bundle — so the frozen build imports the ASGI app
directly and serves it in-process. It is the entry point declared by
``aerotwin_backend.spec``.

Responsibilities:
* Guard ``sys.stdout``/``sys.stderr`` — a windowed (``console=False``) Windows
  binary may have no streams until Electron attaches its pipes, and uvicorn's
  logging must never crash on ``None``.
* Keep all path defaults in :mod:`app.core.config` (``sys._MEIPASS``-aware) and
  all tunables in ``AEROTWIN_*`` environment variables, which Electron sets
  when it spawns this executable (``AEROTWIN_HOST``/``AEROTWIN_PORT`` are
  aligned with the backend URL the renderer polls).

Usage (frozen)::

    aerotwin_backend.exe            # reads AEROTWIN_* env if provided
"""

from __future__ import annotations

import os
import sys


def _ensure_streams() -> None:
    """Replace missing stdout/stderr with ``NUL`` handles (GUI subsystem)."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def main() -> None:
    """Start the FastAPI sidecar with uvicorn (no reload; frozen-safe)."""
    _ensure_streams()
    # Import after the stream guard so logging during import is safe.
    import uvicorn

    from app.core.config import settings
    from app.main import app

    print(
        f"[AeroTwin] Frozen backend starting on http://{settings.host}:{settings.port} "
        f"| data={settings.data_dir} | weights={settings.weights_dir}",
        flush=True,
    )
    # Pass the app object directly: string-based imports ("app.main:app") would
    # re-import through the filesystem, which does not exist in a frozen bundle.
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
