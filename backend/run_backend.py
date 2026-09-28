"""Development launcher for the AeroTwin AI backend.

Usage (from the ``backend/`` directory):
    python run_backend.py
"""

from __future__ import annotations

import uvicorn

from app.core.config import settings


def main() -> None:
    """Run the FastAPI sidecar on the configured host/port with auto-reload."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
