"""FastAPI entry point for the AeroTwin AI backend.

Run it from the ``backend/`` directory with ``python run_backend.py`` or:
``uvicorn app.main:app --reload``.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .core.config import settings
from .core.database import get_connection, init_database

logger = logging.getLogger("aerotwin.backend")

API_PREFIX = "/api"


class DatabaseHealth(BaseModel):
    """SQLite reachability details reported by the health endpoint."""

    path: str = Field(description="Absolute path of the SQLite database file.")
    reachable: bool = Field(description="Whether a trivial query succeeded.")
    journal_mode: str | None = Field(
        default=None,
        description="Active journal mode; 'wal' when correctly configured.",
    )
    error: str | None = Field(default=None, description="Error message when unreachable.")


class HealthResponse(BaseModel):
    """Payload returned by ``GET /api/health``."""

    status: str = Field(description="'healthy' or 'degraded'.")
    service: str
    version: str
    timestamp: str = Field(description="UTC ISO-8601 timestamp of the probe.")
    database: DatabaseHealth


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise data directories and the SQLite database at startup."""
    settings.ensure_directories()
    db_path = init_database()
    logger.info("AeroTwin backend ready | database=%s (WAL enabled)", db_path)
    yield
    logger.info("AeroTwin backend shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="UAV-based post-disaster structural health monitoring sidecar.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["health"])
async def health_check(response: Response) -> HealthResponse:
    """Liveness probe: reports service status and SQLite/WAL reachability.

    Returns HTTP 200 when the database answers and HTTP 503 (with the same
    structured body) when it does not, so callers never see a raw traceback.
    """
    try:
        with get_connection() as connection:
            connection.execute("SELECT 1").fetchone()
            row = connection.execute("PRAGMA journal_mode").fetchone()
        database_health = DatabaseHealth(
            path=str(settings.database_path),
            reachable=True,
            journal_mode=str(row[0]).lower() if row is not None else None,
        )
        response.status_code = status.HTTP_200_OK
        health_status = "healthy"
    except sqlite3.Error as exc:
        database_health = DatabaseHealth(
            path=str(settings.database_path),
            reachable=False,
            error=str(exc),
        )
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        health_status = "degraded"

    return HealthResponse(
        status=health_status,
        service=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        database=database_health,
    )
