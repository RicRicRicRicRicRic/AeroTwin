"""SQLite connection handling for AeroTwin AI.

Design notes:
* **WAL journal mode** is enforced for the database file so concurrent reads
  (health checks, report generation) never block the writer during
  long-running inference jobs, and vice versa.
* Connections are short-lived and always managed by context managers:
  commit on success, rollback on error, close in ``finally``.
* A connection is created and consumed inside a single thread; background
  workers that log metrics must open their own connection via
  :func:`get_connection` instead of sharing one across threads.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import HTTPException

from .config import settings

#: Seconds a connection waits for a competing writer before raising ``SQLITE_BUSY``.
BUSY_TIMEOUT_SECONDS: float = 5.0
_BUSY_TIMEOUT_MS: int = 5000

#: Idempotent DDL applied on every startup (``IF NOT EXISTS`` throughout).
SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS videos (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL UNIQUE,
        stored_path TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        fps REAL NOT NULL,
        frame_count INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        duration_seconds REAL NOT NULL,
        codec TEXT,
        ingested_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS processing_jobs (
        id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL,
        video_id TEXT,
        video_filename TEXT NOT NULL,
        status TEXT NOT NULL,
        params_json TEXT NOT NULL,
        output_dir TEXT,
        frames_written INTEGER,
        total_video_frames INTEGER,
        processing_time_seconds REAL,
        error TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        metrics_json TEXT,
        FOREIGN KEY (video_id) REFERENCES videos (id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS frame_defect_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        frame_filename TEXT NOT NULL,
        crack_pixel_count INTEGER NOT NULL,
        crack_area_ratio REAL NOT NULL,
        crack_length_px REAL NOT NULL,
        mean_width_px REAL NOT NULL,
        component_count INTEGER NOT NULL,
        computed_at TEXT NOT NULL,
        UNIQUE (job_id, frame_filename),
        FOREIGN KEY (job_id) REFERENCES processing_jobs (id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON processing_jobs (status)",
    "CREATE INDEX IF NOT EXISTS idx_processing_jobs_video ON processing_jobs (video_filename)",
    "CREATE INDEX IF NOT EXISTS idx_defect_metrics_job ON frame_defect_metrics (job_id)",
)

#: Additive migrations for databases created before a column existed
#: (table, column, statement). Checked against ``PRAGMA table_info``.
MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    (
        "processing_jobs",
        "metrics_json",
        "ALTER TABLE processing_jobs ADD COLUMN metrics_json TEXT",
    ),
)


def _configure_connection(connection: sqlite3.Connection) -> None:
    """Apply the pragmas every AeroTwin connection must run with."""
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    # WAL is persistent per database file, but requesting it on every connect
    # guarantees WAL even if the file was created by an external tool.
    connection.execute("PRAGMA journal_mode = WAL")
    # Recommended pairing with WAL: durable enough for our write pattern and
    # far faster than synchronous = FULL.
    connection.execute("PRAGMA synchronous = NORMAL")


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a configured connection, committing on success and rolling back on error.

    Usage::

        with get_connection() as conn:
            conn.execute("INSERT INTO ...", params)
    """
    path = db_path if db_path is not None else settings.database_path
    connection = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_SECONDS)
    _configure_connection(connection)
    try:
        yield connection
    except BaseException:
        connection.rollback()
        raise
    else:
        connection.commit()
    finally:
        connection.close()


def init_database(db_path: Path | None = None) -> Path:
    """Create the database file, enable WAL, and apply the schema DDL.

    Idempotent: safe to call on every application start (all statements use
    ``IF NOT EXISTS``). Returns the absolute path of the database and raises
    ``RuntimeError`` if WAL cannot be activated, so misconfiguration fails
    loudly instead of silently.
    """
    path = db_path if db_path is not None else settings.database_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with get_connection(path) as connection:
        row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        journal_mode = str(row[0]).lower() if row is not None else ""
        if journal_mode != "wal":
            raise RuntimeError(
                f"Failed to enable WAL journal mode on {path} (got {journal_mode!r})"
            )
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        _apply_migrations(connection)
    return path


def _apply_migrations(connection: sqlite3.Connection) -> None:
    """Apply additive column migrations for pre-existing databases."""
    for table, column, statement in MIGRATIONS:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(statement)


# ---------------------------------------------------------------------------
# Generic processing-job persistence (shared by every pipeline phase)
# ---------------------------------------------------------------------------
#: Columns the job-status updater may write (identifiers are never user input).
JOB_UPDATABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "status",
        "output_dir",
        "frames_written",
        "total_video_frames",
        "processing_time_seconds",
        "error",
        "completed_at",
        "metrics_json",
    }
)

_JOB_INITIAL_STATUS: str = "pending"


def create_job_record(
    *,
    job_type: str,
    video_filename: str,
    params_json: str,
    job_id: str | None = None,
    video_id: str | None = None,
    db_path: Path | None = None,
) -> str:
    """Insert a ``pending`` processing job and return its id (uuid4 hex)."""
    new_id = job_id if job_id is not None else uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO processing_jobs (
                id, job_type, video_id, video_filename, status, params_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (new_id, job_type, video_id, video_filename, _JOB_INITIAL_STATUS, params_json, created_at),
        )
    return new_id


def update_job_record(
    job_id: str,
    values: dict[str, Any],
    db_path: Path | None = None,
) -> None:
    """Persist selected job columns (``values`` must use whitelisted names).

    Raises:
        KeyError: Unknown job id. ValueError: Non-whitelisted column.
    """
    if not values:
        return
    unknown = set(values) - JOB_UPDATABLE_COLUMNS
    if unknown:
        raise ValueError(f"Cannot update job columns: {sorted(unknown)}")
    assignments = ", ".join(f"{column} = ?" for column in values)
    sql = f"UPDATE processing_jobs SET {assignments} WHERE id = ?"  # noqa: S608 (whitelisted identifiers)
    with get_connection(db_path) as connection:
        cursor = connection.execute(sql, (*values.values(), job_id))
        if cursor.rowcount == 0:
            raise KeyError(f"Unknown processing job: {job_id}")


def get_job_record(job_id: str, db_path: Path | None = None) -> sqlite3.Row | None:
    """Fetch a raw job row by id, or ``None`` when it does not exist."""
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT * FROM processing_jobs WHERE id = ?", (job_id,)
        ).fetchone()


def list_job_records(
    limit: int = 50,
    db_path: Path | None = None,
) -> list[sqlite3.Row]:
    """Return the most recent job rows, newest first."""
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT * FROM processing_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


def job_record_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a raw job row into a dict matching ``ProcessingJobResponse``.

    ``params_json``/``metrics_json`` are parsed to objects and timestamps to
    ``datetime`` so any job type (extraction, segmentation, crack mapping…)
    can be serialised through one generic response model.
    """
    return {
        "job_id": row["id"],
        "job_type": row["job_type"],
        "status": row["status"],
        "video_filename": row["video_filename"],
        "params": json.loads(row["params_json"]) if row["params_json"] else {},
        "output_dir": row["output_dir"],
        "frames_written": row["frames_written"],
        "total_video_frames": row["total_video_frames"],
        "processing_time_seconds": row["processing_time_seconds"],
        "metrics": json.loads(row["metrics_json"]) if row["metrics_json"] else None,
        "error": row["error"],
        "created_at": datetime.fromisoformat(row["created_at"]),
        "completed_at": (
            datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
        ),
    }


def get_journal_mode(db_path: Path | None = None) -> str:
    """Return the database's journal mode (``wal`` when correctly configured)."""
    path = db_path if db_path is not None else settings.database_path
    with get_connection(path) as connection:
        row = connection.execute("PRAGMA journal_mode").fetchone()
    return str(row[0]).lower() if row is not None else "unknown"


def get_db() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency yielding a request-scoped database connection.

    Maps SQLite failures to a clean HTTP 503 response with a readable detail
    message instead of an unhandled traceback (AeroTwin error-handling rule).
    """
    try:
        with get_connection() as connection:
            yield connection
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503,
            detail=f"SQLite database unavailable: {exc}",
        ) from exc
