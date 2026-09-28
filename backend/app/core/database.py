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

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi import HTTPException

from .config import settings

#: Seconds a connection waits for a competing writer before raising ``SQLITE_BUSY``.
BUSY_TIMEOUT_SECONDS: float = 5.0
_BUSY_TIMEOUT_MS: int = 5000


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
    """Create the database file (and its parent directory) and enable WAL.

    Idempotent: safe to call on every application start. Returns the absolute
    path of the database and raises ``RuntimeError`` if WAL cannot be
    activated, so misconfiguration fails loudly instead of silently.
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
    return path


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
