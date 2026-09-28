"""Unit tests for SQLite connection handling (WAL mode, commit/rollback).

These cover the Phase 1 database layer; pipeline tests (preprocessing,
segmentation, crack mapping, seismic calculation) are added in later phases.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.core.database import get_connection, get_journal_mode, init_database


def test_init_database_creates_file_and_enables_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "aerotwin.db"

    returned = init_database(db_path)

    assert returned == db_path
    assert db_path.is_file()
    assert get_journal_mode(db_path) == "wal"


def test_wal_is_idempotent_across_repeated_starts(tmp_path: Path) -> None:
    db_path = tmp_path / "aerotwin.db"
    init_database(db_path)
    init_database(db_path)

    assert get_journal_mode(db_path) == "wal"


def test_connection_commits_on_success(tmp_path: Path) -> None:
    db_path = init_database(tmp_path / "aerotwin.db")
    with get_connection(db_path) as connection:
        connection.execute("CREATE TABLE samples (id INTEGER PRIMARY KEY, note TEXT)")
    with get_connection(db_path) as connection:
        connection.execute("INSERT INTO samples (note) VALUES (?)", ("crack-01",))
    with get_connection(db_path) as connection:
        rows = connection.execute("SELECT note FROM samples").fetchall()

    assert [row["note"] for row in rows] == ["crack-01"]


def test_connection_rolls_back_on_error(tmp_path: Path) -> None:
    db_path = init_database(tmp_path / "aerotwin.db")
    with get_connection(db_path) as connection:
        connection.execute("CREATE TABLE samples (id INTEGER PRIMARY KEY, note TEXT)")

    with pytest.raises(RuntimeError, match="boom"):
        with get_connection(db_path) as connection:
            connection.execute("INSERT INTO samples (note) VALUES (?)", ("never-persisted",))
            raise RuntimeError("boom")

    with get_connection(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    assert count == 0


def test_connection_enables_foreign_keys_and_row_factory(tmp_path: Path) -> None:
    db_path = init_database(tmp_path / "aerotwin.db")

    with get_connection(db_path) as connection:
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        row = connection.execute("SELECT 1 AS probe").fetchone()

    assert foreign_keys == 1
    assert isinstance(row, sqlite3.Row)
    assert row["probe"] == 1
