"""SQLite structured data store for the normalized inventory."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from fedai_atlas.config import get_settings

TABLE_NAME = "use_cases"


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a SQLite connection. Creates the file and parent dirs if needed."""
    if db_path is None:
        db_path = get_settings().sqlite_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def store_inventory(df: pd.DataFrame, db_path: Path | None = None) -> int:
    """Write normalized inventory DataFrame to SQLite.

    Replaces the existing table on each call (full refresh).
    Returns the number of rows written.
    """
    conn = get_connection(db_path)
    try:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        return len(df)
    finally:
        conn.close()


def load_all(db_path: Path | None = None) -> pd.DataFrame:
    """Load the full inventory table as a DataFrame."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)  # noqa: S608
    finally:
        conn.close()


def query_by_agency(agency: str, db_path: Path | None = None) -> pd.DataFrame:
    """Return all use cases for a given agency abbreviation."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql(
            f"SELECT * FROM {TABLE_NAME} WHERE agency = ?",  # noqa: S608
            conn,
            params=(agency,),
        )
    finally:
        conn.close()


def query_by_id(use_case_id: str, db_path: Path | None = None) -> pd.DataFrame:
    """Return a single use case by its ID."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql(
            f"SELECT * FROM {TABLE_NAME} WHERE id = ?",  # noqa: S608
            conn,
            params=(use_case_id,),
        )
    finally:
        conn.close()


def list_agencies(db_path: Path | None = None) -> list[str]:
    """Return sorted list of distinct agency abbreviations."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(f"SELECT DISTINCT agency FROM {TABLE_NAME} ORDER BY agency")  # noqa: S608
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
