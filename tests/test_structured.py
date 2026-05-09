"""Tests for the SQLite structured store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SAMPLE_DF = pd.DataFrame({
    "agency": ["DOE", "DOE", "NASA"],
    "agency_name": ["Dept of Energy", "Dept of Energy", "NASA"],
    "id": ["DOE-1", "DOE-2", "NASA-1"],
    "use_case_name": ["Solar forecast", "Grid optimization", "Mars weather"],
})


def test_store_and_load_roundtrip(tmp_path: Path) -> None:
    """store_inventory + load_all roundtrip preserves data."""
    from fedai_atlas.store.structured import load_all, store_inventory

    db = tmp_path / "test.db"
    count = store_inventory(SAMPLE_DF, db_path=db)
    assert count == 3

    result = load_all(db_path=db)
    assert len(result) == 3
    assert list(result["id"]) == ["DOE-1", "DOE-2", "NASA-1"]


def test_query_by_agency(tmp_path: Path) -> None:
    """query_by_agency filters correctly."""
    from fedai_atlas.store.structured import query_by_agency, store_inventory

    db = tmp_path / "test.db"
    store_inventory(SAMPLE_DF, db_path=db)

    result = query_by_agency("DOE", db_path=db)
    assert len(result) == 2
    assert all(result["agency"] == "DOE")


def test_query_by_id(tmp_path: Path) -> None:
    """query_by_id returns a single matching row."""
    from fedai_atlas.store.structured import query_by_id, store_inventory

    db = tmp_path / "test.db"
    store_inventory(SAMPLE_DF, db_path=db)

    result = query_by_id("NASA-1", db_path=db)
    assert len(result) == 1
    assert result.iloc[0]["use_case_name"] == "Mars weather"


def test_list_agencies(tmp_path: Path) -> None:
    """list_agencies returns sorted unique values."""
    from fedai_atlas.store.structured import list_agencies, store_inventory

    db = tmp_path / "test.db"
    store_inventory(SAMPLE_DF, db_path=db)

    result = list_agencies(db_path=db)
    assert result == ["DOE", "NASA"]
