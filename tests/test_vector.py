"""Tests for the LanceDB vector store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_store_and_search_roundtrip(tmp_path: Path) -> None:
    """store_vectors + search returns nearest results."""
    from fedai_atlas.store.vector import search, store_vectors

    db_path = tmp_path / "test_lancedb"
    df = pd.DataFrame({
        "id": ["A-1", "A-2", "A-3"],
        "use_case_name": ["Alpha", "Beta", "Gamma"],
        "agency": ["DOE", "NASA", "DOE"],
        "vector": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
    })

    count = store_vectors(df, db_path=db_path)
    assert count == 3

    # Search near [1, 0, 0] — should return A-1 and A-3 as top results
    results = search([1.0, 0.0, 0.0], limit=2, db_path=db_path)
    assert len(results) == 2
    assert "A-1" in results["id"].values


def test_store_overwrites(tmp_path: Path) -> None:
    """Calling store_vectors twice replaces the table."""
    from fedai_atlas.store.vector import store_vectors

    db_path = tmp_path / "test_lancedb"
    df1 = pd.DataFrame({"id": ["X"], "vector": [[1.0, 0.0]]})
    df2 = pd.DataFrame({"id": ["Y", "Z"], "vector": [[0.0, 1.0], [1.0, 1.0]]})

    store_vectors(df1, db_path=db_path)
    count = store_vectors(df2, db_path=db_path)
    assert count == 2
