"""LanceDB vector store for semantic search over use cases."""

from __future__ import annotations

from pathlib import Path

import lancedb
import pandas as pd
import voyageai

from fedai_atlas.config import get_settings

TABLE_NAME = "use_cases"


def get_vector_db(db_path: Path | None = None) -> lancedb.DBConnection:
    """Open or create a LanceDB database."""
    if db_path is None:
        db_path = get_settings().vector_db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(db_path))


def store_vectors(df: pd.DataFrame, db_path: Path | None = None) -> int:
    """Write a DataFrame with a ``vector`` column to LanceDB.

    Overwrites the existing table. Returns the number of rows stored.
    """
    db = get_vector_db(db_path)
    try:
        db.drop_table(TABLE_NAME)
    except Exception:
        pass
    db.create_table(TABLE_NAME, data=df)
    return len(df)


def append_vectors(df: pd.DataFrame, db_path: Path | None = None) -> int:
    """Append rows to an existing LanceDB table, or create it if it doesn't exist.

    Returns the number of rows appended.
    """
    db = get_vector_db(db_path)
    try:
        table = db.open_table(TABLE_NAME)
        table.add(df)
    except Exception:
        db.create_table(TABLE_NAME, data=df)
    return len(df)


def count_vectors(db_path: Path | None = None) -> int:
    """Return the number of rows in the vector table, or 0 if it doesn't exist."""
    db = get_vector_db(db_path)
    try:
        table = db.open_table(TABLE_NAME)
        return table.count_rows()
    except Exception:
        return 0


def search(
    query_vector: list[float],
    limit: int = 10,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Vector similarity search. Returns top-k results with metadata."""
    db = get_vector_db(db_path)
    table = db.open_table(TABLE_NAME)
    results = table.search(query_vector).limit(limit).to_pandas()
    return results


def semantic_search(
    query: str,
    limit: int = 10,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """End-to-end semantic search: embed the query, then find nearest vectors."""
    settings = get_settings()
    client = voyageai.Client(api_key=settings.voyage_api_key)
    result = client.embed([query], model=settings.embedding_model, input_type="query")
    query_vector = result.embeddings[0]
    return search(query_vector, limit=limit, db_path=db_path)


def find_similar(
    use_case_id: str,
    limit: int = 10,
    cross_agency: bool = False,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Find most similar use cases by embedding distance.

    Args:
        use_case_id: The target use case ID.
        limit: Number of results to return.
        cross_agency: If True, exclude results from the same agency.

    Returns:
        DataFrame with id, use_case_name, agency, _distance, and other metadata.
    """
    db = get_vector_db(db_path)
    table = db.open_table(TABLE_NAME)
    all_df = table.to_pandas()

    # Find the target use case
    target = all_df[all_df["id"] == use_case_id]
    if target.empty:
        msg = f"Use case not found in vector DB: {use_case_id}"
        raise ValueError(msg)

    target_vector = target.iloc[0]["vector"]
    target_agency = target.iloc[0].get("agency", "")

    # Search for more than needed to account for filtering
    fetch_limit = limit + 50 if cross_agency else limit + 1
    results = table.search(target_vector).limit(fetch_limit).to_pandas()

    # Exclude the target itself
    results = results[results["id"] != use_case_id]

    # Optionally exclude same agency
    if cross_agency and target_agency:
        results = results[results["agency"] != target_agency]

    return results.head(limit).reset_index(drop=True)


def find_duplicates(
    threshold: float = 0.2,
    limit: int = 30,
    agency: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Find cross-agency near-duplicate use cases by embedding distance.

    Uses numpy matrix operations for efficiency (~14MB for 3611×1024).

    Args:
        threshold: Maximum distance to consider a duplicate (lower = more similar).
        limit: Number of duplicate pairs to return.
        agency: If set, only return pairs involving this agency.

    Returns:
        DataFrame with columns: id_a, name_a, agency_a, id_b, name_b, agency_b, distance.
    """
    import numpy as np

    db = get_vector_db(db_path)
    table = db.open_table(TABLE_NAME)
    all_df = table.to_pandas()

    if all_df.empty or "vector" not in all_df.columns:
        return pd.DataFrame()

    ids = all_df["id"].values.astype(str)
    names = all_df["use_case_name"].values.astype(str)
    agencies = np.array(all_df["agency"].values.astype(str))
    vectors = np.stack(all_df["vector"].values)

    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    normed = vectors / norms

    # Cosine distance = 1 - cosine_similarity
    # Process in chunks to limit memory (3611×3611 float32 = ~50MB, fine)
    similarity = normed @ normed.T
    distance = 1.0 - similarity

    # Mask: only upper triangle (avoid duplicates), exclude self, exclude same-agency
    n = len(ids)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    # Exclude same-agency pairs (vectorized)
    agency_grid = agencies[:, None] == agencies[None, :]
    mask = mask & ~agency_grid

    # Apply agency filter if specified
    if agency:
        agency_involved = (agencies == agency)
        agency_pair_mask = agency_involved[:, None] | agency_involved[None, :]
        mask = mask & agency_pair_mask

    # Apply threshold
    mask = mask & (distance < threshold)

    # Extract pairs
    rows_i, rows_j = np.where(mask)
    distances = distance[rows_i, rows_j]

    # Sort by distance
    sort_idx = np.argsort(distances)[:limit]
    rows_i = rows_i[sort_idx]
    rows_j = rows_j[sort_idx]
    distances = distances[sort_idx]

    pairs = pd.DataFrame({
        "id_a": ids[rows_i],
        "name_a": names[rows_i],
        "agency_a": agencies[rows_i],
        "id_b": ids[rows_j],
        "name_b": names[rows_j],
        "agency_b": agencies[rows_j],
        "distance": distances,
    })

    return pairs


def find_most_connected(
    threshold: float = 0.3,
    limit: int = 20,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Find use cases with the most cross-agency similar neighbors.

    Args:
        threshold: Maximum distance to count as a neighbor.
        limit: Number of top results to return.

    Returns:
        DataFrame with: id, use_case_name, agency, neighbor_count, neighbor_agencies.
    """
    import numpy as np

    db = get_vector_db(db_path)
    table = db.open_table(TABLE_NAME)
    all_df = table.to_pandas()

    if all_df.empty or "vector" not in all_df.columns:
        return pd.DataFrame()

    ids = all_df["id"].values.astype(str)
    names = all_df["use_case_name"].values.astype(str)
    agencies = np.array(all_df["agency"].values.astype(str))
    vectors = np.stack(all_df["vector"].values)

    # Normalize + cosine distance
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = vectors / norms
    distance = 1.0 - (normed @ normed.T)

    # Cross-agency mask
    same_agency = agencies[:, None] == agencies[None, :]
    np.fill_diagonal(same_agency, True)  # also exclude self

    # Count cross-agency neighbors within threshold per use case
    neighbors = (distance < threshold) & ~same_agency
    counts = neighbors.sum(axis=1)

    # For each use case, collect which agencies its neighbors belong to
    top_idx = np.argsort(counts)[::-1][:limit]

    rows = []
    for i in top_idx:
        if counts[i] == 0:
            break
        neighbor_mask = neighbors[i]
        neighbor_agys = sorted(set(agencies[neighbor_mask]))
        rows.append({
            "id": ids[i],
            "use_case_name": names[i],
            "agency": agencies[i],
            "neighbor_count": int(counts[i]),
            "neighbor_agencies": ", ".join(neighbor_agys[:8]) + ("..." if len(neighbor_agys) > 8 else ""),
        })

    return pd.DataFrame(rows)
