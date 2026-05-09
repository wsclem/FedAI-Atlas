"""Generate embeddings for use case records via the Voyage API."""

from __future__ import annotations

import time

import pandas as pd
import voyageai
from rich.progress import Progress

from fedai_atlas.config import get_settings

# Fields whose content forms the core of the embedding
EMBED_FIELDS = ["use_case_name", "problem_solved", "benefits", "system_outputs"]

# Structured context fields prepended as metadata
CONTEXT_FIELDS = {
    "agency_name": "Agency",
    "topic_area_short": "Topic",
    "development_stage_short": "Stage",
}

BATCH_SIZE = 20  # Small batches to stay under free-tier 10K TPM
REQUEST_DELAY = 21.0  # 3 RPM = 1 request per 20s; add 1s margin


def build_embed_text(row: pd.Series) -> str:
    """Combine key fields into a single text block for embedding.

    Format::

        {use_case_name}

        Agency: ... | Topic: ... | Stage: ...

        Problem: {problem_solved}

        Benefits: {benefits}

        Outputs: {system_outputs}

    Null or empty fields are omitted.
    """
    parts: list[str] = []

    # Title
    name = _safe(row, "use_case_name")
    if name:
        parts.append(name)

    # Context line
    ctx_parts = []
    for field, label in CONTEXT_FIELDS.items():
        val = _safe(row, field)
        if val:
            ctx_parts.append(f"{label}: {val}")
    if ctx_parts:
        parts.append(" | ".join(ctx_parts))

    # Main content sections
    section_labels = {
        "problem_solved": "Problem",
        "benefits": "Benefits",
        "system_outputs": "Outputs",
    }
    for field, label in section_labels.items():
        val = _safe(row, field)
        if val:
            parts.append(f"{label}: {val}")

    return "\n\n".join(parts)


def _safe(row: pd.Series, field: str) -> str:
    """Return the string value of a field, or empty string if missing/null."""
    if field not in row.index:
        return ""
    val = row[field]
    if pd.isna(val):
        return ""
    return str(val).strip()


def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
    input_type: str = "document",
) -> list[list[float]]:
    """Embed a list of texts via the Voyage API in batches.

    Args:
        texts: Texts to embed.
        model: Voyage model name. Defaults to config setting.
        input_type: 'document' for indexing, 'query' for search queries.

    Returns:
        List of embedding vectors (one per input text).
    """
    settings = get_settings()
    if model is None:
        model = settings.embedding_model

    client = voyageai.Client(api_key=settings.voyage_api_key)
    all_vectors: list[list[float]] = []

    with Progress() as progress:
        task = progress.add_task("Embedding", total=len(texts))
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            # Retry with exponential backoff on rate limit errors
            for attempt in range(5):
                try:
                    result = client.embed(batch, model=model, input_type=input_type)
                    break
                except Exception as e:
                    if "rate" in str(e).lower() and attempt < 4:
                        wait = 20 * (2**attempt)
                        progress.console.print(
                            f"[yellow]Rate limited, waiting {wait}s...[/yellow]"
                        )
                        time.sleep(wait)
                    else:
                        raise
            all_vectors.extend(result.embeddings)
            progress.advance(task, len(batch))
            # Rate limit: pause between batches (skip after last batch)
            if i + BATCH_SIZE < len(texts):
                time.sleep(REQUEST_DELAY)

    return all_vectors


def embed_inventory(df: pd.DataFrame, db_path: "Path | None" = None) -> int:
    """Embed all use cases and store vectors incrementally to LanceDB.

    Each batch is written to LanceDB immediately, so progress is preserved
    if the process is interrupted. On resume, already-embedded IDs are skipped.

    Returns the total number of vectors in the store after completion.
    """
    from pathlib import Path

    from fedai_atlas.store.vector import append_vectors, count_vectors, get_vector_db, TABLE_NAME

    # Build embed text for all rows
    df = df.copy()
    df["embed_text"] = df.apply(build_embed_text, axis=1)

    # Check what's already embedded (for resume support)
    existing_ids: set[str] = set()
    try:
        db = get_vector_db(db_path)
        table = db.open_table(TABLE_NAME)
        existing_df = table.to_pandas()
        if "id" in existing_df.columns:
            existing_ids = set(existing_df["id"].tolist())
    except Exception:
        pass

    remaining = df[~df["id"].isin(existing_ids)]
    if len(remaining) == 0:
        return count_vectors(db_path)

    settings = get_settings()
    model = settings.embedding_model
    client = voyageai.Client(api_key=settings.voyage_api_key)

    total_stored = len(existing_ids)
    with Progress() as progress:
        if existing_ids:
            progress.console.print(
                f"[dim]Resuming: {len(existing_ids)} already embedded, "
                f"{len(remaining)} remaining[/dim]"
            )
        task = progress.add_task("Embedding", total=len(remaining))

        for i in range(0, len(remaining), BATCH_SIZE):
            batch_df = remaining.iloc[i : i + BATCH_SIZE]
            texts = batch_df["embed_text"].tolist()

            # Retry with exponential backoff on rate limit errors
            result = None
            for attempt in range(5):
                try:
                    result = client.embed(texts, model=model, input_type="document")
                    break
                except Exception as e:
                    if "rate" in str(e).lower() and attempt < 4:
                        wait = 20 * (2**attempt)
                        progress.console.print(
                            f"[yellow]Rate limited, waiting {wait}s...[/yellow]"
                        )
                        time.sleep(wait)
                    else:
                        raise

            # Write this batch to LanceDB immediately
            batch_df = batch_df.copy()
            batch_df["vector"] = result.embeddings
            append_vectors(batch_df, db_path=db_path)
            total_stored += len(batch_df)
            progress.advance(task, len(batch_df))

            # Rate limit: pause between batches
            if i + BATCH_SIZE < len(remaining):
                time.sleep(REQUEST_DELAY)

    return total_stored
