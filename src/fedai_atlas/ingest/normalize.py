"""Normalize and clean raw inventory records using the OMB data dictionary."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_recode_lookup(data_dict: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Build a {column: {variant_lower: canonical_value}} lookup from the data dictionary.

    The data dictionary maps canonical → [variants]. This inverts that so we can
    look up any messy value and find its canonical form.
    """
    lookup: dict[str, dict[str, str]] = {}
    for col, meta in data_dict.items():
        recode_map = meta.get("recoding_map")
        if not recode_map:
            continue
        col_lookup: dict[str, str] = {}
        for canonical, variants in recode_map.items():
            # The canonical value itself should also map to itself
            col_lookup[canonical.strip().lower()] = canonical
            for variant in variants:
                col_lookup[variant.strip().lower()] = canonical
        lookup[col] = col_lookup
    return lookup


def _build_short_label_map(data_dict: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Build a {column: {canonical_value: short_label}} map."""
    result: dict[str, dict[str, str]] = {}
    for col, meta in data_dict.items():
        short_map = meta.get("short_hand_recodings")
        if short_map:
            result[col] = short_map
    return result


def normalize_inventory(
    df: pd.DataFrame, data_dict: dict[str, Any]
) -> pd.DataFrame:
    """Apply normalization to a raw inventory DataFrame.

    Steps:
        1. Strip whitespace from all string columns
        2. Apply recoding_map for multiple-choice fields (case-insensitive)
        3. Parse operational_date to datetime
        4. Add *_short columns where short_hand_recodings exist

    Returns a new DataFrame (does not mutate the input).
    """
    df = df.copy()

    # 1. Strip whitespace
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().replace("nan", pd.NA)

    # 2. Apply recoding maps
    recode = build_recode_lookup(data_dict)
    for col, col_lookup in recode.items():
        if col not in df.columns:
            continue
        original = df[col].copy()
        df[col] = df[col].map(
            lambda val, lk=col_lookup: _recode_value(val, lk)  # noqa: B023
        )
        # Log values that didn't match any recode entry
        changed = original != df[col]
        unmatched = df[col].isna() & original.notna()
        if unmatched.any():
            # These were non-null but couldn't be recoded — restore originals
            df.loc[unmatched, col] = original.loc[unmatched]
            samples = original.loc[unmatched].unique()[:3]
            logger.warning(
                "Column '%s': %d values not in recoding_map (e.g. %s)",
                col,
                unmatched.sum(),
                list(samples),
            )

    # 3. Parse operational_date
    if "operational_date" in df.columns:
        df["operational_date"] = pd.to_datetime(df["operational_date"], errors="coerce")

    # 4. Add short label columns
    short_maps = _build_short_label_map(data_dict)
    for col, smap in short_maps.items():
        if col in df.columns:
            df[f"{col}_short"] = df[col].map(smap)

    # 5. Generate synthetic IDs for rows missing an id, and prefix numeric-only IDs with agency
    if "id" in df.columns:
        # Prefix numeric-only IDs with agency abbreviation
        if "agency" in df.columns:
            numeric_mask = df["id"].str.match(r"^\d+$", na=False)
            if numeric_mask.any():
                df.loc[numeric_mask, "id"] = (
                    df.loc[numeric_mask, "agency"].str.upper() + "-" + df.loc[numeric_mask, "id"]
                )

        missing_mask = df["id"].isna() | (df["id"] == "")
        if missing_mask.any() and "agency" in df.columns:
            for agency, group in df[missing_mask].groupby("agency"):
                prefix = str(agency).upper()
                # Find max existing numeric suffix for this agency
                existing = df[(~missing_mask) & (df["agency"] == agency)]["id"]
                max_num = 0
                for eid in existing:
                    parts = str(eid).rsplit("-", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        max_num = max(max_num, int(parts[1]))
                # Assign sequential IDs
                for i, idx in enumerate(group.index, start=max_num + 1):
                    df.loc[idx, "id"] = f"{prefix}-{i:04d}"

    return df


def _recode_value(val: object, lookup: dict[str, str]) -> object:
    """Map a single cell value to its canonical form, or return it unchanged."""
    if pd.isna(val):
        return val
    key = str(val).strip().lower()
    return lookup.get(key, val)
