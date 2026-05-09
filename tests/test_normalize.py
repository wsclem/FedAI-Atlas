"""Tests for the normalize module."""

from __future__ import annotations

import pandas as pd


SAMPLE_DATA_DICT = {
    "development_stage": {
        "data_type": "multiple_choice",
        "recoding_map": {
            "a) Pre-deployment": ["pre-deployment", "Acquisition and/or Development"],
            "b) Pilot": ["pilot"],
            "c) Deployed": ["deployed", "Operation and Maintenance"],
        },
        "short_hand_recodings": {
            "a) Pre-deployment": "Pre-deployment",
            "b) Pilot": "Pilot",
            "c) Deployed": "Deployed",
        },
    },
    "is_high_impact": {
        "data_type": "multiple_choice",
        "recoding_map": {
            "a) Yes": ["yes", "Yes"],
            "b) No": ["no", "No"],
        },
    },
}


def test_build_recode_lookup() -> None:
    """Inverts recoding_map correctly: variant → canonical."""
    from fedai_atlas.ingest.normalize import build_recode_lookup

    lookup = build_recode_lookup(SAMPLE_DATA_DICT)
    assert "development_stage" in lookup
    ds = lookup["development_stage"]
    assert ds["pre-deployment"] == "a) Pre-deployment"
    assert ds["acquisition and/or development"] == "a) Pre-deployment"
    assert ds["deployed"] == "c) Deployed"
    # Canonical value itself maps to itself
    assert ds["a) pre-deployment"] == "a) Pre-deployment"


def test_normalize_recodes_messy_values() -> None:
    """normalize_inventory recodes known messy values to canonical."""
    from fedai_atlas.ingest.normalize import normalize_inventory

    df = pd.DataFrame({
        "development_stage": ["pre-deployment", "deployed", "pilot"],
        "is_high_impact": ["Yes", "No", "yes"],
    })
    result = normalize_inventory(df, SAMPLE_DATA_DICT)
    assert list(result["development_stage"]) == [
        "a) Pre-deployment",
        "c) Deployed",
        "b) Pilot",
    ]
    assert list(result["is_high_impact"]) == ["a) Yes", "b) No", "a) Yes"]


def test_normalize_preserves_unrecognized_values() -> None:
    """Unrecognized values pass through unchanged."""
    from fedai_atlas.ingest.normalize import normalize_inventory

    df = pd.DataFrame({"development_stage": ["something unexpected"]})
    result = normalize_inventory(df, SAMPLE_DATA_DICT)
    assert result["development_stage"].iloc[0] == "something unexpected"


def test_normalize_strips_whitespace() -> None:
    """Leading/trailing whitespace is stripped from string columns."""
    from fedai_atlas.ingest.normalize import normalize_inventory

    df = pd.DataFrame({"development_stage": ["  deployed  "]})
    result = normalize_inventory(df, SAMPLE_DATA_DICT)
    assert result["development_stage"].iloc[0] == "c) Deployed"


def test_normalize_adds_short_labels() -> None:
    """Short label columns are added where short_hand_recodings exist."""
    from fedai_atlas.ingest.normalize import normalize_inventory

    df = pd.DataFrame({"development_stage": ["deployed"]})
    result = normalize_inventory(df, SAMPLE_DATA_DICT)
    assert "development_stage_short" in result.columns
    assert result["development_stage_short"].iloc[0] == "Deployed"


def test_normalize_does_not_mutate_input() -> None:
    """Input DataFrame is not modified."""
    from fedai_atlas.ingest.normalize import normalize_inventory

    df = pd.DataFrame({"development_stage": ["pre-deployment"]})
    original_val = df["development_stage"].iloc[0]
    normalize_inventory(df, SAMPLE_DATA_DICT)
    assert df["development_stage"].iloc[0] == original_val
