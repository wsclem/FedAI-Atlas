"""Tests for the agency briefer module."""

from __future__ import annotations

import json

import pandas as pd
import pytest


SAMPLE_DF = pd.DataFrame({
    "id": ["AGY-1", "AGY-2", "AGY-3"],
    "use_case_name": ["Doc Summarizer", "Fraud Detector", "Chatbot"],
    "agency": ["AGY", "AGY", "AGY"],
    "agency_name": ["Agency Test", "Agency Test", "Agency Test"],
    "agency_bureau": ["Bureau A", "Bureau A", "Bureau B"],
    "development_stage_short": ["Deployed", "Pre-deployment", "Pilot"],
    "topic_area_short": ["Admin Functions", "Financial", "Admin Functions"],
    "classification_short": ["NLP", "Classical ML", "Generative AI"],
    "is_high_impact_short": ["Not High-impact", "High-impact", "Not High-impact"],
    "problem_solved": [
        "Staff spend hours reading documents.",
        "Fraudulent transactions are hard to detect manually.",
        "Citizens need quick answers to common questions.",
    ],
})

MOCK_BRIEF_JSON = {
    "executive_summary": "Agency Test has a small but diverse AI portfolio.",
    "portfolio_overview": "3 use cases spanning admin and financial topics.",
    "approach_family_distribution": "Primarily LLM-based and classical ML.",
    "key_themes": ["Document automation", "Fraud detection"],
    "notable_use_cases": [
        {"id": "AGY-1", "name": "Doc Summarizer", "why_notable": "Deployed and operational."}
    ],
    "architecture_recommendations": "NLP use cases fit LLM-based approaches well. Fraud detection is better served by classical ML.",
    "maturity_assessment": "Early stage with one deployed system.",
}


def test_compute_agency_stats() -> None:
    """_compute_agency_stats includes key breakdowns."""
    from fedai_atlas.claude.agency_briefer import _compute_agency_stats

    stats = _compute_agency_stats(SAMPLE_DF)
    assert "Total use cases: 3" in stats
    assert "Deployed: 1" in stats
    assert "Pre-deployment: 1" in stats
    assert "Bureau A: 2" in stats


def test_build_use_case_table_limits() -> None:
    """_build_use_case_table respects max_cases limit."""
    from fedai_atlas.claude.agency_briefer import _build_use_case_table

    # With max_cases=2 on a 3-row df, should sample
    table = _build_use_case_table(SAMPLE_DF, max_cases=2)
    assert "2 of 3" in table
    # Should contain use case entries
    assert "AGY-" in table


def test_build_use_case_table_no_limit() -> None:
    """_build_use_case_table includes all when under limit."""
    from fedai_atlas.claude.agency_briefer import _build_use_case_table

    table = _build_use_case_table(SAMPLE_DF, max_cases=10)
    assert "of" not in table  # no sampling notice
    assert "AGY-1" in table
    assert "AGY-2" in table
    assert "AGY-3" in table


def test_brief_agency_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """brief_agency calls Claude and returns AgencyBrief."""
    from unittest.mock import MagicMock

    from fedai_atlas.claude import agency_briefer

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    # Mock query_by_agency
    monkeypatch.setattr(agency_briefer, "query_by_agency", lambda a: SAMPLE_DF)

    # Mock client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(MOCK_BRIEF_JSON))]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    monkeypatch.setattr(agency_briefer, "get_client", lambda: mock_client)

    from fedai_atlas.config import get_settings

    get_settings.cache_clear()

    result = agency_briefer.brief_agency("AGY")
    assert result.agency == "AGY"
    assert result.total_use_cases == 3
    assert "diverse" in result.executive_summary
    assert len(result.notable_use_cases) == 1
    assert result.notable_use_cases[0].id == "AGY-1"
