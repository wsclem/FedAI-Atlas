"""Tests for the fit analyzer module."""

from __future__ import annotations

import json

import pandas as pd
import pytest


SAMPLE_ROW = pd.Series({
    "id": "TEST-1",
    "use_case_name": "Document Summarization Tool",
    "agency_name": "Dept of Test",
    "agency_bureau": "Bureau of Testing",
    "topic_area_short": "Administrative",
    "classification": "Natural Language Processing",
    "development_stage_short": "Deployed",
    "problem_solved": "Staff spend hours reading long policy documents.",
    "benefits": "Reduces document review time by 60%.",
    "system_outputs": "Summaries of policy documents with key points highlighted.",
    "data_description": "Internal policy documents",
    "has_custom_code": "Yes",
    "contracting_usage_short": "No contracting",
})

MOCK_RESPONSE_JSON = {
    "approach_family": "LLM-based approach",
    "capability_pattern": "Long-context document reasoning",
    "verdict": "Recommended",
    "reasoning": "This is a classic document summarization task requiring long-context reasoning over policy documents.",
    "integration_pattern": "Process policy documents through LLM API with structured output extraction.",
    "risks": "May need fine-tuning for agency-specific terminology. Cost at scale for large document volumes.",
    "hybrid_decomposition": None,
    "rights_impacting": False,
    "proposed_eval_dimensions": ["Summary accuracy vs human baseline", "Key point coverage"],
    "notes": None,
}


def test_build_user_message() -> None:
    """_build_user_message includes key fields in formatted output."""
    from fedai_atlas.claude.fit_analyzer import _build_user_message

    msg = _build_user_message(SAMPLE_ROW)
    assert "TEST-1" in msg
    assert "Document Summarization Tool" in msg
    assert "Staff spend hours" in msg
    assert "Reduces document review" in msg


def test_build_user_message_handles_nulls() -> None:
    """_build_user_message skips null/nan fields."""
    from fedai_atlas.claude.fit_analyzer import _build_user_message

    row = pd.Series({
        "id": "TEST-2",
        "use_case_name": "Minimal Case",
        "problem_solved": None,
        "benefits": float("nan"),
    })
    msg = _build_user_message(row)
    assert "TEST-2" in msg
    assert "Minimal Case" in msg
    assert "Problem Solved" not in msg
    assert "Benefits" not in msg


def test_fit_analysis_model_validates() -> None:
    """FitAnalysis pydantic model validates correct data."""
    from fedai_atlas.claude.fit_analyzer import FitAnalysis

    analysis = FitAnalysis(
        use_case_id="TEST-1",
        use_case_name="Test Case",
        **MOCK_RESPONSE_JSON,
    )
    assert analysis.approach_family == "LLM-based approach"
    assert analysis.verdict == "Recommended"
    assert analysis.capability_pattern == "Long-context document reasoning"


def test_fit_analysis_rejects_cross_family_pattern() -> None:
    """FitAnalysis rejects a pattern not valid for the chosen family."""
    from fedai_atlas.claude.fit_analyzer import FitAnalysis

    with pytest.raises(Exception):
        FitAnalysis(
            use_case_id="TEST-1",
            use_case_name="Test Case",
            **{**MOCK_RESPONSE_JSON, "approach_family": "Classical machine learning"},
        )


def test_analyze_use_case_from_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """analyze_use_case_from_row calls Claude and returns FitAnalysis."""
    from unittest.mock import MagicMock

    from fedai_atlas.claude import fit_analyzer

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    # Mock the client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(MOCK_RESPONSE_JSON))]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    monkeypatch.setattr(fit_analyzer, "get_client", lambda: mock_client)

    from fedai_atlas.config import get_settings

    get_settings.cache_clear()

    result = fit_analyzer.analyze_use_case_from_row(SAMPLE_ROW)
    assert result.use_case_id == "TEST-1"
    assert result.approach_family == "LLM-based approach"
    assert result.capability_pattern == "Long-context document reasoning"

    # Verify prompt caching was used
    call_kwargs = mock_client.messages.create.call_args
    system_arg = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
    assert system_arg[0]["cache_control"] == {"type": "ephemeral"}
