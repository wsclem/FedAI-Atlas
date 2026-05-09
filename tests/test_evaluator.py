"""Tests for the evaluator module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


SAMPLE_SEED_MD = """## Analysis 1: Source Code Development Tool - DHS-2452

**Source.**
Agency: DHS. Use case ID: DHS-2452
Stage: Pre-Deployment
Impact level: Not High-impact

**Verdict.**
A strong LLM fit.

**Capability category.**
Coding assistance.

**Reasoning.**
Standard coding assistant use case.

**Top risks.**
Bad code or insecure code.
"""

MOCK_EVAL_JSON = {
    "verdict_accuracy": 5,
    "category_accuracy": 4,
    "reasoning_quality": 3,
    "completeness": 3,
    "risk_awareness": 2,
    "overall": 0.70,
    "commentary": "The automated analysis correctly identifies this as a strong fit but lacks specificity on risks.",
}


def test_parse_seed_analyses(tmp_path: Path) -> None:
    """parse_seed_analyses extracts use case IDs and text."""
    from fedai_atlas.claude.evaluator import parse_seed_analyses

    seed_file = tmp_path / "seeds.md"
    seed_file.write_text(SAMPLE_SEED_MD)

    seeds = parse_seed_analyses(seed_file)
    assert "DHS-2452" in seeds
    assert "strong LLM fit" in seeds["DHS-2452"]


def test_parse_seed_analyses_missing_file(tmp_path: Path) -> None:
    """parse_seed_analyses returns empty dict for missing file."""
    from fedai_atlas.claude.evaluator import parse_seed_analyses

    seeds = parse_seed_analyses(tmp_path / "nonexistent.md")
    assert seeds == {}


def test_evaluate_analysis_no_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluate_analysis raises if no seed exists."""
    from fedai_atlas.claude import evaluator

    monkeypatch.setattr(evaluator, "parse_seed_analyses", lambda: {})

    with pytest.raises(ValueError, match="No seed analysis found"):
        evaluator.evaluate_analysis("FAKE-001")


def test_evaluate_analysis_no_automated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """evaluate_analysis raises if no automated analysis exists."""
    from fedai_atlas.claude import evaluator

    monkeypatch.setattr(evaluator, "parse_seed_analyses", lambda: {"DHS-2452": "seed text"})
    monkeypatch.setattr(evaluator, "FIT_ANALYSES_PATH", tmp_path / "missing.json")

    with pytest.raises(ValueError, match="No automated analysis found"):
        evaluator.evaluate_analysis("DHS-2452")


def test_evaluate_analysis_with_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """evaluate_analysis calls Claude and returns EvalResult."""
    from unittest.mock import MagicMock

    from fedai_atlas.claude import evaluator

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    # Mock seeds
    monkeypatch.setattr(evaluator, "parse_seed_analyses", lambda: {"DHS-2452": SAMPLE_SEED_MD})

    # Mock automated analysis file
    fit_path = tmp_path / "fit_analyses.json"
    fit_path.write_text(json.dumps([{
        "use_case_id": "DHS-2452",
        "use_case_name": "Source Code Development Tool",
        "capability_category": "Software dev / Claude Code - generation",
        "fit_verdict": "Strong fit",
        "confidence": 0.9,
        "reasoning": "Standard coding tool.",
        "claude_approach": "Use Claude Code.",
        "limitations": "None significant.",
        "recommended_model": "sonnet",
        "recommended_products": ["Claude Code"],
    }]))
    monkeypatch.setattr(evaluator, "FIT_ANALYSES_PATH", fit_path)

    # Mock query_by_id
    import pandas as pd
    monkeypatch.setattr(
        "fedai_atlas.store.structured.query_by_id",
        lambda _: pd.DataFrame({"id": ["DHS-2452"], "use_case_name": ["Source Code Development Tool"]}),
    )

    # Mock Claude client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=json.dumps(MOCK_EVAL_JSON))]
    mock_response.usage.input_tokens = 500
    mock_response.usage.output_tokens = 100
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    monkeypatch.setattr(evaluator, "get_client", lambda: mock_client)

    from fedai_atlas.config import get_settings
    get_settings.cache_clear()

    result = evaluator.evaluate_analysis("DHS-2452", show_tokens=True)
    assert result.use_case_id == "DHS-2452"
    assert result.scores.verdict_accuracy == 5
    assert result.scores.overall == 0.70
    assert "correctly identifies" in result.scores.commentary
    assert result.token_usage is not None
    assert result.token_usage.input_tokens == 500
