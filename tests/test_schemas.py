"""Tests for schema validation (approach family + capability pattern constraints)."""

from __future__ import annotations

import pytest

from fedai_atlas.claude.fit_analyzer import (
    CAPABILITY_PATTERNS,
    ApproachFamily,
    FitAnalysis,
    HybridComponent,
)


def _base_analysis(**overrides) -> dict:
    """Build a valid FitAnalysis dict with optional overrides."""
    base = {
        "use_case_id": "TEST-1",
        "use_case_name": "Test Case",
        "approach_family": "LLM-based approach",
        "capability_pattern": "Long-context document reasoning",
        "verdict": "Recommended",
        "reasoning": "Test reasoning.",
        "integration_pattern": "Test integration.",
        "risks": "Test risks.",
        "hybrid_decomposition": None,
        "rights_impacting": None,
        "proposed_eval_dimensions": ["dim1"],
        "notes": None,
    }
    base.update(overrides)
    return base


class TestApproachFamilyPatternValidation:
    """Test that capability_pattern must be valid for approach_family."""

    def test_valid_llm_pattern(self) -> None:
        """LLM family with valid LLM pattern passes."""
        a = FitAnalysis(**_base_analysis())
        assert a.approach_family == "LLM-based approach"

    def test_valid_classical_ml_pattern(self) -> None:
        """Classical ML with valid pattern passes."""
        a = FitAnalysis(**_base_analysis(
            approach_family="Classical machine learning",
            capability_pattern="Tabular prediction (classification or regression)",
        ))
        assert a.approach_family == "Classical machine learning"

    def test_valid_cv_pattern(self) -> None:
        """Computer vision with valid pattern passes."""
        a = FitAnalysis(**_base_analysis(
            approach_family="Computer vision",
            capability_pattern="Image classification",
        ))
        assert a.capability_pattern == "Image classification"

    def test_valid_time_series_pattern(self) -> None:
        """Time-series with valid pattern passes."""
        a = FitAnalysis(**_base_analysis(
            approach_family="Time-series and forecasting",
            capability_pattern="Statistical forecasting",
        ))
        assert a.approach_family == "Time-series and forecasting"

    def test_valid_or_pattern(self) -> None:
        """Operations research with valid pattern passes."""
        a = FitAnalysis(**_base_analysis(
            approach_family="Operations research / optimization",
            capability_pattern="Routing and scheduling",
        ))
        assert a.capability_pattern == "Routing and scheduling"

    def test_valid_not_ai_ml_pattern(self) -> None:
        """Not AI/ML with valid pattern passes."""
        a = FitAnalysis(**_base_analysis(
            approach_family="Not appropriate for AI/ML",
            capability_pattern="Data quality / process problem",
        ))
        assert a.approach_family == "Not appropriate for AI/ML"

    def test_rejects_cross_family_pattern(self) -> None:
        """Classical ML family rejects LLM-family pattern."""
        with pytest.raises(ValueError, match="not valid for"):
            FitAnalysis(**_base_analysis(
                approach_family="Classical machine learning",
                capability_pattern="Long-context document reasoning",
            ))

    def test_rejects_cv_pattern_in_llm_family(self) -> None:
        """LLM family rejects computer vision pattern."""
        with pytest.raises(ValueError, match="not valid for"):
            FitAnalysis(**_base_analysis(
                approach_family="LLM-based approach",
                capability_pattern="Image classification",
            ))


class TestHybridDecomposition:
    """Test hybrid decomposition on any family."""

    def test_any_family_accepts_hybrid_decomposition(self) -> None:
        """Any family can have hybrid_decomposition."""
        components = [
            HybridComponent(
                family="Classical machine learning",
                pattern="Anomaly detection on structured signals",
                role="Flag anomalous transactions",
                justification="Structured tabular data",
            ),
        ]
        a = FitAnalysis(**_base_analysis(
            approach_family="LLM-based approach",
            capability_pattern="Classification with nuance",
            hybrid_decomposition=components,
        ))
        assert len(a.hybrid_decomposition) == 1

    def test_family_works_without_decomposition(self) -> None:
        """Hybrid decomposition is optional."""
        a = FitAnalysis(**_base_analysis(
            approach_family="Classical machine learning",
            capability_pattern="Tabular prediction (classification or regression)",
        ))
        assert a.hybrid_decomposition is None


class TestAllFamiliesHavePatterns:
    """Ensure every family has at least one valid pattern."""

    def test_all_families_have_patterns(self) -> None:
        """Every family has defined patterns."""
        for family in ApproachFamily:
            patterns = CAPABILITY_PATTERNS.get(family, [])
            assert len(patterns) > 0, f"{family.value} has no patterns defined"
