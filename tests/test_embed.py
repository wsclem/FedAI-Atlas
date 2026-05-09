"""Tests for the embedding module."""

from __future__ import annotations

import pandas as pd


def test_build_embed_text_full_row() -> None:
    """build_embed_text produces expected format with all fields present."""
    from fedai_atlas.ingest.embed import build_embed_text

    row = pd.Series({
        "use_case_name": "Fraud Detection System",
        "agency_name": "Dept of Treasury",
        "topic_area_short": "Financial",
        "development_stage_short": "Deployed",
        "problem_solved": "Detect fraudulent transactions in real time.",
        "benefits": "Reduced fraud losses by 40%.",
        "system_outputs": "Risk scores for each transaction.",
    })
    text = build_embed_text(row)
    assert text.startswith("Fraud Detection System")
    assert "Agency: Dept of Treasury" in text
    assert "Topic: Financial" in text
    assert "Stage: Deployed" in text
    assert "Problem: Detect fraudulent" in text
    assert "Benefits: Reduced fraud" in text
    assert "Outputs: Risk scores" in text


def test_build_embed_text_handles_nulls() -> None:
    """build_embed_text skips null/missing fields gracefully."""
    from fedai_atlas.ingest.embed import build_embed_text

    row = pd.Series({
        "use_case_name": "Simple Tool",
        "problem_solved": None,
        "benefits": "",
        "system_outputs": "Reports",
    })
    text = build_embed_text(row)
    assert "Simple Tool" in text
    assert "Problem" not in text
    assert "Benefits" not in text
    assert "Outputs: Reports" in text


def test_embed_texts_batches(monkeypatch: "pytest.MonkeyPatch") -> None:
    """embed_texts calls Voyage API in batches and collects vectors."""
    import pytest  # noqa: F811

    from fedai_atlas.ingest import embed as embed_mod

    calls: list[int] = []

    class FakeResult:
        def __init__(self, n: int) -> None:
            self.embeddings = [[0.1] * 3 for _ in range(n)]

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def embed(self, texts: list[str], **kwargs: object) -> FakeResult:
            calls.append(len(texts))
            return FakeResult(len(texts))

    monkeypatch.setattr(embed_mod, "voyageai", type("M", (), {"Client": FakeClient})())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "va-test")

    from fedai_atlas.config import get_settings

    get_settings.cache_clear()

    # Monkeypatch sleep to avoid real delays in tests
    monkeypatch.setattr(embed_mod.time, "sleep", lambda _: None)

    texts = [f"text {i}" for i in range(50)]
    vectors = embed_mod.embed_texts(texts, model="voyage-3")
    assert len(vectors) == 50
    # Should have made 3 batches: 20 + 20 + 10
    assert calls == [20, 20, 10]
