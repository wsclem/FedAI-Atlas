"""Smoke test: verify the Anthropic client can be instantiated."""

from __future__ import annotations

import pytest


def test_anthropic_client_instantiates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic client can be created with a test API key (no actual API call)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    import anthropic

    from fedai_atlas.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )
    assert client is not None
