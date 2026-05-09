"""Tests for configuration loading."""

from __future__ import annotations

import pytest


def test_config_loads_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings loads successfully when ANTHROPIC_API_KEY is set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    from fedai_atlas.config import Settings

    settings = Settings()
    assert settings.anthropic_api_key == "sk-test-key"
    assert settings.model_opus == "claude-opus-4-7"
    assert settings.model_sonnet == "claude-sonnet-4-6"
    assert settings.model_haiku == "claude-haiku-4-5-20251001"
    assert settings.embedding_model == "voyage-3"


def test_config_fails_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings raises a validation error when ANTHROPIC_API_KEY is missing."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from fedai_atlas.config import Settings

    with pytest.raises(Exception):  # noqa: B017
        Settings(_env_file=None)
