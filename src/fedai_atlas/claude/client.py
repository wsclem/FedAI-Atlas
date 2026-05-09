"""Shared Anthropic client wrapper."""

from __future__ import annotations

from functools import lru_cache

import anthropic

from fedai_atlas.config import get_settings


@lru_cache
def get_client() -> anthropic.Anthropic:
    """Return a configured Anthropic client (cached)."""
    settings = get_settings()
    return anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )
