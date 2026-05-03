"""CLI entry point for FedAI Atlas."""

from __future__ import annotations

import typer
from rich import print as rprint

app = typer.Typer(name="fedai", help="FedAI Atlas CLI")


@app.command()
def hello() -> None:
    """Smoke test: verify config loads and Anthropic client can be instantiated."""
    from fedai_atlas.config import get_settings

    settings = get_settings()
    rprint("[green]✓[/green] Config loaded successfully")
    rprint(f"  Opus model:   {settings.model_opus}")
    rprint(f"  Sonnet model: {settings.model_sonnet}")
    rprint(f"  Haiku model:  {settings.model_haiku}")

    import anthropic

    _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    rprint("[green]✓[/green] Anthropic client instantiated (no API call made)")


if __name__ == "__main__":
    app()
