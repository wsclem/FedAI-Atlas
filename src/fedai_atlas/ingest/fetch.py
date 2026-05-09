"""Fetch the OMB AI Use Case Inventory CSVs from GitHub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from rich.console import Console

from fedai_atlas.config import get_settings

console = Console()

INDIVIDUAL_FILENAME = "2025_individually_reported_AI_use_cases.csv"
COTS_FILENAME = "2025_consolidated_COTS_AI_use_cases.csv"
DATA_DICT_FILENAME = "data_dictionary.json"


def _download(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download a file from *url* to *dest*. Skips if *dest* exists unless *force*."""
    if dest.exists() and not force:
        console.print(f"[dim]Skipping (already exists):[/dim] {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Downloading[/bold] {url}")

    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)

    console.print(f"[green]✓[/green] Saved to {dest}")
    return dest


def fetch_inventory(*, force: bool = False) -> Path:
    """Download the individual use cases CSV. Returns the local path."""
    settings = get_settings()
    dest = settings.data_dir / INDIVIDUAL_FILENAME
    return _download(settings.inventory_url, dest, force=force)


def fetch_cots(*, force: bool = False) -> Path:
    """Download the consolidated COTS CSV. Returns the local path."""
    settings = get_settings()
    dest = settings.data_dir / COTS_FILENAME
    return _download(settings.cots_url, dest, force=force)


def load_inventory(*, force: bool = False) -> pd.DataFrame:
    """Load the individual use cases CSV as a DataFrame. Fetches if missing."""
    path = fetch_inventory(force=force)
    return pd.read_csv(path)


def load_cots(*, force: bool = False) -> pd.DataFrame:
    """Load the COTS CSV as a DataFrame. Fetches if missing."""
    path = fetch_cots(force=force)
    return pd.read_csv(path)


def fetch_data_dictionary(*, force: bool = False) -> Path:
    """Download the data dictionary JSON. Returns the local path."""
    settings = get_settings()
    dest = settings.data_dir / DATA_DICT_FILENAME
    return _download(settings.data_dictionary_url, dest, force=force)


def load_data_dictionary(*, force: bool = False) -> dict[str, Any]:
    """Load the data dictionary as parsed JSON. Fetches if missing."""
    path = fetch_data_dictionary(force=force)
    return json.loads(path.read_text(encoding="utf-8"))
