"""Tests for the fetch module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture()
def _mock_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Patch get_settings to use tmp_path as data_dir."""
    from fedai_atlas.config import Settings, get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = Settings(
        _env_file=None,
        anthropic_api_key="sk-test",
        data_dir=tmp_path,
        inventory_url="https://example.com/inventory.csv",
        cots_url="https://example.com/cots.csv",
    )
    monkeypatch.setattr("fedai_atlas.ingest.fetch.get_settings", lambda: settings)


def test_download_skips_existing(tmp_path: Path) -> None:
    """_download returns early without a network call when the file already exists."""
    from fedai_atlas.ingest.fetch import _download

    dest = tmp_path / "test.csv"
    dest.write_text("existing data")

    result = _download("https://example.com/test.csv", dest, force=False)
    assert result == dest
    assert dest.read_text() == "existing data"


def test_download_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_download re-downloads when force=True even if file exists."""
    from fedai_atlas.ingest.fetch import _download

    dest = tmp_path / "test.csv"
    dest.write_text("old data")

    # Mock httpx.stream
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_bytes.return_value = [b"new data"]
    mock_response.__enter__ = lambda self: self
    mock_response.__exit__ = MagicMock(return_value=False)

    mock_stream = MagicMock(return_value=mock_response)
    monkeypatch.setattr("fedai_atlas.ingest.fetch.httpx.stream", mock_stream)

    result = _download("https://example.com/test.csv", dest, force=True)
    assert result == dest
    assert dest.read_text() == "new data"


@pytest.mark.usefixtures("_mock_settings")
def test_load_inventory_reads_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_inventory returns a DataFrame from a local CSV."""
    from fedai_atlas.ingest.fetch import INDIVIDUAL_FILENAME, load_inventory

    csv_path = tmp_path / INDIVIDUAL_FILENAME
    csv_path.write_text("agency,use_case_name\nDOE,Test Case\n")

    df = load_inventory()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["agency"] == "DOE"
