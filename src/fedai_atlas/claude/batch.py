"""Anthropic Message Batches API integration for bulk analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from fedai_atlas.claude.client import get_client
from fedai_atlas.claude.fit_analyzer import (
    FitAnalysis,
    _build_user_message,
    _get_system_prompt,
)
from fedai_atlas.config import get_settings

BATCH_JOBS_PATH = Path("data") / "batch_jobs.json"


class BatchJob(BaseModel):
    """Tracked batch job metadata."""

    batch_id: str
    submitted_at: str
    model: str
    count: int
    status: str = "submitted"


def _load_batch_jobs() -> list[dict]:
    """Load batch job tracking file."""
    if BATCH_JOBS_PATH.exists():
        return json.loads(BATCH_JOBS_PATH.read_text(encoding="utf-8"))
    return []


def _save_batch_jobs(jobs: list[dict]) -> None:
    """Save batch job tracking file."""
    BATCH_JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BATCH_JOBS_PATH.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def submit_batch(df: pd.DataFrame, *, model: str | None = None) -> str:
    """Submit all rows as a Message Batch. Returns batch_id."""
    settings = get_settings()
    if model is None:
        model = settings.model_sonnet

    system_prompt = _get_system_prompt()

    requests = []
    for _, row in df.iterrows():
        user_message = _build_user_message(row)
        requests.append({
            "custom_id": str(row.get("id", "")),
            "params": {
                "model": model,
                "max_tokens": 4096,
                "system": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": user_message}],
            },
        })

    client = get_client()
    batch = client.messages.batches.create(requests=requests)

    # Track the job
    jobs = _load_batch_jobs()
    jobs.append({
        "batch_id": batch.id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "count": len(requests),
        "status": "submitted",
    })
    _save_batch_jobs(jobs)

    return batch.id


def get_batch_status(batch_id: str) -> dict:
    """Check batch status. Returns status dict."""
    client = get_client()
    batch = client.messages.batches.retrieve(batch_id)
    return {
        "id": batch.id,
        "status": batch.processing_status,
        "request_counts": {
            "succeeded": batch.request_counts.succeeded,
            "errored": batch.request_counts.errored,
            "canceled": batch.request_counts.canceled,
            "expired": batch.request_counts.expired,
            "processing": batch.request_counts.processing,
        },
        "created_at": batch.created_at.isoformat() if batch.created_at else None,
        "ended_at": batch.ended_at.isoformat() if batch.ended_at else None,
    }


def retrieve_batch_results(batch_id: str) -> list[dict]:
    """Stream results from completed batch. Returns list of {custom_id, result_data}."""
    client = get_client()
    results = []

    for entry in client.messages.batches.results(batch_id):
        custom_id = entry.custom_id
        if entry.result.type == "succeeded":
            raw_text = entry.result.message.content[0].text.strip()
            # Handle markdown fences
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            try:
                data = json.loads(raw_text)
                results.append({"custom_id": custom_id, "data": data, "status": "succeeded"})
            except json.JSONDecodeError as e:
                results.append({"custom_id": custom_id, "error": f"JSON parse error: {e}", "status": "parse_error"})
        else:
            results.append({"custom_id": custom_id, "error": str(entry.result), "status": "failed"})

    return results


def save_batch_results(batch_id: str, output_path: Path | None = None) -> tuple[int, int]:
    """Retrieve results, validate, save to JSON. Returns (success_count, fail_count)."""
    if output_path is None:
        output_path = Path("data") / "fit_analyses.json"

    raw_results = retrieve_batch_results(batch_id)

    # Load existing analyses
    existing = []
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    existing_ids = {a["use_case_id"] for a in existing}

    succeeded = 0
    failed = 0

    for entry in raw_results:
        if entry["status"] != "succeeded":
            failed += 1
            continue

        custom_id = entry["custom_id"]
        data = entry["data"]

        try:
            analysis = FitAnalysis(
                use_case_id=custom_id,
                use_case_name=data.pop("use_case_name", custom_id),
                **data,
            )
            # Remove old entry if exists
            existing = [a for a in existing if a["use_case_id"] != custom_id]
            existing.append(analysis.model_dump())
            succeeded += 1
        except Exception:
            failed += 1

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    return succeeded, failed
