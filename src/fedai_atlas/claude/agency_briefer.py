"""Generate per-agency AI portfolio briefs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from fedai_atlas.claude.client import get_client
from fedai_atlas.config import get_settings
from fedai_atlas.store.structured import query_by_agency

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "agency_briefer.md"
MAX_USE_CASES_IN_CONTEXT = 100


class NotableUseCase(BaseModel):
    """A notable use case highlighted in the brief."""

    id: str
    name: str
    why_notable: str


class AgencyBrief(BaseModel):
    """Structured output from the agency briefer."""

    agency: str
    agency_name: str
    total_use_cases: int
    executive_summary: str
    portfolio_overview: str
    approach_family_distribution: str = ""
    key_themes: list[str]
    notable_use_cases: list[NotableUseCase]
    architecture_recommendations: str
    maturity_assessment: str


# Cache the system prompt
_system_prompt: str | None = None


def _get_system_prompt() -> str:
    """Load the system prompt from the markdown file (cached)."""
    global _system_prompt  # noqa: PLW0603
    if _system_prompt is None:
        _system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return _system_prompt


def _compute_agency_stats(df: pd.DataFrame) -> str:
    """Compute structured statistics for the agency."""
    lines = [f"Total use cases: {len(df)}"]

    # Development stage
    if "development_stage_short" in df.columns:
        stage_counts = df["development_stage_short"].value_counts()
        lines.append("\nDevelopment Stage Distribution:")
        for stage, count in stage_counts.items():
            lines.append(f"  - {stage}: {count}")

    # Topic areas
    if "topic_area_short" in df.columns:
        topic_counts = df["topic_area_short"].value_counts()
        lines.append("\nTopic Areas:")
        for topic, count in topic_counts.items():
            if str(topic) != "nan":
                lines.append(f"  - {topic}: {count}")

    # Classification
    if "classification_short" in df.columns:
        class_counts = df["classification_short"].value_counts()
        lines.append("\nAI Classification:")
        for cls, count in class_counts.items():
            if str(cls) != "nan":
                lines.append(f"  - {cls}: {count}")

    # High-impact
    if "is_high_impact_short" in df.columns:
        hi_counts = df["is_high_impact_short"].value_counts()
        lines.append("\nHigh-Impact AI:")
        for val, count in hi_counts.items():
            if str(val) != "nan":
                lines.append(f"  - {val}: {count}")

    # Bureaus
    if "agency_bureau" in df.columns:
        bureau_counts = df["agency_bureau"].value_counts()
        lines.append(f"\nBureaus/Components ({len(bureau_counts)} total):")
        for bureau, count in bureau_counts.head(10).items():
            if str(bureau) != "nan":
                lines.append(f"  - {bureau}: {count}")
        if len(bureau_counts) > 10:
            lines.append(f"  - ... and {len(bureau_counts) - 10} more")

    return "\n".join(lines)


def _build_use_case_table(df: pd.DataFrame, max_cases: int = MAX_USE_CASES_IN_CONTEXT) -> str:
    """Build a condensed table of use cases for context."""
    sampled = df.head(max_cases) if len(df) <= max_cases else df.sample(n=max_cases, random_state=42)

    lines = []
    if len(df) > max_cases:
        lines.append(
            f"(Showing {max_cases} of {len(df)} use cases — representative sample)\n"
        )

    for _, row in sampled.iterrows():
        use_id = row.get("id", "?")
        name = str(row.get("use_case_name", "")).strip()
        problem = str(row.get("problem_solved", "")).strip()
        if problem == "nan":
            problem = ""
        # Truncate problem to ~150 chars
        if len(problem) > 150:
            problem = problem[:147] + "..."

        stage = row.get("development_stage_short", "")
        topic = row.get("topic_area_short", "")
        classification = row.get("classification_short", "")

        entry = f"- **{use_id}**: {name}"
        meta_parts = []
        if stage and str(stage) != "nan":
            meta_parts.append(str(stage))
        if topic and str(topic) != "nan":
            meta_parts.append(str(topic))
        if classification and str(classification) != "nan":
            meta_parts.append(str(classification))
        if meta_parts:
            entry += f" [{', '.join(meta_parts)}]"
        if problem:
            entry += f"\n  {problem}"
        lines.append(entry)

    return "\n".join(lines)


def _build_agency_context(agency: str) -> str:
    """Build the full user message with stats + use case summaries."""
    df = query_by_agency(agency)
    if df.empty:
        msg = f"No use cases found for agency: {agency}"
        raise ValueError(msg)

    agency_name = df.iloc[0].get("agency_name", agency)
    stats = _compute_agency_stats(df)
    table = _build_use_case_table(df)

    return (
        f"Generate an AI portfolio brief for: **{agency_name} ({agency})**\n\n"
        f"## Portfolio Statistics\n\n{stats}\n\n"
        f"## Use Cases\n\n{table}"
    )


def brief_agency(agency: str, *, model: str | None = None) -> AgencyBrief:
    """Generate a full agency AI portfolio brief via Claude."""
    settings = get_settings()
    if model is None:
        model = settings.model_sonnet

    df = query_by_agency(agency)
    if df.empty:
        msg = f"No use cases found for agency: {agency}"
        raise ValueError(msg)

    agency_name = str(df.iloc[0].get("agency_name", agency))

    client = get_client()
    system_prompt = _get_system_prompt()
    user_message = _build_agency_context(agency)

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = response.content[0].text.strip()

    # Parse JSON — handle potential markdown fences
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    data: dict[str, Any] = json.loads(raw_text)

    return AgencyBrief(
        agency=agency,
        agency_name=agency_name,
        total_use_cases=len(df),
        **data,
    )
