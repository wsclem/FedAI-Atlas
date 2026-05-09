"""Analyze AI/ML architecture fit for a given federal AI use case."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from fedai_atlas.claude.client import get_client
from fedai_atlas.config import get_settings
from fedai_atlas.store.structured import query_by_id

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "fit_analyzer.md"


class ApproachFamily(str, Enum):
    """Top-level AI/ML approach families."""

    LLM_BASED = "LLM-based approach"
    CLASSICAL_ML = "Classical machine learning"
    COMPUTER_VISION = "Computer vision"
    TIME_SERIES = "Time-series and forecasting"
    OPERATIONS_RESEARCH = "Operations research / optimization"
    SPEECH_AUDIO = "Speech and audio processing"
    SPECIALIZED_NLP = "Specialized NLP (non-LLM)"
    RULES_BASED = "Rules-based / deterministic systems"
    NOT_AI_ML = "Not appropriate for AI/ML"


CAPABILITY_PATTERNS: dict[str, list[str]] = {
    ApproachFamily.LLM_BASED: [
        "Long-context document reasoning",
        "Structured extraction",
        "Drafting / generation",
        "Agentic workflow (non-coding)",
        "Software dev — agentic code generation",
        "Software dev — legacy modernization",
        "Software dev — code review and security analysis",
        "Software dev — automated testing and documentation",
        "Classification with nuance",
        "RAG / grounded QA",
    ],
    ApproachFamily.CLASSICAL_ML: [
        "Tabular prediction (classification or regression)",
        "Anomaly detection on structured signals",
        "Recommendation / ranking",
        "Clustering and segmentation",
    ],
    ApproachFamily.COMPUTER_VISION: [
        "Image classification",
        "Object detection and tracking",
        "Document image processing (OCR-plus-reasoning)",
        "Geospatial / satellite imagery",
        "Medical imaging",
    ],
    ApproachFamily.TIME_SERIES: [
        "Statistical forecasting",
        "Time-series ML / deep learning forecasting",
        "Streaming anomaly detection",
    ],
    ApproachFamily.OPERATIONS_RESEARCH: [
        "Linear / mixed-integer programming",
        "Routing and scheduling",
        "Simulation and what-if analysis",
    ],
    ApproachFamily.SPEECH_AUDIO: [
        "Speech recognition",
        "Speaker identification / diarization",
        "Audio classification",
    ],
    ApproachFamily.SPECIALIZED_NLP: [
        "Translation (high-volume, well-resourced language pairs)",
        "Named entity recognition at scale",
        "Topic modeling at scale",
    ],
    ApproachFamily.RULES_BASED: [
        "Rules engine for eligibility / compliance",
        "Workflow automation without AI",
    ],
    ApproachFamily.NOT_AI_ML: [
        "Data quality / process problem",
        "Human judgment problem",
        "Traditional software engineering problem",
    ],
}

# Flat set of all valid patterns for validation
ALL_PATTERNS: set[str] = set()
for patterns in CAPABILITY_PATTERNS.values():
    ALL_PATTERNS.update(patterns)

VERDICTS = ["Recommended", "Recommended with caveats", "Not recommended"]


class TokenUsage(BaseModel):
    """Token counts from a Claude API call."""

    input_tokens: int
    output_tokens: int


class HybridComponent(BaseModel):
    """A component within a hybrid architecture decomposition."""

    family: str
    pattern: str
    role: str
    justification: str


class FitAnalysis(BaseModel):
    """Structured output from the architecture decomposer."""

    use_case_id: str
    use_case_name: str
    approach_family: str
    capability_pattern: str
    verdict: str
    reasoning: str
    integration_pattern: str
    risks: str
    hybrid_decomposition: list[HybridComponent] | None = None
    rights_impacting: bool | None = None
    proposed_eval_dimensions: list[str] = Field(default_factory=list)
    notes: str | None = None
    token_usage: TokenUsage | None = None

    @model_validator(mode="after")
    def validate_pattern_for_family(self) -> FitAnalysis:
        """Ensure capability_pattern is valid for the chosen approach_family."""
        family_str = self.approach_family
        pattern = self.capability_pattern

        # Find matching family
        matched_family = None
        for fam in ApproachFamily:
            if fam.value == family_str:
                matched_family = fam
                break

        if matched_family is None:
            return self  # let pydantic handle unknown family values

        # Check pattern is valid for this family
        valid_patterns = CAPABILITY_PATTERNS.get(matched_family, [])
        if valid_patterns and pattern not in valid_patterns:
            msg = (
                f"capability_pattern '{pattern}' is not valid for "
                f"approach_family '{family_str}'. "
                f"Valid patterns: {valid_patterns}"
            )
            raise ValueError(msg)

        return self


# Cache the system prompt text
_system_prompt: str | None = None


def _get_system_prompt() -> str:
    """Load the system prompt from the markdown file (cached in memory)."""
    global _system_prompt  # noqa: PLW0603
    if _system_prompt is None:
        _system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return _system_prompt


def _build_user_message(row: pd.Series) -> str:
    """Format use case fields into the user message."""
    fields = {
        "Use Case ID": row.get("id", ""),
        "Use Case Name": row.get("use_case_name", ""),
        "Agency": row.get("agency_name", ""),
        "Bureau": row.get("agency_bureau", ""),
        "Topic Area": row.get("topic_area_short", row.get("topic_area", "")),
        "Classification": row.get("classification", ""),
        "Development Stage": row.get("development_stage_short", ""),
        "Problem Solved": row.get("problem_solved", ""),
        "Benefits": row.get("benefits", ""),
        "System Outputs": row.get("system_outputs", ""),
        "Data Description": row.get("data_description", ""),
        "Has Custom Code": row.get("has_custom_code", ""),
        "Contracting Usage": row.get("contracting_usage_short", row.get("contracting_usage", "")),
    }

    lines = []
    for label, val in fields.items():
        val_str = str(val).strip() if val is not None and str(val) != "nan" else ""
        if val_str:
            lines.append(f"**{label}**: {val_str}")

    return "Analyze this federal AI use case:\n\n" + "\n".join(lines)


def analyze_use_case(use_case_id: str, *, model: str | None = None, show_tokens: bool = False) -> FitAnalysis:
    """Load a use case from SQLite and run architecture analysis."""
    result = query_by_id(use_case_id)
    if result.empty:
        msg = f"Use case not found: {use_case_id}"
        raise ValueError(msg)
    return analyze_use_case_from_row(result.iloc[0], model=model, show_tokens=show_tokens)


def analyze_use_case_from_row(
    row: pd.Series, *, model: str | None = None, show_tokens: bool = False, _retries: int = 1
) -> FitAnalysis:
    """Run architecture analysis on a pre-loaded use case row.

    Retries once on validation error (e.g., model paraphrases a pattern).
    """
    settings = get_settings()
    if model is None:
        model = settings.model_sonnet

    client = get_client()
    system_prompt = _get_system_prompt()
    user_message = _build_user_message(row)

    response = client.messages.create(
        model=model,
        max_tokens=4096,
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

    data = json.loads(raw_text)

    token_usage = None
    if show_tokens:
        token_usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    try:
        return FitAnalysis(
            use_case_id=str(row.get("id", "")),
            use_case_name=str(row.get("use_case_name", "")),
            token_usage=token_usage,
            **data,
        )
    except (ValueError, Exception) as e:
        if _retries > 0:
            # Retry — the strengthened prompt usually fixes paraphrasing on second attempt
            return analyze_use_case_from_row(row, model=model, show_tokens=show_tokens, _retries=_retries - 1)
        raise
