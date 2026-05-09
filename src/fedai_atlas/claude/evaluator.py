"""LLM-as-judge evaluator for fit analyzer output."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from fedai_atlas.claude.client import get_client
from fedai_atlas.claude.fit_analyzer import TokenUsage
from fedai_atlas.config import get_settings

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evaluator.md"
SEED_PATH = Path(__file__).parent.parent.parent.parent / "docs" / "seed-analyses.md"
FIT_ANALYSES_PATH = Path("data") / "fit_analyses.json"


class EvalScore(BaseModel):
    """Rubric scores from the LLM judge."""

    verdict_accuracy: int = Field(ge=0, le=5)
    category_accuracy: int = Field(ge=0, le=5)
    reasoning_quality: int = Field(ge=0, le=5)
    completeness: int = Field(ge=0, le=5)
    risk_awareness: int = Field(ge=0, le=5)
    overall: float = Field(ge=0.0, le=1.0)
    commentary: str


class EvalResult(BaseModel):
    """Full evaluation result for a use case."""

    use_case_id: str
    scores: EvalScore
    token_usage: TokenUsage | None = None
    analysis_token_usage: TokenUsage | None = None  # from original fit_analyzer run


# Cache
_system_prompt: str | None = None


def _get_system_prompt() -> str:
    """Load the evaluator system prompt (cached)."""
    global _system_prompt  # noqa: PLW0603
    if _system_prompt is None:
        _system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return _system_prompt


def parse_seed_analyses(path: Path | None = None) -> dict[str, str]:
    """Parse docs/seed-analyses.md into {use_case_id: raw_text} dict.

    Splits on '## Analysis' headers and extracts the use case ID from
    the 'Use case ID:' line in the Source block.
    """
    if path is None:
        path = SEED_PATH

    if not path.exists():
        return {}

    content = path.read_text(encoding="utf-8")

    # Split on ## Analysis headers
    sections = re.split(r"(?=^## Analysis \d+:)", content, flags=re.MULTILINE)

    seeds: dict[str, str] = {}
    for section in sections:
        if not section.strip() or not section.strip().startswith("## Analysis"):
            continue

        # Extract use case ID from "Use case ID: XXX" line
        id_match = re.search(r"Use case ID:\s*(\S+)", section)
        if id_match:
            use_case_id = id_match.group(1).strip()
            seeds[use_case_id] = section.strip()

    return seeds


def _load_automated_analysis(use_case_id: str) -> dict | None:
    """Load the automated analysis for a use case from fit_analyses.json."""
    if not FIT_ANALYSES_PATH.exists():
        return None

    analyses = json.loads(FIT_ANALYSES_PATH.read_text(encoding="utf-8"))
    for a in analyses:
        if a.get("use_case_id") == use_case_id:
            return a
    return None


def _build_user_message(use_case_id: str, seed_text: str, automated: dict) -> str:
    """Build the user message for the judge."""
    from fedai_atlas.store.structured import query_by_id

    # Get original use case data
    result = query_by_id(use_case_id)
    if result.empty:
        use_case_data = "(Original use case data not available)"
    else:
        row = result.iloc[0]
        fields = []
        for col in row.index:
            val = row[col]
            if val is not None and str(val).strip() and str(val) != "nan":
                fields.append(f"**{col}**: {val}")
        use_case_data = "\n".join(fields)

    # Format automated analysis
    auto_lines = []
    for key, val in automated.items():
        if key == "token_usage" or val is None:
            continue
        auto_lines.append(f"**{key}**: {val}")
    automated_text = "\n".join(auto_lines)

    return (
        f"## Original Use Case Data\n\n{use_case_data}\n\n"
        f"## Human Seed Analysis (Gold Standard)\n\n{seed_text}\n\n"
        f"## Automated Analysis (To Evaluate)\n\n{automated_text}"
    )


def evaluate_analysis(
    use_case_id: str, *, model: str | None = None, show_tokens: bool = False
) -> EvalResult:
    """Run LLM-as-judge evaluation on an automated analysis.

    Requires both a seed analysis in docs/seed-analyses.md and an
    automated analysis in data/fit_analyses.json for the given use case.
    """
    settings = get_settings()
    if model is None:
        model = settings.model_sonnet

    # Load seed
    seeds = parse_seed_analyses()
    if use_case_id not in seeds:
        msg = f"No seed analysis found for {use_case_id}. Available: {list(seeds.keys())}"
        raise ValueError(msg)

    # Load automated
    automated = _load_automated_analysis(use_case_id)
    if automated is None:
        msg = f"No automated analysis found for {use_case_id}. Run `fedai analyze {use_case_id}` first."
        raise ValueError(msg)

    client = get_client()
    system_prompt = _get_system_prompt()
    user_message = _build_user_message(use_case_id, seeds[use_case_id], automated)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
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

    # Handle markdown fences
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    data = json.loads(raw_text)

    token_usage = None
    if show_tokens:
        token_usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    # Extract original analysis token usage if available
    analysis_token_usage = None
    if show_tokens and automated.get("token_usage"):
        tu = automated["token_usage"]
        analysis_token_usage = TokenUsage(
            input_tokens=tu["input_tokens"],
            output_tokens=tu["output_tokens"],
        )

    return EvalResult(
        use_case_id=use_case_id,
        scores=EvalScore(**data),
        token_usage=token_usage,
        analysis_token_usage=analysis_token_usage,
    )
