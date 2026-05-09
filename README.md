# FedAI Atlas

AI/ML architecture analysis over OMB's 2025 Federal Agency AI Use Case Inventory — surfacing which use cases are well-suited to LLMs, classical ML, computer vision, time-series methods, operations research, or other approach families, and which might be better served by non-AI investments entirely.

> 🚧 **Under construction** — this project is in active early development.

## What it does

FedAI Atlas decomposes 3,611 federal AI use cases into their most appropriate AI/ML approach families using a two-level taxonomy:

1. **Approach family** — LLM-based, classical ML, computer vision, time-series, operations research, speech/audio, specialized NLP, rules-based, hybrid, or not-AI/ML
2. **Capability pattern** — specific technique within each family (e.g., "Anomaly detection on structured signals" within classical ML)

The system produces per-use-case architecture analyses and per-agency portfolio briefs, identifying opportunities, gaps, and mismatches across approach families.

## Implementation

The reasoning runtime uses Claude (Anthropic) as the LLM for architecture decomposition. The methodology is vendor-agnostic — the taxonomy and routing logic apply regardless of which frontier LLM performs the analysis. Claude is an implementation choice, not the project's identity.

**Stack**: Python 3.11+, uv, Typer CLI, SQLite + LanceDB (dual-store retrieval), Voyage AI embeddings, Streamlit UI, Pydantic for structured outputs.

## Usage

```bash
uv run fedai ingest          # Download, normalize, store in SQLite
uv run fedai embed           # Generate embeddings → LanceDB
uv run fedai search "query"  # Semantic search
uv run fedai analyze NASA-1  # Architecture analysis on a use case
uv run fedai brief NASA      # Agency portfolio brief
uv run fedai ui              # Launch Streamlit dashboard
```

## Acknowledgments

This project builds on the [2025 Federal Agency AI Use Case Inventory](https://github.com/ombegov/2025-Federal-Agency-AI-Use-Case-Inventory) published by the OMB Digital Corps team.
