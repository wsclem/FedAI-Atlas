# Eval Gold Datasets

Hand-labeled gold records for evaluating the architecture decomposer. Each record pairs a use case ID with its correct approach family, capability pattern, and verdict.

## Taxonomy (two-level)

**Approach families**: LLM-based · Classical ML · Computer vision · Time-series · Operations research · Speech/audio · Specialized NLP · Rules-based · Hybrid · Not AI/ML

**Verdicts**: Recommended · Recommended with caveats · Not recommended

Each family has specific capability patterns (see `src/fedai_atlas/claude/fit_analyzer.py` for the full list).

## Gold CSV columns

`use_case_id, approach_family, capability_pattern, verdict`

## Target distribution

- ~30-35 LLM-based cases
- ~5-8 classical ML
- ~3-5 computer vision
- ~3-5 time-series
- ~3-5 operations research
- ~3-5 hybrid (architecture decomposition demonstrations)
- ~2-3 not-AI/ML

Gold records are hand-labeled. Place dataset files (CSV or JSON) alongside this README.
