# Eval Gold Datasets

This folder holds gold datasets for evaluating `fit_analyzer` and `agency_briefer` outputs.

Each gold record pairs a use case (by ID) with a human-assigned **capability category** and **fit verdict**.

## Capability categories

- Software dev / Claude Code - generation
- Software dev / Claude Code - modernization
- Software dev / Claude Code - review & security
- Software dev / Claude Code - test & docs
- Long-context document reasoning
- Structured extraction
- Drafting / generation
- Agentic workflow (non-coding)
- Classification w/ nuance
- RAG / grounded QA
- Hybrid (Claude + classical ML)
- Not a Claude fit

## Fit verdicts

Strong fit · Partial fit (hybrid) · Weak fit · Wrong tool

Gold records are hand-labeled. Place dataset files (CSV or JSON) alongside this README.
