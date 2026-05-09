"""CLI entry point for FedAI Atlas."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

app = typer.Typer(name="fedai", help="FedAI Atlas CLI")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """FedAI Atlas — Claude-powered intelligence over federal AI use cases."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(hello)


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

    _client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
    )
    rprint("[green]✓[/green] Anthropic client instantiated (no API call made)")
    if settings.anthropic_base_url:
        rprint(f"  Base URL:     {settings.anthropic_base_url}")


@app.command()
def fetch(
    force: bool = typer.Option(False, "--force", "-f", help="Re-download even if files exist"),
) -> None:
    """Download the OMB AI Use Case Inventory CSVs."""
    from fedai_atlas.ingest.fetch import fetch_cots, fetch_data_dictionary, fetch_inventory

    fetch_inventory(force=force)
    fetch_cots(force=force)
    fetch_data_dictionary(force=force)


@app.command()
def ingest(
    force: bool = typer.Option(False, "--force", "-f", help="Re-download even if files exist"),
) -> None:
    """Download, normalize, and store the inventory in SQLite."""
    from fedai_atlas.ingest.fetch import load_data_dictionary, load_inventory
    from fedai_atlas.ingest.normalize import normalize_inventory
    from fedai_atlas.store.structured import store_inventory

    rprint("[bold]1/3[/bold] Fetching data...")
    df = load_inventory(force=force)
    data_dict = load_data_dictionary(force=force)

    rprint("[bold]2/3[/bold] Normalizing...")
    df = normalize_inventory(df, data_dict)

    rprint("[bold]3/3[/bold] Storing in SQLite...")
    count = store_inventory(df)
    rprint(f"[green]✓[/green] Stored {count:,} use cases in SQLite")


@app.command()
def embed(
    force: bool = typer.Option(False, "--force", "-f", help="Wipe existing vectors and re-embed all"),
) -> None:
    """Embed all use cases and store vectors in LanceDB (resumable)."""
    from fedai_atlas.ingest.embed import embed_inventory
    from fedai_atlas.store.structured import load_all

    if force:
        from fedai_atlas.store.vector import get_vector_db, TABLE_NAME
        rprint("[yellow]Wiping existing vectors...[/yellow]")
        try:
            db = get_vector_db()
            db.drop_table(TABLE_NAME)
        except Exception:
            pass

    rprint("[bold]1/2[/bold] Loading from SQLite...")
    df = load_all()
    rprint(f"  Loaded {len(df):,} use cases")

    rprint("[bold]2/2[/bold] Embedding + storing to LanceDB (incremental)...")
    count = embed_inventory(df)
    rprint(f"[green]✓[/green] {count:,} vectors in LanceDB")


@app.command()
def lookup(
    use_case_id: str = typer.Argument(..., help="Use case ID (e.g. NASA-1)"),
) -> None:
    """Look up a single use case by ID."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    from fedai_atlas.store.structured import query_by_id

    result = query_by_id(use_case_id)
    if result.empty:
        rprint(f"[red]No use case found with ID '{use_case_id}'[/red]")
        raise typer.Exit(1)

    row = result.iloc[0]
    lines: list[str] = []
    for col in row.index:
        val = row[col]
        if val is not None and str(val).strip() and str(val) != "nan":
            lines.append(f"[bold]{col}[/bold]: {val}")

    Console().print(Panel("\n".join(lines), title=str(row.get("use_case_name", use_case_id))))


@app.command()
def agency(
    agency_abbr: str = typer.Argument(..., help="Agency abbreviation (e.g. NASA, DOE)"),
) -> None:
    """List all use cases for an agency."""
    from rich.console import Console
    from rich.table import Table

    from fedai_atlas.store.structured import query_by_agency

    results = query_by_agency(agency_abbr)
    if results.empty:
        rprint(f"[red]No use cases found for agency '{agency_abbr}'[/red]")
        raise typer.Exit(1)

    table = Table(title=f"{agency_abbr} — {len(results)} use cases", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Use Case", max_width=50)
    table.add_column("Stage", style="dim")
    table.add_column("Topic", style="green")

    for _, row in results.iterrows():
        table.add_row(
            str(row.get("id", "")),
            str(row.get("use_case_name", "")),
            str(row.get("development_stage_short", "")),
            str(row.get("topic_area_short", "")),
        )

    Console().print(table)


@app.command()
def agencies() -> None:
    """List all agencies in the inventory."""
    from fedai_atlas.store.structured import list_agencies

    for abbr in list_agencies():
        rprint(abbr)


@app.command(name="search")
def search_cmd(
    query: str = typer.Argument(..., help="Search query text"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
) -> None:
    """Semantic search across federal AI use cases."""
    from rich.table import Table

    from fedai_atlas.store.vector import semantic_search

    results = semantic_search(query, limit=limit)

    table = Table(title=f"Search: {query!r}", show_lines=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Agency", style="bold")
    table.add_column("Use Case", max_width=50)
    table.add_column("Stage", style="dim")
    table.add_column("Dist", justify="right", style="magenta")

    for _, row in results.iterrows():
        table.add_row(
            str(row.get("id", "")),
            str(row.get("agency", "")),
            str(row.get("use_case_name", "")),
            str(row.get("development_stage_short", "")),
            f"{row.get('_distance', 0):.3f}",
        )

    from rich.console import Console

    Console().print(table)


@app.command()
def analyze(
    use_case_id: str = typer.Argument(..., help="Use case ID (e.g. NASA-1)"),
    model: str = typer.Option(None, "--model", "-m", help="Model override (opus/sonnet/haiku)"),
    tokens: bool = typer.Option(False, "--tokens", "-t", help="Show input/output token counts"),
) -> None:
    """Run architecture analysis on a single use case."""
    from rich.console import Console
    from rich.panel import Panel

    from fedai_atlas.claude.fit_analyzer import analyze_use_case

    console = Console()
    console.print(f"[bold]Analyzing {use_case_id}...[/bold]")

    result = analyze_use_case(use_case_id, model=model, show_tokens=tokens)

    verdict_color = {
        "Recommended": "green",
        "Recommended with caveats": "yellow",
        "Not recommended": "red",
    }.get(result.verdict, "white")

    lines = [
        f"[bold]Approach Family:[/bold] {result.approach_family}",
        f"[bold]Pattern:[/bold] {result.capability_pattern}",
        f"[bold]Verdict:[/bold] [{verdict_color}]{result.verdict}[/{verdict_color}]",
        "",
        f"[bold]Reasoning:[/bold] {result.reasoning}",
        "",
        f"[bold]Integration:[/bold] {result.integration_pattern}",
        "",
        f"[bold]Risks:[/bold] {result.risks}",
    ]

    if result.hybrid_decomposition:
        lines.append("")
        lines.append("[bold]Hybrid Decomposition:[/bold]")
        for comp in result.hybrid_decomposition:
            lines.append(f"  • {comp.family} → {comp.pattern}: {comp.role}")

    if result.proposed_eval_dimensions:
        lines.append("")
        lines.append(f"[bold]Eval Dimensions:[/bold] {', '.join(result.proposed_eval_dimensions)}")

    if result.notes:
        lines.append("")
        lines.append(f"[bold]Notes:[/bold] {result.notes}")

    if result.token_usage:
        lines.append("")
        lines.append(
            f"[bold]Tokens:[/bold] {result.token_usage.input_tokens:,} in / "
            f"{result.token_usage.output_tokens:,} out"
        )

    console.print(Panel("\n".join(lines), title=result.use_case_name))

    # Save/update in fit_analyses.json
    import json as json_mod

    output_path = Path("data") / "fit_analyses.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analyses: list = []
    if output_path.exists():
        analyses = json_mod.loads(output_path.read_text(encoding="utf-8"))
    # Remove existing entry for this use case, then append new
    analyses = [a for a in analyses if a.get("use_case_id") != use_case_id]
    analyses.append(result.model_dump())
    output_path.write_text(json_mod.dumps(analyses, indent=2), encoding="utf-8")
    rprint(f"[green]✓[/green] Saved to {output_path}")


@app.command(name="analyze-batch")
def analyze_batch(
    agency_abbr: str = typer.Option(None, "--agency", "-a", help="Filter by agency"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max use cases to analyze"),
    offset: int = typer.Option(0, "--offset", "--skip", help="Skip first N use cases"),
    model: str = typer.Option(None, "--model", "-m", help="Model override"),
    tokens: bool = typer.Option(False, "--tokens", "-t", help="Show input/output token counts"),
) -> None:
    """Run Claude fit analysis on multiple use cases."""
    import json as json_mod

    from rich.console import Console

    from fedai_atlas.claude.fit_analyzer import analyze_use_case_from_row
    from fedai_atlas.store.structured import load_all, query_by_agency

    console = Console()

    if agency_abbr:
        df = query_by_agency(agency_abbr)
        if df.empty:
            rprint(f"[red]No use cases found for agency '{agency_abbr}'[/red]")
            raise typer.Exit(1)
    else:
        df = load_all()

    df = df.iloc[offset:offset + limit]
    console.print(f"[bold]Analyzing {len(df)} use cases (offset={offset})...[/bold]\n")

    results = []
    total_input = 0
    total_output = 0
    for i, (_, row) in enumerate(df.iterrows(), 1):
        use_case_id = row.get("id", "?")
        console.print(f"[dim][{i}/{len(df)}][/dim] {use_case_id}: {row.get('use_case_name', '')}")
        try:
            analysis = analyze_use_case_from_row(row, model=model, show_tokens=tokens)
            results.append(analysis)
            verdict_color = {
                "Recommended": "green",
                "Recommended with caveats": "yellow",
                "Not recommended": "red",
            }.get(analysis.verdict, "white")
            token_str = ""
            if analysis.token_usage:
                total_input += analysis.token_usage.input_tokens
                total_output += analysis.token_usage.output_tokens
                token_str = f" [{analysis.token_usage.input_tokens:,}+{analysis.token_usage.output_tokens:,} tok]"
            console.print(
                f"  → [{verdict_color}]{analysis.verdict}[/{verdict_color}] "
                f"({analysis.approach_family} / {analysis.capability_pattern}){token_str}"
            )
        except Exception as e:
            console.print(f"  → [red]Error: {e}[/red]")

    # Summary
    console.print(f"\n[bold]Completed {len(results)}/{len(df)} analyses[/bold]")
    if results:
        from collections import Counter

        verdicts = Counter(r.verdict for r in results)
        for verdict, count in verdicts.most_common():
            console.print(f"  {verdict}: {count}")

        families = Counter(r.approach_family for r in results)
        console.print("\n[bold]Approach families:[/bold]")
        for family, count in families.most_common():
            console.print(f"  {family}: {count}")

        if tokens and total_input > 0:
            console.print(
                f"\n[bold]Total tokens:[/bold] {total_input:,} in / {total_output:,} out "
                f"({total_input + total_output:,} total)"
            )

        # Save results to JSON
        output_path = Path("data") / "fit_analyses.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json_mod.dumps([r.model_dump() for r in results], indent=2),
            encoding="utf-8",
        )
        console.print(f"\n[green]✓[/green] Results saved to {output_path}")


@app.command()
def brief(
    agency_abbr: str = typer.Argument(..., help="Agency abbreviation (e.g. NASA, DOE)"),
    model: str = typer.Option(None, "--model", "-m", help="Model override"),
) -> None:
    """Generate an AI portfolio brief for an agency."""
    import json as json_mod

    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    from fedai_atlas.claude.agency_briefer import brief_agency

    console = Console()
    console.print(f"[bold]Generating brief for {agency_abbr}...[/bold]\n")

    result = brief_agency(agency_abbr, model=model)

    # Display
    console.print(Panel(result.executive_summary, title=f"{result.agency_name} — Executive Summary"))
    console.print()
    console.print("[bold]Portfolio Overview[/bold]")
    console.print(result.portfolio_overview)
    console.print()
    if result.approach_family_distribution:
        console.print("[bold]Approach Family Distribution[/bold]")
        console.print(result.approach_family_distribution)
        console.print()
    console.print("[bold]Key Themes[/bold]")
    for theme in result.key_themes:
        console.print(f"  • {theme}")
    console.print()
    console.print("[bold]Notable Use Cases[/bold]")
    for uc in result.notable_use_cases:
        console.print(f"  [cyan]{uc.id}[/cyan]: {uc.name}")
        console.print(f"    {uc.why_notable}")
    console.print()
    console.print("[bold]Architecture Recommendations[/bold]")
    console.print(result.architecture_recommendations)
    console.print()
    console.print("[bold]Maturity Assessment[/bold]")
    console.print(result.maturity_assessment)

    # Save to file
    output_dir = Path("data") / "briefs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{agency_abbr}.json"
    output_path.write_text(
        json_mod.dumps(result.model_dump(), indent=2),
        encoding="utf-8",
    )
    console.print(f"\n[green]✓[/green] Brief saved to {output_path}")


@app.command(name="eval")
def eval_cmd(
    use_case_id: str = typer.Argument(None, help="Use case ID to evaluate (must have seed)"),
    all_seeds: bool = typer.Option(False, "--all", help="Evaluate all seeded use cases"),
    model: str = typer.Option(None, "--model", "-m", help="Model override"),
    tokens: bool = typer.Option(False, "--tokens", "-t", help="Show token counts"),
) -> None:
    """Evaluate analyzer output against human seed analyses (LLM-as-judge)."""
    from rich.console import Console
    from rich.table import Table

    from fedai_atlas.claude.evaluator import evaluate_analysis, parse_seed_analyses

    console = Console()

    seeds = parse_seed_analyses()
    if not seeds:
        rprint("[red]No seed analyses found in docs/seed-analyses.md[/red]")
        raise typer.Exit(1)

    if all_seeds:
        ids_to_eval = list(seeds.keys())
    elif use_case_id:
        ids_to_eval = [use_case_id]
    else:
        rprint("[yellow]Specify a use case ID or --all[/yellow]")
        rprint(f"Available seeds: {', '.join(seeds.keys())}")
        raise typer.Exit(1)

    results = []
    for uid in ids_to_eval:
        console.print(f"[bold]Evaluating {uid}...[/bold]")
        try:
            result = evaluate_analysis(uid, model=model, show_tokens=tokens)
            results.append(result)
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")

    if not results:
        return

    # Display results table
    table = Table(title="Evaluation Results", show_lines=True)
    table.add_column("Use Case", style="cyan")
    table.add_column("Verdict", justify="center")
    table.add_column("Category", justify="center")
    table.add_column("Reasoning", justify="center")
    table.add_column("Complete", justify="center")
    table.add_column("Risk", justify="center")
    table.add_column("Overall", justify="center", style="bold")

    for r in results:
        s = r.scores
        table.add_row(
            r.use_case_id,
            f"{s.verdict_accuracy}/5",
            f"{s.category_accuracy}/5",
            f"{s.reasoning_quality}/5",
            f"{s.completeness}/5",
            f"{s.risk_awareness}/5",
            f"{s.overall:.0%}",
        )

    console.print(table)

    # Commentary
    for r in results:
        console.print(f"\n[bold]{r.use_case_id}:[/bold] {r.scores.commentary}")
        if r.token_usage:
            eval_str = (
                f"Eval: {r.token_usage.input_tokens:,} in / "
                f"{r.token_usage.output_tokens:,} out"
            )
            if r.analysis_token_usage:
                anal_str = (
                    f"Analysis: {r.analysis_token_usage.input_tokens:,} in / "
                    f"{r.analysis_token_usage.output_tokens:,} out"
                )
                total_in = r.token_usage.input_tokens + r.analysis_token_usage.input_tokens
                total_out = r.token_usage.output_tokens + r.analysis_token_usage.output_tokens
                console.print(f"  [dim]{anal_str}[/dim]")
                console.print(f"  [dim]{eval_str}[/dim]")
                console.print(
                    f"  [dim]Total: {total_in:,} in / {total_out:,} out[/dim]"
                )
            else:
                console.print(f"  [dim]{eval_str}[/dim]")
                console.print(
                    "  [dim]Analysis: (not available — re-run analyze with --tokens)[/dim]"
                )


@app.command()
def similar(
    use_case_id: str = typer.Argument(..., help="Use case ID to find similar cases for"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    cross_agency: bool = typer.Option(False, "--cross-agency", "-x", help="Exclude same-agency results"),
) -> None:
    """Find use cases most similar to a given one (by embedding distance)."""
    from rich.console import Console
    from rich.table import Table

    from fedai_atlas.store.vector import find_similar

    console = Console()

    try:
        results = find_similar(use_case_id, limit=limit, cross_agency=cross_agency)
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if results.empty:
        rprint("[yellow]No similar use cases found.[/yellow]")
        return

    table = Table(title=f"Similar to {use_case_id}" + (" (cross-agency)" if cross_agency else ""))
    table.add_column("ID", style="cyan")
    table.add_column("Agency")
    table.add_column("Use Case Name")
    table.add_column("Distance", justify="right")

    for _, row in results.iterrows():
        table.add_row(
            str(row.get("id", "")),
            str(row.get("agency", "")),
            str(row.get("use_case_name", ""))[:80],
            f"{row.get('_distance', 0):.4f}",
        )

    console.print(table)


@app.command()
def duplicates(
    threshold: float = typer.Option(0.25, "--threshold", "-t", help="Max distance (lower = more similar)"),
    limit: int = typer.Option(30, "--limit", "-n", help="Number of duplicate pairs"),
    agency_abbr: str = typer.Option(None, "--agency", "-a", help="Filter to pairs involving this agency"),
) -> None:
    """Detect potential cross-agency duplicate AI use cases."""
    from rich.console import Console
    from rich.table import Table

    from fedai_atlas.store.vector import find_duplicates

    console = Console()

    with console.status("Computing cross-agency similarities..."):
        pairs = find_duplicates(threshold=threshold, limit=limit, agency=agency_abbr)

    if pairs.empty:
        rprint("[yellow]No duplicates found at threshold {threshold}. Try increasing --threshold.[/yellow]")
        return

    table = Table(title=f"Cross-Agency Duplicates (threshold={threshold})")
    table.add_column("Use Case A", style="cyan")
    table.add_column("Agency A")
    table.add_column("Use Case B", style="cyan")
    table.add_column("Agency B")
    table.add_column("Distance", justify="right")

    for _, row in pairs.iterrows():
        table.add_row(
            f"{row['id_a']}: {str(row['name_a'])[:40]}",
            str(row["agency_a"]),
            f"{row['id_b']}: {str(row['name_b'])[:40]}",
            str(row["agency_b"]),
            f"{row['distance']:.4f}",
        )

    console.print(table)
    console.print(f"\n[dim]{len(pairs)} pairs found[/dim]")


@app.command(name="most-connected")
def most_connected(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of use cases to show"),
    threshold: float = typer.Option(0.25, "--threshold", "-t", help="Distance threshold for counting neighbors"),
) -> None:
    """Show use cases with the most cross-agency similar neighbors."""
    from rich.console import Console
    from rich.table import Table

    from fedai_atlas.store.vector import find_most_connected

    console = Console()

    with console.status("Computing neighbor counts..."):
        results = find_most_connected(threshold=threshold, limit=limit)

    if results.empty:
        rprint("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Most Connected Use Cases (threshold={threshold})")
    table.add_column("ID", style="cyan")
    table.add_column("Agency")
    table.add_column("Use Case Name")
    table.add_column("Cross-Agency\nNeighbors", justify="right", style="bold")
    table.add_column("Agencies", style="dim")

    for _, row in results.iterrows():
        table.add_row(
            str(row["id"]),
            str(row["agency"]),
            str(row["use_case_name"])[:60],
            str(row["neighbor_count"]),
            str(row["neighbor_agencies"]),
        )

    console.print(table)


@app.command(name="batch-submit")
def batch_submit(
    agency_abbr: str = typer.Option(None, "--agency", "-a", help="Filter to one agency"),
    limit: int = typer.Option(None, "--limit", "-n", help="Limit number of use cases"),
    model: str = typer.Option(None, "--model", "-m", help="Model override"),
) -> None:
    """Submit use cases to Anthropic Batches API for bulk analysis (50% discount)."""
    from fedai_atlas.claude.batch import submit_batch
    from fedai_atlas.store.structured import load_all, query_by_agency

    if agency_abbr:
        df = query_by_agency(agency_abbr)
    else:
        df = load_all()

    if df.empty:
        rprint("[red]No use cases found.[/red]")
        raise typer.Exit(1)

    if limit:
        df = df.head(limit)

    rprint(f"Submitting {len(df)} use cases to Batches API...")
    try:
        batch_id = submit_batch(df, model=model)
    except Exception as e:
        rprint(f"[red]Submission failed: {e}[/red]")
        raise typer.Exit(1)

    rprint(f"[green]✓[/green] Batch submitted: [bold]{batch_id}[/bold]")
    rprint(f"  Use cases: {len(df)}")
    rprint(f"  Check status: [cyan]fedai batch-status {batch_id}[/cyan]")


@app.command(name="batch-status")
def batch_status(
    batch_id: str = typer.Argument(None, help="Batch ID (omit to show all tracked)"),
) -> None:
    """Check status of a batch job or list all tracked batches."""
    from rich.console import Console
    from rich.table import Table

    from fedai_atlas.claude.batch import BATCH_JOBS_PATH, get_batch_status, _load_batch_jobs

    console = Console()

    if batch_id:
        try:
            status = get_batch_status(batch_id)
        except Exception as e:
            rprint(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        console.print(f"[bold]Batch:[/bold] {status['id']}")
        console.print(f"[bold]Status:[/bold] {status['status']}")
        counts = status["request_counts"]
        console.print(f"  Succeeded: [green]{counts['succeeded']}[/green]")
        console.print(f"  Errored:   [red]{counts['errored']}[/red]")
        console.print(f"  Processing: {counts['processing']}")
        if status["ended_at"]:
            console.print(f"  Ended: {status['ended_at']}")
        if status["status"] == "ended":
            console.print(f"\n  Retrieve: [cyan]fedai batch-retrieve {batch_id}[/cyan]")
    else:
        jobs = _load_batch_jobs()
        if not jobs:
            rprint("No tracked batch jobs. Submit one with [cyan]fedai batch-submit[/cyan]")
            return

        table = Table(title="Tracked Batch Jobs")
        table.add_column("Batch ID", style="cyan")
        table.add_column("Submitted")
        table.add_column("Count")
        table.add_column("Model")

        for job in jobs[-10:]:
            table.add_row(
                job["batch_id"],
                job["submitted_at"][:19],
                str(job["count"]),
                job.get("model", ""),
            )
        console.print(table)


@app.command(name="batch-retrieve")
def batch_retrieve(
    batch_id: str = typer.Argument(..., help="Batch ID to retrieve results from"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
) -> None:
    """Retrieve and save results from a completed batch."""
    from fedai_atlas.claude.batch import get_batch_status, save_batch_results

    # Check status first
    try:
        status = get_batch_status(batch_id)
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    if status["status"] != "ended":
        rprint(f"[yellow]Batch not complete yet. Status: {status['status']}[/yellow]")
        counts = status["request_counts"]
        rprint(f"  Processing: {counts['processing']}, Succeeded: {counts['succeeded']}")
        raise typer.Exit(1)

    output_path = Path(output) if output else None
    rprint("Retrieving and validating results...")

    try:
        succeeded, failed = save_batch_results(batch_id, output_path=output_path)
    except Exception as e:
        rprint(f"[red]Retrieval failed: {e}[/red]")
        raise typer.Exit(1)

    rprint(f"[green]✓[/green] Results saved: {succeeded} succeeded, {failed} failed")
    rprint(f"  Output: {output_path or 'data/fit_analyses.json'}")


@app.command()
def ui() -> None:
    """Launch the Streamlit web UI."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "ui" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    app()
