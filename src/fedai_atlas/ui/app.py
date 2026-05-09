"""Streamlit application for FedAI Atlas."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="FedAI Atlas",
    page_icon="🏛️",
    layout="wide",
)

# Inject compact CSS
st.markdown("""
<style>
    /* Reduce chart padding */
    .stPlotlyChart { margin-top: -10px; }
    /* Tighter subheaders */
    h3 { margin-top: 0.5rem !important; margin-bottom: 0.2rem !important; font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)

FIT_PATH = Path("data") / "fit_analyses.json"

# Color palette
PALETTE = [
    "#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626",
    "#0891B2", "#4F46E5", "#65A30D", "#CA8A04", "#9333EA",
]
VERDICT_COLORS = {"Recommended": "#059669", "Recommended with caveats": "#D97706", "Not recommended": "#DC2626"}


@st.cache_data(ttl=300)
def _load_inventory():
    """Load all use cases from SQLite (cached)."""
    from fedai_atlas.store.structured import load_all
    return load_all()


@st.cache_data(ttl=300)
def _load_analyses():
    """Load fit analyses from JSON (cached)."""
    if FIT_PATH.exists():
        return json.loads(FIT_PATH.read_text(encoding="utf-8"))
    return []


def dashboard_page() -> None:
    """Data-dense dashboard."""
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    st.header("FedAI Atlas")
    st.caption("2025 Federal Agency AI Use Case Inventory — Architecture Analysis")

    try:
        df = _load_inventory()
    except Exception:
        st.error("No data available. Run `fedai ingest` first.")
        return

    if df.empty:
        st.warning("No use cases in database. Run `fedai ingest` first.")
        return

    analyses = _load_analyses()
    analyses_df = pd.DataFrame(analyses) if analyses else pd.DataFrame()
    n_agencies = df["agency"].nunique()

    # ── KPI row ──
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Use Cases", f"{len(df):,}")
    k2.metric("Agencies", n_agencies)
    k3.metric("Analyzed", f"{len(analyses):,}")
    coverage = f"{len(analyses)/len(df)*100:.0f}%" if len(df) > 0 else "0%"
    k4.metric("Coverage", coverage)
    high_impact = df["is_high_impact_short"].eq("High-impact").sum() if "is_high_impact_short" in df.columns else 0
    k5.metric("High-Impact", f"{high_impact:,}")

    st.markdown("")

    # ── Row 1: AI Categories + Capability Patterns ──
    if not analyses_df.empty:
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            st.subheader("AI Category")
            cat_counts = analyses_df["approach_family"].value_counts().sort_values()
            fig = px.bar(
                x=cat_counts.values, y=cat_counts.index,
                orientation="h",
                labels={"x": "", "y": ""},
                color=cat_counts.index,
                color_discrete_sequence=PALETTE,
            )
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            st.subheader("Top Capability Patterns")
            pattern_counts = analyses_df["capability_pattern"].value_counts().head(12).sort_values()
            fig = px.bar(
                x=pattern_counts.values, y=pattern_counts.index,
                orientation="h",
                labels={"x": "", "y": ""},
            )
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            fig.update_traces(marker_color="#7C3AED")
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Top Agencies + Lifecycle Stage ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("Top 15 Agencies")
        agency_counts = df["agency"].value_counts().head(15).sort_values()
        fig = px.bar(
            x=agency_counts.values, y=agency_counts.index,
            orientation="h",
            labels={"x": "", "y": ""},
        )
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        fig.update_traces(marker_color="#2563EB")
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        st.subheader("Lifecycle Stage")
        if "development_stage_short" in df.columns:
            stage_counts = df["development_stage_short"].value_counts()
            fig = go.Figure(go.Pie(
                labels=stage_counts.index,
                values=stage_counts.values,
                hole=0.45,
                marker=dict(colors=PALETTE),
                textinfo="label+percent",
                textposition="outside",
            ))
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: High-Impact by Agency ──
    st.subheader("High-Impact Share (Top 15)")
    if "is_high_impact_short" in df.columns:
            top_agencies = df["agency"].value_counts().head(15).index
            filtered = df[df["agency"].isin(top_agencies)]
            cross = pd.crosstab(filtered["agency"], filtered["is_high_impact_short"])
            cross = cross.loc[cross.sum(axis=1).sort_values(ascending=True).index]
            fig = px.bar(
                cross, orientation="h",
                labels={"value": "", "agency": ""},
                color_discrete_map={"High-impact": "#DC2626", "Not High-impact": "#D1D5DB"},
            )
            fig.update_layout(
                barmode="stack", height=320, margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Cross-Agency Replicable Solutions ──
    try:
        import numpy as np
        from fedai_atlas.store.vector import get_vector_db, TABLE_NAME

        db = get_vector_db()
        tbl = db.open_table(TABLE_NAME)
        vdf = tbl.to_pandas()

        if not vdf.empty and "vector" in vdf.columns:
            st.subheader("Cross-Agency Replicable Solutions")
            st.caption("Use cases with similar counterparts at other agencies (distance < 0.25)")

            vectors = np.stack(vdf["vector"].values)
            agencies_arr = np.array(vdf["agency"].values.astype(str))

            # Normalize + cosine distance
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normed = vectors / norms

            # Cross-agency neighbors at 0.25
            # Process in chunks to keep memory reasonable
            threshold = 0.25
            n = len(normed)
            neighbor_counts = np.zeros(n, dtype=int)
            chunk_size = 500
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk_dists = 1.0 - (normed[start:end] @ normed.T)
                cross_agency = agencies_arr[start:end, None] != agencies_arr[None, :]
                neighbors = (chunk_dists < threshold) & cross_agency
                neighbor_counts[start:end] = neighbors.sum(axis=1)

            vdf["cross_agency_neighbors"] = neighbor_counts
            has_neighbors = vdf[vdf["cross_agency_neighbors"] > 0]

            r4c1, r4c2 = st.columns(2)

            with r4c1:
                # Bar chart: agencies with most replicable use cases
                agency_replicable = has_neighbors.groupby("agency").size().sort_values(ascending=True).tail(15)
                fig = px.bar(
                    x=agency_replicable.values, y=agency_replicable.index,
                    orientation="h",
                    labels={"x": "Use cases with cross-agency matches", "y": ""},
                )
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                fig.update_traces(marker_color="#059669")
                st.plotly_chart(fig, use_container_width=True)

            with r4c2:
                # Histogram of neighbor counts
                fig = px.histogram(
                    has_neighbors, x="cross_agency_neighbors",
                    nbins=30,
                    labels={"cross_agency_neighbors": "Number of cross-agency neighbors"},
                )
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                fig.update_traces(marker_color="#2563EB")
                st.plotly_chart(fig, use_container_width=True)

            st.metric(
                "Replicable Use Cases",
                f"{len(has_neighbors):,} of {len(vdf):,}",
                delta=f"{len(has_neighbors)/len(vdf)*100:.0f}% have cross-agency matches",
            )

            # Top 10 most connected use cases with drill-down
            st.subheader("Top Replicable Use Cases")
            top = vdf.nlargest(10, "cross_agency_neighbors")
            for i, (_, r) in enumerate(top.iterrows()):
                uc_id = str(r.get("id", ""))
                with st.expander(
                    f"**{uc_id}** ({r.get('agency', '')}) — {r.get('use_case_name', '')} · "
                    f"{int(r['cross_agency_neighbors'])} neighbors",
                    expanded=(i < 3),
                ):
                    detail = _get_use_case_detail(uc_id)
                    if detail:
                        _render_use_case_card(detail)
                    else:
                        st.markdown(f"**Problem:** {r.get('problem_solved', '')}")
    except Exception:
        pass


def search_page() -> None:
    """Semantic search across federal AI use cases."""
    st.header("Semantic Search")
    st.caption("Search across federal AI use cases by meaning, not just keywords.")

    query = st.text_input("Search query", placeholder="e.g. fraud detection in benefits payments")
    limit = st.slider("Results", min_value=5, max_value=50, value=10, step=5)

    if query:
        from fedai_atlas.store.vector import semantic_search

        with st.spinner("Searching..."):
            results = semantic_search(query, limit=limit)

        if results.empty:
            st.warning("No results found.")
            return

        for i, (_, row) in enumerate(results.iterrows()):
            dist = row.get("_distance", 0)
            col1, col2 = st.columns([4, 1])
            with col1:
                with st.expander(
                    f"**{row.get('use_case_name', '')}** — {row.get('agency', '')} ({row.get('id', '')})",
                    expanded=(i == 0),
                ):
                    st.markdown(f"**Stage:** {row.get('development_stage_short', '')}")
                    st.markdown(f"**Topic:** {row.get('topic_area_short', '')}")
                    st.markdown(f"**Classification:** {row.get('classification_short', '')}")
                    st.markdown(f"**Problem:** {row.get('problem_solved', '')}")
                    st.markdown(f"**Benefits:** {row.get('benefits', '')}")
                    st.markdown(f"**Outputs:** {row.get('system_outputs', '')}")

                    if st.button(f"Analyze", key=f"analyze_{row.get('id', i)}"):
                        _run_inline_analysis(str(row.get("id", "")))
            with col2:
                st.metric("Distance", f"{dist:.3f}")


def agency_page() -> None:
    """Explore agencies and generate briefs."""
    st.header("Agency Explorer")

    from fedai_atlas.store.structured import list_agencies, query_by_agency

    agencies = list_agencies()
    selected = st.selectbox("Select an agency", agencies)

    if not selected:
        return

    df = query_by_agency(selected)
    if df.empty:
        st.warning(f"No use cases found for {selected}")
        return

    agency_name = df.iloc[0].get("agency_name", selected)
    st.subheader(f"{agency_name} ({selected}) — {len(df)} use cases")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Development Stage**")
        if "development_stage_short" in df.columns:
            st.dataframe(
                df["development_stage_short"].value_counts().reset_index(),
                hide_index=True, use_container_width=True,
            )
    with col2:
        st.markdown("**Topic Areas**")
        if "topic_area_short" in df.columns:
            st.dataframe(
                df["topic_area_short"].value_counts().reset_index(),
                hide_index=True, use_container_width=True,
            )
    with col3:
        st.markdown("**AI Classification**")
        if "classification_short" in df.columns:
            st.dataframe(
                df["classification_short"].value_counts().reset_index(),
                hide_index=True, use_container_width=True,
            )

    st.markdown("---")
    display_cols = ["id", "use_case_name", "development_stage_short", "topic_area_short"]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols], hide_index=True, use_container_width=True)

    st.markdown("---")
    if st.button("Generate Agency Brief", type="primary"):
        _run_agency_brief(selected)


def analyze_page() -> None:
    """Run fit analysis on a use case."""
    st.header("Fit Analyzer")
    st.caption("Classify a federal AI use case by AI/ML category and capability pattern.")

    use_case_id = st.text_input("Use Case ID", placeholder="e.g. NASA-1")

    if st.button("Analyze", type="primary") and use_case_id:
        _run_inline_analysis(use_case_id)

    if "analyses" in st.session_state and st.session_state.analyses:
        st.markdown("---")
        st.subheader("Recent Analyses")
        for analysis in reversed(st.session_state.analyses[-10:]):
            verdict_colors = {
                "Recommended": "🟢",
                "Recommended with caveats": "🟡",
                "Not recommended": "🔴",
            }
            icon = verdict_colors.get(analysis.verdict, "⚪")
            with st.expander(
                f"{icon} {analysis.use_case_id}: {analysis.use_case_name} — {analysis.verdict}"
            ):
                st.markdown(f"**Category:** {analysis.approach_family}")
                st.markdown(f"**Pattern:** {analysis.capability_pattern}")
                st.markdown(f"**Reasoning:** {analysis.reasoning}")
                st.markdown(f"**Integration:** {analysis.integration_pattern}")
                st.markdown(f"**Risks:** {analysis.risks}")


def _run_inline_analysis(use_case_id: str) -> None:
    """Run fit analysis and display results inline."""
    from fedai_atlas.claude.fit_analyzer import analyze_use_case

    with st.spinner(f"Analyzing {use_case_id}..."):
        try:
            result = analyze_use_case(use_case_id)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    if "analyses" not in st.session_state:
        st.session_state.analyses = []
    st.session_state.analyses.append(result)

    st.success(f"**{result.verdict}** — {result.approach_family} / {result.capability_pattern}")
    st.markdown(f"**Reasoning:** {result.reasoning}")
    st.markdown(f"**Integration:** {result.integration_pattern}")
    st.markdown(f"**Risks:** {result.risks}")
    if result.hybrid_decomposition:
        st.markdown("**Hybrid Decomposition:**")
        for comp in result.hybrid_decomposition:
            st.markdown(f"- {comp.family} → {comp.pattern}: {comp.role}")
    if result.proposed_eval_dimensions:
        st.markdown(f"**Eval Dimensions:** {', '.join(result.proposed_eval_dimensions)}")


def _run_agency_brief(agency: str) -> None:
    """Generate and display an agency brief."""
    from fedai_atlas.claude.agency_briefer import brief_agency

    with st.spinner(f"Generating brief for {agency}... (this takes ~30s)"):
        try:
            result = brief_agency(agency)
        except Exception as e:
            st.error(f"Brief generation failed: {e}")
            return

    st.markdown("---")
    st.subheader("Agency Brief")
    st.info(result.executive_summary)

    st.markdown("**Portfolio Overview**")
    st.markdown(result.portfolio_overview)

    st.markdown("**Key Themes**")
    for theme in result.key_themes:
        st.markdown(f"- {theme}")

    st.markdown("**Notable Use Cases**")
    for uc in result.notable_use_cases:
        st.markdown(f"- **{uc.id}**: {uc.name} — {uc.why_notable}")

    if result.approach_family_distribution:
        st.markdown("**AI Category Distribution**")
        st.markdown(result.approach_family_distribution)

    st.markdown("**Architecture Recommendations**")
    st.markdown(result.architecture_recommendations)

    st.markdown("**Maturity Assessment**")
    st.markdown(result.maturity_assessment)


def _get_use_case_detail(use_case_id: str) -> dict | None:
    """Load full use case details from SQLite + fit analysis if available."""
    from fedai_atlas.store.structured import query_by_id
    row = query_by_id(use_case_id)
    if row.empty:
        return None
    detail = row.iloc[0].to_dict()

    # Attach fit analysis if available
    if FIT_PATH.exists():
        analyses = _load_analyses()
        for a in analyses:
            if a.get("use_case_id") == use_case_id:
                detail["_analysis"] = a
                break
    return detail


def _render_use_case_card(detail: dict) -> None:
    """Render a detailed use case card."""
    st.markdown(f"**Problem:** {detail.get('problem_solved', 'N/A')}")
    st.markdown(f"**Benefits:** {detail.get('benefits', 'N/A')}")
    st.markdown(f"**Outputs:** {detail.get('system_outputs', 'N/A')}")
    st.markdown(f"**Stage:** {detail.get('development_stage_short', 'N/A')} · "
                f"**Topic:** {detail.get('topic_area_short', 'N/A')} · "
                f"**Classification:** {detail.get('classification_short', 'N/A')}")
    if "_analysis" in detail:
        a = detail["_analysis"]
        verdict_icon = {"Recommended": "🟢", "Recommended with caveats": "🟡", "Not recommended": "🔴"}
        icon = verdict_icon.get(a.get("verdict", ""), "⚪")
        st.markdown(f"**AI Category:** {a.get('approach_family', 'N/A')} / {a.get('capability_pattern', 'N/A')}")
        st.markdown(f"**Verdict:** {icon} {a.get('verdict', 'N/A')}")
        st.markdown(f"**Reasoning:** {a.get('reasoning', '')}")


def similar_page() -> None:
    """Find similar use cases and detect duplicates."""
    import pandas as pd

    st.header("Similar Use Cases & Duplicates")

    tab1, tab2, tab3 = st.tabs(["Find Similar", "Duplicate Detection", "Most Connected"])

    with tab1:
        st.caption("Find use cases most similar to a given one across all agencies.")
        col1, col2 = st.columns([3, 1])
        with col1:
            use_case_id = st.text_input("Use Case ID", placeholder="e.g. DHS-2452", key="similar_id")
        with col2:
            sim_limit = st.number_input("Results", min_value=5, max_value=50, value=10, key="sim_limit")

        cross_agency = st.checkbox("Cross-agency only (exclude same agency)")

        if st.button("Find Similar", type="primary") and use_case_id:
            from fedai_atlas.store.vector import find_similar

            with st.spinner("Searching..."):
                try:
                    results = find_similar(use_case_id, limit=sim_limit, cross_agency=cross_agency)
                except ValueError as e:
                    st.error(str(e))
                    return

            if results.empty:
                st.warning("No similar use cases found.")
                return

            for i, (_, row) in enumerate(results.iterrows()):
                dist = row.get("_distance", 0)
                sim_pct = max(0, (1 - dist)) * 100
                with st.expander(
                    f"**{row.get('use_case_name', '')}** — {row.get('agency', '')} ({row.get('id', '')}) · {sim_pct:.0f}%",
                    expanded=(i < 3),
                ):
                    detail = _get_use_case_detail(str(row.get("id", "")))
                    if detail:
                        _render_use_case_card(detail)
                    else:
                        st.markdown(f"**Problem:** {row.get('problem_solved', '')}")
                        st.markdown(f"**Benefits:** {row.get('benefits', '')}")
                        st.markdown(f"**Stage:** {row.get('development_stage_short', '')}")

    with tab2:
        st.caption("Detect potential cross-agency duplicate AI efforts.")
        col1, col2, col3 = st.columns(3)
        with col1:
            threshold = st.slider("Distance threshold", 0.05, 0.5, 0.25, 0.05,
                                  help="Lower = more similar. 0.1 = near-identical, 0.3 = related.")
        with col2:
            dup_limit = st.number_input("Max pairs", min_value=10, max_value=200, value=30, key="dup_limit")
        with col3:
            from fedai_atlas.store.structured import list_agencies
            agencies = ["All agencies"] + list_agencies()
            selected_agency = st.selectbox("Filter by agency", agencies, key="dup_agency")

        if st.button("Detect Duplicates", type="primary"):
            from fedai_atlas.store.vector import find_duplicates

            agency_filter = None if selected_agency == "All agencies" else selected_agency

            with st.spinner("Computing cross-agency similarities..."):
                pairs = find_duplicates(threshold=threshold, limit=dup_limit, agency=agency_filter)

            if pairs.empty:
                st.warning(f"No duplicates found at threshold {threshold}. Try increasing the threshold.")
                return

            st.success(f"Found {len(pairs)} potential duplicate pairs")

            for i, (_, row) in enumerate(pairs.iterrows()):
                sim_pct = max(0, (1 - row["distance"])) * 100
                with st.expander(
                    f"{row['agency_a']}/{row['id_a']} ↔ {row['agency_b']}/{row['id_b']} — {sim_pct:.0f}% similar",
                    expanded=(i < 5),
                ):
                    c1, c2 = st.columns(2)
                    detail_a = _get_use_case_detail(str(row["id_a"]))
                    detail_b = _get_use_case_detail(str(row["id_b"]))
                    with c1:
                        st.markdown(f"### {row['id_a']} ({row['agency_a']})")
                        st.markdown(f"**{row['name_a']}**")
                        if detail_a:
                            _render_use_case_card(detail_a)
                    with c2:
                        st.markdown(f"### {row['id_b']} ({row['agency_b']})")
                        st.markdown(f"**{row['name_b']}**")
                        if detail_b:
                            _render_use_case_card(detail_b)

    with tab3:
        st.caption("Use cases with the most cross-agency similar neighbors.")
        col1, col2 = st.columns(2)
        with col1:
            mc_threshold = st.slider("Distance threshold", 0.1, 0.5, 0.25, 0.05,
                                     help="Max distance to count as a neighbor.", key="mc_thresh")
        with col2:
            mc_limit = st.number_input("Top N", min_value=5, max_value=100, value=20, key="mc_limit")

        if st.button("Find Most Connected", type="primary"):
            from fedai_atlas.store.vector import find_most_connected

            with st.spinner("Computing neighbor counts..."):
                results = find_most_connected(threshold=mc_threshold, limit=mc_limit)

            if results.empty:
                st.warning("No results found.")
            else:
                for i, (_, row) in enumerate(results.iterrows()):
                    with st.expander(
                        f"**{row['id']}** ({row['agency']}) — {row['neighbor_count']} cross-agency neighbors",
                        expanded=(i < 5),
                    ):
                        detail = _get_use_case_detail(str(row["id"]))
                        if detail:
                            _render_use_case_card(detail)
                        else:
                            st.markdown(f"**{row['use_case_name']}**")
                        st.markdown(f"**Neighbor agencies:** {row['neighbor_agencies']}")


# --- Main ---
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Search", "Agency Explorer", "Fit Analyzer", "Similar & Duplicates"],
)

if page == "Dashboard":
    dashboard_page()
elif page == "Search":
    search_page()
elif page == "Agency Explorer":
    agency_page()
elif page == "Fit Analyzer":
    analyze_page()
elif page == "Similar & Duplicates":
    similar_page()
