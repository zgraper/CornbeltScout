"""
streamlit_app.py
----------------
Streamlit control panel for the AgriIndex Web Intelligence Crawler.

Sections
--------
1. Title / description
2. Query Runner  – run a Phase 1 crawl job from the browser
3. Crawl Results Summary – counts of discovered / processed / failed / skipped URLs
4. Recent Pages Table – last 50 rows from the SQLite database
5. Page Detail Viewer  – drill into a single page record
6. Log output  – scrollable log tail

Tabs are pre-structured so future views (Investor Leads, Domain Analysis,
Deep Crawl Candidates, Entity Explorer) can be added without restructuring.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so agriindex imports work when the app
# is launched from any working directory (e.g. `streamlit run ui/streamlit_app.py`)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agriindex.config import settings  # noqa: E402  (after sys.path patch)
from agriindex.db import database  # noqa: E402


# ---------------------------------------------------------------------------
# In-memory log handler so we can surface log output inside the UI
# ---------------------------------------------------------------------------

class _StreamlitLogHandler(logging.Handler):
    """Accumulate log records in a session-state list for display."""

    def emit(self, record: logging.LogRecord) -> None:
        if "log_messages" not in st.session_state:
            st.session_state["log_messages"] = []
        st.session_state["log_messages"].append(self.format(record))


def _setup_logging() -> None:
    root = logging.getLogger()
    # Avoid adding the handler multiple times across Streamlit reruns
    if not any(isinstance(h, _StreamlitLogHandler) for h in root.handlers):
        handler = _StreamlitLogHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))


# ---------------------------------------------------------------------------
# Database helper functions
# ---------------------------------------------------------------------------

def load_recent_pages(db_path: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
    """
    Return the most recent *limit* processed pages from the database.

    Columns returned
    ----------------
    page_id, url, title, domain, page_type,
    relevance_cornbelt_ai, relevance_investor, fetched_at
    """
    path = db_path or settings.DB_PATH
    if not os.path.exists(path):
        return pd.DataFrame()

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT
                p.id            AS page_id,
                u.canonical_url AS url,
                p.page_title    AS title,
                u.domain        AS domain,
                p.page_type,
                p.relevance_cornbelt_ai,
                p.relevance_investor,
                p.fetched_at
            FROM pages p
            JOIN urls u ON u.id = p.url_id
            ORDER BY p.fetched_at DESC
            LIMIT ?
        """
        rows = conn.execute(query, (limit,)).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
    finally:
        conn.close()


def load_page_details(page_id: int, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Return full details for a single page record.

    Returns
    -------
    dict with keys:
        title, url, summary, keywords, entities, contacts, cleaned_text
    """
    path = db_path or settings.DB_PATH
    if not os.path.exists(path):
        return {}

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        page_row = conn.execute(
            """
            SELECT p.*, u.canonical_url AS url
            FROM pages p
            JOIN urls u ON u.id = p.url_id
            WHERE p.id = ?
            """,
            (page_id,),
        ).fetchone()
        if page_row is None:
            return {}

        details: Dict[str, Any] = dict(page_row)

        # Decode JSON columns
        for col in ("keywords", "topics"):
            raw = details.get(col)
            if raw and isinstance(raw, str):
                try:
                    details[col] = json.loads(raw)
                except json.JSONDecodeError:
                    pass

        # Fetch entities linked to this page
        entity_rows = conn.execute(
            """
            SELECT e.text, e.label, pe.count
            FROM page_entities pe
            JOIN entities e ON e.id = pe.entity_id
            WHERE pe.page_id = ?
            ORDER BY pe.count DESC
            """,
            (page_id,),
        ).fetchall()
        details["entities"] = [dict(r) for r in entity_rows]

        # Fetch contacts linked to this page
        contact_rows = conn.execute(
            "SELECT type, value FROM contacts WHERE page_id = ? ORDER BY type",
            (page_id,),
        ).fetchall()
        details["contacts"] = [dict(r) for r in contact_rows]

        return details
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Crawl execution helper
# ---------------------------------------------------------------------------

def _run_crawl(
    query: str,
    limit: int,
    use_llm: bool,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Initialise the database and run the Phase 1 pipeline.

    Parameters
    ----------
    query : str
    limit : int
    use_llm : bool
        When False the LLM summarisation step is skipped by patching the
        environment variable that ``settings`` reads.
    db_path : str, optional

    Returns
    -------
    list of context dicts (one per URL attempted)
    """
    # Toggle LLM via environment variable so settings.LLAMA_ENABLED reflects it
    os.environ["AGRIINDEX_LLAMA_ENABLED"] = "true" if use_llm else "false"

    # Lazy import so the module picks up the updated env var
    from agriindex.pipeline.phase1_pipeline import run_pipeline  # noqa: PLC0415

    database.init_db(db_path=db_path)
    results = run_pipeline(query=query, limit=limit, db_path=db_path)
    return results


def _summarise_results(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Count discovered / processed / failed / skipped URLs from pipeline output.

    A URL is considered:
    - *discovered*  – present in results list
    - *processed*   – has cleaned_text (fetch + parse succeeded)
    - *failed*      – has a fetch_error
    - *skipped*     – fetch succeeded but no cleaned_text (e.g. binary content)
    """
    discovered = len(results)
    failed = sum(1 for r in results if r.get("fetch_error"))
    processed = sum(1 for r in results if r.get("cleaned_text"))
    skipped = discovered - failed - processed
    if skipped < 0:
        logging.getLogger(__name__).warning(
            "Unexpected negative skipped count (%d); clamping to 0. "
            "discovered=%d, failed=%d, processed=%d",
            skipped, discovered, failed, processed,
        )
        skipped = 0
    return {
        "discovered": discovered,
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Page layout helpers
# ---------------------------------------------------------------------------

def _render_query_runner() -> Optional[List[Dict[str, Any]]]:
    """Render the Query Runner section and return pipeline results if a run was triggered."""
    st.subheader("🔍 Query Runner")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Search query",
            placeholder="e.g. corn belt precision agriculture Iowa",
            help="DuckDuckGo search query used to discover agricultural webpages.",
        )
    with col2:
        limit = st.number_input(
            "Result limit",
            min_value=1,
            max_value=100,
            value=10,
            step=5,
            help="Maximum number of search results to crawl.",
        )

    use_llm = st.checkbox(
        "Enable LLM summarisation",
        value=False,
        help="Run llama.cpp on each page to generate a short summary (requires a local model).",
    )

    run_clicked = st.button("▶ Run Crawl", type="primary", disabled=not query.strip())

    if run_clicked and query.strip():
        st.session_state["log_messages"] = []  # reset log buffer for this run
        progress_bar = st.progress(0, text="Initialising…")
        status_area = st.empty()

        with st.spinner("Crawling — this may take a minute…"):
            try:
                progress_bar.progress(10, text="Starting pipeline…")
                results = _run_crawl(
                    query=query.strip(),
                    limit=int(limit),
                    use_llm=use_llm,
                )
                progress_bar.progress(100, text="Done ✓")
                status_area.success(f"Crawl complete — {len(results)} URL(s) attempted.")
                return results
            except Exception as exc:  # noqa: BLE001
                progress_bar.progress(100, text="Error ✗")
                status_area.error(
                    f"Pipeline error: {str(exc)}\n\n"
                    "Check the Log Output section below for details, or verify that "
                    "your search query is valid and the database path is writable."
                )
                return []

    return None


def _render_results_summary(results: List[Dict[str, Any]]) -> None:
    """Render the four-metric crawl summary."""
    st.subheader("📊 Crawl Results Summary")
    counts = _summarise_results(results)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Discovered", counts["discovered"])
    c2.metric("Processed", counts["processed"])
    c3.metric("Failed", counts["failed"])
    c4.metric("Skipped", counts["skipped"])


def _render_recent_pages_tab() -> None:
    """Render the Recent Pages table and Page Detail Viewer."""
    st.subheader("📄 Recent Pages")

    df = load_recent_pages()
    if df.empty:
        st.info("No pages in the database yet. Run a crawl to populate results.")
        return

    # Format float columns for readability
    for col in ("relevance_cornbelt_ai", "relevance_investor"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Page detail selector
    st.subheader("🔎 Page Detail Viewer")
    page_ids = df["page_id"].tolist()
    labels = {
        row["page_id"]: f"[{row['page_id']}] {row['title'] or row['url']}"
        for _, row in df.iterrows()
    }
    selected_id = st.selectbox(
        "Select a page to inspect",
        options=page_ids,
        format_func=lambda pid: labels.get(pid, str(pid)),
    )

    if selected_id:
        details = load_page_details(int(selected_id))
        if not details:
            st.warning("Could not load page details.")
            return

        st.markdown(f"### {details.get('page_title') or '(no title)'}")
        st.markdown(f"**URL:** {details.get('url', '—')}")

        tabs = st.tabs(["Summary", "Keywords", "Entities", "Contacts", "Text Preview"])

        with tabs[0]:
            summary = details.get("summary") or "_No summary available._"
            st.write(summary)

        with tabs[1]:
            kw = details.get("keywords")
            if kw and isinstance(kw, dict):
                kw_df = pd.DataFrame(
                    [{"Category": k, "Hits": v} for k, v in kw.items() if v]
                ).sort_values("Hits", ascending=False)
                st.dataframe(kw_df, use_container_width=True, hide_index=True)
            else:
                st.info("No keyword data.")

        with tabs[2]:
            entities = details.get("entities", [])
            if entities:
                st.dataframe(
                    pd.DataFrame(entities),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No entities extracted.")

        with tabs[3]:
            contacts = details.get("contacts", [])
            if contacts:
                st.dataframe(
                    pd.DataFrame(contacts),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No contacts found.")

        with tabs[4]:
            text = details.get("cleaned_text") or ""
            preview = text[:2000] + ("…" if len(text) > 2000 else "")
            st.text_area("cleaned_text (first 2,000 chars)", preview, height=300)


def _render_log_section() -> None:
    """Render the scrollable log output section."""
    messages: List[str] = st.session_state.get("log_messages", [])
    with st.expander("📋 Log Output", expanded=False):
        if messages:
            log_text = "\n".join(messages[-200:])  # cap at last 200 lines
            st.text_area("Log", log_text, height=300, label_visibility="collapsed")
        else:
            st.caption("No log output yet. Run a crawl to see activity here.")


# ---------------------------------------------------------------------------
# Future tab stubs
# ---------------------------------------------------------------------------

def _render_investor_leads_tab() -> None:
    st.info("🚧 Investor Leads tab — coming in a future phase.")


def _render_domain_analysis_tab() -> None:
    st.info("🚧 Domain Analysis tab — coming in a future phase.")


def _render_deep_crawl_tab() -> None:
    st.info("🚧 Deep Crawl Candidates tab — coming in a future phase.")


def _render_entity_explorer_tab() -> None:
    st.info("🚧 Entity Explorer tab — coming in a future phase.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="AgriIndex Crawler",
        page_icon="🌽",
        layout="wide",
    )

    _setup_logging()

    # -----------------------------------------------------------------------
    # 1. Title
    # -----------------------------------------------------------------------
    st.title("🌽 AgriIndex Web Intelligence Crawler")
    st.markdown(
        "AgriIndex discovers agricultural webpages using DuckDuckGo searches, "
        "fetches and parses page content, extracts structured metadata "
        "(keywords, entities, contacts, relevance scores), and stores results "
        "in a local SQLite database."
    )
    st.divider()

    # -----------------------------------------------------------------------
    # 2. Query Runner
    # -----------------------------------------------------------------------
    crawl_results = _render_query_runner()

    # -----------------------------------------------------------------------
    # 3. Crawl Results Summary (shown only after a run)
    # -----------------------------------------------------------------------
    if crawl_results is not None:
        _render_results_summary(crawl_results)
        st.divider()

    # -----------------------------------------------------------------------
    # Main navigation tabs
    # -----------------------------------------------------------------------
    tab_pages, tab_investor, tab_domain, tab_deep, tab_entities = st.tabs([
        "📄 Recent Pages",
        "💼 Investor Leads",
        "🌐 Domain Analysis",
        "🔗 Deep Crawl Candidates",
        "🏷️ Entity Explorer",
    ])

    # -----------------------------------------------------------------------
    # 4 & 5. Recent Pages table + Page Detail Viewer
    # -----------------------------------------------------------------------
    with tab_pages:
        _render_recent_pages_tab()

    with tab_investor:
        _render_investor_leads_tab()

    with tab_domain:
        _render_domain_analysis_tab()

    with tab_deep:
        _render_deep_crawl_tab()

    with tab_entities:
        _render_entity_explorer_tab()

    # -----------------------------------------------------------------------
    # Log output
    # -----------------------------------------------------------------------
    st.divider()
    _render_log_section()


if __name__ == "__main__":
    main()
