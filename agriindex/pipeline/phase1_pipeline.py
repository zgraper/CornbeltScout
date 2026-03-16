"""
phase1_pipeline.py
------------------
Orchestrates the Phase 1 AgriIndex processing flow.

Pipeline order
--------------
1.  run_search()          – query DuckDuckGo and collect raw URLs
2.  filter_urls()         – normalise, deduplicate, reject blocked domains
3.  fetch_page()          – download HTML for each URL
4.  parse_html()          – extract title, meta, and clean text
5.  extract_contacts()    – find emails and phone numbers
6.  extract_keywords()    – count per-set keyword hits
7.  run_rule_classifier() – score relevance and determine page type
8.  run_llama_summary()   – generate a short summary (stub in Phase 1)
9.  store_results()       – persist everything to the database

Each step receives the accumulated context dictionary for a single URL so
individual steps can be tested in isolation without running the full pipeline.

Phase 2+ could extend this module to:
- Parallelise the fetch/parse/extract steps across multiple workers.
- Add a robots.txt check before fetching.
- Schedule re-crawls based on content freshness.
- Feed discovered outbound links into a deeper crawl queue.
"""

from typing import Any, Dict, List, Optional

import yaml

from agriindex.classifiers.rule_classifier import run_rule_classifier
from agriindex.config import settings
from agriindex.db import database
from agriindex.extractors.contact_extractor import extract_contacts
from agriindex.extractors.entity_extractor import extract_entities
from agriindex.extractors.keyword_extractor import extract_keywords
from agriindex.fetchers.page_fetcher import fetch_page
from agriindex.filters.url_filters import filter_urls
from agriindex.llm.llama_processor import run_llama_summary
from agriindex.parsers.html_parser import parse_html
from agriindex.search.duckduckgo_search import run_search
from agriindex.utils.hashing import sha256_text
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step: store results
# ---------------------------------------------------------------------------

def store_results(
    context: Dict[str, Any],
    query: str,
    result_rank: Optional[int] = None,
    db_path: Optional[str] = None,
) -> None:
    """
    Persist the fully enriched page context to the database.

    This function is the final step of the Phase 1 pipeline.  It writes to
    the following tables: ``urls``, ``pages``, ``entities``, ``contacts``,
    ``search_discovery``.

    Parameters
    ----------
    context : dict
        Accumulated pipeline context for a single URL.  Must contain at
        least ``raw_url``, ``canonical_url``, and ``domain``.
    query : str
        The DDG search query that led to this URL.
    result_rank : int, optional
        Position of this URL in the search result list.
    db_path : str, optional
        Override the default database path from ``settings``.
    """
    raw_url = context["raw_url"]
    canonical_url = context["canonical_url"]
    domain = context["domain"]

    # -- urls table --
    url_id = database.insert_url(raw_url, canonical_url, domain, db_path=db_path)

    # -- pages table --
    page_record = {
        "url_id": url_id,
        "fetched_at": context.get("fetched_at"),
        "http_status": context.get("status_code"),
        "page_title": context.get("page_title"),
        "meta_description": context.get("meta_description"),
        "cleaned_text": context.get("cleaned_text"),
        "word_count": context.get("word_count"),
        "summary": context.get("summary"),
        "topics": context.get("topics"),
        "keywords": context.get("keyword_hits"),
        "relevance_cornbelt_ai": context.get("relevance_cornbelt_ai"),
        "relevance_investor": context.get("relevance_investor"),
        "page_type": context.get("page_type"),
        "content_hash": context.get("content_hash"),
    }
    page_id = database.insert_page(page_record, db_path=db_path)

    # -- entities table --
    entities = context.get("entities", [])
    if entities:
        database.insert_entities(page_id, entities, db_path=db_path)

    # -- contacts table --
    contacts = context.get("contacts", [])
    if contacts:
        database.insert_contacts(page_id, contacts, db_path=db_path)

    # -- search_discovery table --
    database.insert_discovery(url_id, query, result_rank=result_rank, db_path=db_path)

    logger.info("Stored results for %s (page_id=%d)", canonical_url, page_id)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    limit: int = 20,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run the full Phase 1 pipeline for a search query.

    Parameters
    ----------
    query : str
        The DuckDuckGo search query string.
    limit : int
        Maximum number of search results to process.
    db_path : str, optional
        Override the default database path from ``settings``.

    Returns
    -------
    list of dict
        One context dictionary per successfully processed URL, containing all
        extracted fields.  URLs that fail to fetch are included with their
        error details but omitted from database storage.
    """
    logger.info("=== Phase 1 pipeline START  query=%r  limit=%d ===", query, limit)
    database.init_db(db_path=db_path)

    # Load the blocked-domain list once for this pipeline run
    try:
        with open(settings.BLOCKED_DOMAINS_PATH, "r", encoding="utf-8") as _fh:
            blocked_domains: List[str] = yaml.safe_load(_fh).get("blocked_domains", [])
    except FileNotFoundError:
        logger.warning("blocked_domains.yaml not found; proceeding with empty block list")
        blocked_domains = []

    # ------------------------------------------------------------------ #
    # 1. Search
    # ------------------------------------------------------------------ #
    search_records = run_search(query, limit=limit)
    logger.info("Search returned %d result records", len(search_records))

    # ------------------------------------------------------------------ #
    # 2. Filter & normalise
    # ------------------------------------------------------------------ #
    # filter_urls accepts and returns lists of dicts; it annotates each
    # surviving record with normalized_url, domain, and kept_reason.
    filtered = filter_urls(search_records, blocked_domains)
    logger.info("After filtering: %d URLs to process", len(filtered))

    results: List[Dict[str, Any]] = []

    for rank, record in enumerate(filtered, start=1):
        raw_url: str = record["url"]
        canonical_url: str = record["normalized_url"]
        domain: str = record.get("domain") or ""

        logger.info("[%d/%d] Processing %s", rank, len(filtered), canonical_url)
        context: Dict[str, Any] = {
            "raw_url": raw_url,
            "canonical_url": canonical_url,
            "domain": domain,
        }

        # ---------------------------------------------------------------- #
        # 3. Fetch
        # ---------------------------------------------------------------- #
        fetch_result = fetch_page(canonical_url)
        context.update({
            "status_code": fetch_result["status_code"],
            "fetched_at": fetch_result["fetched_at"],
            "html": fetch_result["html"],
            "fetch_error": fetch_result["error"],
        })

        if fetch_result["error"] or not fetch_result["html"]:
            logger.warning("Skipping %s: %s", canonical_url, fetch_result["error"])
            results.append(context)
            continue

        # ---------------------------------------------------------------- #
        # 4. Parse HTML
        # ---------------------------------------------------------------- #
        parsed = parse_html(fetch_result["html"], url=canonical_url)
        context.update(parsed)

        cleaned_text = parsed.get("cleaned_text", "")
        context["content_hash"] = sha256_text(cleaned_text)

        # ---------------------------------------------------------------- #
        # 5. Extract contacts
        # ---------------------------------------------------------------- #
        context["contacts"] = extract_contacts(cleaned_text)

        # ---------------------------------------------------------------- #
        # 6. Extract keywords
        # ---------------------------------------------------------------- #
        context["keyword_hits"] = extract_keywords(cleaned_text)

        # ---------------------------------------------------------------- #
        # 7. Classify
        # ---------------------------------------------------------------- #
        classification = run_rule_classifier(
            canonical_url,
            context["keyword_hits"],
            page_data=parsed,
        )
        context.update(classification)

        # ---------------------------------------------------------------- #
        # 8. LLM summary
        # ---------------------------------------------------------------- #
        llm_result = run_llama_summary(cleaned_text)
        context.update(llm_result)

        # ---------------------------------------------------------------- #
        # 9. Entity extraction (optional — requires spaCy)
        # ---------------------------------------------------------------- #
        context["entities"] = extract_entities(cleaned_text)

        # ---------------------------------------------------------------- #
        # 10. Store
        # ---------------------------------------------------------------- #
        store_results(context, query=query, result_rank=rank, db_path=db_path)

        results.append(context)

    logger.info("=== Phase 1 pipeline DONE  processed=%d ===", len(results))
    return results
