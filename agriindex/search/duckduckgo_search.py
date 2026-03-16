"""
duckduckgo_search.py
--------------------
Run DuckDuckGo searches and return structured result records.

Phase 1 uses the ``duckduckgo_search`` PyPI package which wraps DDG's
unofficial API without requiring an API key.

Phase 2+ could extend this module to:
- Support additional search engines (Bing, Google CSE, etc.)
- Implement pagination to retrieve more than one "page" of results
- Cache recent queries to avoid redundant network calls

Swap point: replace ``_fetch_ddg_results`` with a different backend
(e.g. Bing, Google CSE) while keeping ``run_search`` and
``normalize_search_result`` unchanged.
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Set

from agriindex.config import settings
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure transformation helper (easy to unit-test without network calls)
# ---------------------------------------------------------------------------

def normalize_search_result(raw_result: Dict, query: str, rank: int) -> Dict:
    """
    Convert a raw DuckDuckGo result dict into the canonical AgriIndex format.

    This function is a pure transformation – it never makes network calls – so
    it is straightforward to unit-test with synthetic inputs.

    Parameters
    ----------
    raw_result : dict
        A single result dict as returned by the ``duckduckgo_search`` package.
        Expected keys: ``href`` (or ``url``), ``title``, ``body`` (or
        ``snippet``).  Missing keys are tolerated and produce empty strings.
    query : str
        The original search query that produced this result.
    rank : int
        1-based position of this result in the ordered result list.

    Returns
    -------
    dict
        Normalised record with the following keys:

        ``query``
            The search query string.
        ``rank``
            1-based result position.
        ``title``
            Page title as reported by DuckDuckGo.
        ``url``
            Result URL (``href`` field preferred; falls back to ``url``).
        ``snippet``
            Short descriptive text (``body`` field preferred; falls back to
            ``snippet``).
        ``discovered_at``
            ISO-8601 UTC timestamp string recording when the result was seen.
    """
    url = raw_result.get("href") or raw_result.get("url", "")
    title = raw_result.get("title", "")
    snippet = raw_result.get("body") or raw_result.get("snippet", "")
    discovered_at = datetime.now(tz=timezone.utc).isoformat()

    return {
        "query": query,
        "rank": rank,
        "title": title,
        "url": url,
        "snippet": snippet,
        "discovered_at": discovered_at,
    }


# ---------------------------------------------------------------------------
# Backend: DuckDuckGo  (swap this function to change search provider)
# ---------------------------------------------------------------------------

def _fetch_ddg_results(query: str, limit: int) -> List[Dict]:
    """
    Fetch raw results from DuckDuckGo via the ``duckduckgo_search`` package.

    This private function isolates all DDG-specific logic so that a different
    search backend can be plugged in by replacing only this function.

    Parameters
    ----------
    query : str
        Search query string.
    limit : int
        Maximum number of results to request from DDG.

    Returns
    -------
    list of dict
        Raw result dicts as returned by ``DDGS.text()``.  Returns an empty
        list if the package is not installed or if the request fails.
    """
    # -- SWAP POINT: replace the block below with a different search backend --
    try:
        from duckduckgo_search import DDGS  # type: ignore[import]
    except ImportError:
        logger.warning(
            "duckduckgo_search package not installed. "
            "Returning empty result list.  "
            "Install with: pip install duckduckgo-search"
        )
        return []

    raw_results: List[Dict] = []
    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=limit):
                raw_results.append(result)
    except Exception as exc:  # noqa: BLE001
        logger.error("DDG search request failed: %s", exc)
    # -- end of swap point ---------------------------------------------------

    return raw_results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_search(query: str, limit: int = settings.DEFAULT_SEARCH_LIMIT) -> List[Dict]:
    """
    Execute a DuckDuckGo web search and return structured result records.

    Each result is normalised into a consistent dictionary format shared across
    all search backends.  Exact-duplicate URLs are removed while the first
    occurrence is kept, preserving the original result order.

    Parameters
    ----------
    query : str
        The search query string (e.g. ``"corn disease management"``).
    limit : int
        Maximum number of result records to return.  Defaults to
        ``settings.DEFAULT_SEARCH_LIMIT``.

    Returns
    -------
    list of dict
        Normalised result records ordered by search rank.  Each dict contains:

        ``query``
            The original search query.
        ``rank``
            1-based position in the result list (after deduplication).
        ``title``
            Page title as reported by DuckDuckGo.
        ``url``
            Result URL.
        ``snippet``
            Short descriptive text for the result.
        ``discovered_at``
            ISO-8601 UTC timestamp string.

        Returns an empty list on total failure rather than raising an
        exception.

    Notes
    -----
    A courtesy sleep of ``settings.DDG_SLEEP_SECONDS`` is applied after the
    request to avoid hammering the search service.

    Page crawling, HTML fetching, and database writes are explicitly **out of
    scope** for this module.
    """
    logger.info("DDG search: query=%r  limit=%d", query, limit)

    raw_results = _fetch_ddg_results(query, limit)
    logger.debug("DDG backend returned %d raw results", len(raw_results))

    results: List[Dict] = []
    seen_urls: Set[str] = set()
    rank = 1

    for raw_index, raw in enumerate(raw_results, start=1):
        record = normalize_search_result(raw, query=query, rank=rank)
        url = record["url"]

        if not url:
            logger.debug("Skipping result with empty URL (raw position %d)", raw_index)
            continue

        # Deduplicate exact duplicate URLs; first occurrence wins
        if url in seen_urls:
            logger.debug("Duplicate URL skipped (raw position %d): %s", raw_index, url)
            continue

        seen_urls.add(url)
        results.append(record)
        rank += 1

    logger.info("run_search returning %d results for query=%r", len(results), query)
    time.sleep(settings.DDG_SLEEP_SECONDS)
    return results


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    test_query = "corn disease management"
    print(f"Running test search: {test_query!r}\n")
    records = run_search(test_query, limit=5)
    if records:
        print(json.dumps(records, indent=2))
    else:
        print("No results returned.")
