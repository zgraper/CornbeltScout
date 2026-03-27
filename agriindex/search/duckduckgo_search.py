"""
duckduckgo_search.py
--------------------
Run DuckDuckGo searches and return structured result records.

Phase 1 initially used the ``duckduckgo_search`` PyPI package, which was
later renamed to ``ddgs``. This module supports both for backwards
compatibility, and includes a fallback to a Selenium-based search if the
API package fails or returns zero results.

Phase 2+ could extend this module to:
- Support additional search engines (Bing, Google CSE, etc.)
- Implement pagination to retrieve more than one "page" of results
- Cache recent queries to avoid redundant network calls
"""

import time
from datetime import datetime, timezone
from typing import Dict, List

from agriindex.config import settings
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure transformation helper (easy to unit-test without network calls)
# ---------------------------------------------------------------------------

def normalize_search_result(raw_result: Dict, query: str, rank: int) -> Dict:
    """
    Convert a raw DuckDuckGo result dict into the canonical CornScout format.

    This function is a pure transformation – it never makes network calls – so
    it is straightforward to unit-test with synthetic inputs.

    Parameters
    ----------
    raw_result : dict
        A single result dict as returned by the search providers.
        Expected keys: ``href`` (or ``url``), ``title``, ``body`` (or
        ``snippet``). Missing keys are tolerated and produce empty strings.
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
    if not isinstance(raw_result, dict):
        logger.warning("Normalizer rejected non-dict result: %r", raw_result)
        return {}

    url = raw_result.get("href") or raw_result.get("url") or ""
    title = raw_result.get("title", "")
    snippet = raw_result.get("body") or raw_result.get("snippet") or ""
    
    # Defensive casting
    url = str(url).strip()
    title = str(title).strip()
    snippet = str(snippet).strip()

    if not url:
        logger.warning("Normalizer rejected result with missing or empty URL: %r", raw_result)
        return {}

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
# Public API
# ---------------------------------------------------------------------------

def run_search(query: str, limit: int = settings.DEFAULT_SEARCH_LIMIT) -> List[Dict]:
    """
    Execute a DuckDuckGo web search and return structured result records.

    This function coordinates search providers. It first tries the primary
    `ddgs` package. If that fails or returns zero results, it falls back
    to a Selenium-based scraper.

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

    # 1. Try DDGS primary provider
    try:
        from agriindex.search.providers.ddgs_provider import fetch_ddgs_results
        results = fetch_ddgs_results(query, limit)
        if results:
            logger.info("DDG primary provider (ddgs) succeeded. Returned %d results.", len(results))
            time.sleep(settings.DDG_SLEEP_SECONDS)
            return results
        else:
            logger.warning("DDG primary provider (ddgs) returned 0 results. Proceeding to fallback.")
    except Exception as exc:
        logger.error("DDG primary provider (ddgs) failed with exception: %s. Proceeding to fallback.", exc)

    # 2. Try Selenium fallback provider
    try:
        from agriindex.search.providers.selenium_provider import fetch_selenium_results
        logger.info("Starting Selenium fallback provider...")
        results = fetch_selenium_results(query, limit)
        if results:
            logger.info("DDG fallback provider (Selenium) succeeded. Returned %d results.", len(results))
            time.sleep(settings.DDG_SLEEP_SECONDS)
            return results
        else:
            logger.warning("DDG fallback provider (Selenium) returned 0 results.")
    except Exception as exc:
        logger.error("DDG fallback provider (Selenium) failed with exception: %s", exc)

    logger.error("All search providers failed or returned 0 results for query=%r", query)
    return []


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(level=logging.INFO)
    test_query = "corn disease management"
    print(f"Running test search: {test_query!r}\n")
    records = run_search(test_query, limit=5)
    if records:
        print(json.dumps(records, indent=2))
    else:
        print("No results returned.")
