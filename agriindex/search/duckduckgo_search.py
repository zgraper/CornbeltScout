"""
duckduckgo_search.py
--------------------
Run DuckDuckGo searches and return raw result URLs.

Phase 1 uses the ``duckduckgo_search`` PyPI package which wraps DDG's
unofficial API without requiring an API key.

Phase 2+ could extend this module to:
- Support additional search engines (Bing, Google CSE, etc.)
- Implement pagination to retrieve more than one "page" of results
- Cache recent queries to avoid redundant network calls
"""

import time
from typing import List

from agriindex.config import settings
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_search(query: str, limit: int = settings.DEFAULT_SEARCH_LIMIT) -> List[str]:
    """
    Execute a DuckDuckGo web search and return a list of result URLs.

    Attempts to import ``duckduckgo_search`` at call-time so that the rest of
    the pipeline can function (with mocked data) even when the package is not
    installed.

    Parameters
    ----------
    query : str
        The search query string (e.g. ``"corn disease management"``).
    limit : int
        Maximum number of URLs to return.  Defaults to
        ``settings.DEFAULT_SEARCH_LIMIT``.

    Returns
    -------
    list of str
        Raw result URLs in the order returned by DuckDuckGo.  May contain
        fewer items than *limit* if DDG returns fewer results.

    Notes
    -----
    A courtesy sleep of ``settings.DDG_SLEEP_SECONDS`` is applied after the
    request to avoid hammering the search service.
    """
    logger.info("DDG search: query=%r  limit=%d", query, limit)

    try:
        from duckduckgo_search import DDGS  # type: ignore[import]
    except ImportError:
        logger.warning(
            "duckduckgo_search package not installed. "
            "Returning empty result list.  "
            "Install with: pip install duckduckgo-search"
        )
        return []

    urls: List[str] = []
    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=limit):
                href = result.get("href") or result.get("url", "")
                if href:
                    urls.append(href)
    except Exception as exc:  # noqa: BLE001
        logger.error("DDG search failed: %s", exc)

    logger.debug("DDG returned %d raw URLs", len(urls))
    time.sleep(settings.DDG_SLEEP_SECONDS)
    return urls
