"""
ddgs_provider.py
----------------
DuckDuckGo search provider using the ``ddgs`` PyPI package.
"""

import warnings
from typing import Dict, List

from agriindex.utils.logging_utils import get_logger
from agriindex.search.duckduckgo_search import normalize_search_result

logger = get_logger(__name__)

def fetch_ddgs_results(query: str, limit: int) -> List[Dict]:
    """
    Fetch normalized results from DuckDuckGo via the ``ddgs`` package.
    
    Returns an empty list on failure or if 0 results are returned.
    """
    DDGS_cls = None
    
    try:
        from ddgs import DDGS as DDGS_cls  # type: ignore[import]
    except ImportError:
        logger.info("Import from 'ddgs' failed. Trying 'duckduckgo_search' fallback.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                from duckduckgo_search import DDGS as DDGS_cls  # type: ignore[import]
        except ImportError:
            logger.error("Neither ddgs nor duckduckgo_search packages are installed.")
            return []

    raw_results: List[Dict] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            client = DDGS_cls()
        
        # Handle versions that support context manager vs those that don't
        if hasattr(client, "__enter__"):
            with client as ddgs_client:
                results_iter = ddgs_client.text(query, max_results=limit)
                for result in results_iter:
                    raw_results.append(result)
        else:
            results_iter = client.text(query, max_results=limit)
            for result in results_iter:
                raw_results.append(result)
                
        if not raw_results:
            logger.warning("DDGS_provider executed successfully but returned 0 raw results.")
            return []
            
    except Exception as exc:  # noqa: BLE001
        logger.error("DDGS_provider request failed: %s", exc)
        return []

    results: List[Dict] = []
    seen_urls = set()
    rank = 1

    for raw_index, raw in enumerate(raw_results, start=1):
        record = normalize_search_result(raw, query=query, rank=rank)
        if not record:
            continue
            
        url = record["url"]
        if url in seen_urls:
            continue

        seen_urls.add(url)
        results.append(record)
        rank += 1

    return results
