"""
page_fetcher.py
---------------
Download HTML content for a given URL.

Responsibilities
----------------
- Send an HTTP GET request with a realistic User-Agent string.
- Respect the configured request timeout.
- Return the raw HTML text and the final (redirected) URL.
- Return a structured result dict so callers can handle errors uniformly.

Phase 2+ could extend this module to:
- Support rotating proxy pools.
- Handle JavaScript-rendered pages via Playwright or Splash.
- Implement retry logic with exponential back-off.
- Honour ``robots.txt`` and crawl-delay directives.
- Cache responses to disk to avoid re-fetching during development.
"""

from datetime import datetime, timezone
from typing import Any, Dict

import requests

from agriindex.config import settings
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)

_HEADERS = {
    "User-Agent": settings.USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def fetch_page(url: str) -> Dict[str, Any]:
    """
    Fetch the HTML content of *url* and return a result dictionary.

    Parameters
    ----------
    url : str
        The URL to fetch.

    Returns
    -------
    dict with keys:
        ``url``         – original requested URL
        ``final_url``   – URL after redirects
        ``status_code`` – HTTP status code (int), or None on connection error
        ``html``        – response body as a string, or empty string on error
        ``fetched_at``  – ISO-8601 UTC timestamp string
        ``error``       – error message string, or None on success
    """
    fetched_at = datetime.now(timezone.utc).isoformat()
    result: Dict[str, Any] = {
        "url": url,
        "final_url": url,
        "status_code": None,
        "html": "",
        "fetched_at": fetched_at,
        "error": None,
    }

    try:
        response = requests.get(
            url,
            headers=_HEADERS,
            timeout=settings.REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        result["final_url"] = response.url
        result["status_code"] = response.status_code
        response.raise_for_status()
        result["html"] = response.text
        logger.debug("Fetched %s  [%d]", url, response.status_code)
    except requests.exceptions.Timeout:
        result["error"] = f"Request timed out after {settings.REQUEST_TIMEOUT}s"
        logger.warning("Timeout fetching %s", url)
    except requests.exceptions.TooManyRedirects:
        result["error"] = "Too many redirects"
        logger.warning("Too many redirects for %s", url)
    except requests.exceptions.HTTPError as exc:
        result["error"] = str(exc)
        logger.warning("HTTP error for %s: %s", url, exc)
    except requests.exceptions.RequestException as exc:
        result["error"] = str(exc)
        logger.warning("Request error for %s: %s", url, exc)

    return result
