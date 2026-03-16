"""
page_fetcher.py
---------------
Download HTML content for a single URL and return structured fetch metadata.

Responsibilities
----------------
- Send an HTTP GET request with a realistic User-Agent string.
- Follow HTTP redirects and record the final URL.
- Validate that the response is HTML (or likely HTML) before returning the body.
- Return a structured result dict so callers can handle errors uniformly.
- Handle timeouts, connection errors, SSL issues, and encoding problems
  gracefully without raising exceptions to callers.

Out of scope for this module
-----------------------------
- HTML parsing or content cleaning  (see ``parsers/`` and ``extractors/``)
- Readability / main-content extraction
- Database writes
- Link extraction / crawling

Phase 2+ could extend this module to:
- Support rotating proxy pools.
- Handle JavaScript-rendered pages via Playwright or Splash.
- Implement retry logic with exponential back-off.
- Honour ``robots.txt`` and crawl-delay directives.
- Cache responses to disk to avoid re-fetching during development.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

from agriindex.config import settings
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MIME types that we consider "HTML" for the purpose of deciding whether to
# keep the response body.
# ---------------------------------------------------------------------------
_HTML_MIME_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "application/xhtml",
    "text/xhtml",
}

# File extensions that strongly suggest the URL points to an HTML page even
# when no Content-Type header is present.
_HTML_EXTENSIONS = {".html", ".htm", ".shtml", ".php", ".asp", ".aspx", ".jsp"}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_headers() -> Dict[str, str]:
    """
    Return a dict of HTTP request headers suitable for fetching web pages.

    The User-Agent is taken from ``agriindex.config.settings.USER_AGENT`` so
    it can be overridden via the ``AGRIINDEX_USER_AGENT`` environment variable
    without touching this module.

    Returns
    -------
    dict
        HTTP header key/value pairs.
    """
    # TODO (Phase 2): add Accept-Encoding for brotli when the ``brotli``
    #                 package is installed.
    return {
        "User-Agent": settings.USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        # ``requests`` automatically decompresses gzip / deflate responses when
        # this header is present, so no manual decompression is needed.
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def is_html_response(content_type: Optional[str], url: str) -> bool:
    """
    Return ``True`` when the response should be treated as HTML.

    The check uses two signals:

    1. The ``Content-Type`` header (primary signal).  A response is considered
       HTML if the MIME type portion (before the first ``';'``) matches one of
       the known HTML MIME types.

    2. The URL file extension (fallback when ``content_type`` is ``None`` or
       empty).  URLs ending with ``.html``, ``.htm``, ``.php``, etc. are
       treated as HTML.

    Parameters
    ----------
    content_type : str or None
        Value of the HTTP ``Content-Type`` response header.
    url : str
        The URL that was fetched (used only as an extension fallback).

    Returns
    -------
    bool
    """
    if content_type:
        # Strip charset and boundary parameters, e.g. "text/html; charset=utf-8"
        mime = content_type.split(";")[0].strip().lower()
        if mime in _HTML_MIME_TYPES:
            return True
        # Reject anything that is clearly not HTML (images, JSON, PDF, …)
        return False

    # No Content-Type header — fall back to URL extension heuristic
    try:
        path = urlparse(url).path.lower()
    except (ValueError, AttributeError):
        return False

    _, dot, ext = path.rpartition(".")
    if dot and f".{ext}" in _HTML_EXTENSIONS:
        return True

    # No extension and no Content-Type: assume it *might* be HTML (e.g. bare
    # domain or path with no extension are common for HTML pages).
    return not dot  # True for paths like "/" or "/about", False for "/file.bin"


def fetch_page(url: str, timeout: int = 15) -> Dict[str, Any]:
    """
    Fetch the HTML content of *url* and return a structured result dictionary.

    The function performs a single HTTP GET request, follows redirects, and
    validates that the response body is HTML before returning it.  All errors
    are caught and recorded in the ``error`` field so callers never need to
    wrap this function in a try/except.

    Parameters
    ----------
    url : str
        The URL to fetch.
    timeout : int, optional
        Request timeout in seconds.  Defaults to 15.

    Returns
    -------
    dict with keys:

    ``url``
        Original requested URL.
    ``final_url``
        URL after following any HTTP redirects.
    ``status_code``
        HTTP status code (``int``), or ``None`` on connection error.
    ``content_type``
        Value of the ``Content-Type`` response header, or ``None``.
    ``html``
        Response body decoded to a string, or ``""`` on error / non-HTML.
    ``fetched_at``
        ISO-8601 UTC timestamp string recording when the request was made.
    ``fetch_success``
        ``True`` if the page was fetched and identified as HTML, else ``False``.
    ``error``
        Human-readable error message string, or ``None`` on success.
    ``response_headers``
        Dict of response HTTP headers, or ``{}`` on connection error.

    Notes
    -----
    - Gzip / deflate decompression is handled automatically by ``requests``.
    - SSL certificate errors are caught and reported in ``error``; the fetch
      is **not** retried with verification disabled.
    - Encoding is handled by ``requests`` (``response.text``); if the server
      sends a non-UTF-8 body without declaring the encoding, ``requests`` will
      fall back to ISO-8859-1 per the HTTP spec.

    TODO (Phase 2): add retry logic with exponential back-off.
    TODO (Phase 2): check robots.txt before fetching.
    TODO (Phase 2): apply per-domain rate limiting / crawl-delay.
    """
    # TODO (Phase 2): consult a robots.txt cache before issuing the request.

    fetched_at = datetime.now(timezone.utc).isoformat()
    result: Dict[str, Any] = {
        "url": url,
        "final_url": url,
        "status_code": None,
        "content_type": None,
        "html": "",
        "fetched_at": fetched_at,
        "fetch_success": False,
        "error": None,
        "response_headers": {},
    }

    try:
        response = requests.get(
            url,
            headers=build_headers(),
            timeout=timeout,
            allow_redirects=True,
            # SSL verification is enabled by default; errors are caught below.
        )

        result["final_url"] = response.url
        result["status_code"] = response.status_code
        result["response_headers"] = dict(response.headers)
        content_type: Optional[str] = response.headers.get("Content-Type")
        result["content_type"] = content_type

        # Raise for 4xx / 5xx so they land in the HTTPError handler below.
        response.raise_for_status()

        if not is_html_response(content_type, response.url):
            result["error"] = (
                f"Non-HTML content-type: {content_type!r}"
                if content_type
                else "Non-HTML response (no Content-Type header)"
            )
            logger.debug(
                "Skipping non-HTML response for %s (Content-Type: %r)",
                url,
                content_type,
            )
            return result

        # ``response.text`` decodes the body using the charset from the
        # Content-Type header, falling back to ISO-8859-1 per the HTTP spec.
        result["html"] = response.text
        result["fetch_success"] = True
        logger.debug("Fetched %s  [%d]", url, response.status_code)

    except requests.exceptions.Timeout:
        result["error"] = f"Request timed out after {timeout}s"
        logger.warning("Timeout fetching %s", url)

    except requests.exceptions.TooManyRedirects:
        result["error"] = "Too many redirects"
        logger.warning("Too many redirects for %s", url)

    except requests.exceptions.SSLError as exc:
        result["error"] = f"SSL error: {exc}"
        logger.warning("SSL error fetching %s: %s", url, exc)

    except requests.exceptions.ConnectionError as exc:
        result["error"] = f"Connection error: {exc}"
        logger.warning("Connection error fetching %s: %s", url, exc)

    except requests.exceptions.HTTPError as exc:
        result["error"] = str(exc)
        logger.warning("HTTP error for %s: %s", url, exc)

    except requests.exceptions.RequestException as exc:
        result["error"] = str(exc)
        logger.warning("Request error for %s: %s", url, exc)

    except UnicodeDecodeError as exc:
        result["error"] = f"Encoding error: {exc}"
        logger.warning("Encoding error for %s: %s", url, exc)

    # TODO (Phase 2): apply per-domain rate limiting here before returning.

    return result


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    _TEST_URL = "https://cropwatch.unl.edu/"

    print("=" * 70)
    print(f"AgriIndex page_fetcher demo — fetching: {_TEST_URL}")
    print("=" * 70)

    _result = fetch_page(_TEST_URL)

    # Pretty-print everything except the full HTML body (too noisy)
    _display = {k: v for k, v in _result.items() if k != "html"}
    _display["html_length"] = len(_result.get("html") or "")
    _display["html_snippet"] = (_result.get("html") or "")[:200]

    print(json.dumps(_display, indent=2, default=str))
    print()
    if _result["fetch_success"]:
        print(f"✓ Successfully fetched {_result['html_length']} characters of HTML.")
    else:
        print(f"✗ Fetch failed: {_result['error']}")
