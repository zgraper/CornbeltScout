"""
html_parser.py
--------------
Extract structured content from raw HTML.

Responsibilities
----------------
- Parse the HTML with BeautifulSoup.
- Extract the page title (``<title>`` tag).
- Extract the meta description (``<meta name="description">``).
- Extract clean, readable body text via ``trafilatura`` (strips nav/ads/boiler-
  plate).  Falls back to BeautifulSoup visible-text extraction when trafilatura
  is unavailable.
- Return word count of the cleaned text.

Phase 2+ could extend this module to:
- Extract Open Graph / Schema.org structured data.
- Extract publication dates and authorship information.
- Detect page language.
- Extract outbound links for deeper crawling.
"""

import re
from typing import Any, Dict

from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_with_trafilatura(html: str) -> str:
    """
    Use ``trafilatura`` to extract main readable text from HTML.

    Returns an empty string if trafilatura is not installed or extraction fails.
    """
    try:
        import trafilatura  # type: ignore[import]

        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        return text or ""
    except ImportError:
        logger.debug("trafilatura not installed; falling back to BeautifulSoup extractor")
        return ""


def _extract_with_beautifulsoup(html: str) -> str:
    """
    Fallback: extract visible text from HTML using BeautifulSoup.

    Removes script, style, and nav elements before extracting text.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore[import]

        soup = BeautifulSoup(html, "html.parser")
        # Remove non-content tags
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except ImportError:
        logger.warning("BeautifulSoup (bs4) not installed; cannot extract text")
        return ""


def _extract_title(html: str) -> str:
    """Return the content of the ``<title>`` tag, or an empty string."""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import]

        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("title")
        return tag.get_text(strip=True) if tag else ""
    except ImportError:
        # Minimal regex fallback when bs4 is absent
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""


def _extract_meta_description(html: str) -> str:
    """Return the content of ``<meta name='description' content='…'>``."""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import]

        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        if tag and tag.get("content"):
            return tag["content"].strip()
        return ""
    except ImportError:
        match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']',
            html,
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_html(html: str, url: str = "") -> Dict[str, Any]:
    """
    Parse *html* and return a dictionary of extracted page data.

    Parameters
    ----------
    html : str
        Raw HTML string fetched from the page.
    url : str, optional
        Source URL (used for logging only).

    Returns
    -------
    dict with keys:
        ``page_title``       – str
        ``meta_description`` – str
        ``cleaned_text``     – str  (main readable body text)
        ``word_count``       – int
    """
    if not html:
        logger.debug("parse_html: empty HTML for %s", url)
        return {
            "page_title": "",
            "meta_description": "",
            "cleaned_text": "",
            "word_count": 0,
        }

    title = _extract_title(html)
    meta_desc = _extract_meta_description(html)

    # Prefer trafilatura; fall back to BeautifulSoup
    cleaned_text = _extract_with_trafilatura(html)
    if not cleaned_text:
        cleaned_text = _extract_with_beautifulsoup(html)

    word_count = len(cleaned_text.split()) if cleaned_text else 0

    logger.debug(
        "parse_html: title=%r  words=%d  url=%s",
        title[:60] if title else "",
        word_count,
        url,
    )

    return {
        "page_title": title,
        "meta_description": meta_desc,
        "cleaned_text": cleaned_text,
        "word_count": word_count,
    }
