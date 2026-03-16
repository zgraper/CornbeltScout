"""
url_filters.py
--------------
URL normalisation, deduplication, and domain-level filtering.

Responsibilities
----------------
- Parse URLs and extract their domain.
- Convert URLs to a canonical form (lowercase scheme/host, sorted query
  parameters, stripped tracking parameters, trailing-slash normalisation).
- Remove duplicate canonical URLs within a batch.
- Reject URLs whose domain appears in ``blocked_domains.yaml``.

Phase 2+ could extend this module to:
- Apply path-pattern allow/block lists.
- Score URL priority based on URL depth or known high-value domains.
- Handle JavaScript single-page-app URL fragments.
"""

import re
from typing import List, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import yaml

from agriindex.config import settings
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tracking / noise query parameters to strip from URLs
# ---------------------------------------------------------------------------
_STRIP_PARAMS: Set[str] = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "ref", "source", "mc_cid", "mc_eid",
}


def _load_blocked_domains() -> Set[str]:
    """Load the blocked domain list from ``blocked_domains.yaml``."""
    try:
        with open(settings.BLOCKED_DOMAINS_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return set(data.get("blocked_domains", []))
    except FileNotFoundError:
        logger.warning("blocked_domains.yaml not found at %s", settings.BLOCKED_DOMAINS_PATH)
        return set()


# Loaded once at import time; reload the module to refresh.
_BLOCKED_DOMAINS: Set[str] = _load_blocked_domains()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def extract_domain(url: str) -> str:
    """
    Return the lower-case registered domain from *url*.

    Leading ``www.`` is stripped so that ``www.example.com`` and
    ``example.com`` are treated as the same domain.

    Parameters
    ----------
    url : str

    Returns
    -------
    str
        Lower-case domain string, or an empty string if parsing fails.
    """
    try:
        host = urlparse(url).netloc.lower()
        return re.sub(r"^www\.", "", host)
    except Exception:  # noqa: BLE001
        return ""


def canonicalize_url(url: str) -> str:
    """
    Return a canonical form of *url* suitable for deduplication.

    Normalisation steps applied
    ---------------------------
    1. Lower-case the scheme and host.
    2. Remove the default port (80 for http, 443 for https).
    3. Strip tracking query parameters.
    4. Sort remaining query parameters alphabetically.
    5. Strip URL fragments (``#anchor``).
    6. Normalise trailing slash on the path (kept only for root paths).

    Parameters
    ----------
    url : str

    Returns
    -------
    str
        Canonical URL, or the original *url* if parsing fails.
    """
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Strip leading www. for canonical form
        netloc = re.sub(r"^www\.", "", netloc)

        # Remove default ports
        netloc = re.sub(r":80$", "", netloc) if scheme == "http" else netloc
        netloc = re.sub(r":443$", "", netloc) if scheme == "https" else netloc

        # Strip tracking params and sort the rest
        qparams = [
            (k, v)
            for k, v in parse_qsl(parsed.query)
            if k.lower() not in _STRIP_PARAMS
        ]
        qparams.sort()

        # Normalise path: strip trailing slash unless root
        path = parsed.path.rstrip("/") or "/"

        canonical = urlunparse((
            scheme,
            netloc,
            path,
            "",                   # params (semicolon-separated; rarely used)
            urlencode(qparams),
            "",                   # fragment stripped
        ))
        return canonical
    except Exception:  # noqa: BLE001
        return url


def is_blocked(url: str) -> bool:
    """
    Return True if the URL's domain is on the blocked-domain list.

    Parameters
    ----------
    url : str
    """
    domain = extract_domain(url)
    # Check both the full subdomain and the parent domain
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in _BLOCKED_DOMAINS:
            return True
    return False


def filter_urls(raw_urls: List[str]) -> List[Tuple[str, str, str]]:
    """
    Filter, normalise, and deduplicate a list of raw URLs.

    Processing order
    ----------------
    1. Reject non-HTTP(S) URLs.
    2. Reject URLs from blocked domains.
    3. Canonicalize each URL.
    4. Deduplicate on the canonical form (first occurrence wins).

    Parameters
    ----------
    raw_urls : list of str
        Raw URLs as returned by the search module.

    Returns
    -------
    list of (raw_url, canonical_url, domain) tuples
        Only accepted, unique URLs are included.
    """
    seen: Set[str] = set()
    results: List[Tuple[str, str, str]] = []

    for raw in raw_urls:
        # Must be an absolute HTTP/HTTPS URL
        scheme = urlparse(raw).scheme.lower()
        if scheme not in ("http", "https"):
            logger.debug("Skipping non-HTTP URL: %s", raw)
            continue

        if is_blocked(raw):
            logger.debug("Skipping blocked domain: %s", raw)
            continue

        canonical = canonicalize_url(raw)
        if canonical in seen:
            logger.debug("Duplicate canonical URL skipped: %s", canonical)
            continue

        seen.add(canonical)
        domain = extract_domain(raw)
        results.append((raw, canonical, domain))

    logger.info("filter_urls: %d raw → %d accepted", len(raw_urls), len(results))
    return results
