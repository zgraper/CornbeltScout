"""
url_filters.py
--------------
URL normalisation, deduplication, and domain-level filtering for CornScout.

Responsibilities
----------------
- Parse URLs and extract their domain.
- Normalize URLs to a canonical form (lowercase scheme/host, stripped
  tracking parameters, sorted query parameters, no fragments).
- Detect and reject obvious non-HTML assets (images, archives, etc.).
- Reject URLs whose domain appears in the caller-supplied blocked list.
- Filter a list of search-result dicts, annotating survivors with metadata.

Phase 2+ could extend this module to:
- Apply path-pattern allow/block lists.
- Score URL priority based on URL depth or known high-value domains.
- Handle JavaScript single-page-app URL fragments.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tracking / noise query parameters to strip during normalization
# ---------------------------------------------------------------------------
_STRIP_PARAMS: Set[str] = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "ref", "source", "mc_cid", "mc_eid",
}

# ---------------------------------------------------------------------------
# File extensions that indicate non-HTML binary assets we skip in Phase 1.
# Raw PDFs are included here because Phase 1 does not yet process them.
# ---------------------------------------------------------------------------
_NON_HTML_EXTENSIONS: Set[str] = {
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp", ".tiff",
    # Documents / archives we are not processing yet
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".rar", ".7z",
    # Executables / installers
    ".exe", ".msi", ".dmg", ".pkg", ".deb", ".rpm",
    # Audio / video
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".wav", ".ogg",
    # Data / feed formats (raw, non-HTML)
    ".xml", ".json", ".csv", ".rss",
    # Other
    ".css", ".js", ".woff", ".woff2", ".ttf",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> Optional[str]:
    """
    Return a normalized form of *url* suitable for deduplication and fetching.

    Normalization steps
    -------------------
    1. Lowercase the scheme and hostname.
    2. Remove default ports (80 for http, 443 for https).
    3. Strip known tracking query parameters (utm_*, fbclid, gclid, etc.).
    4. Sort remaining query parameters alphabetically for stable comparison.
    5. Remove URL fragments (``#anchor``).
    6. Strip trailing slashes from the path (root ``/`` is preserved).

    Parameters
    ----------
    url : str
        Raw URL string to normalize.

    Returns
    -------
    str or None
        Normalized URL string, or ``None`` if *url* is malformed or does not
        have an http/https scheme.
    """
    if not url or not url.strip():
        return None

    try:
        parsed = urlparse(url.strip())
    except Exception:  # noqa: BLE001
        logger.debug("normalize_url: failed to parse %r", url)
        return None

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        # Reject mailto:, tel:, javascript:, and other non-web schemes
        logger.debug("normalize_url: non-http scheme %r in %r", scheme, url)
        return None

    netloc = parsed.netloc.lower()
    if not netloc:
        logger.debug("normalize_url: empty netloc in %r", url)
        return None

    # Remove default ports to keep the canonical form clean
    netloc = re.sub(r":80$", "", netloc) if scheme == "http" else netloc
    netloc = re.sub(r":443$", "", netloc) if scheme == "https" else netloc

    # Strip tracking parameters; sort the survivors for a stable canonical key
    qparams = [
        (k, v)
        for k, v in parse_qsl(parsed.query)
        if k.lower() not in _STRIP_PARAMS
    ]
    qparams.sort()

    # Normalize path: remove trailing slash unless we are at the root
    path = parsed.path.rstrip("/") or "/"

    normalized = urlunparse((
        scheme,
        netloc,
        path,
        "",                   # params field (semicolon-separated; rarely used)
        urlencode(qparams),
        "",                   # fragment stripped intentionally
    ))
    return normalized


def extract_domain(url: str) -> Optional[str]:
    """
    Return the lowercase hostname from *url*, with a leading ``www.`` removed.

    ``www.example.com`` and ``example.com`` are treated as the same domain so
    that blocked-domain lookups and deduplication work correctly.

    Parameters
    ----------
    url : str

    Returns
    -------
    str or None
        Lowercase domain string (e.g. ``"example.com"``), or ``None`` if the
        URL cannot be parsed or yields an empty hostname.
    """
    try:
        host = urlparse(url).netloc.lower()
        if not host:
            return None
        # Strip leading www. subdomain for normalized comparisons
        return re.sub(r"^www\.", "", host)
    except Exception:  # noqa: BLE001
        logger.debug("extract_domain: failed for %r", url)
        return None


def is_blocked_domain(url: str, blocked_domains: List[str]) -> bool:
    """
    Return ``True`` if *url*'s domain (or any parent domain) is blocked.

    Subdomain matching is intentional: if ``facebook.com`` is blocked then
    ``apps.facebook.com`` is also blocked.

    Parameters
    ----------
    url : str
        URL to check.
    blocked_domains : list of str
        Lowercase domain strings to reject (e.g. ``["facebook.com", "x.com"]``).

    Returns
    -------
    bool
    """
    domain = extract_domain(url)
    if not domain:
        return False

    blocked_set: Set[str] = set(blocked_domains)
    # Walk from full subdomain up to the registerable domain and check each
    parts = domain.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in blocked_set:
            logger.debug("is_blocked_domain: %r matched blocked entry %r", url, candidate)
            return True
    return False


def is_probably_non_html_asset(url: str) -> bool:
    """
    Return ``True`` when the URL path ends with a known non-HTML file extension.

    This is a fast, heuristic check based on the URL path alone — no HTTP
    HEAD request is made.  It catches the most common cases (images, PDFs,
    zip files, executables) without being exhaustive.

    Parameters
    ----------
    url : str

    Returns
    -------
    bool
    """
    try:
        path = urlparse(url).path.lower()
    except Exception:  # noqa: BLE001
        return False

    # Strip query-string edge case: take only the path component
    # (urlparse already handles this, but be explicit)
    _, _, ext = path.rpartition(".")
    if ext and f".{ext}" in _NON_HTML_EXTENSIONS:
        logger.debug("is_probably_non_html_asset: %r matched extension .%s", url, ext)
        return True
    return False


def is_valid_candidate_url(
    url: str,
    blocked_domains: List[str],
) -> Tuple[bool, str]:
    """
    Perform all pre-fetch validation checks on a single URL.

    Checks applied (in order)
    -------------------------
    1. Empty or whitespace-only string.
    2. Non-HTTP(S) schemes (mailto:, tel:, javascript:, etc.).
    3. Malformed URL (cannot be parsed).
    4. Non-HTML asset extension (.pdf, .jpg, .zip, etc.).
    5. Blocked domain.

    Parameters
    ----------
    url : str
        Raw URL candidate.
    blocked_domains : list of str
        Lowercase domain strings to reject.

    Returns
    -------
    (bool, str)
        A ``(valid, reason)`` pair.  When *valid* is ``True``, *reason* is a
        short human-readable explanation of why the URL was kept.  When
        *valid* is ``False``, *reason* describes why it was rejected.
    """
    # --- Check 1: empty URL ---
    if not url or not url.strip():
        return False, "empty or whitespace URL"

    # --- Check 2 & 3: scheme and parsability ---
    normalized = normalize_url(url)
    if normalized is None:
        # normalize_url already rejects bad schemes and malformed URLs
        try:
            scheme = urlparse(url).scheme.lower()
        except Exception:  # noqa: BLE001
            scheme = ""
        if scheme and scheme not in ("http", "https"):
            return False, f"non-http scheme: {scheme!r}"
        return False, "malformed or unparseable URL"

    # --- Check 4: non-HTML asset ---
    if is_probably_non_html_asset(url):
        _, _, ext = urlparse(url).path.lower().rpartition(".")
        return False, f"non-HTML asset extension: .{ext}"

    # --- Check 5: blocked domain ---
    if is_blocked_domain(url, blocked_domains):
        domain = extract_domain(url) or url
        return False, f"blocked domain: {domain}"

    return True, "passed all filters"


def filter_urls(
    results: List[Dict],
    blocked_domains: List[str],
) -> List[Dict]:
    """
    Filter, normalize, and deduplicate a list of search-result dictionaries.

    Each dict in *results* must contain at least a ``"url"`` key.  The
    function annotates every surviving dict with:

    ``normalized_url``
        The normalized form of the URL (tracking params removed, lowercase
        scheme/host, no fragment).
    ``domain``
        Lowercase hostname with leading ``www.`` stripped.
    ``kept_reason``
        Short explanation of why this URL passed all filters.

    Rejected results are logged at DEBUG level and excluded from the return
    value.  Duplicate normalized URLs are silently deduplicated (first
    occurrence wins).

    Parameters
    ----------
    results : list of dict
        Search-result records, each containing at least a ``"url"`` field.
        The dicts are **not modified in place**; annotated copies are returned.
    blocked_domains : list of str
        Lowercase domain names to reject (e.g. ``["linkedin.com", "x.com"]``).

    Returns
    -------
    list of dict
        Only valid, unique results are returned, each annotated with
        ``normalized_url``, ``domain``, and ``kept_reason``.
    """
    seen_normalized: Set[str] = set()
    accepted: List[Dict] = []

    for record in results:
        raw_url: str = record.get("url", "") or ""

        # Run all pre-fetch validation checks
        valid, reason = is_valid_candidate_url(raw_url, blocked_domains)

        if not valid:
            logger.debug("filter_urls: REJECT %r — %s", raw_url, reason)
            continue

        # normalize_url is guaranteed non-None here because is_valid_candidate_url passed
        normalized = normalize_url(raw_url)  # type: ignore[assignment]

        # Deduplicate on the normalized URL (first occurrence wins)
        if normalized in seen_normalized:
            logger.debug("filter_urls: DUPLICATE %r (normalized: %r)", raw_url, normalized)
            continue

        seen_normalized.add(normalized)

        domain = extract_domain(raw_url)

        # Build an annotated copy so the original dict is not mutated
        annotated = dict(record)
        annotated["normalized_url"] = normalized
        annotated["domain"] = domain
        annotated["kept_reason"] = reason
        accepted.append(annotated)

    logger.info(
        "filter_urls: %d input → %d accepted, %d rejected/duplicated",
        len(results),
        len(accepted),
        len(results) - len(accepted),
    )
    return accepted


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml  # only needed here for loading the demo blocked list
    from agriindex.config import settings  # noqa: PLC0415

    # Load the default blocked-domain list for the demo
    try:
        with open(settings.BLOCKED_DOMAINS_PATH, "r", encoding="utf-8") as _fh:
            _blocked: List[str] = yaml.safe_load(_fh).get("blocked_domains", [])
    except FileNotFoundError:
        _blocked = ["linkedin.com", "facebook.com", "x.com", "twitter.com",
                    "instagram.com", "tiktok.com"]

    _sample_results = [
        {"url": "https://www.extension.iastate.edu/corn/disease", "title": "Corn Disease – ISU Extension"},
        {"url": "https://www.linkedin.com/posts/cornfarmer/tips", "title": "LinkedIn post"},
        {"url": "https://usda.gov/nass/reports/corn-yield.pdf", "title": "USDA PDF report"},
        {"url": "https://twitter.com/cornbelt_news", "title": "Twitter / X page"},
        {"url": "https://cropwatch.unl.edu/corn/?utm_source=newsletter&utm_medium=email", "title": "CropWatch"},
        {"url": "https://cropwatch.unl.edu/corn/", "title": "CropWatch (duplicate after normalize)"},
        {"url": "mailto:info@example.com", "title": "Mailto link"},
        {"url": "tel:+15555555555", "title": "Phone link"},
        {"url": "javascript:void(0)", "title": "JS link"},
        {"url": "https://example.com/photo.jpg", "title": "Image asset"},
        {"url": "https://example.com/installer.exe", "title": "Executable"},
        {"url": "", "title": "Empty URL"},
        {"url": "not-a-url-at-all", "title": "Garbage string"},
        {"url": "https://agupdate.com/iowa-farmer/2024/corn-market-trends", "title": "Ag Update article"},
    ]

    print("=" * 70)
    print("CornScout URL filter demo")
    print("=" * 70)
    kept = filter_urls(_sample_results, _blocked)
    kept_raw_urls = {r["url"] for r in kept}

    # Pre-compute normalized forms so we can detect duplicates in the display
    _seen_norm: set = set()

    for r in _sample_results:
        raw = r.get("url", "")
        if raw in kept_raw_urls:
            match = next(k for k in kept if k["url"] == raw)
            status = f"KEPT   → {match['normalized_url']}  [{match['kept_reason']}]"
        else:
            valid, reason = is_valid_candidate_url(raw, _blocked)
            if valid:
                # URL passed the checks but was deduplicated
                norm = normalize_url(raw)
                if norm in _seen_norm:
                    reason = f"duplicate of already-accepted normalized URL: {norm}"
                else:
                    reason = "duplicate of already-accepted normalized URL"
            status = f"REJECT → {reason}"
        _norm = normalize_url(raw)
        if _norm:
            _seen_norm.add(_norm)
        print(f"  {status}")
        print(f"    input: {raw!r}")
        print()

    print(f"Result: {len(kept)}/{len(_sample_results)} URLs accepted")
