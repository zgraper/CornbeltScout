"""
rule_classifier.py
------------------
Classify web pages by type and compute relevance scores using deterministic,
rule-based heuristics.  No machine-learning models are required.

Responsibilities
----------------
- Assign a ``page_type`` label from a fixed vocabulary (e.g.
  ``"agronomy_guidance"``, ``"ag_news"``, ``"investor_page"``).
- Compute a ``relevance_cornbelt_ai`` score (0.0–1.0) measuring how strongly
  the page relates to corn-belt agriculture and AI/data topics.
- Compute a ``relevance_investor`` score (0.0–1.0) measuring how likely the
  page will interest agricultural investors.
- Return ``confidence_score``, ``quality_flags``, ``why_relevant`` strings,
  and a ``classification_version`` tag for downstream traceability.

All scoring weights are module-level constants so they can be tuned without
touching the logic.

Phase 2+ could extend this module to:
- Replace or augment rule scores with a trained ML classifier.
- Add domain-authority weighting once a DA lookup service is available.
- Incorporate freshness scoring once publication-date extraction is implemented.
- Add geography-relevance scoring (Corn Belt state bias, etc.).
- Support additional page-type categories (regulatory, weather, marketplace).
"""

import re
from typing import Any, Dict, List, Optional

from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema version — bump when the output dictionary shape changes
# ---------------------------------------------------------------------------

CLASSIFICATION_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# Page-type vocabulary
# ---------------------------------------------------------------------------

PAGE_TYPE_AGRONOMY_GUIDANCE: str = "agronomy_guidance"
PAGE_TYPE_AG_NEWS: str = "ag_news"
PAGE_TYPE_INVESTOR_PAGE: str = "investor_page"
PAGE_TYPE_AG_COMPANY: str = "ag_company"
PAGE_TYPE_UNIVERSITY_EXTENSION: str = "university_extension"
PAGE_TYPE_GOVERNMENT_RESOURCE: str = "government_resource"
PAGE_TYPE_RESEARCH_PUBLICATION: str = "research_publication"
PAGE_TYPE_CONTACT_PAGE: str = "contact_page"
PAGE_TYPE_DIRECTORY_LISTING: str = "directory_listing"
PAGE_TYPE_JOB_POSTING: str = "job_posting"
PAGE_TYPE_GENERAL_OTHER: str = "general_other"

# ---------------------------------------------------------------------------
# Scoring weights (tune these constants to adjust classifier behaviour)
# ---------------------------------------------------------------------------

# Keyword-category weights for the Cornbelt-AI relevance score.
# Each category contributes: min(hits / _NORMALISE_AT, 1.0) * weight
_CORNBELT_AI_WEIGHTS: Dict[str, float] = {
    "crops":           0.30,
    "agronomy":        0.25,
    "pest_management": 0.15,
    "ag_tech":         0.30,
}

# Keyword-category weights for the investor relevance score.
_INVESTOR_WEIGHTS: Dict[str, float] = {
    "investor": 0.70,
    "markets":  0.30,
}

# Keyword hits per category at which the contribution saturates (score = 1.0).
_NORMALISE_AT: float = 5.0

# Bonus score added per cornbelt keyword hit in free text (capped at this max).
_CORNBELT_KW_BONUS_CAP: float = 0.20

# Bonus score added per investor keyword hit in free text (capped at this max).
_INVESTOR_KW_BONUS_CAP: float = 0.30

# Score bonus for .edu domain / extension signals in cornbelt scoring.
_EDU_DOMAIN_BONUS: float = 0.10
_EXTENSION_SIGNAL_BONUS: float = 0.05

# Minimum contact count + maximum word count to trigger contact_page type.
_CONTACT_PAGE_MIN_CONTACTS: int = 3
_CONTACT_PAGE_MAX_WORDS: int = 300

# Minimum agronomy keyword hits in body text to assign agronomy_guidance type.
_AGRONOMY_GUIDANCE_MIN_HITS: int = 5

# Minimum agronomy keyword hits to prefer ag_company over general_other.
_AG_COMPANY_MIN_HITS: int = 2

# Confidence base value and adjustment constants.
_CONFIDENCE_BASE: float = 0.50
_CONFIDENCE_SCORE_FACTOR: float = 0.30
_CONFIDENCE_SPECIFIC_TYPE_BONUS: float = 0.10
_CONFIDENCE_QUALITY_PENALTY: float = 0.05
_CONFIDENCE_MANY_HITS_BONUS: float = 0.10
_CONFIDENCE_SOME_HITS_BONUS: float = 0.05
_CONFIDENCE_MANY_HITS_THRESHOLD: int = 10
_CONFIDENCE_SOME_HITS_THRESHOLD: int = 5

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Domain / URL structure signals
_EDU_PATTERN: re.Pattern = re.compile(r"\.edu(/|$)", re.IGNORECASE)
_GOV_PATTERN: re.Pattern = re.compile(r"\.gov(/|$)", re.IGNORECASE)
_EXTENSION_DOMAIN_PATTERN: re.Pattern = re.compile(r"extension\.", re.IGNORECASE)

# URL-path → page-type patterns (evaluated in order; first match wins)
_PAGE_TYPE_URL_PATTERNS: List[tuple] = [
    (PAGE_TYPE_AG_NEWS,             re.compile(r"/news/|/article/|/press-release/|/blog/", re.I)),
    (PAGE_TYPE_RESEARCH_PUBLICATION, re.compile(r"/research/|/study/|/report/|/publication/", re.I)),
    (PAGE_TYPE_DIRECTORY_LISTING,   re.compile(r"/directory/|/listing/|/members/", re.I)),
    (PAGE_TYPE_JOB_POSTING,         re.compile(r"/jobs?/|/careers?/|/hire/|/positions?/", re.I)),
    (PAGE_TYPE_CONTACT_PAGE,        re.compile(r"/contact(-us)?/|/get-in-touch/|/reach-us/", re.I)),
]

# Text-based signals for page-type inference
_EXTENSION_TEXT_PATTERN: re.Pattern = re.compile(
    r"\b(extension|cooperative extension|university extension|land.grant)\b",
    re.IGNORECASE,
)
_GOV_TEXT_PATTERN: re.Pattern = re.compile(
    r"\b(usda|fsa|nrcs|epa|regulation|regulatory|federal|state agency|"
    r"department of agriculture|government)\b",
    re.IGNORECASE,
)
_RESEARCH_TEXT_PATTERN: re.Pattern = re.compile(
    r"\b(abstract|methodology|hypothesis|conclusion|findings|"
    r"doi:|journal|peer.reviewed|citation|references|experiment|trial)\b",
    re.IGNORECASE,
)
_INVESTOR_TEXT_PATTERN: re.Pattern = re.compile(
    r"\b(portfolio|investment thesis|venture capital|fund|limited partner|"
    r"private equity|our investments|fund size|aum|assets under management)\b",
    re.IGNORECASE,
)
_JOB_TEXT_PATTERN: re.Pattern = re.compile(
    r"\b(job title|job description|apply now|qualifications|responsibilities|"
    r"compensation|benefits package|full.time|part.time|position available|"
    r"we are hiring|join our team)\b",
    re.IGNORECASE,
)
_DIRECTORY_TEXT_PATTERN: re.Pattern = re.compile(
    r"\b(member directory|find a|search results|browse by|filter by|"
    r"listing(s)?|directory)\b",
    re.IGNORECASE,
)
_NEWS_TEXT_PATTERN: re.Pattern = re.compile(
    r"\b(press release|breaking news|published on|staff writer|written by|"
    r"byline|news release|announcement|wire service)\b",
    re.IGNORECASE,
)
_AG_COMPANY_TEXT_PATTERN: re.Pattern = re.compile(
    r"\b(our products|our services|about us|company overview|"
    r"seed company|chemical company|equipment|dealer|distributor|"
    r"request a quote|contact sales)\b",
    re.IGNORECASE,
)

# Broad agronomy keyword counter (used for scoring + type inference)
_AGRONOMY_BROAD_PATTERN: re.Pattern = re.compile(
    r"\b(corn|maize|soybean|soy|wheat|sorghum|alfalfa|oats|barley|"
    r"disease|pest|nutrient deficiency|nutrient|rootworm|aphid|fungal|blight|"
    r"field management|crop rotation|no.till|tillage|soil health|drainage|"
    r"extension|agronomy|agronomic|herbicide|insecticide|fungicide|"
    r"research|yield|bushel|planting|harvest)\b",
    re.IGNORECASE,
)

# Investor-specific keyword counter (used for bonus scoring)
_INVESTOR_BROAD_PATTERN: re.Pattern = re.compile(
    r"\b(investment|investor|portfolio|fund|venture capital|thesis|"
    r"private equity|limited partner|strategic investment|startup|series a|"
    r"seed round|equity|return on investment|roi|vc firm|pe firm|"
    r"institutional investor|endowment|family office|deal flow)\b",
    re.IGNORECASE,
)

# Quality flag patterns
_QUALITY_FLAGS_PENALTY_SET: frozenset = frozenset(
    {"thin_content", "no_title", "no_meta_description"}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_lower(value: Any) -> str:
    """Return *value* as a lowercase string, or ``""`` if falsy."""
    if not value:
        return ""
    return str(value).lower()


def _combine_text_fields(parsed_page: Dict[str, Any]) -> str:
    """
    Concatenate title, meta description, headings, and cleaned text into one
    string for pattern matching.

    Accepts both the ``parse_html`` output schema (``page_title``,
    ``meta_description``, ``cleaned_text``) and looser schemas with ``title``
    / ``text`` aliases.
    """
    parts = [
        parsed_page.get("page_title") or parsed_page.get("title") or "",
        parsed_page.get("meta_description") or "",
        " ".join(parsed_page.get("headings") or []),
        parsed_page.get("cleaned_text") or parsed_page.get("text") or "",
    ]
    return " ".join(p for p in parts if p)


def _get_hit_counts(keyword_data: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract per-category hit counts from *keyword_data*.

    Accepts both the flat ``{category: count}`` dict returned by
    :func:`agriindex.extractors.keyword_extractor.extract_keywords` and the
    full result dict returned by ``extract_keyword_matches`` (which has a
    ``keyword_hit_count`` key).
    """
    if "keyword_hit_count" in keyword_data:
        return keyword_data["keyword_hit_count"]
    return keyword_data


def _count_pattern_hits(text: str, pattern: re.Pattern) -> int:
    """Return the number of non-overlapping regex matches in *text*."""
    if not text:
        return 0
    return len(pattern.findall(text))


# ---------------------------------------------------------------------------
# Helper functions (part of the public API)
# ---------------------------------------------------------------------------

def score_cornbelt_relevance(
    parsed_page: Dict[str, Any],
    keyword_data: Dict[str, Any],
) -> float:
    """
    Compute a 0.0–1.0 relevance score for Cornbelt-AI topics.

    Combines per-category keyword hits (weighted by ``_CORNBELT_AI_WEIGHTS``)
    with direct agronomy-keyword hits in the page text, plus a bonus for
    ``.edu`` and extension signals.

    Parameters
    ----------
    parsed_page : dict
        Parsed page dictionary containing at least one of ``page_title``,
        ``meta_description``, or ``cleaned_text``.
    keyword_data : dict
        Keyword hit data as returned by ``extract_keywords`` or
        ``extract_keyword_matches``.

    Returns
    -------
    float
        Cornbelt-AI relevance score clamped to [0.0, 1.0].
    """
    combined = _combine_text_fields(parsed_page)
    hit_counts = _get_hit_counts(keyword_data)

    score = 0.0

    # Weighted keyword-category contribution
    for category, weight in _CORNBELT_AI_WEIGHTS.items():
        hits = hit_counts.get(category, 0)
        score += min(hits / _NORMALISE_AT, 1.0) * weight

    # Bonus for broad agronomy keyword density in free text
    kw_hits = _count_pattern_hits(combined, _AGRONOMY_BROAD_PATTERN)
    score += min(kw_hits / (_NORMALISE_AT * 2), _CORNBELT_KW_BONUS_CAP)

    # Domain / signal boosts
    url = _safe_lower(parsed_page.get("url") or parsed_page.get("canonical_url") or "")
    domain = _safe_lower(parsed_page.get("domain") or "")
    if _EDU_PATTERN.search(url) or _EDU_PATTERN.search(domain):
        score += _EDU_DOMAIN_BONUS
    if (
        _EXTENSION_DOMAIN_PATTERN.search(url)
        or _EXTENSION_TEXT_PATTERN.search(combined)
    ):
        score += _EXTENSION_SIGNAL_BONUS

    # TODO: Add geography-relevance bonus (e.g., Corn Belt state mentions) once
    #       a geography keyword set is available.
    # TODO: Add freshness decay once publication-date extraction is implemented.

    return round(min(score, 1.0), 4)


def score_investor_relevance(
    parsed_page: Dict[str, Any],
    keyword_data: Dict[str, Any],
) -> float:
    """
    Compute a 0.0–1.0 relevance score for agricultural investor audiences.

    Combines per-category keyword hits (weighted by ``_INVESTOR_WEIGHTS``) with
    direct investor-keyword hits in the page text.

    Parameters
    ----------
    parsed_page : dict
        Parsed page dictionary.
    keyword_data : dict
        Keyword hit data.

    Returns
    -------
    float
        Investor relevance score clamped to [0.0, 1.0].
    """
    combined = _combine_text_fields(parsed_page)
    hit_counts = _get_hit_counts(keyword_data)

    score = 0.0

    # Weighted keyword-category contribution
    for category, weight in _INVESTOR_WEIGHTS.items():
        hits = hit_counts.get(category, 0)
        score += min(hits / _NORMALISE_AT, 1.0) * weight

    # Bonus for investor keyword density in free text
    kw_hits = _count_pattern_hits(combined, _INVESTOR_BROAD_PATTERN)
    score += min(kw_hits / _NORMALISE_AT, _INVESTOR_KW_BONUS_CAP)

    # TODO: Add domain-authority weighting once a DA lookup service is available.

    return round(min(score, 1.0), 4)


def infer_page_type(
    parsed_page: Dict[str, Any],
    keyword_data: Dict[str, Any],
    contact_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Infer the most likely page type from URL, domain, text signals, and contact data.

    Signals are evaluated in priority order:

    1. Domain TLD (``.gov``, ``.edu``) — strong structural signal.
    2. URL-path patterns (``/news/``, ``/jobs/``, etc.).
    3. Contact density vs. word count — indicates a contact directory page.
    4. Body-text patterns (research markers, investor language, job listing
       phrases, news bylines, directory structures, agronomy density,
       company language, extension mentions).

    Parameters
    ----------
    parsed_page : dict
        Parsed page data.  Should include ``url`` or ``canonical_url`` for
        domain-based inference.
    keyword_data : dict
        Keyword hit data.
    contact_data : dict or None
        Contact extraction results (``contact_count``, ``emails``,
        ``phone_numbers``).

    Returns
    -------
    str
        One of the ``PAGE_TYPE_*`` module constants.
    """
    url = _safe_lower(
        parsed_page.get("url") or parsed_page.get("canonical_url") or ""
    )
    domain = _safe_lower(parsed_page.get("domain") or "")
    combined = _combine_text_fields(parsed_page)
    word_count: int = parsed_page.get("word_count") or len(combined.split())

    # -- 1. Domain TLD (strongest signal) ------------------------------------
    if _GOV_PATTERN.search(url) or _GOV_PATTERN.search(domain):
        return PAGE_TYPE_GOVERNMENT_RESOURCE

    if _EDU_PATTERN.search(url) or _EDU_PATTERN.search(domain):
        if (
            _EXTENSION_TEXT_PATTERN.search(combined)
            or _EXTENSION_DOMAIN_PATTERN.search(url)
        ):
            return PAGE_TYPE_UNIVERSITY_EXTENSION
        return PAGE_TYPE_RESEARCH_PUBLICATION

    # -- 2. URL-path patterns ------------------------------------------------
    for page_type_label, pattern in _PAGE_TYPE_URL_PATTERNS:
        if pattern.search(url):
            return page_type_label

    # -- 3. Contact-heavy + thin content -------------------------------------
    if contact_data:
        contact_count: int = contact_data.get("contact_count", 0)
        if contact_count >= _CONTACT_PAGE_MIN_CONTACTS and word_count < _CONTACT_PAGE_MAX_WORDS:
            return PAGE_TYPE_CONTACT_PAGE

    # -- 4. Body-text patterns (in rough priority order) ---------------------
    if _RESEARCH_TEXT_PATTERN.search(combined):
        return PAGE_TYPE_RESEARCH_PUBLICATION

    if _INVESTOR_TEXT_PATTERN.search(combined):
        return PAGE_TYPE_INVESTOR_PAGE

    if _JOB_TEXT_PATTERN.search(combined):
        return PAGE_TYPE_JOB_POSTING

    if _NEWS_TEXT_PATTERN.search(combined):
        return PAGE_TYPE_AG_NEWS

    if _DIRECTORY_TEXT_PATTERN.search(combined):
        return PAGE_TYPE_DIRECTORY_LISTING

    ag_hits = _count_pattern_hits(combined, _AGRONOMY_BROAD_PATTERN)
    if ag_hits >= _AGRONOMY_GUIDANCE_MIN_HITS:
        return PAGE_TYPE_AGRONOMY_GUIDANCE

    if _AG_COMPANY_TEXT_PATTERN.search(combined) or ag_hits >= _AG_COMPANY_MIN_HITS:
        return PAGE_TYPE_AG_COMPANY

    if _EXTENSION_TEXT_PATTERN.search(combined):
        return PAGE_TYPE_UNIVERSITY_EXTENSION

    # TODO: Add geography-aware inference once Corn Belt state signals are available.
    # TODO: Add domain-authority bucket rule (e.g., DA > 60 + agronomy → trusted source).

    return PAGE_TYPE_GENERAL_OTHER


def build_quality_flags(
    parsed_page: Dict[str, Any],
    contact_data: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Build a list of quality and warning flags for the page.

    Flags are free-form lowercase strings that downstream consumers can
    filter or act on.  Current flags:

    - ``"no_title"``            – ``<title>`` tag is absent or empty.
    - ``"no_meta_description"`` – meta description is absent or empty.
    - ``"thin_content"``        – fewer than 100 words of body text.
    - ``"low_word_count"``      – between 100 and 299 words (borderline thin).
    - ``"contact_heavy"``       – page has 5 or more contact entries.

    Parameters
    ----------
    parsed_page : dict
        Parsed page data.
    contact_data : dict or None
        Contact extraction results.

    Returns
    -------
    list of str
        Zero or more flag strings.
    """
    flags: List[str] = []

    title = parsed_page.get("page_title") or parsed_page.get("title") or ""
    meta = parsed_page.get("meta_description") or ""
    word_count: int = parsed_page.get("word_count") or len(
        (parsed_page.get("cleaned_text") or parsed_page.get("text") or "").split()
    )

    if not title:
        flags.append("no_title")
    if not meta:
        flags.append("no_meta_description")
    if word_count < 100:
        flags.append("thin_content")
    elif word_count < 300:
        flags.append("low_word_count")

    if contact_data:
        if contact_data.get("contact_count", 0) >= 5:
            flags.append("contact_heavy")

    # TODO: Add "stale_content" flag once publication-date extraction is in place.
    # TODO: Add "duplicate_content" flag via content-hash comparison across the DB.
    # TODO: Add "low_domain_authority" flag once a DA lookup service is wired in.

    return flags


def compute_confidence(
    page_type: str,
    cornbelt_score: float,
    investor_score: float,
    quality_flags: List[str],
    keyword_data: Dict[str, Any],
) -> float:
    """
    Compute a 0.0–1.0 confidence score for the overall classification.

    Higher confidence means the page type and relevance scores are more
    strongly supported by the available signals.

    Parameters
    ----------
    page_type : str
        Inferred page type from :func:`infer_page_type`.
    cornbelt_score : float
        Cornbelt-AI relevance score.
    investor_score : float
        Investor relevance score.
    quality_flags : list of str
        Quality flags from :func:`build_quality_flags`.
    keyword_data : dict
        Keyword hit data.

    Returns
    -------
    float
        Confidence score clamped to [0.0, 1.0].
    """
    confidence = _CONFIDENCE_BASE

    # Strong relevance signals raise confidence
    max_score = max(cornbelt_score, investor_score)
    confidence += max_score * _CONFIDENCE_SCORE_FACTOR

    # A specific page type (not "general_other") raises confidence
    if page_type != PAGE_TYPE_GENERAL_OTHER:
        confidence += _CONFIDENCE_SPECIFIC_TYPE_BONUS

    # Quality issues reduce confidence
    for flag in quality_flags:
        if flag in _QUALITY_FLAGS_PENALTY_SET:
            confidence -= _CONFIDENCE_QUALITY_PENALTY

    # High keyword density raises confidence
    hit_counts = _get_hit_counts(keyword_data)
    total_hits = sum(hit_counts.values()) if hit_counts else 0
    if total_hits > _CONFIDENCE_MANY_HITS_THRESHOLD:
        confidence += _CONFIDENCE_MANY_HITS_BONUS
    elif total_hits > _CONFIDENCE_SOME_HITS_THRESHOLD:
        confidence += _CONFIDENCE_SOME_HITS_BONUS

    return round(min(max(confidence, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Internal: human-readable explanation builder
# ---------------------------------------------------------------------------

def _build_why_relevant(
    page_type: str,
    cornbelt_score: float,
    investor_score: float,
    keyword_data: Dict[str, Any],
    parsed_page: Dict[str, Any],
) -> List[str]:
    """
    Build a list of human-readable reasons explaining the page's relevance.

    Returns at most 6 reason strings so the list stays compact.
    """
    reasons: List[str] = []

    if cornbelt_score >= 0.5:
        reasons.append(f"High Cornbelt-AI relevance (score={cornbelt_score:.2f})")
    elif cornbelt_score >= 0.2:
        reasons.append(f"Moderate Cornbelt-AI relevance (score={cornbelt_score:.2f})")

    if investor_score >= 0.5:
        reasons.append(f"High investor relevance (score={investor_score:.2f})")
    elif investor_score >= 0.2:
        reasons.append(f"Moderate investor relevance (score={investor_score:.2f})")

    _PAGE_TYPE_REASONS: Dict[str, str] = {
        PAGE_TYPE_UNIVERSITY_EXTENSION: "University extension / land-grant content",
        PAGE_TYPE_GOVERNMENT_RESOURCE:  "Government or regulatory resource",
        PAGE_TYPE_RESEARCH_PUBLICATION: "Research or academic publication",
        PAGE_TYPE_AGRONOMY_GUIDANCE:    "Agronomy or field management guidance",
        PAGE_TYPE_INVESTOR_PAGE:        "Investor or fund-related page",
        PAGE_TYPE_AG_NEWS:              "Agricultural news or press release",
        PAGE_TYPE_AG_COMPANY:           "Agricultural company or product page",
        PAGE_TYPE_JOB_POSTING:          "Job posting or careers page",
        PAGE_TYPE_DIRECTORY_LISTING:    "Directory or member listing",
        PAGE_TYPE_CONTACT_PAGE:         "Contact or directory page (thin content)",
    }
    if page_type in _PAGE_TYPE_REASONS:
        reasons.append(_PAGE_TYPE_REASONS[page_type])

    hit_counts = _get_hit_counts(keyword_data)
    for cat, cnt in sorted(hit_counts.items(), key=lambda x: -x[1]):
        if cnt > 0 and len(reasons) < 6:
            reasons.append(f"Keyword category '{cat}': {cnt} hit(s)")

    return reasons


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def classify_page(
    parsed_page: Dict[str, Any],
    keyword_data: Dict[str, Any],
    contact_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classify a parsed web page and return a structured result dictionary.

    This is the primary entry point for the rule-based classifier.  It
    orchestrates the helper functions and returns a single dictionary that
    downstream consumers (pipeline, database, LLM prompt builder) can use
    directly.

    Parameters
    ----------
    parsed_page : dict
        Parsed page data produced by ``html_parser.parse_html``.  Expected
        keys: ``page_title``, ``meta_description``, ``cleaned_text``,
        ``word_count``.  Optional keys that improve classification accuracy:
        ``url`` / ``canonical_url``, ``domain``, ``headings``.
    keyword_data : dict
        Keyword hit data produced by ``keyword_extractor.extract_keywords``
        (a ``{category: count}`` mapping) or the full result from
        ``extract_keyword_matches`` (which includes a ``keyword_hit_count``
        sub-dict).
    contact_data : dict or None
        Contact extraction results from ``contact_extractor.extract_contacts``
        (``emails``, ``phone_numbers``, ``contact_count``).

    Returns
    -------
    dict with keys:
        ``page_type``              – str, one of the PAGE_TYPE_* constants
        ``relevance_cornbelt_ai``  – float (0.0–1.0)
        ``relevance_investor``     – float (0.0–1.0)
        ``confidence_score``       – float (0.0–1.0)
        ``quality_flags``          – list of str
        ``why_relevant``           – list of str (human-readable reasons)
        ``classification_version`` – str (schema version tag)
    """
    cornbelt_score = score_cornbelt_relevance(parsed_page, keyword_data)
    investor_score = score_investor_relevance(parsed_page, keyword_data)
    page_type = infer_page_type(parsed_page, keyword_data, contact_data)
    quality_flags = build_quality_flags(parsed_page, contact_data)
    confidence = compute_confidence(
        page_type, cornbelt_score, investor_score, quality_flags, keyword_data
    )
    why_relevant = _build_why_relevant(
        page_type, cornbelt_score, investor_score, keyword_data, parsed_page
    )

    result: Dict[str, Any] = {
        "page_type":              page_type,
        "relevance_cornbelt_ai":  cornbelt_score,
        "relevance_investor":     investor_score,
        "confidence_score":       confidence,
        "quality_flags":          quality_flags,
        "why_relevant":           why_relevant,
        "classification_version": CLASSIFICATION_VERSION,
    }

    logger.debug(
        "classify_page: type=%s  cornbelt_ai=%.3f  investor=%.3f  confidence=%.3f  flags=%s",
        page_type,
        cornbelt_score,
        investor_score,
        confidence,
        quality_flags,
    )

    return result


# ---------------------------------------------------------------------------
# Backward-compatible helpers (retained for callers that use the old API)
# ---------------------------------------------------------------------------

def classify_page_type(url: str, keyword_hits: Dict[str, int]) -> str:
    """
    Determine a page-type label from the URL and keyword hit counts.

    .. deprecated::
        Prefer :func:`infer_page_type` for richer, multi-signal inference.
        This function is retained for backward compatibility.

    Parameters
    ----------
    url : str
        The canonical URL of the page.
    keyword_hits : dict
        Per-set keyword hit counts (output of ``keyword_extractor``).

    Returns
    -------
    str
        A PAGE_TYPE_* constant, or ``PAGE_TYPE_GENERAL_OTHER`` if no URL
        pattern matches.
    """
    for label, pattern in _PAGE_TYPE_URL_PATTERNS:
        if pattern.search(url):
            return label
    return PAGE_TYPE_GENERAL_OTHER


def _weighted_score(keyword_hits: Dict[str, int], weights: Dict[str, float]) -> float:
    """
    Compute a 0.0–1.0 score from keyword hit counts and per-set weights.

    Each set contributes ``min(hits / _NORMALISE_AT, 1.0) * weight`` to the
    total.
    """
    total = 0.0
    for set_name, weight in weights.items():
        hits = keyword_hits.get(set_name, 0)
        contribution = min(hits / _NORMALISE_AT, 1.0) * weight
        total += contribution
    return round(min(total, 1.0), 4)


def run_rule_classifier(
    url: str,
    keyword_hits: Dict[str, int],
    page_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the rule-based classifier and return classification results.

    This function wraps :func:`classify_page` and injects *url* into the
    page data dict so domain-based rules can fire.  The ``topics`` key is
    appended for callers (e.g., ``phase1_pipeline``) that still rely on it.

    Parameters
    ----------
    url : str
        Canonical URL of the page.
    keyword_hits : dict
        Per-set keyword hit counts.
    page_data : dict, optional
        Parsed page data (title, meta, text).

    Returns
    -------
    dict with keys:
        ``page_type``              – str
        ``relevance_cornbelt_ai``  – float (0.0–1.0)
        ``relevance_investor``     – float (0.0–1.0)
        ``confidence_score``       – float (0.0–1.0)
        ``quality_flags``          – list of str
        ``why_relevant``           – list of str
        ``classification_version`` – str
        ``topics``                 – list of str (keyword sets with ≥1 hit)
    """
    enriched_page: Dict[str, Any] = dict(page_data or {})
    enriched_page.setdefault("url", url)

    result = classify_page(enriched_page, keyword_hits)

    # Append legacy ``topics`` field for pipeline backward compatibility
    result["topics"] = [name for name, cnt in keyword_hits.items() if cnt > 0]

    logger.debug(
        "run_rule_classifier: url=%s  type=%s  cornbelt_ai=%.3f  investor=%.3f",
        url,
        result["page_type"],
        result["relevance_cornbelt_ai"],
        result["relevance_investor"],
    )

    return result


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    _SAMPLES = [
        {
            "label": "Extension agronomy page",
            "parsed_page": {
                "page_title": "Corn Rootworm Management – Purdue Extension",
                "meta_description": (
                    "Guidance on scouting and managing corn rootworm using "
                    "integrated pest management practices."
                ),
                "cleaned_text": (
                    "Corn rootworm is one of the most economically damaging pests "
                    "in the U.S. Corn Belt. This guide covers scouting thresholds, "
                    "crop rotation, soil insecticide options, and resistance "
                    "management for both western and northern corn rootworm. "
                    "Published by Purdue Cooperative Extension as part of the "
                    "land-grant university outreach mission. Yield losses from "
                    "rootworm feeding can exceed 15 bushels per acre when populations "
                    "are high. Soil sampling and agronomic research support decisions."
                ),
                "word_count": 85,
                "url": "https://extension.purdue.edu/pest-management/corn-rootworm",
                "domain": "extension.purdue.edu",
            },
            "keyword_data": {
                "crops": 3,
                "agronomy": 2,
                "pest_management": 4,
                "ag_tech": 0,
                "investor": 0,
                "markets": 0,
            },
            "contact_data": {"emails": [], "phone_numbers": [], "contact_count": 0},
        },
        {
            "label": "Agtech venture capital investor page",
            "parsed_page": {
                "page_title": "AgriVentures Fund III – Portfolio & Investment Thesis",
                "meta_description": (
                    "AgriVentures is a private equity fund focused on strategic "
                    "investment in agriculture technology startups."
                ),
                "cleaned_text": (
                    "Our portfolio includes precision agriculture, crop analytics, "
                    "and supply chain software companies. We target Series A and "
                    "Series B rounds with fund sizes ranging from $2M to $20M. "
                    "As a limited partner-friendly fund, we provide venture capital "
                    "to founders solving real-world challenges in the food and "
                    "agriculture sector. Our investment thesis centers on technology "
                    "that measurably improves yield, reduces input costs, or "
                    "expands access to capital for farmers. Institutional investors "
                    "and family offices are welcome to review our fund deck."
                ),
                "word_count": 105,
                "url": "https://agriventures.com/fund/thesis",
                "domain": "agriventures.com",
            },
            "keyword_data": {
                "crops": 1,
                "agronomy": 0,
                "pest_management": 0,
                "ag_tech": 2,
                "investor": 5,
                "markets": 1,
            },
            "contact_data": {"emails": ["info@agriventures.com"], "phone_numbers": [], "contact_count": 1},
        },
        {
            "label": "USDA government resource",
            "parsed_page": {
                "page_title": "USDA NRCS – Conservation Programs for Corn Belt Farmers",
                "meta_description": (
                    "Federal conservation programs administered by NRCS to support "
                    "soil health, drainage management, and nutrient stewardship."
                ),
                "cleaned_text": (
                    "The Natural Resources Conservation Service (NRCS) offers "
                    "multiple programs to help farmers manage soil health and "
                    "nutrient runoff. The Environmental Quality Incentives Program "
                    "(EQIP) provides cost-share payments for eligible conservation "
                    "practices. Regulatory compliance with state and federal "
                    "guidelines is required. Contact your local USDA service center "
                    "to learn about eligibility and application deadlines."
                ),
                "word_count": 72,
                "url": "https://www.nrcs.usda.gov/programs/conservation",
                "domain": "nrcs.usda.gov",
            },
            "keyword_data": {
                "crops": 1,
                "agronomy": 2,
                "pest_management": 0,
                "ag_tech": 0,
                "investor": 0,
                "markets": 0,
            },
            "contact_data": None,
        },
        {
            "label": "Generic contact page",
            "parsed_page": {
                "page_title": "Contact Us",
                "meta_description": "",
                "cleaned_text": (
                    "For inquiries call (515) 555-0100 or email hello@acmefarm.com. "
                    "Sales: sales@acmefarm.com | (515) 555-0101. "
                    "Support: support@acmefarm.com"
                ),
                "word_count": 22,
                "url": "https://acmefarm.com/contact-us/",
                "domain": "acmefarm.com",
            },
            "keyword_data": {
                "crops": 0,
                "agronomy": 0,
                "pest_management": 0,
                "ag_tech": 0,
                "investor": 0,
                "markets": 0,
            },
            "contact_data": {
                "emails": ["hello@acmefarm.com", "sales@acmefarm.com", "support@acmefarm.com"],
                "phone_numbers": ["(515) 555-0100", "(515) 555-0101"],
                "contact_count": 5,
            },
        },
    ]

    for sample in _SAMPLES:
        print(f"\n{'=' * 60}")
        print(f"Sample: {sample['label']}")
        print(f"{'=' * 60}")
        result = classify_page(
            sample["parsed_page"],
            sample["keyword_data"],
            sample.get("contact_data"),
        )
        print(json.dumps(result, indent=2))
