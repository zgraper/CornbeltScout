"""
rule_classifier.py
------------------
Classify pages by type and compute relevance scores using simple rule-based
heuristics.

Responsibilities
----------------
- Determine a ``page_type`` label (e.g. ``"news"``, ``"product"``, ``"research"``,
  ``"directory"``, ``"unknown"``) from URL patterns and keyword signals.
- Compute a ``cornbelt_ai`` relevance score (0.0–1.0) measuring how strongly the
  page relates to corn-belt agriculture and AI/data topics.
- Compute an ``investor`` relevance score (0.0–1.0) measuring how likely the page
  is of interest to agricultural investors.

Phase 2+ could extend this module to:
- Replace or augment rule scores with a trained ML classifier.
- Add more page-type categories (regulatory, weather, marketplace, etc.).
- Combine multiple scoring signals with learned weights.
"""

import re
from typing import Any, Dict

from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page-type patterns
# ---------------------------------------------------------------------------

_PAGE_TYPE_PATTERNS = [
    ("news",      re.compile(r"/news/|/article/|/press-release/|/blog/", re.I)),
    ("research",  re.compile(r"/research/|/study/|/report/|/publication/|\.edu/", re.I)),
    ("product",   re.compile(r"/product/|/shop/|/store/|/buy/|/catalog/", re.I)),
    ("directory", re.compile(r"/directory/|/listing/|/members/", re.I)),
    ("event",     re.compile(r"/event/|/conference/|/webinar/|/field-day/", re.I)),
]

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

# Each keyword set contributes a weight to the cornbelt_ai score
_CORNBELT_AI_WEIGHTS: Dict[str, float] = {
    "crops":           0.30,
    "agronomy":        0.25,
    "pest_management": 0.15,
    "ag_tech":         0.30,
}

# investor score is driven primarily by the investor keyword set
_INVESTOR_WEIGHTS: Dict[str, float] = {
    "investor": 0.70,
    "markets":  0.30,
}

# Maximum keyword hits per set used to normalise the score
_NORMALISE_AT = 5.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_page_type(url: str, keyword_hits: Dict[str, int]) -> str:
    """
    Determine a page-type label from the URL and keyword hit counts.

    Parameters
    ----------
    url : str
        The canonical URL of the page.
    keyword_hits : dict
        Per-set keyword hit counts (output of ``keyword_extractor``).

    Returns
    -------
    str
        One of: ``"news"``, ``"research"``, ``"product"``, ``"directory"``,
        ``"event"``, or ``"unknown"``.
    """
    for label, pattern in _PAGE_TYPE_PATTERNS:
        if pattern.search(url):
            return label
    return "unknown"


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
    page_data: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Run the rule-based classifier and return classification results.

    Parameters
    ----------
    url : str
        Canonical URL of the page.
    keyword_hits : dict
        Per-set keyword hit counts.
    page_data : dict, optional
        Parsed page data (title, meta, text).  Reserved for future use.

    Returns
    -------
    dict with keys:
        ``page_type``           – str
        ``relevance_cornbelt_ai`` – float (0.0–1.0)
        ``relevance_investor``    – float (0.0–1.0)
        ``topics``              – list of str  (keyword sets with at least 1 hit)
    """
    page_type = classify_page_type(url, keyword_hits)
    cornbelt_ai_score = _weighted_score(keyword_hits, _CORNBELT_AI_WEIGHTS)
    investor_score = _weighted_score(keyword_hits, _INVESTOR_WEIGHTS)

    topics = [set_name for set_name, hits in keyword_hits.items() if hits > 0]

    logger.debug(
        "rule_classifier: url=%s  type=%s  cornbelt_ai=%.3f  investor=%.3f",
        url,
        page_type,
        cornbelt_ai_score,
        investor_score,
    )

    return {
        "page_type": page_type,
        "relevance_cornbelt_ai": cornbelt_ai_score,
        "relevance_investor": investor_score,
        "topics": topics,
    }
