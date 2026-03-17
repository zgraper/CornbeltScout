"""
keyword_extractor.py
--------------------
Detect agriculture-related keywords in page text using configurable keyword
dictionaries loaded from a YAML file.

Responsibilities
----------------
- Load keyword sets from a YAML file at a caller-supplied path.
- Match keywords case-insensitively against page text.
- Return per-category match details and aggregate signals.
- Provide fast, local, CPU-only keyword detection — no ML models required.

Phase 2+ could extend this module to:
- Use token-level matching to avoid false hits (e.g. "corn" in "unicorn").
- Weight keywords by importance within a set.
- Support stemming / lemmatisation with NLTK or spaCy.
- Add fuzzy / phonetic matching for misspelt agricultural terms.
"""

import re
from typing import Any, Dict, List

import yaml

from agriindex.config import settings
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API — keyword loading
# ---------------------------------------------------------------------------

def load_keyword_sets(path: str) -> Dict[str, List[str]]:
    """
    Load keyword sets from a YAML file at *path*.

    The YAML file must map category names to lists of keyword strings.
    Keywords are lowercased for consistent case-insensitive matching.

    Parameters
    ----------
    path : str
        Absolute or relative path to the ``keyword_sets.yaml`` file.

    Returns
    -------
    dict
        Maps category name (str) → list of lowercase keyword strings.
        Returns an empty dict if the file is missing or malformed.

    Example YAML structure::

        crops:
          - corn
          - soybean
        investor_terms:
          - venture capital
          - series a
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            logger.warning("load_keyword_sets: unexpected YAML structure in %s", path)
            return {}
        keyword_sets = {
            k: [str(kw).lower() for kw in v]
            for k, v in data.items()
            if isinstance(v, list)
        }
        logger.debug(
            "load_keyword_sets: loaded %d categories from %s",
            len(keyword_sets),
            path,
        )
        return keyword_sets
    except FileNotFoundError:
        logger.warning("load_keyword_sets: file not found at %s", path)
        return {}
    except yaml.YAMLError as exc:
        logger.warning("load_keyword_sets: YAML parse error in %s: %s", path, exc)
        return {}


# ---------------------------------------------------------------------------
# Module-level default keyword sets (loaded once at import time)
# ---------------------------------------------------------------------------

_DEFAULT_KEYWORD_SETS: Dict[str, List[str]] = load_keyword_sets(settings.KEYWORD_SETS_PATH)


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _build_pattern(keyword: str) -> re.Pattern:
    """
    Compile a case-insensitive regex for *keyword*.

    Single-word keywords use word-boundary anchors (``\\b``) to avoid partial
    matches (e.g. ``"corn"`` does not match inside ``"unicorn"``).  Multi-word
    phrases use plain substring matching.

    Parameters
    ----------
    keyword : str
        Lowercase keyword string.

    Returns
    -------
    re.Pattern
    """
    escaped = re.escape(keyword)
    if " " in keyword:
        return re.compile(escaped, re.IGNORECASE)
    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API — extraction
# ---------------------------------------------------------------------------

def extract_keyword_matches(
    text: str,
    keyword_sets: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Match all keywords in *keyword_sets* against *text*.

    Matching is case-insensitive and deduplicates repeated terms.  Designed
    for local CPU use — no ML models or network calls are required.

    Parameters
    ----------
    text : str
        Cleaned page text.
    keyword_sets : dict
        Category → list of keywords mapping, as returned by
        :func:`load_keyword_sets`.

    Returns
    -------
    dict with keys:
        ``matched_keywords_by_category``
            dict mapping each category name to the list of distinct keywords
            that appeared in *text*.
        ``all_matched_keywords``
            Flat deduplicated list of every matched keyword across all
            categories, in order of first appearance.
        ``keyword_hit_count``
            dict mapping each category name to the integer count of distinct
            matched keywords (zero-hit categories are included).
    """
    lower_text = text.lower() if text else ""
    matched_by_category: Dict[str, List[str]] = {}
    all_matched: List[str] = []
    seen_all: set = set()
    hit_count: Dict[str, int] = {}

    for category, keywords in keyword_sets.items():
        matched: List[str] = []
        for kw in keywords:
            pattern = _build_pattern(kw)
            if pattern.search(lower_text):
                matched.append(kw)
                if kw not in seen_all:
                    seen_all.add(kw)
                    all_matched.append(kw)
        matched_by_category[category] = matched
        hit_count[category] = len(matched)

    logger.debug(
        "extract_keyword_matches: %d categories, %d total unique hits",
        len(keyword_sets),
        len(all_matched),
    )

    return {
        "matched_keywords_by_category": matched_by_category,
        "all_matched_keywords": all_matched,
        "keyword_hit_count": hit_count,
    }


def extract_basic_signals(text: str) -> Dict[str, bool]:
    """
    Calculate simple boolean page-level signals from *text*.

    Each signal indicates whether the page mentions a key agricultural topic.
    These signals are fast to compute and require no external data.

    Parameters
    ----------
    text : str
        Cleaned page text.

    Returns
    -------
    dict with bool values for keys:
        ``mentions_corn``
        ``mentions_soybean``
        ``mentions_agriculture``
        ``mentions_investment``
        ``mentions_extension``
        ``mentions_research``
    """
    lower_text = text.lower() if text else ""

    # Phase 2+ could expand or tune these patterns per crop/topic
    signals: Dict[str, bool] = {
        "mentions_corn": bool(
            re.search(r"\b(corn|maize)\b", lower_text)
        ),
        "mentions_soybean": bool(
            re.search(r"\bsoybean(s)?\b", lower_text)
        ),
        "mentions_agriculture": bool(
            re.search(r"\b(agriculture|agricultural|agronomy|farming)\b", lower_text)
        ),
        "mentions_investment": bool(
            re.search(
                r"\b(investment|investor|funding|venture capital|startup)\b",
                lower_text,
            )
        ),
        "mentions_extension": bool(
            re.search(
                r"\b(extension|cooperative extension|university extension)\b",
                lower_text,
            )
        ),
        "mentions_research": bool(
            re.search(r"\b(research|study|experiment|trial|findings)\b", lower_text)
        ),
    }

    logger.debug("extract_basic_signals: %s", signals)
    return signals


# ---------------------------------------------------------------------------
# Convenience wrapper (uses module-level default keyword sets)
# ---------------------------------------------------------------------------

def extract_keywords(text: str) -> Dict[str, int]:
    """
    Count keyword hits per category using the default keyword sets.

    Thin convenience wrapper around :func:`extract_keyword_matches` that uses
    the keyword sets loaded from ``keyword_sets.yaml`` at import time and
    returns only the per-category hit counts.  Suitable for use by the rule
    classifier and database storage.

    Parameters
    ----------
    text : str
        Cleaned page text.

    Returns
    -------
    dict
        Maps keyword-set name → count of distinct keywords matched.
    """
    result = extract_keyword_matches(text, _DEFAULT_KEYWORD_SETS)
    return result["keyword_hit_count"]


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _SAMPLE = """
    Corn and soybean prices remain under pressure following last week's USDA report.
    Farmers across the Corn Belt are adopting precision agriculture and no-till
    practices to improve soil health.  Venture capital funding in agtech startups
    jumped 18% in Q3, according to a new research study released by Purdue Extension.
    Rootworm resistance to insecticide is a growing concern in Illinois.
    """

    kw_sets = load_keyword_sets(settings.KEYWORD_SETS_PATH)
    matches = extract_keyword_matches(_SAMPLE, kw_sets)

    print("=== Keyword Matches by Category ===")
    for cat, kws in matches["matched_keywords_by_category"].items():
        if kws:
            print(f"  {cat}: {kws}")

    print("\n=== All Matched Keywords ===")
    print(" ", matches["all_matched_keywords"])

    print("\n=== Hit Count per Category ===")
    print(" ", matches["keyword_hit_count"])

    print("\n=== Basic Signals ===")
    signals = extract_basic_signals(_SAMPLE)
    for k, v in signals.items():
        print(f"  {k}: {v}")
