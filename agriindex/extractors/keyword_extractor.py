"""
keyword_extractor.py
--------------------
Detect agriculture-related keywords in page text using the keyword
dictionaries defined in ``config/keyword_sets.yaml``.

Responsibilities
----------------
- Load keyword sets from YAML at startup.
- For each keyword set, count how many distinct keywords appear in the text.
- Return a per-set hit count dictionary for storage and scoring.

Phase 2+ could extend this module to:
- Use token-level matching to avoid false hits (e.g. "corn" inside "concern").
- Weight keywords by importance within a set.
- Return matched keyword lists (not just counts) for highlighted display.
- Support stemming / lemmatisation with NLTK or spaCy.
"""

import re
from typing import Dict, List

import yaml

from agriindex.config import settings
from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Load keyword sets once at import time
# ---------------------------------------------------------------------------

def _load_keyword_sets() -> Dict[str, List[str]]:
    """Load keyword sets from ``keyword_sets.yaml``."""
    try:
        with open(settings.KEYWORD_SETS_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return {k: [kw.lower() for kw in v] for k, v in data.items() if isinstance(v, list)}
    except FileNotFoundError:
        logger.warning("keyword_sets.yaml not found at %s", settings.KEYWORD_SETS_PATH)
        return {}


_KEYWORD_SETS: Dict[str, List[str]] = _load_keyword_sets()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_keywords(text: str) -> Dict[str, int]:
    """
    Count keyword hits for each configured keyword set in *text*.

    The matching is case-insensitive and uses whole-word boundaries so that,
    for example, ``"corn"`` does not match inside ``"unicorn"``.

    Parameters
    ----------
    text : str
        Cleaned page text.

    Returns
    -------
    dict
        Maps keyword-set name to the number of distinct keywords that appear
        at least once in *text*.  Zero-hit sets are included.

        Example::

            {
                "crops": 3,
                "agronomy": 1,
                "pest_management": 0,
                "ag_tech": 2,
                "markets": 0,
                "investor": 1,
            }
    """
    lower_text = text.lower() if text else ""
    result: Dict[str, int] = {}

    for set_name, keywords in _KEYWORD_SETS.items():
        hits = 0
        for kw in keywords:
            # Use word-boundary matching for single-word keywords;
            # for multi-word phrases use plain substring matching.
            if " " in kw:
                pattern = re.escape(kw)
            else:
                pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, lower_text):
                hits += 1
        result[set_name] = hits

    logger.debug("extract_keywords: %s", result)
    return result
