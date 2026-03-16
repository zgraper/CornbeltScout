"""
entity_extractor.py
-------------------
Extract named entities from page text using spaCy.

Responsibilities
----------------
- Load a spaCy language model (``en_core_web_sm`` by default).
- Run the NER pipeline on cleaned page text.
- Return a list of (text, label) tuples for storage.

**Placeholder status:** spaCy and the language model are optional dependencies
in Phase 1.  When they are not installed this module returns an empty list so
the rest of the pipeline continues without interruption.

Phase 2+ could extend this module to:
- Use a larger, more accurate spaCy model (``en_core_web_lg``).
- Fine-tune a custom NER model on agricultural text.
- Deduplicate entities by normalising case and removing punctuation.
- Link entities to knowledge bases (Wikidata, USDA PLANTS, etc.).
"""

from typing import List, Tuple

from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_nlp = None  # type: ignore[assignment]


def _get_nlp():
    """
    Lazily load and cache the spaCy NLP model.

    Returns None (with a warning) when spaCy or the model are not available.
    """
    global _nlp  # noqa: PLW0603
    if _nlp is not None:
        return _nlp

    try:
        import spacy  # type: ignore[import]

        _nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model 'en_core_web_sm' loaded")
    except ImportError:
        logger.warning(
            "spaCy is not installed.  Entity extraction is disabled.  "
            "Install with: pip install spacy && python -m spacy download en_core_web_sm"
        )
        _nlp = None
    except OSError:
        logger.warning(
            "spaCy model 'en_core_web_sm' not found.  "
            "Download with: python -m spacy download en_core_web_sm"
        )
        _nlp = None

    return _nlp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """
    Extract named entities from *text* using spaCy.

    Parameters
    ----------
    text : str
        Cleaned page text.

    Returns
    -------
    list of (entity_text, entity_label) tuples
        Returns an empty list when spaCy is unavailable or when *text* is empty.

        Entity labels follow spaCy conventions:
        ``ORG``, ``PERSON``, ``GPE``, ``LOC``, ``PRODUCT``, ``DATE``, etc.

    Notes
    -----
    Very long texts are truncated to 100,000 characters before processing to
    avoid excessive memory use and processing time in Phase 1.
    """
    if not text:
        return []

    nlp = _get_nlp()
    if nlp is None:
        return []

    # Truncate to avoid runaway processing on huge pages
    truncated = text[:100_000]

    doc = nlp(truncated)
    entities: List[Tuple[str, str]] = [
        (ent.text.strip(), ent.label_)
        for ent in doc.ents
        if ent.text.strip()
    ]

    logger.debug("extract_entities: %d entities found", len(entities))
    return entities
