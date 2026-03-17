"""
text_utils.py
-------------
General-purpose text processing utilities shared across CornScout modules.

Phase 2+ could extend this module to:
- Add language detection.
- Provide sentence tokenisation helpers for LLM prompt construction.
- Implement text chunking for models with limited context windows.
"""

import re
import unicodedata
from typing import List


def normalise_whitespace(text: str) -> str:
    """
    Collapse all runs of whitespace (including newlines and tabs) into a
    single space and strip leading/trailing whitespace.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return re.sub(r"\s+", " ", text).strip()


def remove_control_characters(text: str) -> str:
    """
    Remove non-printable and control characters from *text*, except for
    standard whitespace (space, tab, newline, carriage return).

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\r", "\t", " ")
    )


def truncate_text(text: str, max_chars: int, suffix: str = "…") -> str:
    """
    Truncate *text* to at most *max_chars* characters.

    If truncation occurs, *suffix* is appended.  The function tries to break
    at a word boundary to avoid cutting mid-word.

    Parameters
    ----------
    text : str
    max_chars : int
    suffix : str

    Returns
    -------
    str
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars - len(suffix)]
    # Try to snap back to the last space
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + suffix


def word_count(text: str) -> int:
    """
    Return the number of whitespace-delimited tokens in *text*.

    Parameters
    ----------
    text : str

    Returns
    -------
    int
    """
    return len(text.split()) if text else 0


def clean_text(text: str) -> str:
    """
    Apply a standard cleaning pipeline to *text*.

    Steps:
    1. Remove control characters.
    2. Normalise whitespace.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """
    text = remove_control_characters(text)
    text = normalise_whitespace(text)
    return text


def extract_sentences(text: str, max_sentences: int = 5) -> List[str]:
    """
    Split *text* into sentences and return up to *max_sentences*.

    Uses a simple regex-based splitter.  For production use, replace with
    a proper sentence tokeniser (e.g. spaCy ``sentencizer``).

    Parameters
    ----------
    text : str
    max_sentences : int

    Returns
    -------
    list of str
    """
    # Split on '.', '!', or '?' followed by whitespace or end-of-string
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences[:max_sentences]
