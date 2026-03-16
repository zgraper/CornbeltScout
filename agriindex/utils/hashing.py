"""
hashing.py
----------
Utility functions for content hashing.

Used primarily for deduplication: two page records with the same
``content_hash`` contain identical cleaned text and the second can be
skipped or flagged as a duplicate.

Phase 2+ could extend this module to:
- Use locality-sensitive hashing (SimHash / MinHash) for near-duplicate
  detection.
- Store multiple hash types (SimHash, MD5) for different use cases.
"""

import hashlib


def sha256_text(text: str) -> str:
    """
    Return the SHA-256 hex digest of *text*.

    Parameters
    ----------
    text : str
        Input string (typically the cleaned page text).

    Returns
    -------
    str
        64-character lowercase hexadecimal digest, or an empty string if
        *text* is empty or None.
    """
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
