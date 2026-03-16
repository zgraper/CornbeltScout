"""
contact_extractor.py
--------------------
Extract email addresses and phone numbers from page text using regular
expressions.

Responsibilities
----------------
- Find all email addresses in cleaned text.
- Find all phone numbers (US-centric patterns) in cleaned text.
- Deduplicate results within a single page.

Phase 2+ could extend this module to:
- Use more sophisticated phone-number parsing (``phonenumbers`` library).
- Support international phone formats.
- Extract social media handles and contact form URLs.
- Validate email addresses with a DNS MX check.
"""

import re
from typing import List, Tuple

from agriindex.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------

# Email: standard RFC-ish pattern; deliberately permissive to reduce false-negatives
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Phone: common North American patterns
# Covers: (555) 555-5555, 555-555-5555, 555.555.5555, +15555555555, etc.
_PHONE_RE = re.compile(
    r"(?:\+?1[\s.\-]?)?"             # optional country code
    r"(?:\(?\d{3}\)?[\s.\-]?)"       # area code
    r"\d{3}[\s.\-]?\d{4}",           # local number
)

# Minimum digit count to avoid matching dates/zip codes as phones
_MIN_PHONE_DIGITS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _digit_count(text: str) -> int:
    """Return the number of digit characters in *text*."""
    return sum(1 for c in text if c.isdigit())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_contacts(text: str) -> List[Tuple[str, str]]:
    """
    Extract emails and phone numbers from *text*.

    Parameters
    ----------
    text : str
        Cleaned page text (output of ``html_parser.parse_html``).

    Returns
    -------
    list of (type, value) tuples
        ``type`` is ``"email"`` or ``"phone"``.
        Duplicates within the same page are removed; order is not guaranteed.
    """
    contacts: List[Tuple[str, str]] = []
    seen_values: set = set()

    # Emails
    for match in _EMAIL_RE.finditer(text):
        value = match.group(0).lower()
        if value not in seen_values:
            seen_values.add(value)
            contacts.append(("email", value))

    # Phones
    for match in _PHONE_RE.finditer(text):
        value = match.group(0).strip()
        # Discard matches that look more like years / zip codes
        if _digit_count(value) < _MIN_PHONE_DIGITS:
            continue
        normalised = re.sub(r"[\s.\-()]", "", value)  # strip formatting for dedup
        if normalised not in seen_values:
            seen_values.add(normalised)
            contacts.append(("phone", value))

    logger.debug(
        "extract_contacts: %d emails, %d phones",
        sum(1 for t, _ in contacts if t == "email"),
        sum(1 for t, _ in contacts if t == "phone"),
    )
    return contacts
