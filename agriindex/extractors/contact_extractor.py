"""
contact_extractor.py
--------------------
Extract email addresses and phone numbers from page text using regular
expressions.

Responsibilities
----------------
- Find all email addresses in cleaned text.
- Find all phone numbers (US-centric patterns) in cleaned text.
- Deduplicate and normalise results within a single page.
- Return a structured dictionary of contact data.

Phase 2+ could extend this module to:
- Use more sophisticated phone-number parsing (``phonenumbers`` library).
- Support international phone formats.
- Extract social media handles and contact form URLs.
- Validate email addresses with a DNS MX check.
"""

import re
from typing import Dict, List

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
# Covers: (555) 555-5555, 555-555-5555, 555.555.5555, +1 555 555 5555, etc.
_PHONE_RE = re.compile(
    r"(?:\+?1[\s.\-]?)?"             # optional country code +1
    r"(?:\(?\d{3}\)?[\s.\-]?)"       # area code
    r"\d{3}[\s.\-]?\d{4}",           # local number
)

# Minimum digit count to avoid matching dates / zip codes as phone numbers
_MIN_PHONE_DIGITS = 10

# Image-extension suffixes that are common false positives in email patterns
_IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "gif", "svg", "webp", "ico"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _digit_count(text: str) -> int:
    """Return the number of digit characters in *text*."""
    return sum(1 for c in text if c.isdigit())


def _normalise_phone(raw: str) -> str:
    """
    Canonicalise a raw phone match to digits only for deduplication.

    Strips all non-digit characters and removes a leading country-code ``1``
    when the result is 11 digits, leaving a 10-digit string.

    Parameters
    ----------
    raw : str
        Raw phone string as captured by ``_PHONE_RE``.

    Returns
    -------
    str
        10-digit canonical form (or 11 digits if country code cannot be
        identified).
    """
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_emails(text: str) -> List[str]:
    """
    Extract deduplicated email addresses from *text*.

    Matching is case-insensitive; results are lowercased.  Addresses whose
    top-level domain looks like an image-file extension are filtered out as
    obvious false positives.

    Parameters
    ----------
    text : str
        Cleaned page text.

    Returns
    -------
    list of str
        Unique email addresses found in *text*, in order of first appearance.
    """
    if not text:
        return []

    seen: set = set()
    results: List[str] = []

    for match in _EMAIL_RE.finditer(text):
        email = match.group(0).lower()
        # Filter addresses whose TLD looks like an image extension
        tld = email.rsplit(".", 1)[-1]
        if tld in _IMAGE_EXTENSIONS:
            continue
        if email not in seen:
            seen.add(email)
            results.append(email)

    logger.debug("extract_emails: found %d unique email(s)", len(results))
    return results


def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract deduplicated US phone numbers from *text*.

    Handles common formats including:

    - ``(812) 555-1212``
    - ``812-555-1212``
    - ``812.555.1212``
    - ``+1 812 555 1212``

    Results are returned with the formatting of their first occurrence.
    Deduplication uses a digits-only canonical form so format variants of the
    same number collapse to one entry.

    Parameters
    ----------
    text : str
        Cleaned page text.

    Returns
    -------
    list of str
        Unique phone numbers found in *text*, in order of first appearance.
    """
    if not text:
        return []

    seen_canonical: set = set()
    results: List[str] = []

    for match in _PHONE_RE.finditer(text):
        raw = match.group(0).strip()
        if _digit_count(raw) < _MIN_PHONE_DIGITS:
            continue
        canonical = _normalise_phone(raw)
        if canonical not in seen_canonical:
            seen_canonical.add(canonical)
            results.append(raw)

    logger.debug("extract_phone_numbers: found %d unique phone number(s)", len(results))
    return results


def extract_contacts(text: str) -> Dict[str, object]:
    """
    Extract all contact information from *text* in a single call.

    Calls :func:`extract_emails` and :func:`extract_phone_numbers` and
    bundles their results into a structured dictionary.

    Parameters
    ----------
    text : str
        Cleaned page text (output of ``html_parser.parse_html``).

    Returns
    -------
    dict with keys:
        ``emails``        – list of str
        ``phone_numbers`` – list of str
        ``contact_count`` – int  (total unique contacts found)
    """
    emails = extract_emails(text)
    phone_numbers = extract_phone_numbers(text)
    contact_count = len(emails) + len(phone_numbers)

    logger.debug(
        "extract_contacts: %d email(s), %d phone(s)  total=%d",
        len(emails),
        len(phone_numbers),
        contact_count,
    )

    return {
        "emails": emails,
        "phone_numbers": phone_numbers,
        "contact_count": contact_count,
    }


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _SAMPLE = """
    Contact us at info@cornbelt.ag or support@agritech.com for more information.
    You can also reach us by phone at (812) 555-1212 or 1-800-555-9999.
    Email: noreply@example.org  |  Office: 812.555.3344
    Duplicate test: info@cornbelt.ag  (812) 555-1212
    """

    result = extract_contacts(_SAMPLE)
    print("Emails       :", result["emails"])
    print("Phone numbers:", result["phone_numbers"])
    print("Contact count:", result["contact_count"])
