"""Regexes for section, circular no., notification no. and issue date.

Gazette formatting is inconsistent, so every pattern is tested against snippets copied from the
real corpus (tests/fixtures/snippets.yaml): ``No.1/2017-Central Tax (Rate)``,
``No. 05/2021 – Central Tax``, ``No. 13/2020– Central Tax``, ``No. 26/2012—Service Tax``.
Numbers are normalised to a canonical form (``3/2025``, ``5/2021-Central Tax``) so they match
the values in data/sources.yaml and the Pinecone metadata filters.
"""

import re
from datetime import date, datetime
from typing import Any

_DASH = r"[-–—]"  # hyphen, en-dash, em-dash: all three occur in real notification numbers

CIRCULAR_RE = re.compile(r"Circular\s+No\.?\s*(\d{1,3})\s*/\s*(\d{4})", re.I)
NOTIFICATION_RE = re.compile(
    r"Notification\s+No\.?\s*(\d{1,3})\s*/\s*(\d{4})\s*" + _DASH + r"\s*"
    r"((?:Central|Integrated|Union\s+Territory|Service)\s+Tax(?:\s*\(\s*Rate\s*\))?"
    r"|Compensation\s+Cess(?:\s*\(\s*Rate\s*\))?)",
    re.I,
)

# Act section heading at line start: "80C. Deduction in respect of ...". Lines the loader
# verified from the page layout start with "§" (taxlens/ingestion/loaders.py).
SECTION_HEADING_RE = re.compile(
    r"^[ \t]*(?:§[ \t]*)?(\d{1,3}[A-Z]{0,4})\.[ \t]+([A-Z][^\n]{3,200})$", re.M
)
MARKED_HEADING_RE = re.compile(r"^[ \t]*§[ \t]*(\d{1,3}[A-Z]{0,4})\.[ \t]*([^\n]*)$", re.M)

# A section number: digits, an optional capital suffix (80C, 80CCD, 194J) and optional
# parenthesised parts (10(23C), 2(91)). Never followed by another digit, so years don't match.
_SECTION_NO = r"(\d{1,3}[A-Z]{0,4}(?:\(\w{1,6}\))*)(?![\d\w])"
SECTION_REF_RE = re.compile(r"(?i:\b(?:sections?|sec\.|u/s\.?))\s*" + _SECTION_NO)
# In a list after the keyword: "sections 80C, 80D and 80E", "sections 230 to 232" (the ends)
_SECTION_LIST_RE = re.compile(
    r"(?i:\bsections)\s*(\d{1,3}[A-Z]{0,4}(?:\(\w{1,6}\))*"
    r"(?:\s*(?:,|and|or|to|&)\s*\d{1,3}[A-Z]{0,4}(?:\(\w{1,6}\))*)+)"
)

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(
    r"(?:dated\s+(?:the\s+)?)?(\d{1,2})(?:st|nd|rd|th)?\s+(" + _MONTHS + r"),?\s+(\d{4})", re.I
)
NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")  # 28.03.2025
# The issue date of a gazette notice follows the place: "New Delhi, the 28th June, 2017"
ISSUE_DATE_RE = re.compile(
    r"New\s+Delhi,?\s*(?:the\s+|dated\s+(?:the\s+)?)?" + DATE_RE.pattern, re.I
)
HEADER_CHARS = 3000  # issue dates appear near the top


def find_circular_no(text: str) -> str | None:
    m = CIRCULAR_RE.search(text)
    return f"{int(m[1])}/{m[2]}" if m else None


def _canonical_notification(number: str, year: str, kind: str) -> str:
    kind = " ".join(kind.split())
    kind = re.sub(r"\s*\(\s*rate\s*\)", " (Rate)", kind, flags=re.I)
    kind = re.sub(r"\bunion\s+territory\b", "Union Territory", kind, flags=re.I)
    words = [w if w in {"(Rate)", "Union", "Territory"} else w.capitalize() for w in kind.split()]
    return f"{int(number)}/{year}-{' '.join(words)}"


def find_notification_no(text: str) -> str | None:
    m = NOTIFICATION_RE.search(text)
    return _canonical_notification(m[1], m[2], m[3]) if m else None


def _date(day: str, month: str, year: str) -> date | None:
    try:
        return datetime.strptime(f"{int(day)} {month} {year}", "%d %B %Y").date()
    except ValueError:
        return None


def find_issue_date(text: str) -> date | None:
    """The date a circular/notification was issued, from its header.

    Prefers "New Delhi, the <date>", then the first written-out date, then ``dd.mm.yyyy``.
    """
    head = text[:HEADER_CHARS]
    for m in (ISSUE_DATE_RE.search(head), DATE_RE.search(head)):
        if m and (d := _date(m[1], m[2], m[3])):
            return d
    if m := NUMERIC_DATE_RE.search(head):
        try:
            return date(int(m[3]), int(m[2]), int(m[1]))
        except ValueError:
            return None
    return None


def _norm_section(ref: str) -> str:
    """``80c`` → ``80C``; parenthesised parts are kept as written (``201(1A)(ii)``)."""
    base, _, rest = ref.partition("(")
    return base.upper() + (f"({rest}" if rest else "")


def base_section(ref: str) -> str:
    """The section a reference belongs to, as stored on chunks: ``10(23C)`` → ``10``."""
    return ref.split("(", 1)[0].upper()


def section_refs(text: str) -> list[str]:
    """Sections cited in running text: ``u/s 194J``, ``section 10(23C)``, ``sections 80C and
    80D``. Sorted and de-duplicated. Amounts ("Rs. 50,000.") and years never match."""
    refs = {_norm_section(r) for r in SECTION_REF_RE.findall(text)}
    for group in _SECTION_LIST_RE.findall(text):
        refs |= {_norm_section(r) for r in re.findall(_SECTION_NO, group)}
    return sorted(refs)


def extract_metadata(text: str) -> dict[str, Any]:
    """Citation metadata found in a document's text. Values not found are left out."""
    meta: dict[str, Any] = {}
    if circular := find_circular_no(text[:HEADER_CHARS]):
        meta["circular_no"] = circular
    if notification := find_notification_no(text[:HEADER_CHARS]):
        meta["notification_no"] = notification
    if issued := find_issue_date(text):
        meta["issued_on"] = issued.isoformat()
    return meta
