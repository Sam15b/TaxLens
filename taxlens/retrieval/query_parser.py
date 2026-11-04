"""Extract section / circular / notification references from a question.

The section boost (docs/01-architecture.md § 1.3 B) runs a second, filtered query when the
question names a section, so an exact match ranks near the top even if the embedding misses it.
Chunks store the base section (``10``, ``80C``), so references are reduced to that for filters.

Questions are looser than statute text: people write "80c" or "194J" without "section". A bare
token counts when it has a letter suffix that isn't an ordinal or a unit ("31st", "50k",
"2BHK") and doesn't follow "Form" ("Form 26AS").
"""

import re
from dataclasses import dataclass, field

from taxlens.ingestion.metadata import (
    CIRCULAR_RE,
    NOTIFICATION_RE,
    base_section,
    canonical_notification,
    section_refs,
)

# Not after a word character, "/", ".", a currency sign, "-" or "(" (a clause like 10(23C)).
_BARE_SECTION_RE = re.compile(r"(?<![\w/.₹$(-])(\d{2,3})([A-Za-z]{1,4})(?:\(\w{1,6}\))*(?![\w/-])")
_ORDINALS = "st nd rd th"
_UNITS = "k l lac lakh lakhs cr crs m mn bn b g kg km cm mm ml am pm hrs yrs x bhk gb mb"
_NOT_A_SECTION = frozenset(f"{_ORDINALS} {_UNITS}".split())


@dataclass(frozen=True)
class QueryRefs:
    sections: list[str] = field(default_factory=list)  # base sections for metadata filters
    circulars: list[str] = field(default_factory=list)
    notifications: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.sections or self.circulars or self.notifications)


def section_refs_in(question: str) -> list[str]:
    """Base section numbers named in a question, e.g. "TDS u/s 194J and 80c?" -> [194J, 80C]."""
    refs = {base_section(r) for r in section_refs(question)}
    for m in _BARE_SECTION_RE.finditer(question):
        before = question[max(0, m.start() - 6) : m.start()].lower()
        if m[2].lower() not in _NOT_A_SECTION and not before.rstrip().endswith("form"):
            refs.add(f"{m[1]}{m[2].upper()}")  # "Form 26AS" / "Form 16A" are forms
    return sorted(refs)


def parse_query(question: str) -> QueryRefs:
    circulars = sorted({f"{int(n)}/{y}" for n, y in CIRCULAR_RE.findall(question)})
    notifications = sorted(
        {canonical_notification(n, y, kind) for n, y, kind in NOTIFICATION_RE.findall(question)}
    )
    return QueryRefs(section_refs_in(question), circulars, notifications)
