"""Section-aware chunking + SentenceSplitter + contextual chunk header.

Acts are split into sections first, so a chunk never mixes two sections. See
docs/04-build-guide.md Week 1 → "Section-aware chunking".
"""

from typing import NamedTuple

from taxlens.ingestion.loaders import SECTION_MARK
from taxlens.ingestion.metadata import MARKED_HEADING_RE, find_section_headings

MIN_SECTION_CHARS = 100  # shorter "sections" are table-of-contents entries or repeated headings


class Section(NamedTuple):
    section_no: str | None  # None for the preamble and for Schedules
    heading: str
    body: str  # starts at the heading line
    start: int  # offset of ``body`` in the full text


def split_act_into_sections(full_text: str) -> list[Section]:
    """Split an Act into sections (and Schedules), in order, without losing text.

    - Headings the loader marked from the page layout (``§ 9. ...``) are trusted as they are.
    - Otherwise every ``80C. Deduction ...`` line is a candidate. Bodies under 100 characters are
      dropped: at the start of an Act the table of contents repeats every heading with no body.
    - A section number seen before is not a new section (a numbered list inside a section, or a
      Schedule restarting at 1.): its text stays with the section before it.
    - Text before the first heading (title, preamble) is kept as a section without a number.
    """
    headings = find_section_headings(full_text)
    if not headings:
        return [Section(None, "", full_text, 0)] if full_text.strip() else []
    marked = MARKED_HEADING_RE.search(full_text) is not None

    candidates: list[Section] = []
    preamble = full_text[: headings[0].start]
    if len(preamble.strip()) >= MIN_SECTION_CHARS:
        candidates.append(Section(None, "", preamble, 0))
    for i, h in enumerate(headings):
        end = headings[i + 1].start if i + 1 < len(headings) else len(full_text)
        candidates.append(Section(h.section_no, h.title, full_text[h.start : end], h.start))

    sections: list[Section] = []
    seen: set[str] = set()
    for s in candidates:
        if not marked and len(s.body.strip()) < MIN_SECTION_CHARS:
            continue
        if s.section_no is not None and s.section_no in seen and sections:
            prev = sections[-1]
            sections[-1] = prev._replace(body=full_text[prev.start : s.start + len(s.body)])
            continue
        if s.section_no is not None:
            seen.add(s.section_no)
        sections.append(s)
    return [_unmark(s) for s in sections]


def _unmark(s: Section) -> Section:
    """Skip the loader's "§ " marker so the body starts "9. Levy ...", keeping offsets exact."""
    mark = f"{SECTION_MARK} "
    if s.body.startswith(mark):
        return s._replace(body=s.body[len(mark) :], start=s.start + len(mark))
    return s
