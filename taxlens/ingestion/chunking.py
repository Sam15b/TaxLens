"""Section-aware chunking + SentenceSplitter + contextual chunk header.

Acts are split into sections first, so a chunk never mixes two sections. See
docs/04-build-guide.md Week 1 → "Section-aware chunking".
"""

from functools import cache
from typing import Any, NamedTuple

from llama_index.core.node_parser import SentenceSplitter

from taxlens.ingestion.loaders import SECTION_MARK
from taxlens.ingestion.metadata import MARKED_HEADING_RE, find_section_headings

MIN_SECTION_CHARS = 100  # shorter "sections" are table-of-contents entries or repeated headings
CHUNK_SIZE = 512  # tokens, including the contextual header
CHUNK_OVERLAP = 64

Chunk = tuple[str, dict[str, Any]]  # (text with header, metadata)


@cache
def splitter() -> SentenceSplitter:
    return SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


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


def contextual_header(meta: dict[str, Any]) -> str:
    """Prepended to every chunk: "Income-tax Act, 1961 | Section 80C: Deduction in respect of ...".

    It helps both embedding and keyword search (a chunk deep inside section 80C still matches
    "80C"), and the LLM sees where a passage comes from. Acts carry no date: they are amended,
    so the enactment date would mislead.
    """
    is_act = meta.get("doc_type") == "act"
    first = (meta.get("act") if is_act else None) or meta.get("title") or ""
    parts = [first]
    if meta.get("section"):
        parts.append(f"Section {meta['section']}: {meta.get('heading') or ''}".rstrip(": "))
    elif meta.get("heading"):
        parts.append(meta["heading"])
    if (circular := meta.get("circular_no")) and circular not in first:
        parts.append(f"Circular No. {circular}")
    if (notification := meta.get("notification_no")) and notification not in first:
        parts.append(f"Notification No. {notification}")
    if meta.get("issued_on") and not is_act:
        parts.append(f"dated {meta['issued_on']}")
    return " | ".join(p for p in parts if p)


def _split(body: str, header: str) -> list[tuple[int, str]]:
    """(offset in body, text) per chunk. Chunks leave room for the header in the token budget."""
    out: list[tuple[int, str]] = []
    cursor = 0
    for piece in splitter().split_text_metadata_aware(body, metadata_str=f"{header}\n\n"):
        found = body.find(piece, cursor)
        offset = found if found >= 0 else cursor
        out.append((offset, piece))
        cursor = offset + 1  # overlapping chunks start after the previous start
    return out


def chunk_document(text: str, base_meta: dict[str, Any]) -> list[Chunk]:
    """(chunk_text, metadata) pairs. Acts are split by section first, so a chunk never mixes
    two sections; other documents are split as one piece.

    Metadata is ``base_meta`` plus ``section``, ``heading``, ``chunk_no`` and ``offset`` (of
    the chunk body in ``text``).
    """
    if base_meta.get("doc_type") == "act":
        sections = split_act_into_sections(text)
    else:
        sections = [Section(None, "", text, 0)] if text.strip() else []
    chunks: list[Chunk] = []
    for section in sections:
        meta = {**base_meta, "section": section.section_no, "heading": section.heading}
        header = contextual_header(meta)
        for offset, piece in _split(section.body, header):
            chunk_meta = {**meta, "chunk_no": len(chunks), "offset": section.start + offset}
            chunks.append((f"{header}\n\n{piece}", chunk_meta))
    return chunks
