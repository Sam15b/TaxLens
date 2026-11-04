"""PDF -> pages with PyMuPDF, low-text/OCR detection, and margin-note section headings.

A page with almost no extractable text is probably a scanned image. Pages like that are flagged
``needs_ocr``; if too many pages of a document are, the whole document is treated as scanned and
skipped until OCR support exists (stretch issue). See docs/04-build-guide.md Week 1.

Acts printed in the Gazette put each section's heading in the page margin, beside the section
number ("Levy and collection." next to "9. (1) Subject to ..."). Plain text extraction loses the
link between the two, and numbered table rows inside sections ("1. More than 10000 ...") look
exactly like section starts. So the loader reads the layout: a margin note that lines up with a
line starting with a section number is moved inline as a marked heading line,

    § 9. Levy and collection.
    (1) Subject to the provisions of sub-section (2), ...

and chunking trusts these ``§`` lines over the plain-text regex (taxlens/ingestion/chunking.py).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

MIN_CHARS_PER_PAGE = 40  # below this, the page is probably a scanned image
SCANNED_DOC_RATIO = 0.3  # above this share of low-text pages, treat the whole document as scanned
SECTION_MARK = "§"  # starts a layout-verified "§ <no>. <heading>" line

# Layout heuristics, as fractions of the page width / in PDF points.
_COLUMN_MIN_WIDTH = 0.4  # lines wider than this define the main text column...
_COLUMN_OUTER = 0.15  # ...unless they reach into the outer 15% (merged margin junk)
_MARGIN_MAX_WIDTH = 0.25  # margin notes are narrow
_NOTE_ABOVE, _NOTE_BELOW = 30.0, 60.0  # how far a note's top may sit from its section's line
# "9. (1) Subject to ...", "8. The tax ...", or "32." alone on its line (text follows below)
_SECTION_START = re.compile(r"^\s*(\d{1,3}[A-Z]{0,4})\.(?:\s+(?=[A-Z(])|\s*$)")
_CITATION = re.compile(r"^\s*(?:\d+\s+of\s+\d{4}\.?\s*)+$", re.I)  # "43 of 1961." beside a ref


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as printed in citations
    text: str
    needs_ocr: bool
    headings: tuple[tuple[str, str], ...] = field(default=())  # (section_no, heading) found here


@dataclass
class _Line:
    block: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    margin: bool = False
    note: int | None = None  # index into the page's notes when this line is part of one


def _lines(page: pymupdf.Page) -> list[_Line]:
    out = []
    data: dict[str, Any] = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
    for bno, block in enumerate(data["blocks"]):
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if text.strip():
                x0, y0, x1, y1 = line["bbox"]
                out.append(_Line(bno, x0, y0, x1, y1, text))
    return out


def _mark_margins(lines: list[_Line], width: float) -> None:
    column = [
        ln
        for ln in lines
        if ln.x1 - ln.x0 > _COLUMN_MIN_WIDTH * width
        and ln.x0 > _COLUMN_OUTER * width
        and ln.x1 < (1 - _COLUMN_OUTER) * width
    ]
    if not column:
        return
    left, right = min(ln.x0 for ln in column), max(ln.x1 for ln in column)
    for ln in lines:
        narrow = ln.x1 - ln.x0 < _MARGIN_MAX_WIDTH * width
        ln.margin = narrow and (ln.x1 <= left + 2 or ln.x0 >= right - 2)


@dataclass
class _Row:
    """Margin fragments on one visual line (justified text splits a line into pieces)."""

    parts: list[_Line]

    @property
    def x0(self) -> float:
        return min(p.x0 for p in self.parts)

    @property
    def y0(self) -> float:
        return self.parts[0].y0

    @property
    def y1(self) -> float:
        return max(p.y1 for p in self.parts)

    @property
    def text(self) -> str:
        return " ".join(p.text.strip() for p in sorted(self.parts, key=lambda p: p.x0))


def _group_notes(lines: list[_Line], width: float) -> list[tuple[float, str, list[_Line]]]:
    """Stack margin lines into notes: (top, text, lines). A note ends at a line ending in '.'."""
    margin = sorted((ln for ln in lines if ln.margin), key=lambda ln: (ln.x0 < 300, ln.y0, ln.x0))
    rows: list[_Row] = []
    for ln in margin:
        last = rows[-1].parts[-1] if rows else None
        side_by_side = last is not None and 0 <= ln.x0 - last.x1 < 0.1 * width
        if last is not None and side_by_side and abs(ln.y0 - last.y0) < 2:
            rows[-1].parts.append(ln)
        else:
            rows.append(_Row([ln]))

    notes: list[list[_Row]] = []
    for row in rows:
        prev = notes[-1][-1] if notes else None
        same_note = (
            prev is not None
            and abs(prev.x0 - row.x0) < 20
            and -0.5 < (row.y0 - prev.y1) / (prev.y1 - prev.y0) < 1.2  # line boxes may overlap
            and not prev.text.endswith(".")
            and not _CITATION.match(prev.text)
            and not _CITATION.match(row.text)
        )
        if same_note:
            notes[-1].append(row)
        else:
            notes.append([row])

    out = []
    for group in notes:
        text = ""
        for row in group:
            part = row.text
            text = f"{text}{part}" if text.endswith("-") else f"{text} {part}".strip()
        if len(re.findall(r"[^\W\d_]", text)) >= 3:  # skip stray "." and similar specks
            out.append((group[0].y0, text, [p for row in group for p in row.parts]))
    return out


def _page_text(page: pymupdf.Page) -> tuple[str, tuple[tuple[str, str], ...]]:
    lines = _lines(page)
    _mark_margins(lines, page.rect.width)
    notes = [n for n in _group_notes(lines, page.rect.width) if not _CITATION.match(n[1])]
    starts = [
        (j, m)
        for j, ln in enumerate(lines)
        if not ln.margin and (m := _SECTION_START.match(ln.text))
    ]
    # Closest pairs first, so a numbered table row near a section can't take its note.
    pairs = sorted(
        (abs(top - lines[j].y0), j, i)
        for j, _ in starts
        for i, (top, _, _) in enumerate(notes)
        if -_NOTE_ABOVE <= top - lines[j].y0 <= _NOTE_BELOW
    )
    match = dict(starts)
    headings: dict[int, tuple[str, str]] = {}  # line index -> (section_no, heading)
    used: set[int] = set()
    for _, j, i in pairs:
        if j in headings or i in used:
            continue
        used.add(i)
        headings[j] = (match[j][1], notes[i][1])
        for part in notes[i][2]:
            part.note = i

    out: list[str] = []
    block = None
    for j, ln in enumerate(lines):
        if ln.note is not None:
            continue  # moved inline as a heading
        if block is not None and ln.block != block:
            out.append("")  # blank line between blocks, like PyMuPDF's plain-text output
        block = ln.block
        if j in headings:
            no, heading = headings[j]
            out.append(f"{SECTION_MARK} {no}. {heading}")
            rest = ln.text[match[j].end() :]
            if rest.strip():
                out.append(rest)
        else:
            out.append(ln.text)
    found = tuple(headings[j] for j in sorted(headings))
    return "\n".join(out) + ("\n" if out else ""), found


def load_pdf(path: Path | str) -> list[Page]:
    """One ``Page`` per PDF page, in order. Raises ``FileNotFoundError`` for a missing file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    pages = []
    with pymupdf.open(path) as doc:
        for i, pdf_page in enumerate(doc, start=1):
            text, headings = _page_text(pdf_page)
            needs = len(text.strip()) < MIN_CHARS_PER_PAGE
            pages.append(Page(number=i, text=text, needs_ocr=needs, headings=headings))
    return pages


def needs_ocr(pages: list[Page]) -> bool:
    """True when the document as a whole looks scanned (or has no pages at all)."""
    if not pages:
        return True
    return sum(p.needs_ocr for p in pages) / len(pages) > SCANNED_DOC_RATIO
