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
from collections.abc import Iterable
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
# "194-IB." too: the Income-tax Act, 1961 hyphenates some inserted sections
_SECTION_START = re.compile(r"^\s*(\d{1,3}(?:-?[A-Z]{1,4})?)\.(?:\s+(?=[A-Z(])|\s*$)")
_HEADING_ABOVE_LINES = 3  # a heading printed above its section wraps onto at most this many lines
_SUPERSCRIPT = 0.8  # spans smaller than this share of the line's largest are footnote markers
_CITATION = re.compile(r"^\s*(?:\d+\s+of\s+\d{4}\.?\s*)+$", re.I)  # "43 of 1961." beside a ref
_GLYPH_CODE = re.compile(r"/c(\d{1,3})")  # Type3 glyph named after its character code
_DIFFERENCES = re.compile(r"/Differences\s*(\[[^\]]*\])")
_PDF_TOKEN = re.compile(r"\d+|/[^\s/\[\]()<>{}%]+")  # integers and /Names inside an array


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
    fonts: tuple[tuple[str, int], ...] = ()  # (font, characters) per span, superscripts left out
    margin: bool = False
    moved: bool = False  # part of a heading that was moved inline (margin note or line above)


def _glyph_code_map(doc: pymupdf.Document, xref: int) -> dict[int, str] | None:
    """``code -> character`` for a font whose ``/Differences`` names every glyph ``/c<code>``.

    Some PageMaker/Distiller PDFs (the official Income-tax Act, 1961) use Type3 fonts with no
    ToUnicode map, whose glyphs are named after their Windows-1252 code: ``/c104`` is "h". PyMuPDF
    can't map those names, so it returns the raw font codes (``"\\x02\\x03..."``) as text.
    """
    kind, value = doc.xref_get_key(xref, "Encoding")
    if kind == "xref":
        kind, value = doc.xref_get_key(int(value.split()[0]), "Differences")
    elif kind == "dict":
        kind, value = "array", (_DIFFERENCES.search(value) or [None, ""])[1]
    else:
        return None
    if kind != "array" or not value:
        return None
    table: dict[int, str] = {}
    code = 0
    for token in _PDF_TOKEN.findall(value):  # "[0/c83 2/c104/c111]": names needn't be spaced
        if token.isdigit():
            code = int(token)
            continue
        named = _GLYPH_CODE.fullmatch(token)
        if named is None:
            return None  # a mixed table: PyMuPDF already maps the real names, leave the font alone
        table[code] = bytes([int(named[1]) % 256]).decode("cp1252", errors="replace")
        code += 1
    return table or None


def _font_maps(
    page: pymupdf.Page, cache: dict[int, dict[int, str] | None]
) -> dict[str, dict[int, str]]:
    """Glyph-code tables for this page's fonts, keyed by the font name spans report: the font's
    ``/Name`` when it has one, else PyMuPDF's ``"Type3 (<xref> 0 R)"``."""
    out: dict[str, dict[int, str]] = {}
    for xref, _ext, kind, name, *_ in page.get_fonts():
        if kind != "Type3":
            continue
        if xref not in cache:
            cache[xref] = _glyph_code_map(page.parent, xref)
        if (table := cache[xref]) is not None:
            out[name or f"Type3 ({xref} 0 R)"] = table
    return out


def _span_text(span: dict[str, Any], fonts: dict[str, dict[int, str]]) -> str:
    if "chars" not in span:  # "dict" output: no glyph-coded fonts on this page
        return str(span["text"])
    table = fonts.get(span["font"], {})
    # PyMuPDF inserts a synthetic " " at wide gaps. In a glyph-coded font 32 is a real code
    # (often a letter), so only the glyphs actually drawn are decoded.
    return "".join(
        " " if ch["synthetic"] else table.get(ord(ch["c"]), ch["c"]) for ch in span["chars"]
    )


def _lines(page: pymupdf.Page, fonts: dict[str, dict[int, str]] | None = None) -> list[_Line]:
    fonts = fonts or {}
    out = []
    kind = "rawdict" if fonts else "dict"  # rawdict: per-character codes, needed for decoding
    data: dict[str, Any] = page.get_text(kind, flags=pymupdf.TEXTFLAGS_TEXT)
    for bno, block in enumerate(data["blocks"]):
        for line in block.get("lines", []):
            parts = [(_span_text(s, fonts), s["font"], s["size"]) for s in line["spans"]]
            text = "".join(t for t, _, _ in parts)
            if text.strip():
                x0, y0, x1, y1 = line["bbox"]
                full = _SUPERSCRIPT * max(size for _, _, size in parts)
                used = tuple(
                    (f, len(t.strip())) for t, f, size in parts if t.strip() and size >= full
                )
                out.append(_Line(bno, x0, y0, x1, y1, text, used))
    return out


def _main_font(fonts: Iterable[tuple[str, int]]) -> str | None:
    """The font most of these characters are in (a line's or a page's ``(font, chars)`` pairs)."""
    count: dict[str, int] = {}
    for font, chars in fonts:
        count[font] = count.get(font, 0) + chars
    return max(count, key=lambda f: count[f]) if count else None


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


def _heading_above(lines: list[_Line], j: int, page_font: str | None) -> list[int]:
    """Lines of a heading printed just above section start ``lines[j]``; ``[]`` when there is none.

    The Income-tax Act, 1961 prints each heading on its own line(s) above the section, mostly in
    the font of the section number and not the body (an italic word, a "." or a superscript
    amendment marker like "77[" may be in another font):

        Fees for professional or technical services.      <- same font as "194J."
        194J. (1) Any person, not being an individual ...  <- body continues in another font
    """
    start = lines[j]
    m = _SECTION_START.match(start.text)
    alone = m is not None and not start.text[m.end() :].strip()  # "80C." with the text below
    if not start.fonts:
        return []
    number_font = start.fonts[0][0]
    if number_font == page_font:
        return []  # numbered paragraphs in the body font ("4. Explanation.-"), not sections
    if alone:  # the body starts on the next line
        body = [_main_font(lines[j + 1].fonts)] if j + 1 < len(lines) else []
    else:
        body = [font for font, _ in start.fonts[1:]]
    if not body or all(font == number_font for font in body):
        return []  # the number and the body share one font: nothing marks a heading out
    picked: list[int] = []
    below = start
    for k in range(j - 1, max(j - 1 - _HEADING_ABOVE_LINES, -1), -1):
        ln = lines[k]
        close = ln.y1 <= below.y0 + 2 and below.y0 - ln.y1 < 1.2 * (ln.y1 - ln.y0)
        if ln.margin or ln.moved or not close or _main_font(ln.fonts) != number_font:
            break
        if _SECTION_START.match(ln.text) or (picked and ln.text.rstrip().endswith(".")):
            break  # a section number, or the end of an earlier bold line: not this heading
        picked.insert(0, k)
        below = ln
    text = " ".join(lines[k].text.strip() for k in picked)
    if not text.endswith(".") or len(re.findall(r"[^\W\d_]", text)) < 3:
        return []
    return picked


def _page_text(
    page: pymupdf.Page, fonts: dict[str, dict[int, str]] | None = None
) -> tuple[str, tuple[tuple[str, str], ...]]:
    lines = _lines(page, fonts)
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
            part.moved = True
    page_font = _main_font(f for ln in lines for f in ln.fonts)
    for j, m in starts:
        if j not in headings and (above := _heading_above(lines, j, page_font)):
            headings[j] = (m[1], " ".join(lines[k].text.strip() for k in above))
            for k in above:
                lines[k].moved = True

    out: list[str] = []
    block = None
    for j, ln in enumerate(lines):
        if ln.moved:
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
    font_cache: dict[int, dict[int, str] | None] = {}
    with pymupdf.open(path) as doc:
        for i, pdf_page in enumerate(doc, start=1):
            text, headings = _page_text(pdf_page, _font_maps(pdf_page, font_cache))
            needs = len(text.strip()) < MIN_CHARS_PER_PAGE
            pages.append(Page(number=i, text=text, needs_ocr=needs, headings=headings))
    return pages


def needs_ocr(pages: list[Page]) -> bool:
    """True when the document as a whole looks scanned (or has no pages at all)."""
    if not pages:
        return True
    return sum(p.needs_ocr for p in pages) / len(pages) > SCANNED_DOC_RATIO
