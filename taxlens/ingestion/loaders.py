"""PDF -> pages with PyMuPDF (LlamaIndex PyMuPDFReader) and low-text/OCR detection.

A page with almost no extractable text is probably a scanned image. Pages like that are flagged
``needs_ocr``; if too many pages of a document are, the whole document is treated as scanned and
skipped until OCR support exists (stretch issue). See docs/04-build-guide.md Week 1.
"""

from dataclasses import dataclass
from pathlib import Path

from llama_index.readers.file import PyMuPDFReader

MIN_CHARS_PER_PAGE = 40  # below this, the page is probably a scanned image
SCANNED_DOC_RATIO = 0.3  # above this share of low-text pages, treat the whole document as scanned


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as printed in citations
    text: str
    needs_ocr: bool


def load_pdf(path: Path | str) -> list[Page]:
    """One ``Page`` per PDF page, in order. Raises ``FileNotFoundError`` for a missing file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    docs = PyMuPDFReader().load_data(file_path=path, metadata=False)  # one Document per page
    return [
        Page(number=i, text=text, needs_ocr=len(text.strip()) < MIN_CHARS_PER_PAGE)
        for i, text in enumerate((d.text or "" for d in docs), start=1)
    ]


def needs_ocr(pages: list[Page]) -> bool:
    """True when the document as a whole looks scanned (or has no pages at all)."""
    if not pages:
        return True
    return sum(p.needs_ocr for p in pages) / len(pages) > SCANNED_DOC_RATIO
