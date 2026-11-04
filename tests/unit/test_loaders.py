from pathlib import Path

import pymupdf
import pytest

from taxlens.ingestion.loaders import MIN_CHARS_PER_PAGE, Page, load_pdf, needs_ocr

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BODY = "Section 194J of the Income-tax Act, 1961 requires deduction of tax at source on fees."


def make_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def test_pages_are_numbered_from_one_with_text(tmp_path: Path) -> None:
    pages = load_pdf(make_pdf(tmp_path / "a.pdf", [BODY, "Second page. " + BODY]))
    assert [p.number for p in pages] == [1, 2]
    assert "194J" in pages[0].text
    assert pages[1].text.strip().startswith("Second page.")
    assert not any(p.needs_ocr for p in pages)


def test_blank_and_near_blank_pages_need_ocr(tmp_path: Path) -> None:
    short = "x" * (MIN_CHARS_PER_PAGE - 1)
    pages = load_pdf(make_pdf(tmp_path / "b.pdf", [BODY, "", short]))
    assert [p.needs_ocr for p in pages] == [False, True, True]


def test_document_level_ocr_threshold() -> None:
    def doc(flags: list[bool]) -> list[Page]:
        return [Page(i, "", f) for i, f in enumerate(flags, start=1)]

    assert not needs_ocr(doc([False, False, False, True]))  # 25% scanned: keep
    assert needs_ocr(doc([False, True, True]))  # 67% scanned: skip until OCR exists
    assert needs_ocr([])


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "nope.pdf")
