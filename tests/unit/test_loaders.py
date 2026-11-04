from pathlib import Path

import pymupdf
import pytest

from taxlens.ingestion.loaders import (
    MIN_CHARS_PER_PAGE,
    SECTION_MARK,
    Page,
    load_pdf,
    needs_ocr,
)

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


def test_sample_circular_fixture() -> None:
    pages = load_pdf(FIXTURES / "sample_circular.pdf")
    assert len(pages) == 3
    assert [p.needs_ocr for p in pages] == [False, False, True]  # page 3 is blank
    assert "Circular No. 3/2025" in pages[0].text
    assert needs_ocr(pages)  # 1 of 3 pages (33%) is over the 30% threshold


def test_sample_notification_fixture() -> None:
    pages = load_pdf(FIXTURES / "sample_notification.pdf")
    assert len(pages) == 2
    assert not any(p.needs_ocr for p in pages)
    text = " ".join(pages[0].text.split())  # this font emits NBSPs between words, like gazettes
    assert "Notification No. 12/2025 – Central Tax (Rate)" in text
    assert "अधिसूचना" in text  # bilingual: the Hindi block survives loading


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "nope.pdf")


def test_margin_notes_become_marked_section_headings() -> None:
    pages = load_pdf(FIXTURES / "sample_act.pdf")
    assert [h for p in pages for h in p.headings] == [
        ("1", "Short title, extent and commencement."),
        ("2", "Definitions."),
        ("80C", "Deduction in respect of life insurance premia, etc."),  # note in 4 lines
        ("194J", "Fees for professional or technical services."),
    ]
    first, second = pages[0].text, pages[1].text
    assert f"{SECTION_MARK} 1. Short title, extent and commencement.\n(1) This Act" in first
    assert f"{SECTION_MARK} 80C. Deduction in respect of life insurance premia, etc.\n" in second
    assert "An individual or a Hindu undivided family" in second  # "80C." was alone on its line


def test_numbered_table_rows_and_citations_are_not_headings() -> None:
    pages = load_pdf(FIXTURES / "sample_act.pdf")
    first, second = pages[0].text, pages[1].text
    assert "\n1. More than 10000 and upto 100000. Two kilometres." in first  # untouched
    assert f"{SECTION_MARK} 1. More" not in first
    assert "43 of 1961." in second  # kept as text, never used as a heading
    assert all("1961" not in heading for _, heading in pages[1].headings)


def test_documents_without_margin_notes_have_no_marks() -> None:
    for name in ("sample_circular.pdf", "sample_notification.pdf"):
        pages = load_pdf(FIXTURES / name)
        assert all(not p.headings and SECTION_MARK not in p.text for p in pages)
