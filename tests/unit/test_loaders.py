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


def make_glyph_coded_pdf(path: Path, codes: bytes, differences: str, *, named: bool) -> Path:
    """A page drawn with a Type3 font whose glyphs are named ``/c<code>`` and that has no
    ToUnicode map, like the official Income-tax Act, 1961 PDF (PageMaker + Distiller)."""
    doc = pymupdf.open()
    page = doc.new_page()
    names = differences.replace("/", " /").split()[1:]
    glyph = doc.get_new_xref()
    doc.update_object(glyph, "<<>>")
    doc.update_stream(glyph, b"500 0 d0")  # an empty glyph: only the codes matter here
    procs = doc.get_new_xref()
    doc.update_object(procs, "<<" + "".join(f"{n} {glyph} 0 R" for n in names) + ">>")
    font = doc.get_new_xref()
    doc.update_object(
        font,
        "<</Type/Font/Subtype/Type3" + ("/Name/T4" if named else "") + "/FontBBox[0 0 500 700]"
        f"/FontMatrix[.001 0 0 .001 0 0]/CharProcs {procs} 0 R"
        f"/Encoding<</Type/Encoding/Differences[{differences}]>>"
        f"/FirstChar 0/LastChar {len(names) - 1}/Widths[{' '.join(['500'] * len(names))}]"
        "/Resources<<>>>>",
    )
    doc.xref_set_key(page.xref, "Resources", f"<</Font<</F1 {font} 0 R>>>>")
    contents = doc.get_new_xref()
    doc.update_object(contents, "<<>>")
    doc.update_stream(contents, b"BT /F1 12 Tf 72 720 Td <" + codes.hex().encode() + b"> Tj ET")
    doc.xref_set_key(page.xref, "Contents", f"{contents} 0 R")
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("named", [True, False])
def test_glyph_coded_type3_text_is_decoded(tmp_path: Path, named: bool) -> None:
    # codes 0..6 -> /c72 "H", /c101 "e", /c108 "l", /c111 "o", /c32 " ", /c33 "!", /c147 "“"
    path = make_glyph_coded_pdf(
        tmp_path / "t3.pdf",
        bytes([6, 0, 1, 2, 2, 3, 4, 5]),
        "0/c72/c101/c108/c111/c32/c33/c147",  # unspaced names, as in the real file
        named=named,
    )
    assert pymupdf.open(path)[0].get_text().startswith("\x06\x00\x01")  # PyMuPDF alone: codes
    assert load_pdf(path)[0].text == "“Hello !\n"


def test_type3_fonts_with_real_glyph_names_are_left_alone(tmp_path: Path) -> None:
    path = make_glyph_coded_pdf(tmp_path / "t3.pdf", bytes([0, 1]), "0/H/i", named=True)
    assert load_pdf(path)[0].text == "Hi\n"  # PyMuPDF maps standard names itself


def _write_line(page: pymupdf.Page, y: float, parts: list[tuple[str, str]]) -> None:
    x = 72.0
    for text, font in parts:
        page.insert_text((x, y), text, fontname=font, fontsize=10)
        x += pymupdf.get_text_length(text, fontname=font, fontsize=10)


def test_bold_headings_above_sections_become_marked_headings(tmp_path: Path) -> None:
    # Income-tax Act, 1961 layout: the heading is printed above the section, in the number's font
    doc = pymupdf.open()
    page = doc.new_page()
    bold, body = "hebo", "helv"
    _write_line(page, 100, [("Payment of rent by certain individuals.", bold)])
    _write_line(page, 113, [("194-IB.", bold), (" (1) Any person shall deduct tax.", body)])
    _write_line(page, 140, [("Fees for professional or technical", bold)])  # wraps
    _write_line(page, 152, [("services.", bold)])
    _write_line(page, 165, [("194J.", bold)])  # the number alone, text on the next line
    _write_line(page, 177, [("(1) Any person paying fees shall deduct tax.", body)])
    _write_line(page, 200, [("2. A bold numbered paragraph, not a section.", bold)])
    _write_line(page, 213, [("(a) a clause in body text.", body)])
    _write_line(page, 240, [("Numbered paragraph in body font.", body)])
    _write_line(page, 253, [("3.", body)])  # the page's body font: a list item, not a section
    _write_line(page, 266, [("Explanation.", "heit")])
    filler = "and the tax so deducted shall be paid to the credit"
    for i in range(6):  # most of a real page is body text
        _write_line(page, 290 + 13 * i, [(filler, body)])
    doc.save(tmp_path / "act.pdf")
    doc.close()

    loaded = load_pdf(tmp_path / "act.pdf")[0]
    assert loaded.headings == (
        ("194-IB", "Payment of rent by certain individuals."),
        ("194J", "Fees for professional or technical services."),
    )
    assert f"{SECTION_MARK} 194-IB. Payment of rent by certain individuals.\n" in loaded.text
    assert "\n(1) Any person paying fees shall deduct tax." in loaded.text
    assert "Fees for professional or technical\n" not in loaded.text  # moved, not duplicated
    assert "\n2. A bold numbered paragraph, not a section." in loaded.text
    assert "Numbered paragraph in body font.\n3.\n" in loaded.text
