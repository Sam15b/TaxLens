from pathlib import Path

from taxlens.ingestion.chunking import Section, split_act_into_sections
from taxlens.ingestion.cleaning import clean_pages
from taxlens.ingestion.loaders import load_pdf

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FILLER = "The assessee shall be allowed a deduction computed in the prescribed manner. " * 2


def act_text() -> str:
    return "\n\n".join(clean_pages([p.text for p in load_pdf(FIXTURES / "sample_act.pdf")]))


def test_sample_act_splits_on_layout_headings() -> None:
    text = act_text()
    sections = split_act_into_sections(text)
    assert [(s.section_no, s.heading) for s in sections] == [
        ("1", "Short title, extent and commencement."),
        ("2", "Definitions."),
        ("80C", "Deduction in respect of life insurance premia, etc."),
        ("194J", "Fees for professional or technical services."),
    ]
    by_no = {s.section_no: s for s in sections}
    assert by_no["80C"].body.startswith("80C. Deduction in respect of life insurance premia")
    assert "An individual or a Hindu undivided family" in by_no["80C"].body
    # the numbered table rows stay inside section 2
    assert "1. More than 10000 and upto 100000" in by_no["2"].body
    assert "Fees for professional" not in by_no["80C"].body  # no mixing of sections
    for s in sections:  # offsets point at the body in the full text
        assert text[s.start : s.start + len(s.body)] == s.body


def test_table_of_contents_is_skipped_for_inline_headings() -> None:
    toc = (
        "ARRANGEMENT OF SECTIONS\n80C. Deduction in respect of life insurance premia.\n"
        "80D. Deduction in respect of health insurance premia.\n"
        "194J. Fees for professional services.\n"
    )
    body = (
        f"80C. Deduction in respect of life insurance premia.\n(1) {FILLER}\n"
        f"80D. Deduction in respect of health insurance premia.\n(1) {FILLER}\n"
        f"194J. Fees for professional services.\n(1) {FILLER}\n"
    )
    sections = split_act_into_sections(toc + body)
    assert [s.section_no for s in sections] == ["80C", "80D", "194J"]
    assert all(s.body.endswith(f"{FILLER}\n") for s in sections)


def test_repeated_numbers_stay_in_the_previous_section() -> None:
    text = (
        f"5. Scope of total income.\n(1) {FILLER}\n"
        f"6. Residence in India.\n(1) {FILLER}\n"
        f"5. Scope of total income.\n{FILLER}\n"  # e.g. a list item restarting the numbering
    )
    sections = split_act_into_sections(text)
    assert [s.section_no for s in sections] == ["5", "6"]
    assert sections[1].body.endswith(f"5. Scope of total income.\n{FILLER}\n")  # nothing lost


def test_schedules_become_their_own_sections() -> None:
    text = (
        f"§ 536. Repeal and savings.\n(1) {FILLER}\n"
        "SCHEDULE XV\n(See section 123)\nDEDUCTION IN RESPECT OF LIFE INSURANCE PREMIA\n"
        f"1. Any sum paid to effect or to keep in force an insurance on the life. {FILLER}\n"
    )
    sections = split_act_into_sections(text)
    assert [(s.section_no, s.heading) for s in sections] == [
        ("536", "Repeal and savings."),
        (None, "Schedule XV: Deduction in respect of life insurance premia"),
    ]


def test_preamble_is_kept_and_text_without_headings_is_one_section() -> None:
    preamble = f"THE SAMPLE ACT, 2025\nAn Act to consolidate the law. {FILLER}\n"
    text = f"{preamble}80C. Heading here.\n{FILLER}"
    sections = split_act_into_sections(text)
    assert sections[0] == Section(None, "", text[: text.index("80C.")], 0)
    assert split_act_into_sections("just a paragraph of text") == [
        Section(None, "", "just a paragraph of text", 0)
    ]
    assert split_act_into_sections("   ") == []
