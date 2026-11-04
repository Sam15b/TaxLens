from pathlib import Path

import pytest

from taxlens.ingestion.chunking import (
    CHUNK_SIZE,
    Section,
    chunk_document,
    contextual_header,
    split_act_into_sections,
    splitter,
)
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


ACT_META = {"doc_type": "act", "act": "Sample Taxation Act, 2025", "title": "Sample Act"}
CIRCULAR_META = {
    "doc_type": "circular",
    "title": "CBDT Circular No. 3/2025 - TDS on fees under section 194J",
    "circular_no": "3/2025",
    "issued_on": "2025-03-15",
}
NOTIFICATION_META = {
    "doc_type": "notification",
    "title": "Notification No. 12/2025-Central Tax (Rate)",
    "notification_no": "12/2025-Central Tax (Rate)",
    "issued_on": "2025-06-28",
}


def test_contextual_header() -> None:
    meta = {**ACT_META, "section": "80C", "heading": "Deduction in respect of life insurance."}
    assert contextual_header(meta) == (
        "Sample Taxation Act, 2025 | Section 80C: Deduction in respect of life insurance."
    )
    assert contextual_header(
        {**ACT_META, "section": None, "heading": "Schedule XV: Deduction"}
    ) == ("Sample Taxation Act, 2025 | Schedule XV: Deduction")
    # the number is already in the title, so it isn't repeated; the date is added
    assert contextual_header(CIRCULAR_META) == (
        "CBDT Circular No. 3/2025 - TDS on fees under section 194J | dated 2025-03-15"
    )
    assert contextual_header(
        {"doc_type": "circular", "title": "Guidance", "circular_no": "3/2025"}
    ) == ("Guidance | Circular No. 3/2025")


def test_act_chunks_never_mix_sections_and_carry_the_header() -> None:
    text = act_text()
    sections = {s.section_no: s for s in split_act_into_sections(text)}
    chunks = chunk_document(text, ACT_META)
    assert {m["section"] for _, m in chunks} == {"1", "2", "80C", "194J"}
    for chunk_text, meta in chunks:
        header, _, body = chunk_text.partition("\n\n")
        assert header == contextual_header(meta)
        assert body in sections[meta["section"]].body  # never crosses a section boundary
        assert text[meta["offset"] :].startswith(body)
    first_80c = next(c for c, m in chunks if m["section"] == "80C")
    assert first_80c.startswith("Sample Taxation Act, 2025 | Section 80C: Deduction in respect of")


def test_long_sections_split_into_overlapping_chunks_within_budget() -> None:
    sentences = " ".join(
        f"Clause {i} allows a deduction of the prescribed amount." for i in range(400)
    )
    text = f"§ 80C. Deduction in respect of life insurance premia.\n{sentences}\n"
    chunks = chunk_document(text, ACT_META)
    assert len(chunks) > 3
    assert [m["chunk_no"] for _, m in chunks] == list(range(len(chunks)))
    tokenizer = splitter()._tokenizer
    assert all(len(tokenizer(c)) <= CHUNK_SIZE for c, _ in chunks)
    offsets = [m["offset"] for _, m in chunks]
    assert offsets == sorted(offsets)
    bodies = [c.partition("\n\n")[2] for c, _ in chunks]
    assert bodies[0][-40:] in bodies[1] or bodies[1][:40] in bodies[0]  # overlap


@pytest.mark.parametrize(
    ("name", "meta"),
    [
        ("sample_circular.pdf", CIRCULAR_META),
        ("sample_notification.pdf", NOTIFICATION_META),
        ("sample_act.pdf", ACT_META),
    ],
)
def test_every_fixture_produces_chunks(name: str, meta: dict[str, str]) -> None:
    text = "\n\n".join(clean_pages([p.text for p in load_pdf(FIXTURES / name)]))
    chunks = chunk_document(text, meta)
    assert len(chunks) > 0
    assert all(c.startswith(contextual_header(m)) for c, m in chunks)


def test_empty_text_has_no_chunks() -> None:
    assert chunk_document("  \n ", CIRCULAR_META) == []
