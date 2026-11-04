"""Checks on the real corpus. data/raw is gitignored, so each test skips when its PDF is absent
(CI, or before `scripts/download_sources.py` / a manual download)."""

from functools import cache
from pathlib import Path

import pytest

from taxlens.ingestion.chunking import Section, split_act_into_sections
from taxlens.ingestion.cleaning import clean_pages
from taxlens.ingestion.loaders import load_pdf

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@cache
def sections_of(source_id: str) -> tuple[str, list[Section]]:
    path = RAW / f"{source_id}.pdf"
    if not path.exists():
        pytest.skip(f"{path.name} not downloaded")
    text = "\n\n".join(clean_pages([p.text for p in load_pdf(path)]))
    return text, split_act_into_sections(text)


@pytest.mark.parametrize(
    ("source_id", "section_no", "heading_start"),
    [
        ("ita-1961", "80C", "Deduction in respect of life insurance premia"),
        ("ita-1961", "194J", "Fees for professional"),
        ("ita-1961", "10", "Incomes not included in total income"),
        ("ita-2025", "123", "Deduction for life insurance premia"),
        ("ita-2025", "393", "Tax to be deducted at source"),
        ("cgst-act-2017", "9", "Levy and collection"),
        ("cgst-act-2017", "16", "Eligibility and conditions for taking input tax credit"),
    ],
)
def test_key_sections_start_at_their_heading(
    source_id: str, section_no: str, heading_start: str
) -> None:
    _, sections = sections_of(source_id)
    found = [s for s in sections if s.section_no == section_no]
    assert len(found) == 1
    assert found[0].heading.startswith(heading_start)
    assert found[0].body.startswith(f"{section_no}. {heading_start}")


@pytest.mark.parametrize(("source_id", "count"), [("cgst-act-2017", 174), ("ita-2025", 530)])
def test_section_coverage(source_id: str, count: int) -> None:
    text, sections = sections_of(source_id)
    numbered = [s.section_no for s in sections if s.section_no is not None]
    assert len(numbered) == len(set(numbered)) >= count
    for s in sections:
        assert text[s.start : s.start + len(s.body)] == s.body
