"""Cleaning tests. Snippets are copied from the real corpus PDFs (data/sources.yaml ids noted)."""

from pathlib import Path

import pytest

from taxlens.ingestion.cleaning import (
    build_vocabulary,
    clean_pages,
    clean_text,
    collapse_whitespace,
    drop_hindi_lines,
    find_repeated_lines,
    fix_hyphenation,
    is_devanagari_line,
    is_legacy_hindi_line,
    normalise_unicode,
)
from taxlens.ingestion.loaders import load_pdf

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# ita-2025 / cgst-act-2017, page 1: e-Gazette masthead, Hindi typed in a legacy (Kruti Dev) font.
LEGACY_HINDI = [
    "jftLVªh lañ Mhñ ,yñ—¼,u½04@0007@2003—25",
    "ubZ fnYyh] c`gLifrokj] vxLr 21] 2025@Jko.k 30] 1947 ¼'kd½",
    'bl Hkkx esa fHkUu i`"B la[;k nh tkrh gS ftlls fd ;g vyx ladyu ds :i esa j[kk tk ldsA',
    "vlk/kkj.k",
    "Hkkx  II — [k.M 1",
    "izkf/kdkj ls izdkf'kr",
    "xxxGIDHxxx",
]
ENGLISH_GAZETTE = [
    "REGISTERED NO. DL—(N)04/0007/2003—25",
    "PART II — Section 1",
    "NEW DELHI, THURSDAY, AUGUST 21, 2025/SHRAVANA 30, 1947 (Saka)",
    "(51) “Fund” means the Consumer Welfare Fund established under section 57;",
    "More than 1000000.",
    "THE INCOME-TAX ACT, 2025",
    "43 of 1961.",
]


@pytest.mark.parametrize("line", LEGACY_HINDI)
def test_legacy_font_hindi_is_detected(line: str) -> None:
    assert is_legacy_hindi_line(line)


@pytest.mark.parametrize("line", ENGLISH_GAZETTE)
def test_english_gazette_lines_are_not_hindi(line: str) -> None:
    assert not is_legacy_hindi_line(line)
    assert not is_devanagari_line(line)


def test_devanagari_heavy_lines_are_dropped_but_english_kept() -> None:
    text = "\n".join(
        [
            "अधिसूचना संख्या 12/2025 - केंद्रीय कर (दर)",
            "Notification No. 12/2025 – Central Tax (Rate)",
            "Notification No. 5/2021 (अधिसूचना)",  # mostly English: kept
        ]
    )
    assert drop_hindi_lines(text).split("\n") == [
        "Notification No. 12/2025 – Central Tax (Rate)",
        "Notification No. 5/2021 (अधिसूचना)",
    ]


def test_unicode_normalisation() -> None:
    # sample_notification.pdf: NBSPs between words, U+2010 for "-"; ita-2025: maths italic 𝐴
    raw = "G.S.R.\xa0401(E).‐\xa0In\xa0exercise of sub‐section (1) where 𝐴 is"
    assert normalise_unicode(raw) == "G.S.R. 401(E).- In exercise of sub-section (1) where A is"
    # en-dash is kept: notification numbers depend on it
    assert "12/2025 – Central" in normalise_unicode("12/2025 – Central")
    # soft hyphen at a line break joins the word
    assert normalise_unicode("deduc­\ntion") == "deduction"


def test_collapse_whitespace() -> None:
    raw = "6 \t\nTHE GAZETTE  OF INDIA\t\n\n\n\n[Part II— \r\nSec. 1]  "
    assert collapse_whitespace(raw) == "6\nTHE GAZETTE OF INDIA\n\n[Part II—\nSec. 1]"


# ita-2025: compounds broken at the line end. Joining them would ruin keyword search.
ACT_SNIPPET = (
    "the provisions of this sub-clause shall apply to a non-resident and to a short-term\n"
    "capital asset held as debenture stock"
)


@pytest.mark.parametrize(
    ("broken", "expected"),
    [
        ("sub-\nclause", "sub-clause"),  # rule 1: "sub-clause" occurs elsewhere
        ("Non-\nresident", "Non-resident"),  # rule 1, case-insensitive
        ("debenture-\nstock", "debenture-stock"),  # rule 3: both halves are words
        ("deduc-\ntion", "deduction"),  # rule 4: a real line-break hyphen
    ],
)
def test_hyphenation_keeps_real_compounds(broken: str, expected: str) -> None:
    vocab = build_vocabulary([ACT_SNIPPET])
    assert fix_hyphenation(f"a {broken} b", vocab) == f"a {expected} b"


def test_hyphenation_joins_when_document_uses_the_joined_form() -> None:
    # ita-2025 writes "semiconductor" elsewhere, so "semi-\nconductor" is a line-break hyphen
    text = "manufacture of semi-\nconductor wafers; the semiconductor unit"
    assert fix_hyphenation(text) == "manufacture of semiconductor wafers; the semiconductor unit"


def test_hyphenation_leaves_numbers_and_capitals_alone() -> None:
    text = "financial year 2017-\n18 and Central Tax-\nRate"
    assert fix_hyphenation(text) == text


def gazette_page(n: int, body: str) -> str:
    # cgst-act-2017 / ita-2025: running header alternates between odd and even pages
    side = "[PART II—" if n % 2 == 0 else "SEC. 1]"
    return f"{n} \t\nTHE GAZETTE OF INDIA EXTRAORDINARY\t\n{side} \n{body}\n43 of 1961."


def test_alternating_gazette_headers_are_removed() -> None:
    bodies = [f"(5{i}) “Fund” means the fund established under section {57 + i};" for i in range(6)]
    pages = [gazette_page(n, b) for n, b in enumerate(bodies, start=2)]
    assert {"the gazette of india extraordinary", "[part ii—", "sec. #]", "#"} <= (
        find_repeated_lines(pages)
    )
    cleaned = clean_pages(pages)
    for page, body in zip(cleaned, bodies, strict=True):
        assert body in page
        assert "GAZETTE" not in page
        assert "PART II" not in page and "SEC. 1]" not in page
        assert not page.split("\n")[0].isdigit()  # page number gone


def test_single_word_table_cells_are_not_headers() -> None:
    # cgst-notif-12-2017-rate: most pages end with the rate column value "Nil"
    items = ["flours, pulses and rice; and", "fresh milk", "curd; lassi", "salt", "eggs"]
    pages = [f"{n}\nServices by way of supply of {item}\nNil\nNil" for n, item in enumerate(items)]
    assert "nil" not in find_repeated_lines(pages)
    assert all(page.endswith("Nil\nNil") for page in clean_pages(pages))


def test_short_documents_have_no_header_detection() -> None:
    assert find_repeated_lines(["Header\nbody one", "Header\nbody two"]) == set()


def test_clean_pages_on_sample_circular() -> None:
    pages = clean_pages([p.text for p in load_pdf(FIXTURES / "sample_circular.pdf")])
    assert len(pages) == 3
    assert pages[2] == ""  # the blank page stays blank, keeping page numbers aligned
    first = pages[0]
    assert first.startswith("Circular No. 3/2025")  # running header removed
    assert "SYNTHETIC TEST FIXTURE" not in first and "Page 1 of 3" not in first  # footers
    assert "such deduction where payments" in first  # "deduc-\ntion" rejoined
    assert "section 194J" in first


def test_clean_text_on_bilingual_notification() -> None:
    page = load_pdf(FIXTURES / "sample_notification.pdf")[0].text
    text = clean_text(page)
    assert "अधिसूचना" not in text
    assert "Notification No. 12/2025 – Central Tax (Rate)" in text
    assert "sub-section (1) of section 9" in text
