from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from taxlens.ingestion.metadata import (
    base_section,
    extract_metadata,
    find_circular_no,
    find_issue_date,
    find_notification_no,
    section_refs,
)

SNIPPETS: list[dict[str, Any]] = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "fixtures" / "snippets.yaml").read_text(encoding="utf-8")
)


def test_there_are_enough_real_snippets() -> None:
    assert len(SNIPPETS) >= 15


@pytest.mark.parametrize("s", SNIPPETS, ids=[f"{i:02d}" for i in range(len(SNIPPETS))])
def test_snippet(s: dict[str, Any]) -> None:
    text = s["text"]
    assert find_circular_no(text) == s.get("circular_no"), s["source"]
    assert find_notification_no(text) == s.get("notification_no"), s["source"]
    assert find_issue_date(text) == s.get("issued_on"), s["source"]
    assert section_refs(text) == s.get("section_refs", []), s["source"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Circular No. 03 / 2025", "3/2025"),
        ("circular no.3/2025", "3/2025"),
        ("Circular No 12/2024", "12/2024"),
    ],
)
def test_circular_number_is_normalised(text: str, expected: str) -> None:
    assert find_circular_no(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Notification No. 01/2024 - Integrated Tax (Rate)", "1/2024-Integrated Tax (Rate)"),
        ("Notification No. 3/2019–Union Territory Tax", "3/2019-Union Territory Tax"),
        ("notification no. 1/2017-central tax ( rate )", "1/2017-Central Tax (Rate)"),
        ("Notification No. 1/2017-Compensation Cess (Rate)", "1/2017-Compensation Cess (Rate)"),
    ],
)
def test_notification_number_is_normalised(text: str, expected: str) -> None:
    assert find_notification_no(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Rs. 50,000.",
        "₹ 1,50,000 for the financial year 2025-26",
        "in the year 2017 and 1961",
        "Sl. No. 10, where the compensation",
        "sub-section (2) of the said section",
        "see paragraph 80C of the guidance",  # no keyword before it
        "section 2025",  # a year, not a section
    ],
)
def test_no_false_section_refs(text: str) -> None:
    assert section_refs(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("TDS u/s 194J on fees", ["194J"]),
        ("deduction u/s. 80CCD(1B)", ["80CCD(1B)"]),
        ("exempt under section 10(23C)(iv) of the Act", ["10(23C)(iv)"]),
        ("as per sec. 115BAC", ["115BAC"]),
        ("Section 80C and section 80c", ["80C"]),
        ("sections 80C, 80D and 80E", ["80C", "80D", "80E"]),
    ],
)
def test_section_refs(text: str, expected: list[str]) -> None:
    assert section_refs(text) == expected


def test_base_section() -> None:
    assert base_section("10(23C)(iv)") == "10"
    assert base_section("80ccd(1B)") == "80CCD"


def test_invalid_and_numeric_dates() -> None:
    assert find_issue_date("dated 31st February, 2025") is None
    assert find_issue_date("dated 28.03.2025") == date(2025, 3, 28)
    assert find_issue_date("no date here") is None


def test_extract_metadata_leaves_out_what_it_cannot_find() -> None:
    header = "Notification No.4/2017-Central Tax (Rate)\nNew Delhi, the 28th June, 2017"
    meta = extract_metadata(header)
    assert meta == {"notification_no": "4/2017-Central Tax (Rate)", "issued_on": "2017-06-28"}
    assert extract_metadata("nothing useful") == {}
