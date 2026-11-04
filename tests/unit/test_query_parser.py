import pytest

from taxlens.retrieval.query_parser import QueryRefs, parse_query, section_refs_in


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the deduction limit under section 80C?", ["80C"]),
        ("Is TDS u/s 194J applicable to a doctor's consultation fee?", ["194J"]),
        ("what's the max i can claim under 80c and 80D?", ["80C", "80D"]),
        ("Am I exempt under section 10(23C)(iv)?", ["10"]),
        ("Does 115BAC allow 80CCD(1B)?", ["115BAC", "80CCD"]),
        ("Compare sections 80C, 80D and 80E", ["80C", "80D", "80E"]),
        ("presumptive tax for doctors under 44ADA", ["44ADA"]),
    ],
)
def test_section_refs_in_questions(question: str, expected: list[str]) -> None:
    assert section_refs_in(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "My salary is Rs. 50,000 a month, how much tax for FY 2025-26?",
        "I earn 50k per month and 12 lakh a year",
        "Is the due date 31st July or 15th September?",
        "I have a 2BHK flat and pay rent of ₹25000",
        "Where do I see TDS in Form 26AS or Form 16A?",
        "Rs 1.5 lakh limit, 10 cr turnover",
    ],
)
def test_no_false_section_refs_in_questions(question: str) -> None:
    assert section_refs_in(question) == []


def test_parse_query_finds_circulars_and_notifications() -> None:
    refs = parse_query(
        "Does Circular No. 03/2025 change Notification No. 12/2017- Central Tax (Rate) or 80C?"
    )
    assert refs == QueryRefs(
        sections=["80C"],
        circulars=["3/2025"],
        notifications=["12/2017-Central Tax (Rate)"],
    )
    assert refs
    assert not parse_query("how do I file my return?")
