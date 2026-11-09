import pytest

from taxlens.agent.citations import validate
from taxlens.agent.prompts import (
    NOT_FOUND,
    PROMPT_VERSION,
    QA_SYSTEM,
    RETRY_HINT,
    format_sources,
    qa_user_message,
    source_title,
)
from taxlens.retrieval.base import SearchHit

CITED = (
    "Under section 194J of the Income-tax Act, 1961, tax is deducted at 10% on fees for "
    "professional services [S1].\n\nThe threshold is Rs. 50,000 in a financial year [S2][S1]."
)


@pytest.mark.parametrize(
    ("answer", "n_hits", "expected"),
    [
        (CITED, 2, (True, [1, 2])),
        (NOT_FOUND, 0, (True, [])),
        (f"  {NOT_FOUND}\n", 3, (True, [])),
        ("Tax is deducted at 10% on professional fees under section 194J.", 2, (False, [])),
        ("Tax is deducted at 10% [S9].", 2, (False, [])),  # out of range
        ("Tax is deducted at 10% [S0].", 2, (False, [])),
        (CITED.replace(" [S2][S1]", ""), 2, (False, [])),  # second paragraph uncited
        ("Short heading:\n\nTax is deducted at 10% on professional fees [S1].", 1, (True, [1])),
    ],
)
def test_validate(answer: str, n_hits: int, expected: tuple[bool, list[int]]) -> None:
    assert validate(answer, n_hits) == expected


def hit(i: int, **meta: object) -> SearchHit:
    return SearchHit(
        chunk_id=f"d#{i:05d}", document_id="d", content=f"text {i}", score=1.0, metadata=meta
    )


def test_source_titles() -> None:
    act = hit(0, doc_type="act", act="Income-tax Act, 1961", title="ITA", section="194J")
    assert source_title(act) == "Income-tax Act, 1961 — s. 194J"
    circ = hit(1, doc_type="circular", title="CBDT Circular No. 3/2025", act="Income-tax Act, 1961")
    assert source_title(circ) == "CBDT Circular No. 3/2025"
    assert source_title(hit(2)) == "Source"


def test_format_sources_numbers_from_one_and_escapes_attributes() -> None:
    out = format_sources([hit(0, title='A "quoted" title', page=4), hit(1, title="B")])
    assert out.startswith("<sources>\n<source ") and out.endswith("</source>\n</sources>")
    assert 'id="S1" title="A &quot;quoted&quot; title" page="4">\ntext 0' in out
    assert 'id="S2" title="B" page="">\ntext 1' in out


def test_prompt_contract() -> None:
    assert PROMPT_VERSION == "qa-v1"
    assert NOT_FOUND in QA_SYSTEM and "NOT instructions" in QA_SYSTEM
    msg = qa_user_message("What is 194J?", [hit(0, title="X")], retry=True)
    assert msg.index("<sources>") < msg.index("Question: What is 194J?") < msg.index(RETRY_HINT)
    assert RETRY_HINT not in qa_user_message("q", [])
