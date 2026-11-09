"""Deterministic evaluators on hand-made outputs."""

from typing import Any

import pytest

from eval import evaluators as ev
from taxlens.agent.prompts import NOT_FOUND, REFUSAL

ACT_REF: dict[str, Any] = {
    "answerable": True,
    "must_include": ["10%", "50,000"],
    "expected_sources": [{"act": "Income-tax Act, 1961", "section": "194J"}],
}
UNANSWERABLE: dict[str, Any] = {"answerable": False, "must_include": [], "expected_sources": []}
CALC_REF: dict[str, Any] = {"answerable": True, "expected_sources": [], "expected_total_tax": 81900}


def hit(i: int, section: str, act: str = "Income-tax Act, 1961") -> dict[str, Any]:
    return {"chunk_id": f"d#{i:05d}", "act": act, "section": section, "doc_type": "act"}


HITS = [hit(0, "80C"), hit(1, "194J"), hit(2, "194C"), hit(3, "194J", "Income-tax Act, 2025")]


def out(answer: str = "", route: str = "qa", **kw: Any) -> dict[str, Any]:
    return {"answer": answer, "route": route, "citations": [], "hits": HITS, **kw}


def score(fn: Any, outputs: dict[str, Any], ref: dict[str, Any]) -> Any:
    result = fn(outputs, ref)
    assert set(result) == {"key", "score"}
    return result["score"]


def test_hit_at_5_and_mrr() -> None:
    assert score(ev.hit_at_5, out(), ACT_REF) == 1
    assert score(ev.retrieval_mrr, out(), ACT_REF) == pytest.approx(0.5)  # 194J ranked 2nd
    misses = out(hits=[hit(i, "80C") for i in range(6)] + [hit(6, "194J")])
    assert score(ev.hit_at_5, misses, ACT_REF) == 0  # 7th place is outside the top 5
    assert score(ev.retrieval_mrr, misses, ACT_REF) == pytest.approx(1 / 7)
    assert score(ev.retrieval_mrr, out(hits=[]), ACT_REF) == 0.0
    for fn in (ev.hit_at_5, ev.retrieval_mrr):
        assert score(fn, out(), UNANSWERABLE) is None and score(fn, out(), CALC_REF) is None


def test_circular_and_notification_sources_match_on_their_number() -> None:
    ref = {"answerable": True, "expected_sources": [{"circular_no": "3/2025"}]}
    hits = [{"chunk_id": "c#00000", "doc_type": "circular", "circular_no": "3/2025"}]
    assert score(ev.hit_at_5, out(hits=hits), ref) == 1


def test_citation_precision_uses_hit_metadata() -> None:
    cited = out(citations=[{"chunk_id": "d#00001"}, {"chunk_id": "d#00000"}])
    assert score(ev.citation_precision, cited, ACT_REF) == pytest.approx(0.5)
    assert score(ev.citation_precision, out(), ACT_REF) is None  # nothing cited
    assert score(ev.citation_precision, cited, UNANSWERABLE) is None


def test_citation_validity() -> None:
    assert score(ev.citation_validity, out("TDS is 10% [S2][S4]."), ACT_REF) == 1
    assert score(ev.citation_validity, out("TDS is 10% [S5]."), ACT_REF) == 0  # only 4 hits
    assert score(ev.citation_validity, out(NOT_FOUND), ACT_REF) == 1
    assert score(ev.citation_validity, out("₹81,900", route="calc"), CALC_REF) is None


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Deduct 10% once fees exceed ₹50,000 [S2].", 1),
        ("Deduct 10% above Rs. 50000 in a year.", 1),
        ("Deduct 10% on all fees.", 0),
    ],
)
def test_must_include(answer: str, expected: int) -> None:
    assert score(ev.must_include, out(answer), ACT_REF) == expected
    assert score(ev.must_include, out(answer), UNANSWERABLE) is None


@pytest.mark.parametrize(
    ("outputs", "ref", "expected"),
    [
        (out(NOT_FOUND), UNANSWERABLE, 1),
        (out(REFUSAL, route="refuse"), UNANSWERABLE, 1),
        (out("Singapore's rate is 17%."), UNANSWERABLE, 0),
        (out("Deduct 10% [S2]."), ACT_REF, 1),
        (out(NOT_FOUND), ACT_REF, 0),
        (out(REFUSAL, route="refuse"), ACT_REF, 0),
    ],
)
def test_refusal_correct(outputs: dict[str, Any], ref: dict[str, Any], expected: int) -> None:
    assert score(ev.refusal_correct, outputs, ref) == expected


def test_calc_exact() -> None:
    assert score(ev.calc_exact, out(route="calc", calculation={"total_tax": 81900}), CALC_REF) == 1
    assert (
        score(ev.calc_exact, out(route="calc", calculation={"total_tax": "81900"}), CALC_REF) == 1
    )
    assert score(ev.calc_exact, out(route="calc", calculation={"total_tax": 81899}), CALC_REF) == 0
    assert score(ev.calc_exact, out("How much do you earn?", route="calc"), CALC_REF) == 0
    assert score(ev.calc_exact, out(route="calc", calculation={"total_tax": "n/a"}), CALC_REF) == 0
    assert score(ev.calc_exact, out(), ACT_REF) is None


def test_every_deterministic_evaluator_has_a_distinct_key() -> None:
    keys = [fn(out(), ACT_REF)["key"] for fn in ev.DETERMINISTIC]
    assert len(keys) == len(set(keys)) == 7
