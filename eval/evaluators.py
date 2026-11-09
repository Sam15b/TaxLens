"""Evaluators with the LangSmith signature ``(outputs, reference_outputs) -> {"key", "score"}``.

The same functions run in the local runner (eval/run_eval.py) and in LangSmith experiments.
``outputs`` is what the runner's target returns:

    {"answer": str, "route": str, "citations": [Citation], "calculation": dict | None,
     "hits": [{"chunk_id": ..., <hit metadata>}]}   # ranked, best first

``reference_outputs`` is a golden row (eval/schema.py). A score of ``None`` means "not applicable"
(e.g. retrieval metrics on a calc question) and is left out of the averages.
See docs/06-evaluation.md § 6.2.
"""

from typing import Any

from eval.schema import normalise
from taxlens.agent.citations import CITE_RE, is_not_found

Result = dict[str, Any]
TOP_K = 5


def _expected(ref: dict[str, Any]) -> list[dict[str, str]]:
    return [{k: str(v) for k, v in s.items() if v} for s in ref.get("expected_sources") or []]


def _matches(meta: dict[str, Any], expected: list[dict[str, str]]) -> bool:
    return any(all(str(meta.get(k)) == v for k, v in exp.items()) for exp in expected)


def _refused(outputs: dict[str, Any]) -> bool:
    return outputs.get("route") == "refuse" or is_not_found(outputs.get("answer") or "")


def hit_at_5(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> Result:
    """1 if any expected source is among the top 5 retrieved chunks."""
    expected = _expected(reference_outputs)
    if not reference_outputs.get("answerable") or not expected:
        return {"key": "hit@5", "score": None}
    top = (outputs.get("hits") or [])[:TOP_K]
    return {"key": "hit@5", "score": int(any(_matches(h, expected) for h in top))}


def retrieval_mrr(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> Result:
    """1 / rank of the first retrieved chunk that matches an expected source (0 if none does)."""
    expected = _expected(reference_outputs)
    if not reference_outputs.get("answerable") or not expected:
        return {"key": "retrieval_mrr", "score": None}
    for rank, hit in enumerate(outputs.get("hits") or [], start=1):
        if _matches(hit, expected):
            return {"key": "retrieval_mrr", "score": 1 / rank}
    return {"key": "retrieval_mrr", "score": 0.0}


def citation_precision(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> Result:
    """Share of cited chunks that match an expected source (Act + section, circular or notice)."""
    expected = _expected(reference_outputs)
    citations = outputs.get("citations") or []
    if not expected or not citations:
        return {"key": "citation_precision", "score": None}
    by_id = {h.get("chunk_id"): h for h in outputs.get("hits") or []}
    good = sum(_matches(by_id.get(c.get("chunk_id"), c), expected) for c in citations)
    return {"key": "citation_precision", "score": good / len(citations)}


def citation_validity(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> Result:
    """1 if every [S#] in a QA answer points at a retrieved chunk (the graph enforces this)."""
    if outputs.get("route") != "qa":
        return {"key": "citation_validity", "score": None}
    answer, n = outputs.get("answer") or "", len(outputs.get("hits") or [])
    refs = [int(x) for x in CITE_RE.findall(answer)]
    return {"key": "citation_validity", "score": int(all(1 <= r <= n for r in refs))}


def must_include(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> Result:
    """1 if the answer contains every must_include fact (₹/Rs, commas, case and spacing ignored)."""
    facts = reference_outputs.get("must_include") or []
    if not facts:
        return {"key": "must_include_rate", "score": None}
    answer = normalise(outputs.get("answer") or "")
    return {"key": "must_include_rate", "score": int(all(normalise(f) in answer for f in facts))}


def refusal_correct(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> Result:
    """Unanswerable → NOT_FOUND or a refusal; answerable → neither."""
    refused = _refused(outputs)
    expected = not reference_outputs.get("answerable", True)
    return {"key": "refusal_accuracy", "score": int(refused == expected)}


def calc_exact(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> Result:
    """1 if the computed total tax equals expected_total_tax exactly."""
    expected = reference_outputs.get("expected_total_tax")
    if expected is None:
        return {"key": "calc_exact", "score": None}
    calculation = outputs.get("calculation") or {}
    total = calculation.get("total_tax")
    try:
        return {"key": "calc_exact", "score": int(total is not None and int(total) == expected)}
    except (TypeError, ValueError):
        return {"key": "calc_exact", "score": 0}


DETERMINISTIC = [
    hit_at_5,
    retrieval_mrr,
    citation_precision,
    citation_validity,
    must_include,
    refusal_correct,
    calc_exact,
]
