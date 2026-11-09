"""Evaluators with the LangSmith signature ``(outputs, reference_outputs) -> {"key", "score"}``.

The same functions run in the local runner (eval/run_eval.py) and in LangSmith experiments.
``outputs`` is what the runner's target returns:

    {"answer": str, "route": str, "citations": [Citation], "calculation": dict | None,
     "hits": [{"chunk_id": ..., <hit metadata>}]}   # ranked, best first

``reference_outputs`` is a golden row (eval/schema.py). A score of ``None`` means "not applicable"
(e.g. retrieval metrics on a calc question) and is left out of the averages.
See docs/06-evaluation.md § 6.2.
"""

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from eval.prompts import JUDGE_PROMPT
from eval.schema import normalise
from taxlens.agent.citations import CITE_RE, is_not_found
from taxlens.agent.nodes.qa import message_text

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


# ── LLM judge ────────────────────────────────────────────────────────────────


class Judgement(BaseModel):
    score: float
    reason: str = ""

    @field_validator("score")
    @classmethod
    def _graded(cls, value: float) -> float:
        if value not in (0, 0.5, 1):
            raise ValueError(f"score must be 0, 0.5 or 1, got {value}")
        return value


def get_judge() -> Any:
    """The judge model: JUDGE_PROVIDER / JUDGE_MODEL, a different model from the answerer."""
    from taxlens.llm.providers import get_judge_model

    return get_judge_model()


def parse_judgement(text: str) -> Judgement:
    """The first JSON object in the judge's reply (models sometimes wrap it in prose or ```)."""
    match = re.search(r"\{.*?\}", text, re.S)
    if match is None:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    return Judgement.model_validate(json.loads(match[0]))


def answer_correctness(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> Result:
    """LLM judge: 1 / 0.5 / 0 against the reference answer. Unanswerable rows are scored by
    refusal_accuracy instead; an unparseable judge reply scores None with a comment."""
    if not reference_outputs.get("answerable") or not reference_outputs.get("expected_answer"):
        return {"key": "answer_correctness", "score": None}
    prompt = JUDGE_PROMPT.format(
        question=reference_outputs.get("question", ""),
        expected_answer=reference_outputs["expected_answer"],
        answer=outputs.get("answer") or "",
    )
    reply = get_judge().invoke(prompt)
    try:
        verdict = parse_judgement(message_text(reply))
    except (ValueError, ValidationError) as exc:
        return {"key": "answer_correctness", "score": None, "comment": f"unparseable: {exc}"}
    return {"key": "answer_correctness", "score": float(verdict.score), "comment": verdict.reason}


# The key each evaluator reports, for when it can't run (e.g. the target raised).
KEYS: dict[str, str] = {
    "hit_at_5": "hit@5",
    "retrieval_mrr": "retrieval_mrr",
    "citation_precision": "citation_precision",
    "citation_validity": "citation_validity",
    "must_include": "must_include_rate",
    "refusal_correct": "refusal_accuracy",
    "calc_exact": "calc_exact",
    "answer_correctness": "answer_correctness",
}
