"""Input and output guards around every request.

Day 7: pass-through. ``input_guard`` copies the question to ``masked_question``; Presidio
masking and the injection detector replace that on days 14–15. ``output_guard`` turns validated
source numbers into citations, answers refusals politely and attaches the disclaimer; unmasking
for the requesting user joins it on day 14.
"""

from typing import Any

from taxlens.agent.nodes.qa import citation_for
from taxlens.agent.prompts import DISCLAIMER, REFUSAL
from taxlens.agent.state import TaxLensState


def input_guard(state: TaxLensState) -> dict[str, Any]:
    return {"masked_question": state.get("question", ""), "attempts": state.get("attempts", 0)}


def output_guard(state: TaxLensState) -> dict[str, Any]:
    out: dict[str, Any] = {"disclaimer": DISCLAIMER}
    if state.get("route") == "refuse":
        return {**out, "answer": REFUSAL, "cited": [], "citations": []}
    hits = state.get("hits") or []
    refs = [r for r in state.get("cited") or [] if 1 <= r <= len(hits)]
    out["citations"] = [citation_for(r, hits[r - 1]) for r in refs]
    return out
