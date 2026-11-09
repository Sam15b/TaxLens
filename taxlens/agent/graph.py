"""The answer graph (docs/01-architecture.md § 1.4).

    input_guard → route, then by route:
      qa:     retrieve → generate → check_citations → ok: output_guard
                                                   → retry: generate (once)
                                                   → fail: not_found → output_guard
      calc:   calc → output_guard
      review: review → output_guard
      refuse: output_guard
    output_guard → END

An answer whose citations don't check out is regenerated once with a retry hint; if it still
fails, the graph answers NOT_FOUND rather than an unsupported claim. Retrieval that finds
nothing skips the LLM. A refusal never reaches the answer model.
"""

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from taxlens.agent.nodes import calc, guards, qa, review, router
from taxlens.agent.state import TaxLensState

MAX_ATTEMPTS = 2


def _after_route(s: TaxLensState) -> str:
    return s.get("route", "refuse")


def _after_retrieve(s: TaxLensState) -> str:
    return "generate" if s.get("hits") else "not_found"


def _after_citations(s: TaxLensState) -> str:
    if s.get("cited") is not None and not s.get("error"):
        return "ok"
    return "retry" if s.get("attempts", 0) < MAX_ATTEMPTS else "fail"


def build_graph() -> Any:
    g = StateGraph(TaxLensState)
    g.add_node("input_guard", guards.input_guard)
    g.add_node("route", router.route)
    g.add_node("retrieve", qa.retrieve)
    g.add_node("generate", qa.generate)
    g.add_node("check_citations", qa.check_citations)
    g.add_node("not_found", qa.not_found)
    g.add_node("calc", calc.run)
    g.add_node("review", review.run)
    g.add_node("output_guard", guards.output_guard)

    g.add_edge(START, "input_guard")
    g.add_edge("input_guard", "route")
    g.add_conditional_edges(
        "route",
        _after_route,
        {"qa": "retrieve", "calc": "calc", "review": "review", "refuse": "output_guard"},
    )
    g.add_conditional_edges(
        "retrieve", _after_retrieve, {"generate": "generate", "not_found": "not_found"}
    )
    g.add_edge("generate", "check_citations")
    g.add_conditional_edges(
        "check_citations",
        _after_citations,
        {"ok": "output_guard", "retry": "generate", "fail": "not_found"},
    )
    g.add_edge("not_found", "output_guard")
    g.add_edge("calc", "output_guard")
    g.add_edge("review", "output_guard")
    g.add_edge("output_guard", END)
    return g.compile()


@lru_cache
def get_graph() -> Any:
    return build_graph()
