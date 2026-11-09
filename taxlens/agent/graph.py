"""The answer graph: retrieve → generate → check_citations, with one retry and a NOT_FOUND exit.

An answer whose citations don't check out is regenerated once with a retry hint; if it still
fails, the graph answers NOT_FOUND rather than an unsupported claim. Retrieval that finds
nothing skips the LLM. The router, calculator and guards join on day 7.
"""

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from taxlens.agent.nodes import qa
from taxlens.agent.state import TaxLensState

MAX_ATTEMPTS = 2


def _after_retrieve(s: TaxLensState) -> str:
    return "generate" if s.get("hits") else "not_found"


def _after_citations(s: TaxLensState) -> str:
    if s.get("cited") is not None and not s.get("error"):
        return "ok"
    return "retry" if s.get("attempts", 0) < MAX_ATTEMPTS else "fail"


def build_graph() -> Any:
    g = StateGraph(TaxLensState)
    g.add_node("retrieve", qa.retrieve)
    g.add_node("generate", qa.generate)
    g.add_node("check_citations", qa.check_citations)
    g.add_node("not_found", qa.not_found)

    g.add_edge(START, "retrieve")
    g.add_conditional_edges(
        "retrieve", _after_retrieve, {"generate": "generate", "not_found": "not_found"}
    )
    g.add_edge("generate", "check_citations")
    g.add_conditional_edges(
        "check_citations", _after_citations, {"ok": END, "retry": "generate", "fail": "not_found"}
    )
    g.add_edge("not_found", END)
    return g.compile()


@lru_cache
def get_graph() -> Any:
    return build_graph()
