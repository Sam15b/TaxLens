from typing import Any

import pytest

from taxlens.agent.nodes import router
from taxlens.agent.nodes.router import RouteDecision
from taxlens.agent.state import TaxLensState


class FakeStructured:
    """A structured-output model that returns a fixed decision and records its prompts."""

    def __init__(self, decision: Any) -> None:
        self.decision = decision
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return self.decision


@pytest.mark.parametrize(
    ("question", "route"),
    [
        ("What is the TDS rate under section 194J?", "qa"),
        ("My salary is 14 lakh, new regime. How much tax do I pay?", "calc"),
        ("Write me a poem about cricket.", "refuse"),
    ],
)
def test_each_route_from_the_llm(
    monkeypatch: pytest.MonkeyPatch, question: str, route: str
) -> None:
    llm = FakeStructured(RouteDecision(route=route, reason="because"))  # type: ignore[arg-type]
    monkeypatch.setattr(router, "get_router_llm", lambda: llm)
    out = router.route(TaxLensState(mode="ask", question=question, masked_question=question))
    assert out == {"route": route, "route_reason": "because"}
    assert question in llm.prompts[0] and "qa:" in llm.prompts[0]


def test_masked_question_is_what_the_llm_sees(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeStructured({"route": "qa", "reason": "dict output is validated"})
    monkeypatch.setattr(router, "get_router_llm", lambda: llm)
    out = router.route(TaxLensState(question="PAN ABCDE1234F?", masked_question="PAN <IN_PAN_1>?"))
    assert out["route"] == "qa" and "<IN_PAN_1>" in llm.prompts[0]
    assert "ABCDE1234F" not in llm.prompts[0]


def test_review_mode_and_prior_refusal_skip_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_llm() -> None:
        raise AssertionError("the router LLM must not be called")

    monkeypatch.setattr(router, "get_router_llm", no_llm)
    assert router.route(TaxLensState(mode="review", question="")) == {"route": "review"}
    assert router.route(TaxLensState(mode="ask", question="x", route="refuse")) == {}
