"""Route a request to qa / calc / refuse with one structured-output LLM call.

Review mode skips the LLM: the endpoint already knows it is a review. A request the input guard
already refused keeps its route.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from taxlens.agent.prompts import ROUTER_PROMPT
from taxlens.agent.state import TaxLensState


class RouteDecision(BaseModel):
    route: Literal["qa", "calc", "refuse"]
    reason: str = Field(description="One short sentence")


def get_router_llm() -> Any:
    from taxlens.llm.providers import get_structured_model

    return get_structured_model(RouteDecision)


def route(state: TaxLensState) -> dict[str, Any]:
    if state.get("mode") == "review":
        return {"route": "review"}
    if state.get("route") == "refuse":  # input_guard already refused it
        return {}
    question = state.get("masked_question") or state["question"]
    decision = get_router_llm().invoke(ROUTER_PROMPT.format(q=question))
    if not isinstance(decision, RouteDecision):
        decision = RouteDecision.model_validate(decision)
    return {"route": decision.route, "route_reason": decision.reason}
