"""``POST /ask``: a cited answer from the indexed corpus.

Week 1 version: no auth yet, and only the ``public`` workspace is searched. Auth, workspaces and
scopes arrive on day 15 (docs/05-api-reference.md § 5.4).
"""

from typing import Any

from fastapi import APIRouter

from taxlens.agent.citations import is_not_found
from taxlens.agent.graph import get_graph
from taxlens.agent.prompts import PROMPT_VERSION
from taxlens.api.schemas import AskRequest, AskResponse, CitationOut, ErrorResponse
from taxlens.llm.providers import model_name
from taxlens.storage.models import PUBLIC_WORKSPACE_ID

router = APIRouter(tags=["ask"])


def to_response(result: dict[str, Any]) -> AskResponse:
    answer = result.get("answer", "")
    route = result.get("route", "qa")
    citations = [CitationOut.model_validate(c) for c in result.get("citations") or []]
    calculation = result.get("calculation")
    if route == "calc":
        found = calculation is not None
    else:
        found = route == "qa" and bool(citations) and not is_not_found(answer)
    return AskResponse(
        answer=answer,
        route=route if route in ("qa", "calc", "refuse") else "qa",
        citations=citations,
        found=found,
        prompt_version=PROMPT_VERSION,
        model=model_name(),
        calculation=calculation,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={503: {"model": ErrorResponse, "description": "No LLM provider available"}},
)
def ask(req: AskRequest) -> AskResponse:
    result = get_graph().invoke(
        {
            "mode": "ask",
            "question": req.question,
            "workspace_ids": [PUBLIC_WORKSPACE_ID],
            "filters": req.filters.to_search_filters() if req.filters else None,
            "top_k": req.top_k,
            "attempts": 0,
        }
    )
    return to_response(result)
