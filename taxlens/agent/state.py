"""The LangGraph state every node reads and updates (docs/01-architecture.md § 1.5).

``pii_map`` (placeholder → original value) is deliberately NOT in the state. LangGraph state is
what checkpointers persist and what tracing exports, so anything in it can end up in Phoenix,
LangSmith or a checkpoint store. The map moves to a request-scoped contextvar on day 14, so the
original PII never leaves the process that is answering the request.
"""

from typing import Any, Literal, TypedDict

from taxlens.retrieval.base import SearchFilters, SearchHit

Route = Literal["qa", "calc", "review", "refuse"]


class Citation(TypedDict):
    ref: str  # "S1"
    chunk_id: str
    title: str  # "Income-tax Act, 2025 — s. 123"
    doc_type: str
    section: str | None
    page: int | None
    issued_on: str | None  # ISO date
    source_url: str | None


class TaxLensState(TypedDict, total=False):
    request_id: str
    user_id: str
    workspace_ids: list[str]  # client workspace + "public"
    mode: Literal["ask", "review"]  # set by the endpoint
    question: str  # original (never sent to an LLM)
    masked_question: str  # what nodes send to LLMs (real masking on day 14)
    document_text: str  # review mode (already masked)
    filters: SearchFilters | None
    top_k: int
    route: Route
    route_reason: str
    hits: list[SearchHit]
    answer: str
    cited: list[int] | None  # validated source numbers; None while the answer fails the check
    citations: list[Citation]
    calculation: dict[str, Any] | None  # calc route: the TaxResult
    findings: list[dict[str, Any]]  # review mode
    attempts: int
    error: str | None
    disclaimer: str
