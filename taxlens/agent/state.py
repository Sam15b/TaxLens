"""The LangGraph state every node reads and updates (docs/01-architecture.md § 1.5)."""

from typing import TypedDict

from taxlens.retrieval.base import SearchFilters, SearchHit


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
    workspace_ids: list[str]  # client workspace + "public"
    question: str
    filters: SearchFilters | None
    top_k: int
    hits: list[SearchHit]
    answer: str
    citations: list[Citation]
    attempts: int
    error: str | None
