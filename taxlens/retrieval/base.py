"""The vector-store contract every backend implements (Pinecone now; Qdrant and Oracle in week 5).

The dense vector is computed once (taxlens/llm/embeddings.py) and shared by all backends. The
keyword side is backend-specific (SPLADE in Pinecone, BM25 in Qdrant, Oracle Text), so each store
encodes it itself. See docs/04-build-guide.md → "The VectorStore contract".
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol


def chunk_id(document_id: str, chunk_no: int) -> str:
    """``{document_id}#{chunk_no:05d}``: a document's chunks share an id prefix, so they can be
    listed and deleted together."""
    return f"{document_id}#{chunk_no:05d}"


@dataclass(frozen=True)
class ChunkRecord:
    id: str  # chunk_id(document_id, chunk_no)
    workspace_id: str
    document_id: str
    content: str
    embedding: list[float]
    doc_type: str
    act: str | None = None
    section: str | None = None
    heading: str | None = None
    title: str | None = None
    circular_no: str | None = None
    notification_no: str | None = None
    issued_on: date | None = None
    page: int | None = None
    source_url: str | None = None
    suspicious: bool = False


@dataclass(frozen=True)
class SearchFilters:
    doc_types: Sequence[str] | None = None
    act: str | None = None
    sections: Sequence[str] | None = None
    issued_after: date | None = None  # inclusive
    include_suspicious: bool = False


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[ChunkRecord]) -> None: ...

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        workspace_ids: Sequence[str],  # REQUIRED, no default: isolation can't be forgotten
        k: int = 8,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]: ...

    def delete_document(self, *, document_id: str, workspace_id: str) -> int: ...
