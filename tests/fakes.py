"""In-memory test doubles, so unit tests need no Pinecone key or LLM."""

import math
from collections.abc import Sequence
from dataclasses import asdict

from taxlens.retrieval.base import ChunkRecord, SearchFilters, SearchHit


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _matches(c: ChunkRecord, f: SearchFilters | None) -> bool:
    f = f or SearchFilters()
    if c.suspicious and not f.include_suspicious:
        return False
    if f.doc_types is not None and c.doc_type not in f.doc_types:
        return False
    if f.act is not None and c.act != f.act:
        return False
    if f.sections is not None and c.section not in f.sections:
        return False
    return f.issued_after is None or (c.issued_on is not None and c.issued_on >= f.issued_after)


class FakeVectorStore:
    """A dict of chunks per workspace with naive cosine scoring (dense only)."""

    def __init__(self) -> None:
        self.by_workspace: dict[str, dict[str, ChunkRecord]] = {}

    def upsert(self, chunks: Sequence[ChunkRecord]) -> None:
        for c in chunks:
            self.by_workspace.setdefault(c.workspace_id, {})[c.id] = c

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        workspace_ids: Sequence[str],
        k: int = 8,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        pool = [
            c
            for ws in workspace_ids
            for c in self.by_workspace.get(ws, {}).values()
            if _matches(c, filters)
        ]
        scored = sorted(pool, key=lambda c: _cosine(query_vector, c.embedding), reverse=True)
        return [
            SearchHit(
                chunk_id=c.id,
                document_id=c.document_id,
                content=c.content,
                score=_cosine(query_vector, c.embedding),
                metadata={k: v for k, v in asdict(c).items() if k not in ("embedding", "content")},
            )
            for c in scored[:k]
        ]

    def delete_document(self, *, document_id: str, workspace_id: str) -> int:
        chunks = self.by_workspace.get(workspace_id, {})
        doomed = [i for i in chunks if i.startswith(f"{document_id}#")]
        for i in doomed:
            del chunks[i]
        return len(doomed)
