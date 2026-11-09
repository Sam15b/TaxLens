"""An in-memory VectorStore: chunks per workspace, dense cosine scoring, the same filters.

Used by unit tests (as ``FakeVectorStore``) and by the eval runner's ``--local-index`` mode, which
measures retrieval without a Pinecone key. There is no sparse side, so it is a dense-only
baseline, not a copy of the hybrid store.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date

from taxlens.retrieval.base import ChunkRecord, SearchFilters, SearchHit


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


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


def _metadata(c: ChunkRecord) -> dict[str, object]:
    meta = {k: v for k, v in asdict(c).items() if k not in ("embedding", "content")}
    if isinstance(meta.get("issued_on"), date):
        meta["issued_on"] = meta["issued_on"].isoformat()  # as Pinecone hits report it
    return meta


class InMemoryStore:
    def __init__(self) -> None:
        self.by_workspace: dict[str, dict[str, ChunkRecord]] = {}
        self._norms: dict[str, float] = {}

    def upsert(self, chunks: Sequence[ChunkRecord]) -> None:
        for c in chunks:
            self.by_workspace.setdefault(c.workspace_id, {})[c.id] = c
            self._norms[c.id] = _norm(c.embedding)

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        workspace_ids: Sequence[str],
        k: int = 8,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        q_norm = _norm(query_vector) or 1.0
        scored = []
        for ws in dict.fromkeys(workspace_ids):
            for c in self.by_workspace.get(ws, {}).values():
                if _matches(c, filters):
                    dot = sum(x * y for x, y in zip(query_vector, c.embedding, strict=True))
                    scored.append((dot / (q_norm * (self._norms.get(c.id) or 1.0)), c))
        scored.sort(key=lambda sc: sc[0], reverse=True)
        return [
            SearchHit(
                chunk_id=c.id,
                document_id=c.document_id,
                content=c.content,
                score=score,
                metadata=_metadata(c),
            )
            for score, c in scored[:k]
        ]

    def delete_document(self, *, document_id: str, workspace_id: str) -> int:
        chunks = self.by_workspace.get(workspace_id, {})
        doomed = [i for i in chunks if i.startswith(f"{document_id}#")]
        for i in doomed:
            del chunks[i]
            self._norms.pop(i, None)
        return len(doomed)

    def count(self, workspace_id: str) -> int:
        return len(self.by_workspace.get(workspace_id, {}))
