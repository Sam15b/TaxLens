"""Pinecone serverless hybrid store: one index, dense (BGE) + sparse (SPLADE) per record.

- **Namespaces** isolate tenants: ``public`` for the government corpus, ``ws-<id>`` per client
  workspace, both behind an optional prefix (CI and tests use a throwaway one).
- **Hybrid scoring**: the query's dense and sparse parts are weighted by ``alpha`` before the
  query, so ``score = alpha·dense + (1 − alpha)·sparse`` on the dotproduct index.
- **Section boost**: when the question names a section, a second query filtered to it runs, and
  the two rankings are merged with RRF, so an exact section match gets close to the top.

See docs/01-architecture.md § 1.3 B.
"""

from collections.abc import Sequence
from functools import cached_property
from itertools import batched
from typing import Any

from taxlens.config import settings
from taxlens.retrieval.base import ChunkRecord, SearchFilters, SearchHit
from taxlens.retrieval.fusion import rrf_fuse
from taxlens.retrieval.pinecone_filters import from_metadata, to_metadata, to_pinecone_filter
from taxlens.retrieval.query_parser import section_refs_in
from taxlens.storage.models import PUBLIC_WORKSPACE_ID

UPSERT_BATCH = 100  # records per request: stays under Pinecone's 2 MB request limit
SPARSE_BATCH = 16
BOOST_TOP_K = 10

Sparse = dict[str, list[Any]]  # {"indices": [...], "values": [...]}


def namespace_for(workspace_id: str) -> str:
    name = PUBLIC_WORKSPACE_ID if workspace_id == PUBLIC_WORKSPACE_ID else f"ws-{workspace_id}"
    return f"{settings.pinecone_namespace_prefix}{name}"


def scale(dense: Sequence[float], sparse: Sparse, alpha: float) -> tuple[list[float], Sparse]:
    """Weight the query: ``alpha`` on the dense part, ``1 − alpha`` on the sparse part."""
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return (
        [v * alpha for v in dense],
        {"indices": list(sparse["indices"]), "values": [v * (1 - alpha) for v in sparse["values"]]},
    )


def _hit(match: Any, score: float) -> SearchHit:
    meta = dict(match.metadata or {})
    return SearchHit(
        chunk_id=match.id,
        document_id=str(meta.get("document_id", match.id.split("#")[0])),
        content=str(meta.get("text", "")),
        score=score,
        metadata=from_metadata(meta),
    )


class PineconeStore:
    def __init__(self, index: Any, *, alpha: float = 0.5) -> None:
        """``index`` is a Pinecone data-plane client (``Pinecone(...).Index(name)``)."""
        self.index = index
        self.alpha = alpha

    @classmethod
    def from_settings(cls) -> "PineconeStore":
        from pinecone import Pinecone

        if settings.pinecone_api_key is None:
            raise RuntimeError("PINECONE_API_KEY is not set (add it to .env)")
        pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        return cls(pc.Index(settings.pinecone_index), alpha=settings.hybrid_alpha)

    @cached_property
    def _splade(self) -> Any:
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(settings.sparse_model, cache_dir=settings.fastembed_cache_path)

    def sparse(self, texts: Sequence[str]) -> list[Sparse]:
        """SPLADE vectors, computed locally (no API quota)."""
        return [
            {"indices": e.indices.tolist(), "values": e.values.tolist()}
            for e in self._splade.embed(list(texts), batch_size=SPARSE_BATCH)
        ]

    # ── writes ──────────────────────────────────────────────────────────────

    def upsert(self, chunks: Sequence[ChunkRecord]) -> None:
        by_ns: dict[str, list[ChunkRecord]] = {}
        for c in chunks:
            by_ns.setdefault(namespace_for(c.workspace_id), []).append(c)
        for ns, group in by_ns.items():
            sparse = self.sparse([c.content for c in group])
            records = [
                {"id": c.id, "values": c.embedding, "sparse_values": s, "metadata": to_metadata(c)}
                for c, s in zip(group, sparse, strict=True)
            ]
            for batch in batched(records, UPSERT_BATCH):
                self.index.upsert(vectors=list(batch), namespace=ns, show_progress=False)

    def delete_document(self, *, document_id: str, workspace_id: str) -> int:
        """Delete every chunk whose id starts with ``{document_id}#``; returns how many."""
        ns, deleted = namespace_for(workspace_id), 0
        for page in self.index.list(prefix=f"{document_id}#", namespace=ns):
            ids = [v.id for v in page.vectors]
            if ids:
                self.index.delete(ids=ids, namespace=ns)
                deleted += len(ids)
        return deleted

    def delete_namespace(self, workspace_id: str) -> None:
        """Delete a whole workspace (``index.delete(delete_all=True)`` on its namespace)."""
        self.index.delete(delete_all=True, namespace=namespace_for(workspace_id))

    def count(self, workspace_id: str) -> int:
        """Vectors in a workspace's namespace (eventually consistent: poll after upserts)."""
        stats = self.index.describe_index_stats()
        summary = stats.namespaces.get(namespace_for(workspace_id))
        return int(summary.vector_count) if summary is not None else 0

    # ── reads ───────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        *,
        query_text: str,
        query_vector: list[float],
        workspace_ids: Sequence[str],
        k: int = 8,
        filters: SearchFilters | None = None,
    ) -> list[SearchHit]:
        if not workspace_ids:
            return []
        dense, sparse = scale(query_vector, self.sparse([query_text])[0], self.alpha)
        flt = to_pinecone_filter(filters)
        namespaces = [namespace_for(w) for w in dict.fromkeys(workspace_ids)]

        main = self._query(namespaces, dense, sparse, flt, top_k=max(3 * k, 20))
        rankings = [[m.id for m in main]]
        pool = {m.id: m for m in main}
        refs = section_refs_in(query_text)
        if refs and not (filters and filters.sections):
            boosted = self._query(
                namespaces, dense, sparse, {**flt, "section": {"$in": refs}}, top_k=BOOST_TOP_K
            )
            rankings.append([m.id for m in boosted])
            pool.update({m.id: m for m in boosted})
        return [_hit(pool[i], score) for i, score in rrf_fuse(*rankings)[:k]]

    def _query(
        self,
        namespaces: Sequence[str],
        dense: list[float],
        sparse: Sparse,
        flt: dict[str, Any],
        top_k: int,
    ) -> list[Any]:
        """Query each namespace and merge by score (scores are comparable within one index)."""
        matches: list[Any] = []
        for ns in namespaces:
            res = self.index.query(
                namespace=ns,
                vector=dense,
                sparse_vector=sparse if sparse["indices"] else None,
                top_k=top_k,
                filter=flt or None,
                include_metadata=True,
            )
            matches.extend(res.matches)
        return sorted(matches, key=lambda m: m.score, reverse=True)[:top_k]
