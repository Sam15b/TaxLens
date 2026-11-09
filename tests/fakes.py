"""In-memory test doubles, so unit tests need no Pinecone key or LLM."""

import math
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from langchain_core.messages import AIMessage

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


class ScriptedChatModel:
    """Replies with the given strings in order and records every prompt it was sent."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[Any] = []

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> AIMessage:
        self.prompts.append(messages)
        return AIMessage(content=self.replies.pop(0))


def corpus_chunks() -> list[ChunkRecord]:
    """A tiny public corpus: 194J, 80C and a GST notification, with 2-d "embeddings"."""
    rows = [
        (
            "ita-1961",
            "194J",
            [1.0, 0.0],
            "Section 194J: TDS at 10% on fees for professional services.",
        ),
        (
            "ita-1961",
            "80C",
            [0.0, 1.0],
            "Section 80C: deduction up to Rs. 1,50,000 for LIC premia.",
        ),
        ("gst-12", None, [0.6, 0.8], "Notification 12/2017-Central Tax (Rate): exempt services."),
    ]
    return [
        ChunkRecord(
            id=f"{doc}#{n:05d}",
            workspace_id="public",
            document_id=doc,
            content=text,
            embedding=vec,
            doc_type="act" if section else "notification",
            act="Income-tax Act, 1961" if section else None,
            section=section,
            title=doc,
            page=10 + n,
            source_url=f"https://example.gov.in/{doc}.pdf",
        )
        for n, (doc, section, vec, text) in enumerate(rows)
    ]


class FixedRouter:
    """A structured router model that always decides ``route`` and counts its calls."""

    def __init__(self, route: str) -> None:
        self.route = route
        self.calls = 0

    def invoke(self, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        from taxlens.agent.nodes.router import RouteDecision

        self.calls += 1
        return RouteDecision(route=self.route, reason="test")  # type: ignore[arg-type]


def wire_qa(
    monkeypatch: Any, store: FakeVectorStore, llm: ScriptedChatModel, route: str = "qa"
) -> FixedRouter:
    """Point the graph at fakes: the router decides ``route``; 'fee'/'TDS' questions embed near
    194J, others near 80C."""
    from taxlens.agent.nodes import qa, router

    def vector(text: str) -> list[float]:
        return [1.0, 0.0] if ("fee" in text.lower() or "tds" in text.lower()) else [0.0, 1.0]

    fixed = FixedRouter(route)
    monkeypatch.setattr(router, "get_router_llm", lambda: fixed)
    monkeypatch.setattr(qa, "get_store", lambda: store)
    monkeypatch.setattr(qa, "get_query_vector", vector)
    monkeypatch.setattr(qa, "get_llm", lambda: llm)
    return fixed
