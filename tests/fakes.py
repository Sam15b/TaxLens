"""In-memory test doubles, so unit tests need no Pinecone key or LLM."""

from typing import Any

from langchain_core.messages import AIMessage

from taxlens.retrieval.base import ChunkRecord
from taxlens.retrieval.memory_store import InMemoryStore

# The in-memory store lives in the package (the eval runner uses it too). Tests use this name.
FakeVectorStore = InMemoryStore


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

    from decimal import Decimal

    from langchain_core.runnables import RunnableLambda

    from taxlens.agent.nodes import calc

    extraction = calc.CalcExtraction(gross_income=Decimal(1_400_000))
    fixed = FixedRouter(route)
    monkeypatch.setattr(router, "get_router_llm", lambda: fixed)
    monkeypatch.setattr(calc, "get_extractor", lambda: RunnableLambda(lambda _: extraction))
    monkeypatch.setattr(qa, "get_store", lambda: store)
    monkeypatch.setattr(qa, "get_query_vector", vector)
    monkeypatch.setattr(qa, "get_llm", lambda: llm)
    return fixed
