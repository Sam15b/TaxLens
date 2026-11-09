"""QA nodes: retrieve → generate → check_citations, and the not_found exit.

Dependencies (vector store, query embedder, chat model) are reached through the small getters
below, so tests swap in fakes with ``monkeypatch.setattr(qa, "get_store", ...)``.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from taxlens.agent.citations import is_not_found, validate
from taxlens.agent.prompts import NOT_FOUND, QA_SYSTEM, qa_user_message, source_title
from taxlens.agent.state import Citation, TaxLensState
from taxlens.config import settings
from taxlens.retrieval.base import SearchHit, VectorStore


def get_store() -> VectorStore:
    from taxlens.retrieval.factory import get_vector_store

    return get_vector_store()


def get_query_vector(text: str) -> list[float]:
    from taxlens.llm.embeddings import embed_query

    return embed_query(text)


def get_llm() -> Any:
    from taxlens.llm.providers import get_chat_model

    return get_chat_model()


def message_text(message: Any) -> str:
    """The text of a chat reply (some providers return a list of content parts)."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else str(p.get("text", "")) for p in content)
    return str(content)


def citation_for(ref: int, hit: SearchHit) -> Citation:
    m = hit.metadata
    return Citation(
        ref=f"S{ref}",
        chunk_id=hit.chunk_id,
        title=source_title(hit),
        doc_type=str(m.get("doc_type", "")),
        section=m.get("section"),
        page=m.get("page"),
        issued_on=m.get("issued_on"),
        source_url=m.get("source_url"),
    )


def _question(state: TaxLensState) -> str:
    """What leaves this process: the masked question once the input guard has run."""
    return state.get("masked_question") or state["question"]


def retrieve(state: TaxLensState) -> dict[str, Any]:
    question = _question(state)
    hits = get_store().hybrid_search(
        query_text=question,
        query_vector=get_query_vector(question),
        workspace_ids=state["workspace_ids"],
        k=state.get("top_k") or settings.retrieval_k,
        filters=state.get("filters"),
    )
    return {"hits": hits}


def generate(state: TaxLensState) -> dict[str, Any]:
    """One answer attempt. From the second attempt on, the prompt carries the retry hint."""
    attempts = state.get("attempts", 0) + 1
    prompt = qa_user_message(_question(state), state["hits"], retry=attempts > 1)
    reply = get_llm().invoke([SystemMessage(QA_SYSTEM), HumanMessage(prompt)])
    return {"answer": message_text(reply).strip(), "attempts": attempts}


def check_citations(state: TaxLensState) -> dict[str, Any]:
    """Sets ``cited`` (the validated source numbers) or ``error``; the graph retries on error."""
    hits = state["hits"]
    ok, refs = validate(state["answer"], len(hits))
    if not ok:
        return {"cited": None, "citations": [], "error": "invalid citations"}
    if is_not_found(state["answer"]):
        return {"answer": NOT_FOUND, "cited": [], "citations": [], "error": None}
    citations = [citation_for(r, hits[r - 1]) for r in refs]
    return {"cited": refs, "citations": citations, "error": None}


def not_found(state: TaxLensState) -> dict[str, Any]:
    """The exit when retrieval finds nothing or citations keep failing: no unsupported claims."""
    return {"answer": NOT_FOUND, "cited": [], "citations": [], "error": None}
