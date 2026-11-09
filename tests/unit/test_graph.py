"""The answer graph end to end with FakeVectorStore and a scripted chat model."""

from typing import Any

import pytest

from taxlens.agent.graph import MAX_ATTEMPTS, build_graph
from taxlens.agent.prompts import NOT_FOUND, RETRY_HINT
from tests.fakes import FakeVectorStore, ScriptedChatModel, corpus_chunks, wire_qa

CITED = "Section 194J of the Income-tax Act, 1961 requires TDS at 10% on professional fees [S1]."


@pytest.fixture
def store() -> FakeVectorStore:
    s = FakeVectorStore()
    s.upsert(corpus_chunks())
    return s


def ask(question: str, workspaces: list[str] | None = None) -> dict[str, Any]:
    state = {"question": question, "workspace_ids": workspaces or ["public"], "attempts": 0}
    return dict(build_graph().invoke(state))


def test_cited_answer(monkeypatch: pytest.MonkeyPatch, store: FakeVectorStore) -> None:
    llm = ScriptedChatModel(CITED)
    wire_qa(monkeypatch, store, llm)
    out = ask("What is the TDS rate on professional fees?")
    assert out["answer"] == CITED and out["attempts"] == 1
    [c] = out["citations"]
    assert c["ref"] == "S1" and c["chunk_id"] == "ita-1961#00000"
    assert c["title"] == "Income-tax Act, 1961 — s. 194J" and c["page"] == 10
    assert c["source_url"] == "https://example.gov.in/ita-1961.pdf"
    system, human = llm.prompts[0]
    assert "Answer ONLY using the sources" in system.content
    assert '<source id="S1" title="Income-tax Act, 1961 — s. 194J"' in human.content
    assert RETRY_HINT not in human.content


def test_uncited_answer_is_retried_once(
    monkeypatch: pytest.MonkeyPatch, store: FakeVectorStore
) -> None:
    llm = ScriptedChatModel("TDS is 10% on professional fees.", CITED)
    wire_qa(monkeypatch, store, llm)
    out = ask("TDS on professional fees?")
    assert out["answer"] == CITED and out["attempts"] == 2
    assert RETRY_HINT in llm.prompts[1][1].content


def test_persistent_bad_citations_end_not_found(
    monkeypatch: pytest.MonkeyPatch, store: FakeVectorStore
) -> None:
    llm = ScriptedChatModel(*["TDS is 10% [S9]."] * MAX_ATTEMPTS)
    wire_qa(monkeypatch, store, llm)
    out = ask("TDS on professional fees?")
    assert (out["answer"], out["citations"], out["attempts"]) == (NOT_FOUND, [], MAX_ATTEMPTS)


def test_model_saying_not_found_is_accepted(
    monkeypatch: pytest.MonkeyPatch, store: FakeVectorStore
) -> None:
    wire_qa(monkeypatch, store, ScriptedChatModel(f'"{NOT_FOUND}"'))
    out = ask("What is the GST rate on gold?")
    assert (out["answer"], out["citations"]) == (NOT_FOUND, [])


def test_no_hits_skips_the_llm(monkeypatch: pytest.MonkeyPatch, store: FakeVectorStore) -> None:
    llm = ScriptedChatModel()
    wire_qa(monkeypatch, store, llm)
    out = ask("TDS on fees?", workspaces=["empty-workspace"])
    assert (out["answer"], out["citations"]) == (NOT_FOUND, []) and llm.prompts == []
