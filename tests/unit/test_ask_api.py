"""POST /ask through the real app, with the graph wired to fakes."""

import pytest
from fastapi.testclient import TestClient

from taxlens.agent.graph import get_graph
from taxlens.agent.nodes import qa
from taxlens.agent.prompts import DISCLAIMER, NOT_FOUND, PROMPT_VERSION
from taxlens.llm.providers import ProviderUnavailableError
from taxlens.main import create_app
from tests.fakes import FakeVectorStore, ScriptedChatModel, corpus_chunks, wire_qa

CITED = "Section 194J requires TDS at 10% on fees for professional services [S1]."


@pytest.fixture
def client() -> TestClient:
    get_graph.cache_clear()
    return TestClient(create_app())


@pytest.fixture
def store() -> FakeVectorStore:
    s = FakeVectorStore()
    s.upsert(corpus_chunks())
    return s


def test_ask_returns_a_cited_answer(
    client: TestClient, store: FakeVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire_qa(monkeypatch, store, ScriptedChatModel(CITED))
    r = client.post("/ask", json={"question": "TDS rate on professional fees?"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["route"] == "qa" and body["answer"] == CITED
    assert body["disclaimer"] == DISCLAIMER and body["prompt_version"] == PROMPT_VERSION
    assert body["model"] and body["calculation"] is None
    [c] = body["citations"]
    assert c == {
        "ref": "S1",
        "chunk_id": "ita-1961#00000",
        "title": "Income-tax Act, 1961 — s. 194J",
        "doc_type": "act",
        "section": "194J",
        "page": 10,
        "issued_on": None,
        "source_url": "https://example.gov.in/ita-1961.pdf",
    }


def test_filters_and_top_k_reach_retrieval(
    client: TestClient, store: FakeVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = ScriptedChatModel(NOT_FOUND)
    wire_qa(monkeypatch, store, llm)
    r = client.post(
        "/ask",
        json={
            "question": "exempt services?",
            "filters": {"doc_types": ["notification"]},
            "top_k": 1,
        },
    )
    assert r.status_code == 200
    assert r.json()["found"] is False and r.json()["citations"] == []
    human = llm.prompts[0][1].content
    assert 'id="S1"' in human and 'id="S2"' not in human and "Notification 12/2017" in human


def test_validation_errors(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid?", "top_k": 50}).status_code == 422


def test_no_llm_provider_is_a_503_envelope(
    client: TestClient, store: FakeVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire_qa(monkeypatch, store, ScriptedChatModel())

    def unavailable() -> None:
        raise ProviderUnavailableError("no LLM provider available")

    monkeypatch.setattr(qa, "get_llm", unavailable)
    r = client.post("/ask", json={"question": "TDS on fees?"}, headers={"X-Request-Id": "req-1"})
    assert r.status_code == 503
    assert r.json() == {
        "error": {
            "code": "llm_unavailable",
            "message": "no LLM provider available",
            "request_id": "req-1",
        }
    }
