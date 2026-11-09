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


def test_calc_route_returns_the_calculation(
    client: TestClient, store: FakeVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire_qa(monkeypatch, store, ScriptedChatModel(), route="calc")  # extracts ₹14 lakh salary
    r = client.post("/ask", json={"question": "Salary 14 lakh, new regime, FY 2025-26: tax?"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "calc" and body["found"] is True and body["citations"] == []
    calc = body["calculation"]
    assert calc["total_tax"] == 81_900 and calc["taxable_income"] == 1_325_000
    assert calc["regime"] == "new" and calc["rules_source"] == "Finance Act, 2025"
    assert calc["breakdown"][-1] == {
        "lower": 1_200_000,
        "upper": 1_325_000,
        "rate": 0.15,
        "tax": 18_750,
    }
    assert "₹81,900" in body["answer"]


def test_calc_clarification_is_not_found(
    client: TestClient, store: FakeVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_core.runnables import RunnableLambda

    from taxlens.agent.nodes import calc

    wire_qa(monkeypatch, store, ScriptedChatModel(), route="calc")
    ask_back = calc.CalcExtraction(clarification="What is your annual income?")
    monkeypatch.setattr(calc, "get_extractor", lambda: RunnableLambda(lambda _: ask_back))
    body = client.post("/ask", json={"question": "How much tax do I pay?"}).json()
    assert (body["route"], body["found"], body["calculation"]) == ("calc", False, None)
    assert body["answer"] == "What is your annual income?"


def test_refuse_route(
    client: TestClient, store: FakeVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire_qa(monkeypatch, store, ScriptedChatModel(), route="refuse")
    body = client.post("/ask", json={"question": "Write a poem"}).json()
    assert (body["route"], body["found"], body["citations"]) == ("refuse", False, [])
