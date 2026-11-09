"""The eval runner with fakes: no keys, no network."""

import json
from pathlib import Path
from typing import Any

import pytest

from eval import run_eval
from eval.evaluators import hit_at_5, refusal_correct
from eval.schema import GoldenItem
from tests.fakes import FakeVectorStore, ScriptedChatModel, corpus_chunks, wire_qa


def item(id_: str, category: str = "act_section", **kw: Any) -> GoldenItem:
    base: dict[str, Any] = {
        "id": id_,
        "category": category,
        "difficulty": "easy",
        "question": "What is the TDS on professional fees?",
        "expected_answer": "10% [194J]",
        "must_include": [],
        "expected_sources": [{"act": "Income-tax Act, 1961", "section": "194J"}],
        "answerable": True,
    }
    return GoldenItem.model_validate(base | kw)


def test_with_retry_backs_off_on_rate_limits_only() -> None:
    waits: list[float] = []
    calls = iter([RuntimeError("429 Too Many Requests"), RuntimeError("RESOURCE_EXHAUSTED"), "ok"])

    def flaky() -> str:
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    assert run_eval.with_retry(flaky, sleep=waits.append) == "ok"
    assert waits == [2.0, 4.0]
    with pytest.raises(ValueError):
        run_eval.with_retry(
            lambda: (_ for _ in ()).throw(ValueError("bad input")), sleep=waits.append
        )
    with pytest.raises(RuntimeError):
        run_eval.with_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("quota exceeded")),
            sleep=lambda _: None,
            retries=1,
        )


def test_score_item_survives_a_failing_target() -> None:
    def boom(_: GoldenItem) -> dict[str, Any]:
        raise RuntimeError("provider down")

    result = run_eval.score_item(item("q001"), boom, [hit_at_5, refusal_correct])
    assert result["error"] == "RuntimeError('provider down')"
    assert result["scores"] == {"hit@5": None, "refusal_accuracy": None}


def test_summary_and_markdown() -> None:
    results = [
        {
            "category": "a",
            "scores": {"hit@5": 1, "x": None},
            "latency_ms": 100,
            "tokens": 5,
            "error": None,
        },
        {
            "category": "a",
            "scores": {"hit@5": 0, "x": 0.5},
            "latency_ms": 300,
            "tokens": 7,
            "error": None,
        },
        {
            "category": "b",
            "scores": {"hit@5": None, "x": 1},
            "latency_ms": 200,
            "tokens": 0,
            "error": "e",
        },
    ]
    s = run_eval.summarise(results)
    assert s["metrics"] == {"hit@5": 0.5, "x": 0.75}
    assert s["by_category"]["a"] == {"n": 2, "hit@5": 0.5, "x": 0.5}
    assert s["by_category"]["b"] == {"n": 1, "hit@5": None, "x": 1.0}
    assert (s["n"], s["errors"], s["total_tokens"]) == (3, 1, 12)
    assert s["latency_p50_ms"] == 200 and 280 <= s["latency_p95_ms"] <= 300
    table = run_eval.markdown(s)
    assert "| hit@5 | 0.50 | 0.50 | – |" in table and "| n | 3 | 2 | 1 |" in table


def test_load_subset() -> None:
    assert len(run_eval.load_subset("all")) == 100
    ci = run_eval.load_subset("ci")
    assert 35 <= len(ci) <= 45
    with pytest.raises(SystemExit):
        run_eval.load_subset("nope")


@pytest.fixture
def store() -> FakeVectorStore:
    s = FakeVectorStore()
    s.upsert(corpus_chunks())
    return s


def test_retrieval_only_run(
    monkeypatch: pytest.MonkeyPatch, store: FakeVectorStore, tmp_path: Path
) -> None:
    wire_qa(monkeypatch, store, ScriptedChatModel())
    monkeypatch.setattr(run_eval, "load_subset", lambda _: [item("q001"), item("q002")])
    out = tmp_path / "r.json"
    assert run_eval.main(["--retrieval-only", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["meta"]["mode"] == "retrieval-only" and report["meta"]["model"] is None
    assert report["summary"]["metrics"] == {"hit@5": 1.0, "retrieval_mrr": 1.0}  # 194J ranks 1st
    assert report["items"][0]["top_hits"][0]["section"] == "194J"


def test_full_run_with_fakes(
    monkeypatch: pytest.MonkeyPatch, store: FakeVectorStore, tmp_path: Path
) -> None:
    llm = ScriptedChatModel("TDS is 10% on professional fees [S1].")
    wire_qa(monkeypatch, store, llm)
    monkeypatch.setattr(run_eval, "load_subset", lambda _: [item("q001", must_include=["10%"])])
    out = tmp_path / "r.json"
    assert run_eval.main(["--no-judge", "--out", str(out), "--concurrency", "1"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    metrics = report["summary"]["metrics"]
    assert metrics["hit@5"] == 1.0 and metrics["citation_precision"] == 1.0
    assert metrics["citation_validity"] == 1 and metrics["must_include_rate"] == 1
    assert metrics["refusal_accuracy"] == 1 and metrics["calc_exact"] is None
    assert report["items"][0]["route"] == "qa" and report["meta"]["judge"] is None


def test_local_index_is_built_once_then_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.seed_index as seed_index

    calls: list[Path] = []

    def fake_seed(repo: Any, store: Any, folder: Path, **_: Any) -> dict[str, int]:
        calls.append(folder)
        store.upsert(corpus_chunks())
        return {"x": 3}

    monkeypatch.setattr(seed_index, "seed", fake_seed)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.pdf").write_bytes(b"%PDF-1.7 fake")
    cache = tmp_path / "idx.json"
    first = run_eval.build_local_index(raw, cache)
    second = run_eval.build_local_index(raw, cache)
    assert first.count("public") == second.count("public") == 3 and len(calls) == 1
    (raw / "a.pdf").write_bytes(b"%PDF-1.7 changed")
    run_eval.build_local_index(raw, cache)
    assert len(calls) == 2  # a changed PDF invalidates the cache
    assert not cache.with_suffix(".db").exists()
