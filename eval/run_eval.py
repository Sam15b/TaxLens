"""Run the golden set through TaxLens and score it.

    uv run python -m eval.run_eval --subset ci                     # graph + Pinecone + LLM judge
    uv run python -m eval.run_eval --subset all --out eval/results.json
    uv run python -m eval.run_eval --local-index --retrieval-only  # offline: no keys needed

Modes:
- default: the full answer graph (router, retrieval, generation, calc) per question, scored by the
  deterministic evaluators and the LLM judge (``--no-judge`` skips the judge).
- ``--retrieval-only``: only the retrieval step, scored by hit@5 and MRR (no LLM calls).
- ``--local-index``: search an in-memory dense index built from ``data/raw`` (cached in
  ``var/``) instead of Pinecone. It has no sparse side, so it is a dense-only baseline.

Free tiers: two questions run at a time, and calls that hit a rate limit (429) are retried with
exponential backoff. Output: per-item results plus a summary (mean per metric, per category,
latency p50/p95, total tokens), written as JSON and printed as a markdown table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from eval.evaluators import (
    DETERMINISTIC,
    KEYS,
    answer_correctness,
    hit_at_5,
    retrieval_mrr,
)
from eval.schema import GOLDEN_VERSION, GoldenItem
from eval.validate_golden import GOLDEN, SUBSETS, load_golden
from taxlens.config import settings
from taxlens.storage.models import PUBLIC_WORKSPACE_ID

ROOT = Path(__file__).resolve().parents[1]
LOCAL_INDEX = ROOT / "var" / "local-index.json"
CONCURRENCY = 2
MAX_RETRIES = 4
BACKOFF_SECONDS = 2.0
_RATE_LIMITED = re.compile(r"429|rate.?limit|resource.?exhausted|quota|too many requests", re.I)

Target = Callable[[GoldenItem], dict[str, Any]]


# ── dataset ──────────────────────────────────────────────────────────────────


def load_subset(name: str) -> list[GoldenItem]:
    items, problems = load_golden(GOLDEN)
    if problems:
        raise SystemExit("golden set is invalid; run: python -m eval.validate_golden")
    if name == "all":
        return items
    ids = set((yaml.safe_load(SUBSETS.read_text(encoding="utf-8")) or {}).get(name) or [])
    if not ids:
        raise SystemExit(f"no subset {name!r} in {SUBSETS}")
    return [i for i in items if i.id in ids]


# ── retries ──────────────────────────────────────────────────────────────────


def with_retry(
    fn: Callable[[], Any],
    *,
    sleep: Callable[[float], None] = time.sleep,
    retries: int = MAX_RETRIES,
) -> Any:
    """Call ``fn``; on a rate-limit error wait 2, 4, 8, ... seconds and try again."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == retries or not _RATE_LIMITED.search(f"{type(exc).__name__} {exc}"):
                raise
            sleep(BACKOFF_SECONDS * 2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover


# ── targets ──────────────────────────────────────────────────────────────────


def _hits(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"chunk_id": h.chunk_id, **h.metadata} for h in state.get("hits") or []]


def graph_target(item: GoldenItem) -> dict[str, Any]:
    from langchain_core.callbacks import get_usage_metadata_callback

    from taxlens.agent.graph import get_graph

    state = {
        "mode": "ask",
        "question": item.question,
        "workspace_ids": [PUBLIC_WORKSPACE_ID],
        "attempts": 0,
    }
    with get_usage_metadata_callback() as usage:
        result = with_retry(lambda: get_graph().invoke(state))
    tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
    return {
        "answer": result.get("answer", ""),
        "route": result.get("route"),
        "citations": result.get("citations") or [],
        "calculation": result.get("calculation"),
        "hits": _hits(result),
        "tokens": tokens,
    }


def retrieval_target(item: GoldenItem) -> dict[str, Any]:
    from taxlens.agent.nodes import qa

    result = with_retry(
        lambda: qa.retrieve({"question": item.question, "workspace_ids": [PUBLIC_WORKSPACE_ID]})
    )
    return {"answer": "", "route": "qa", "citations": [], "hits": _hits(result), "tokens": 0}


# ── local dense index ────────────────────────────────────────────────────────


def _corpus_key(raw: Path) -> str:
    h = hashlib.sha256(settings.embedding_model.encode())
    for pdf in sorted(raw.glob("*.pdf")):
        h.update(pdf.name.encode() + hashlib.sha256(pdf.read_bytes()).digest())
    return h.hexdigest()


def build_local_index(raw: Path, cache: Path = LOCAL_INDEX) -> Any:
    """Every listed PDF in ``raw`` run through the real ingestion pipeline into an InMemoryStore.
    Embedding the corpus takes minutes, so the chunks are cached until a PDF or the model changes.
    """
    from taxlens.retrieval.memory_store import InMemoryStore

    store = InMemoryStore()
    key = _corpus_key(raw)
    if cache.is_file():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            store.upsert([_record(c) for c in saved["chunks"]])
            print(f"local index: {len(saved['chunks'])} chunks from {cache}", file=sys.stderr)
            return store
    from scripts.seed_index import seed
    from taxlens.storage.sqlite_repo import SqliteRepository

    cache.parent.mkdir(parents=True, exist_ok=True)
    db = cache.with_suffix(".db")
    db.unlink(missing_ok=True)
    repo = SqliteRepository(db, migrate=True)
    try:
        seed(repo, store, raw)
    finally:
        repo.dispose()
        db.unlink(missing_ok=True)
    chunks = [asdict(c) for ws in store.by_workspace.values() for c in ws.values()]
    cache.write_text(json.dumps({"key": key, "chunks": chunks}, default=str), encoding="utf-8")
    return store


def _record(d: dict[str, Any]) -> Any:
    from taxlens.retrieval.base import ChunkRecord

    if d.get("issued_on"):
        d["issued_on"] = date.fromisoformat(d["issued_on"])
    return ChunkRecord(**d)


# ── running and scoring ──────────────────────────────────────────────────────


def score_item(
    item: GoldenItem, target: Target, evaluators: list[Callable[..., dict[str, Any]]]
) -> dict[str, Any]:
    ref = item.model_dump(mode="json")
    started = time.perf_counter()
    try:
        outputs = target(item)
        error = None
    except Exception as exc:  # one failing question must not sink the run
        outputs, error = {"answer": "", "route": None, "citations": [], "hits": []}, repr(exc)
    latency_ms = round((time.perf_counter() - started) * 1000)
    scores: dict[str, Any] = {}
    comments: dict[str, str] = {}
    for fn in evaluators:
        result: dict[str, Any] = (
            fn(outputs, ref)
            if error is None
            else {"key": KEYS.get(fn.__name__, fn.__name__), "score": None}
        )
        scores[result["key"]] = result["score"]
        if result.get("comment"):
            comments[result["key"]] = result["comment"]
    return {
        "id": item.id,
        "category": item.category,
        "question": item.question,
        "answer": outputs.get("answer", ""),
        "route": outputs.get("route"),
        "top_hits": [
            {k: h.get(k) for k in ("act", "section", "circular_no", "notification_no", "page")}
            for h in (outputs.get("hits") or [])[:5]
        ],
        "scores": scores,
        "comments": comments,
        "latency_ms": latency_ms,
        "tokens": outputs.get("tokens", 0),
        "error": error,
    }


def _mean(values: list[Any]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


def _percentile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1])


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for r in results for k in r["scores"]})
    by_category: dict[str, dict[str, Any]] = defaultdict(dict)
    for cat in sorted({r["category"] for r in results}):
        rows = [r for r in results if r["category"] == cat]
        by_category[cat] = {"n": len(rows)} | {
            k: _mean([r["scores"].get(k) for r in rows]) for k in keys
        }
    latencies = [r["latency_ms"] for r in results]
    return {
        "n": len(results),
        "errors": sum(r["error"] is not None for r in results),
        "metrics": {k: _mean([r["scores"].get(k) for r in results]) for k in keys},
        "by_category": dict(by_category),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "total_tokens": sum(r["tokens"] for r in results),
    }


def markdown(summary: dict[str, Any]) -> str:
    keys = list(summary["metrics"])
    fmt = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
    lines = [
        "| metric | all | " + " | ".join(summary["by_category"]) + " |",
        "|---|---:|" + "---:|" * len(summary["by_category"]),
    ]
    for k in keys:
        cats = " | ".join(fmt(c.get(k)) for c in summary["by_category"].values())
        lines.append(f"| {k} | {fmt(summary['metrics'][k])} | {cats} |")
    n_by_cat = " | ".join(str(c["n"]) for c in summary["by_category"].values())
    lines.append(f"| n | {summary['n']} | {n_by_cat} |")
    lines.append("")
    lines.append(
        f"latency p50 {summary['latency_p50_ms']} ms · p95 {summary['latency_p95_ms']} ms · "
        f"tokens {summary['total_tokens']} · errors {summary['errors']}"
    )
    return "\n".join(lines)


def run(
    items: list[GoldenItem],
    target: Target,
    evaluators: list[Callable[..., dict[str, Any]]],
    concurrency: int = CONCURRENCY,
) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda i: score_item(i, target, evaluators), items))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the golden set and score it.")
    p.add_argument("--subset", default="all", help="all, or a subset name in eval/subsets.yaml")
    p.add_argument("--out", type=Path, default=ROOT / "eval" / "results.json")
    p.add_argument("--retrieval-only", action="store_true", help="retrieval metrics, no LLM")
    p.add_argument("--local-index", action="store_true", help="in-memory dense index, no Pinecone")
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw", help="PDFs for --local-index")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY)
    a = p.parse_args(argv)

    items = load_subset(a.subset)
    if a.local_index:
        from taxlens.agent.nodes import qa

        store = build_local_index(a.raw)
        qa.get_store = lambda: store  # the graph's retrieval now searches the local index
    if a.retrieval_only:
        target: Target = retrieval_target
        evaluators: list[Callable[..., dict[str, Any]]] = [hit_at_5, retrieval_mrr]
    else:
        target = graph_target
        evaluators = [*DETERMINISTIC, *([] if a.no_judge else [answer_correctness])]

    started = time.monotonic()
    results = run(items, target, evaluators, a.concurrency)
    summary = summarise(results)
    report = {
        "meta": {
            "golden_version": GOLDEN_VERSION,
            "subset": a.subset,
            "mode": "retrieval-only" if a.retrieval_only else "full",
            "store": "local dense (BGE only)"
            if a.local_index
            else f"pinecone {settings.pinecone_index}",
            "model": None if a.retrieval_only else settings.llm_model,
            "judge": None if (a.retrieval_only or a.no_judge) else settings.judge_model,
            "embedding": settings.embedding_model,
            "hybrid_alpha": None if a.local_index else settings.hybrid_alpha,
            "k": settings.retrieval_k,
            "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "seconds": round(time.monotonic() - started, 1),
        },
        "summary": summary,
        "items": results,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(markdown(summary))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
