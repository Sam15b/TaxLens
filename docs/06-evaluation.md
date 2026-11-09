# 6. Evaluation and observability

Most portfolio RAG projects never measure their answers. TaxLens measures every change and blocks
merges that make it worse. This is the part to lead with in interviews.

## 6.1 Golden set

`eval/golden_set.jsonl` has one JSON object per line:

```json
{"id": "q001", "category": "act_section", "difficulty": "easy",
 "question": "What is the maximum deduction allowed under section 80C of the Income-tax Act, 1961?",
 "expected_answer": "₹1,50,000 in aggregate for the specified investments/payments (old regime).",
 "must_include": ["1,50,000"],
 "expected_sources": [{"act": "Income-tax Act, 1961", "section": "80C"}],
 "answerable": true,
 "notes": "Also covered by 80CCE aggregate limit"}
{"id": "q087", "category": "out_of_scope", "difficulty": "easy",
 "question": "What is the corporate tax rate in Singapore?",
 "expected_answer": null, "must_include": [], "expected_sources": [],
 "answerable": false}
```

Target mix (100 questions):

| Category | Count | Example |
|----------|------:|---------|
| `act_section`: what a section says | 25 | "Who must deduct TDS under 194J?" |
| `act_mapping`: old ↔ new Act | 10 | "Which section of the 2025 Act corresponds to old 80C?" |
| `circular`: CBDT circular content | 20 | "What did Circular 3/2025 clarify about ...?" |
| `gst_notification`: rates, thresholds | 20 | "What is the e-invoicing turnover threshold?" |
| `calc`: tax computation | 10 | "Salary ₹14 lakh, new regime, FY 2025-26: tax?" |
| `multi_hop`: needs 2+ sources | 7 | "Circular X modifies section Y how?" |
| `out_of_scope` / unanswerable | 8 | Must return NOT_FOUND or refuse |

How to build it well:

1. Write questions **while reading the source page**, and record the exact page or section.
2. Write some questions the way a user would ("my CA said...") and some formally.
3. Write `must_include` as short facts (numbers, section ids). These are cheap, deterministic
   checks that complement the LLM judge.
4. Tag about 40 questions as the CI subset in `eval/subsets.yaml`, spread across categories.
5. Version the set: when you change it, bump `GOLDEN_VERSION` and re-run the baseline.

## 6.2 Metrics

| Metric | Definition | LLM needed? | Gate in CI |
|--------|------------|:-----------:|:----------:|
| `retrieval_hit@5` | Share of answerable questions where at least one expected source is in the top 5 hits | no | ✅ |
| `retrieval_mrr` | Mean of 1/rank of the first expected source | no | report only |
| `citation_precision` | Share of cited chunks that match an expected source (act+section / circular / notification) | no | ✅ |
| `citation_validity` | Share of answers whose `[S#]` refs all exist (should be 1.0, since the graph enforces it) | no | ✅ |
| `must_include_rate` | Share of answers containing every `must_include` string (after normalising ₹ and commas) | no | ✅ |
| `answer_correctness` | LLM judge scores 0/0.5/1 against `expected_answer` | yes | ✅ |
| `faithfulness` | LLM judge: is every claim supported by the cited chunks? | yes | report only |
| `refusal_accuracy` | Unanswerable → NOT_FOUND / refuse; answerable → not refused | no | ✅ |
| `calc_exact` | `total_tax` equals the expected value exactly | no | ✅ |
| `latency_p95_ms`, `tokens_per_answer`, `cost_per_100` | from traces / usage metadata | no | report only |

Most of the gate uses deterministic metrics, so CI isn't flaky. The LLM judge gets a looser
threshold.

### Judge prompt

```text
You grade answers about Indian tax law.
Question: {question}
Reference answer: {expected_answer}
Candidate answer: {answer}

Score 1 if the candidate states the same key facts as the reference (numbers, sections, conditions)
and nothing contradicting it. Score 0.5 if partially correct or missing an important condition.
Score 0 if wrong, contradicting, or not answering.
Return JSON: {"score": 0|0.5|1, "reason": "<one sentence>"}
```

The judge uses a **different model** from the answer model (`JUDGE_PROVIDER`), always at temperature 0.

## 6.3 Running evals

The local runner is built (day 10):

```powershell
uv run python -m eval.validate_golden                          # 100 rows, 7 categories
uv run python -m eval.run_eval --subset ci                     # graph + Pinecone + LLM judge
uv run python -m eval.run_eval --subset all --out eval/results.json
uv run python -m eval.run_eval --local-index --retrieval-only  # offline: no keys needed
```

`--local-index` searches an in-memory **dense-only** index built from `data/raw` (cached in
`var/`), and `--retrieval-only` scores retrieval without calling an LLM. Together they give a
retrieval baseline with no keys at all. The LangSmith path below (`--langsmith`) comes with the
experiments.

```python
# eval/run_eval.py (sketch)
import argparse, json
from langsmith import Client, evaluate
from eval.evaluators import (hit_at_5, citation_precision, must_include,
                             refusal_correct, answer_correctness)
from taxlens.agent.graph import get_graph
from taxlens.config import settings

def target(inputs: dict) -> dict:
    s = get_graph().invoke({"mode": "ask", "question": inputs["question"],
                            "workspace_ids": [PUBLIC_WORKSPACE_ID], "attempts": 0})
    return {"answer": s.get("answer"), "citations": s.get("citations", []),
            "hits": [h.metadata for h in s.get("hits", [])], "route": s.get("route")}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="all")          # all | ci
    p.add_argument("--langsmith", action="store_true")
    p.add_argument("--experiment", default="baseline")
    p.add_argument("--out", default="eval/results.json")
    a = p.parse_args()
    evaluators = [hit_at_5, citation_precision, must_include, refusal_correct, answer_correctness]

    if a.langsmith:
        evaluate(target, data=f"taxlens-golden-{a.subset}", evaluators=evaluators,
                 experiment_prefix=a.experiment, max_concurrency=2,
                 metadata={"prompt_version": PROMPT_VERSION, "model": settings.llm_model,
                           "k": settings.retrieval_k, "embedding": settings.embedding_model})
    else:
        results = run_locally(target, load_subset(a.subset), evaluators)   # plain loop
        json.dump(results, open(a.out, "w"), indent=2, default=str)
```

Evaluators use the LangSmith signature, so the same functions work locally and in LangSmith:

```python
# eval/evaluators.py
def hit_at_5(outputs: dict, reference_outputs: dict) -> dict:
    if not reference_outputs["answerable"]:
        return {"key": "hit@5", "score": None}
    expected = reference_outputs["expected_sources"]
    top = outputs["hits"][:5]
    hit = any(all(h.get(k) == v for k, v in exp.items()) for exp in expected for h in top)
    return {"key": "hit@5", "score": int(hit)}
```

Upload the dataset once with `scripts/upload_golden_to_langsmith.py`, using `Client().create_dataset`
and `create_examples`.

## 6.4 Experiments worth running (week 2)

Put these in a results table in the README. Interviewers like seeing the reasoning behind each choice.

| Experiment | Variable | Expectation |
|------------|----------|-------------|
| E1 | Pinecone `HYBRID_ALPHA` sweep: 1.0 (dense only), 0.7, 0.5, 0.3, 0.0 (sparse only) | Mid values win; sparse-heavy helps section/circular questions |
| E1b | section-boost query on/off | Boost lifts hit@5 on questions that name a section |
| E2 | chunk size 256 / 512 / 1024 | 512 is a good starting guess; measure it |
| E3 | with vs without contextual chunk header | The header helps keyword search on section numbers |
| E4 | k = 4 / 8 / 12 | Recall vs prompt cost |
| E5 | BGE query prefix on/off; local SPLADE vs Pinecone `pinecone-sparse-english-v0` | Small but measurable |
| E6 | Gemini vs Groq-Llama vs Ollama-Llama-8B | Quality / latency / cost trade-off |
| E7 | citation retry on/off | Validity goes to ~1.0 at some latency cost |
| E8 | Pinecone hosted reranker on top 20 → top 8 | Better precision, more latency + rerank quota |

## 6.5 The CI eval gate

```toml
# eval/thresholds.toml
[absolute]                       # hard floor
"hit@5"              = 0.85
citation_precision   = 0.85
citation_validity    = 0.98
must_include_rate    = 0.75
answer_correctness   = 0.70
refusal_accuracy     = 0.85
calc_exact           = 1.00

[regression]                     # max allowed drop vs eval/baseline.json
default = 0.03
answer_correctness = 0.05        # judge is noisier
```

```python
# eval/check_thresholds.py
import json, sys, tomllib
from pathlib import Path

res = json.loads(Path(sys.argv[1]).read_text())["summary"]
base = json.loads(Path("eval/baseline.json").read_text())["summary"]
cfg = tomllib.loads(Path("eval/thresholds.toml").read_text())

failures, lines = [], ["| metric | value | baseline | floor | status |", "|---|---|---|---|---|"]
for metric, floor in cfg["absolute"].items():
    v = res[metric]
    b = base.get(metric, v)          # no baseline yet -> only the floor applies
    max_drop = cfg["regression"].get(metric, cfg["regression"]["default"])
    ok = v >= floor and (b - v) <= max_drop
    lines.append(f"| {metric} | {v:.3f} | {b:.3f} | {floor} | {'✅' if ok else '❌'} |")
    if not ok:
        failures.append(metric)

Path("eval/summary.md").write_text("\n".join(lines))       # CI posts this to the PR / step summary
sys.exit(1 if failures else 0)
```

Flakiness controls:

- temperature 0 everywhere, with model versions pinned in `.env.ci`
- local dense + sparse embeddings (deterministic), and the CI corpus is fixed by `data/sources.yaml` checksums
- each CI run writes to its own Pinecone namespace (`ci-<run_id>`) in the `taxlens-ci` index, waits
  until `describe_index_stats()` shows all vectors, and deletes the namespace at the end
  (`if: always()`), so parallel PRs never see each other's data
- deterministic metrics carry most of the gate; the judge gets a 0.05 regression tolerance
- `max_concurrency=2` so free-tier rate limits don't turn into errors that count as failures
- when `main` gets better, a manual workflow updates `baseline.json` (so the baseline can't drift silently)

The full 100-question set plus a LangSmith experiment runs **nightly** (`nightly-eval.yml`). The CI
subset (about 40 questions) runs on every PR.

## 6.6 Observability with Phoenix

What to look at, and where it comes from:

| Signal | Source |
|--------|--------|
| Latency per node (route, retrieve, generate) | LangChain/LangGraph spans via OpenInference |
| Tokens in/out per call | LLM span attributes `llm.token_count.*` |
| Cost | Phoenix model pricing, or compute it from tokens × your price table |
| Errors / fallbacks used | span status + a `fallback_used` attribute you set |
| Retrieval quality per request | retriever span: query, returned chunk ids and scores |
| Prompt version | add `prompt_version` as a span attribute / metadata |

Useful habits:

- Save a few **Phoenix screenshots** for the README: a trace waterfall, and a token/cost view.
- Export traces of failing eval questions into a Phoenix dataset, which helps debugging.
- Check that PII doesn't appear in traces (see [Security § 7.2](07-security-compliance.md)).
  Phoenix stores every prompt, so a PII leak in a trace is still a leak.
- **LangSmith** is for datasets and experiment comparison. **Phoenix** is for runtime tracing. You
  can enable LangSmith tracing as well (`LANGSMITH_TRACING=true`), but use one tracer as your
  source of truth for latency and cost numbers.
