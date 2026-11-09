# Sprint 1 review and retro: ingestion, hybrid retrieval, `/ask`

**Dates:** 2026-11-02 → 2026-11-09 · **Goal:** a question to `POST /ask` returns an answer with
valid `[S#]` citations from the indexed public corpus.

## What shipped

| Area | Delivered |
|------|-----------|
| Tooling | uv, ruff, mypy, pytest, pre-commit (ruff + gitleaks), CI on every PR |
| Data layer | Pydantic domain models, `Repository` protocol, SQLite backend + Alembic migration, contract tests |
| Corpus | 17 public sources in `data/sources.yaml` (Income-tax Acts 1961 and 2025, CGST Act, 5 CBDT circulars, 9 GST notifications), sha256-verified downloader |
| Loading | PyMuPDF loader with margin-note and heading-above layouts, low-text/OCR detection, and decoding of the glyph-coded fonts in the official 1961 Act PDF |
| Chunking | Section-aware chunks with a contextual header and page numbers. The corpus comes to 3,670 chunks (1961 Act: 794 sections) |
| Retrieval | Local BGE dense + SPLADE sparse vectors, Pinecone hybrid store (namespaces per workspace, alpha weighting, RRF section boost, delete by id prefix) |
| Answering | Gemini → Groq fallback factory, QA prompt `qa-v1`, citation validator, retrieve → generate → check_citations graph with one retry and a NOT_FOUND exit |
| API | `POST /ask` (public workspace, no auth yet), 503 error envelope when no LLM is available |

## Demo

Swagger → `POST /ask` "What is the maximum deduction under section 80C?" → a cited answer.

> **Pending:** the live demo and the 10 manual questions below need `PINECONE_API_KEY`,
> `GOOGLE_API_KEY` and `GROQ_API_KEY` in `.env`, then `scripts/init_pinecone.py` and
> `scripts/seed_index.py data/raw --wait`. The whole path is covered by tests with fakes
> (`tests/unit/test_graph.py`, `tests/unit/test_ask_api.py`).

## Ten hand-written questions (seed for the eval set)

| # | Question | Expect | Result |
|--:|----------|--------|--------|
| 1 | What is the maximum deduction under section 80C? | 80C, ₹1,50,000 | pending |
| 2 | What is the TDS rate on fees for professional services? | 194J | pending |
| 3 | Which incomes are not included in total income? | s. 10 (1961) | pending |
| 4 | Which section of the Income-tax Act, 2025 covers the deduction for life insurance premia? | s. 123 (2025) | pending |
| 5 | What does CBDT Circular 3/2025 cover? | TDS on salaries u/s 192, FY 2024-25 | pending |
| 6 | What relief does Circular 9/2025 give when a PAN was inoperative? | Rule 114AAA, 206AA/206CC | pending |
| 7 | From when must businesses above ₹5 crore turnover issue e-invoices? | Notification 10/2023-CT, 1 Aug 2023 | pending |
| 8 | Which services are taxed under reverse charge in GST? | Notification 13/2017-CT (Rate) | pending |
| 9 | What is the eligibility for input tax credit under the CGST Act? | CGST s. 16 | pending |
| 10 | What is the income tax rate in Singapore? | NOT_FOUND | pending |

## Keep / change / try

- **Keep:** testing against the real corpus. The 1961 Act's glyph-coded fonts and its
  heading-above layout only showed up once the real PDF was loaded, and the fixture-based unit
  tests would never have caught them.
- **Keep:** fakes behind small getters (`FakeVectorStore`, scripted chat model). Every graph and
  API path is tested without keys.
- **Change:** set up external accounts and keys on day 1, not day 5. The live Pinecone and LLM
  checks slipped because the keys weren't in place.
- **Try:** time-box layout heuristics. Record the misses (two 1961-Act sections whose heading
  ends the previous page) as issues instead of chasing every case.

## Velocity

- 6 sprint days, 43 commits (excluding merges), 235 unit tests passing.
- Carry-over into Sprint 2:
  - the live demo and the manual questions (needs API keys)
  - 18 chunks over BGE's 512-token limit
  - GST "@" lines dropped as Hindi
  - `SECTION_REF_RE` reads "section 194-IA" as `194`
