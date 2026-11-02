# 2. Project structure

TaxLens uses a basic Python layout: one importable package (`taxlens/`) at the repo root, with
tests, evals, scripts, infra and docs next to it. There is no `src/` folder and no build step.
`uv` manages the virtualenv, and you run modules with `python -m taxlens....`.

## 2.1 Folder tree

```text
TaxLens/
├── pyproject.toml              # dependencies + ruff/mypy/pytest config (managed by uv)
├── uv.lock                     # pinned versions — commit this
├── .python-version             # 3.12
├── .env.example                # every setting, no real secrets — copy to .env
├── .gitignore
├── .dockerignore
├── Dockerfile                  # one image for API, worker and MCP server
├── docker-compose.yml          # api, worker, azurite, phoenix (+ optional ollama, cosmos-emulator, qdrant, oracle)
├── alembic.ini                 # migrations for the SQLite repository
├── README.md                   # project pitch, architecture diagram, eval results, how to run
│
├── taxlens/                    # ── the application package ──
│   ├── __init__.py
│   ├── config.py               # pydantic-settings Settings; the ONLY place env vars are read
│   ├── main.py                 # FastAPI app factory, router registration, tracing setup
│   ├── logging_config.py       # structured logging (never logs raw request text)
│   │
│   ├── api/                    # HTTP layer only: validation, auth, mapping to services
│   │   ├── __init__.py
│   │   ├── deps.py             # get_db, get_current_user, require_scope, get_workspace
│   │   ├── schemas.py          # Pydantic request/response models (AskRequest, AskResponse, ...)
│   │   ├── errors.py           # exception handlers -> consistent error JSON
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── auth.py         # POST /auth/token
│   │       ├── ingest.py       # POST /ingest, GET /documents/{id}
│   │       ├── ask.py          # POST /ask
│   │       ├── review.py       # POST /review
│   │       ├── redact.py       # POST /redact
│   │       └── health.py       # GET /health, GET /ready
│   │
│   ├── agent/                  # LangGraph orchestration
│   │   ├── __init__.py
│   │   ├── state.py            # TaxLensState, Citation
│   │   ├── graph.py            # build_graph() -> compiled graph
│   │   ├── prompts.py          # all prompt templates in one place (versioned)
│   │   ├── citations.py        # parse [S#] refs, validate against hits
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── guards.py       # input_guard, output_guard
│   │       ├── router.py       # route (RouteDecision structured output)
│   │       ├── qa.py           # retrieve, generate
│   │       ├── calc.py         # extract TaxInput + call tools.tax_calculator
│   │       └── review.py       # document review
│   │
│   ├── ingestion/              # LlamaIndex-based pipeline
│   │   ├── __init__.py
│   │   ├── loaders.py          # PDF -> pages (PyMuPDF), OCR fallback detection
│   │   ├── cleaning.py         # headers/footers, hyphenation, Hindi stripping
│   │   ├── metadata.py         # regexes: section, circular no., notification no., dates
│   │   ├── chunking.py         # section-aware split + SentenceSplitter + contextual header
│   │   └── pipeline.py         # ingest_document(doc_id) — orchestrates the above + embed + upsert
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── base.py             # VectorStore Protocol, ChunkRecord, SearchHit, SearchFilters
│   │   ├── factory.py          # get_vector_store() based on settings.vector_backend
│   │   ├── fusion.py           # rrf_fuse() — shared by backends that fuse in Python
│   │   ├── query_parser.py     # pulls "80C", "Circular 3/2025" out of the question
│   │   ├── pinecone_store.py   # primary backend: dense + SPLADE hybrid, namespaces
│   │   ├── pinecone_filters.py # SearchFilters -> Pinecone metadata filter dict
│   │   ├── qdrant_store.py     # extension (week 5)
│   │   └── oracle_store.py     # extension (week 5)
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── providers.py        # get_chat_model() with fallbacks
│   │   └── embeddings.py       # dense: FastEmbed bge-small (sparse SPLADE lives in the Pinecone store)
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── tax_calculator.py   # compute_tax(TaxInput) -> TaxResult (pure, no I/O)
│   │   └── tax_rules/
│   │       └── fy2025_26.yaml  # slabs, rebate, cess — data, not code
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   ├── pii.py              # PIIMasker.mask()/unmask() — Presidio + your Redact rules
│   │   ├── recognizers.py      # GSTIN, IFSC, Indian phone, your SIH Redact patterns
│   │   ├── injection.py        # heuristic + optional classifier score
│   │   ├── auth.py             # password hashing, JWT encode/decode
│   │   └── audit.py            # write_audit_event()
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py             # Repository Protocol (users, workspaces, documents, audit)
│   │   ├── models.py           # Pydantic domain models: User, Workspace, Membership, Document, AuditEvent
│   │   ├── factory.py          # get_repository() based on settings.db_backend
│   │   ├── sqlite_repo.py      # SQLAlchemy Core + SQLite (local, CI, tests)
│   │   ├── cosmos_repo.py      # azure-cosmos SDK (Azure)
│   │   ├── blob.py             # upload/download (Azurite locally, managed identity in Azure)
│   │   └── queue.py            # enqueue/receive/delete/poison
│   │
│   ├── worker/
│   │   ├── __init__.py
│   │   └── indexer.py          # python -m taxlens.worker.indexer --loop | --once
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   └── tracing.py          # Phoenix register() + OpenInference instrumentors
│   │
│   └── mcp_server/
│       ├── __init__.py
│       └── server.py           # FastMCP tools -> calls the REST API
│
├── migrations/                 # Alembic — SQLite schema only (Cosmos containers come from infra/)
│   ├── env.py
│   └── versions/
│
├── tests/
│   ├── conftest.py             # fixtures: settings override, temp SQLite repo, fake vector store, test client, fake LLM
│   ├── fixtures/
│   │   ├── sample_circular.pdf
│   │   ├── sample_notification.pdf
│   │   └── pii_samples.txt     # synthetic PAN/Aadhaar/GSTIN — never real data
│   ├── unit/
│   │   ├── test_metadata.py
│   │   ├── test_chunking.py
│   │   ├── test_fusion.py
│   │   ├── test_citations.py
│   │   ├── test_pii.py
│   │   ├── test_injection.py
│   │   └── test_tax_calculator.py
│   ├── integration/
│   │   ├── test_vector_store_contract.py   # same tests for Pinecone / Qdrant / Oracle
│   │   ├── test_repository_contract.py     # same tests for SQLite / Cosmos (emulator)
│   │   ├── test_ingest_pipeline.py
│   │   └── test_api.py
│   └── eval/
│       └── test_eval_gate.py   # marked @pytest.mark.eval, run only in the eval CI job
│
├── eval/
│   ├── golden_set.jsonl        # ~100 questions with expected answers + sources
│   ├── subsets.yaml            # which ids form the fast "ci" subset (~40)
│   ├── thresholds.toml         # gate values
│   ├── baseline.json           # last accepted results from main
│   ├── evaluators.py           # retrieval, citation, correctness (LLM judge), refusal
│   ├── run_eval.py             # runs locally or as a LangSmith experiment
│   └── check_thresholds.py     # exits non-zero if the gate fails
│
├── data/
│   ├── sources.yaml            # manifest: url, doc_type, act, sha256 for every public doc
│   ├── raw/                    # downloaded PDFs (gitignored)
│   └── eval_corpus/            # small subset used in CI (or downloaded + cached)
│
├── scripts/
│   ├── download_sources.py     # reads data/sources.yaml, verifies sha256
│   ├── init_pinecone.py        # create the index if missing (dimension 384, dotproduct)
│   ├── seed_index.py           # ingest a folder synchronously (no queue) — local + CI
│   └── create_user.py          # bootstrap an admin user + workspace
│
├── infra/
│   ├── deploy.sh               # az CLI provisioning (week 4)
│   └── main.bicep              # optional: same thing as IaC
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # lint -> test -> eval -> build -> deploy
│   │   └── nightly-eval.yml    # full 100-question eval + LangSmith experiment
│   ├── ISSUE_TEMPLATE/
│   │   ├── feature.yml
│   │   └── bug.yml
│   └── pull_request_template.md
│
└── docs/                       # you are here
```

## 2.2 Dependency rules

Imports only point down this list. Nothing imports from a layer above it.

```text
api/  ─────────────►  agent/  ─────►  retrieval/, llm/, tools/, security/
  │                                         │
  └──►  storage/, security/                 └──►  storage/
worker/  ──►  ingestion/  ──►  retrieval/, llm/, storage/, security/
mcp_server/  ──►  (HTTP only — no taxlens imports except config)
config.py, logging_config.py  ◄── everyone
```

- `api/` never talks to Pinecone, SQLite or Cosmos directly, and never builds prompts. It validates input, checks auth and calls a service.
- `agent/`, `ingestion/` and `tools/` never import FastAPI, so they can be tested without HTTP.
- `tools/tax_calculator.py` has no I/O at all, so it is easy to unit-test exhaustively.
- Only `config.py` reads environment variables.

## 2.3 Configuration pattern

```python
# taxlens/config.py
from functools import lru_cache
from typing import Literal
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["local", "ci", "prod"] = "local"
    log_level: str = "INFO"

    # LLM
    llm_provider: Literal["gemini", "groq", "ollama"] = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_fallback_provider: Literal["gemini", "groq", "ollama"] | None = "groq"
    llm_fallback_model: str = "llama-3.3-70b-versatile"
    judge_provider: Literal["gemini", "groq", "ollama"] = "groq"   # eval judge ≠ answer model
    judge_model: str = "llama-3.3-70b-versatile"
    ollama_base_url: str = "http://localhost:11434"

    # Embeddings / retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    sparse_model: str = "prithivida/Splade_PP_en_v1"
    vector_backend: Literal["pinecone", "qdrant", "oracle"] = "pinecone"
    pinecone_index: str = "taxlens-dev"                          # -ci / -prod in other envs
    hybrid_alpha: float = 0.5                                     # 1 = dense only, 0 = sparse only
    pinecone_namespace_prefix: str = ""                           # CI sets "ci-<run_id>-"
    retrieval_k: int = 8
    qdrant_url: str = "http://localhost:6333"                    # week 5
    oracle_dsn: str = "localhost:1521/FREEPDB1"                  # week 5
    oracle_user: str = "taxlens"
    oracle_password: SecretStr = SecretStr("taxlens")

    # Data
    db_backend: Literal["sqlite", "cosmos"] = "sqlite"
    sqlite_path: str = "var/taxlens.db"
    cosmos_endpoint: str | None = None                            # https://<acct>.documents.azure.com:443/
    cosmos_key: SecretStr | None = None                           # emulator only; Azure uses managed identity
    cosmos_database: str = "taxlens"
    azure_storage_connection_string: SecretStr | None = None   # local (Azurite)
    azure_storage_account_url: str | None = None               # cloud (managed identity)
    blob_container: str = "documents"
    ingest_queue: str = "ingest-jobs"
    poison_queue: str = "ingest-poison"
    max_upload_mb: int = 25

    # Auth
    jwt_secret: SecretStr = SecretStr("change-me")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Observability
    phoenix_collector_endpoint: str | None = "http://localhost:6006/v1/traces"
    langsmith_project: str = "taxlens"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

Provider API keys (`GOOGLE_API_KEY`, `GROQ_API_KEY`, `PINECONE_API_KEY`, `LANGSMITH_API_KEY`) are read by their SDKs
straight from the environment. In Azure they are injected from Key Vault.

## 2.4 Naming and code conventions

| Thing | Convention |
|-------|------------|
| Modules | `snake_case.py`, one responsibility each |
| Pydantic API models | `XxxRequest` / `XxxResponse` in `api/schemas.py` |
| Internal data | `@dataclass(frozen=True)` in the module that owns it |
| Graph nodes | plain functions `def node(state: TaxLensState) -> dict` returning only the changed keys |
| Prompts | constants in `agent/prompts.py` with a `PROMPT_VERSION` string logged on every trace |
| Tests | mirror the package path, e.g. `taxlens/ingestion/metadata.py` → `tests/unit/test_metadata.py` |
| Branches | `feat/12-hybrid-search`, `fix/31-citation-regex` (the number is the issue number) |
| Commits | Conventional Commits: `feat(retrieval): add RRF fusion` |

## 2.5 "Where does this go?" cheat sheet

| I'm writing... | Put it in |
|----------------|-----------|
| A new endpoint | `api/routes/<name>.py` + schemas in `api/schemas.py` + register in `main.py` |
| A new regex for circular numbers | `ingestion/metadata.py` + a case in `tests/unit/test_metadata.py` |
| A new prompt | `agent/prompts.py`, then bump `PROMPT_VERSION` and re-run the eval |
| A new LLM provider | one branch in `llm/providers.py` + the extra dependency |
| A new vector DB | `retrieval/<name>_store.py` implementing `VectorStore` + add it to the contract-test params |
| A new field on documents/users | `storage/models.py` + Alembic migration (SQLite) + bump `schema_version` (Cosmos) + repository contract test |
| New tax-year slabs | `tools/tax_rules/fy20XX_YY.yaml` + test cases |
| A new PII type | `security/recognizers.py` + synthetic samples in `tests/fixtures/pii_samples.txt` |
| A secret | Key Vault in the cloud, `.env` locally, a field in `config.py`. Never in code |
