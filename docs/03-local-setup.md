# 3. Local setup

## 3.1 Prerequisites

| Tool | Why | Install (Windows) |
|------|-----|-------------------|
| `uv` | Python versions, virtualenv, dependencies, lockfile | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| Python 3.12 | Widest wheel support for spaCy, ONNX Runtime and PyMuPDF | `uv python install 3.12` (your system 3.14 can stay) |
| Docker Desktop | Azurite (Blob + Queue emulator), Phoenix, optional Cosmos emulator | docker.com |
| Pinecone account | Vector database (free Starter plan). Create an API key in the console | pinecone.io |
| Git + GitHub CLI | Repo, PRs, Projects board | `winget install Git.Git GitHub.cli` |
| Azure CLI | Week 4 only | `winget install Microsoft.AzureCLI` |
| Ollama (optional) | Local Llama | ollama.com, then `ollama pull llama3.1:8b` |

> Python 3.14 is new, and some ML libraries may not have wheels for it yet. Pin 3.12 for this
> project with `.python-version`. `uv` downloads it automatically and leaves your system Python alone.

## 3.2 Create the project

```powershell
cd E:\TaxLens
git init
uv init --app --python 3.12 --name taxlens   # basic layout, no src/, no build backend
Remove-Item main.py                          # uv's hello-world; we use taxlens/main.py
mkdir taxlens, tests, eval, data, scripts, infra, migrations
```

### Dependencies

The full list is already in `pyproject.toml` / `uv.lock`, with a pip-friendly copy in
`requirements.txt` and `requirements-dev.txt`. Run `uv sync` (or
`python -m pip install -r requirements-dev.txt`). The commands below show how each group was added;
use the same pattern (`uv add <pkg>`, then update `requirements.txt`) for new packages.

```powershell
# API + data
uv add fastapi "uvicorn[standard]" pydantic-settings python-multipart sqlalchemy alembic pyyaml httpx

# Vector DB + cloud app database
uv add pinecone azure-cosmos

# Ingestion (LlamaIndex) + embeddings
uv add llama-index-core llama-index-readers-file llama-index-embeddings-fastembed pymupdf fastembed

# Orchestration + LLM providers
uv add langgraph langchain-core langchain-google-genai langchain-groq langchain-ollama

# Security
uv add pyjwt "pwdlib[argon2]" presidio-analyzer presidio-anonymizer spacy
uv add "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# Azure (Blob, Queue, Key Vault, managed identity)
uv add azure-storage-blob azure-storage-queue azure-identity azure-keyvault-secrets

# Observability + eval
uv add arize-phoenix-otel openinference-instrumentation-langchain openinference-instrumentation-llama-index langsmith

# MCP
uv add mcp

# Dev
uv add --dev pytest pytest-asyncio pytest-cov ruff mypy types-PyYAML
```

> The spaCy model is installed as a URL dependency so it lands in `uv.lock` and in the Docker
> image. `python -m spacy download` would need `pip`, which `uv` virtualenvs don't include. Check
> the spaCy models release page for the version that matches your installed spaCy.

### Tool config in `pyproject.toml`

Add these sections under what `uv` generated:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "S", "ASYNC"]   # S = bandit security rules
ignore = ["S101"]                                     # allow assert in tests

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
strict_optional = true

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = [
  "integration: needs Azurite and a Pinecone API key (uses a throwaway namespace)",
  "cosmos: needs the Cosmos DB emulator (docker compose --profile cosmos up -d)",
  "eval: calls real LLMs; run only in the eval job",
]
addopts = "-m 'not eval'"
```

## 3.3 `.env.example`

Copy this to `.env` and fill in the keys. Commit `.env.example` only.

```dotenv
APP_ENV=local
LOG_LEVEL=INFO

# ── LLM ─────────────────────────────────────────────
LLM_PROVIDER=gemini                 # gemini | groq | ollama
LLM_MODEL=gemini-2.5-flash
LLM_FALLBACK_PROVIDER=groq
LLM_FALLBACK_MODEL=llama-3.3-70b-versatile
JUDGE_PROVIDER=groq
JUDGE_MODEL=llama-3.3-70b-versatile
GOOGLE_API_KEY=
GROQ_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434

# ── Embeddings / retrieval ──────────────────────────
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=384
SPARSE_MODEL=prithivida/Splade_PP_en_v1
VECTOR_BACKEND=pinecone             # pinecone | qdrant | oracle
PINECONE_API_KEY=
PINECONE_INDEX=taxlens-dev
HYBRID_ALPHA=0.5                    # 1 = dense only, 0 = sparse only
FASTEMBED_CACHE_PATH=./.models

# ── Data ────────────────────────────────────────────
DB_BACKEND=sqlite                   # sqlite | cosmos
SQLITE_PATH=var/taxlens.db
COSMOS_ENDPOINT=                    # cloud: https://<account>.documents.azure.com:443/
COSMOS_DATABASE=taxlens
# Azurite well-known dev account (public, safe to commit — local only)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://localhost:10000/devstoreaccount1;QueueEndpoint=http://localhost:10001/devstoreaccount1;
BLOB_CONTAINER=documents
INGEST_QUEUE=ingest-jobs
POISON_QUEUE=ingest-poison

# ── Auth ────────────────────────────────────────────
# generate: uv run python -c "import secrets; print(secrets.token_urlsafe(48))"
JWT_SECRET=change-me
JWT_EXPIRE_MINUTES=60

# ── Observability ───────────────────────────────────
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=taxlens
```

`.gitignore` must include `.env`, `.venv/`, `.models/`, `var/`, `data/raw/`, `__pycache__/`,
`.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` and `eval/results*.json`.

## 3.4 `Dockerfile`

One image runs three ways: the API (default `CMD`), the worker and the MCP server.

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv   # pin a version tag in practice

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/.models \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) dependencies (cached layer — only rebuilt when the lockfile changes)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# 2) bake the embedding model into the image so containers never download at startup
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('prithivida/Splade_PP_en_v1')"

# 3) application code
COPY taxlens ./taxlens
COPY migrations ./migrations
COPY alembic.ini ./

RUN mkdir -p /app/var && useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "taxlens.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore`: `.git`, `.venv`, `.env`, `var`, `data/raw`, `tests`, `eval`, `docs`, `.models`.

## 3.5 `docker-compose.yml`

```yaml
x-app-env: &app-env
  DB_BACKEND: sqlite
  SQLITE_PATH: /app/var/taxlens.db
  AZURE_STORAGE_CONNECTION_STRING: "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://azurite:10000/devstoreaccount1;QueueEndpoint=http://azurite:10001/devstoreaccount1;"
  PHOENIX_COLLECTOR_ENDPOINT: http://phoenix:6006/v1/traces
  OLLAMA_BASE_URL: http://ollama:11434
  # PINECONE_API_KEY / PINECONE_INDEX come from .env (Pinecone is a cloud service, nothing to run locally)

services:
  azurite:
    image: mcr.microsoft.com/azure-storage/azurite
    command: azurite --blobHost 0.0.0.0 --queueHost 0.0.0.0 --skipApiVersionCheck --loose
    ports: ["10000:10000", "10001:10001"]

  phoenix:
    image: arizephoenix/phoenix:latest
    ports: ["6006:6006", "4317:4317"]

  api:
    build: .
    env_file: .env
    environment: *app-env
    ports: ["8000:8000"]
    volumes: [appdata:/app/var]          # SQLite file shared with the worker
    depends_on:
      azurite: { condition: service_started }
      phoenix: { condition: service_started }
    command: >
      sh -c "alembic upgrade head &&
             uvicorn taxlens.main:app --host 0.0.0.0 --port 8000"

  worker:
    build: .
    env_file: .env
    environment: *app-env
    volumes: [appdata:/app/var]
    depends_on:
      api: { condition: service_started }   # api runs the migrations first
    command: python -m taxlens.worker.indexer --loop

  # ── optional profiles ─────────────────────────────
  cosmos:                                # test the Cosmos repository locally
    image: mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:vnext-preview
    profiles: ["cosmos"]
    ports: ["8081:8081", "1234:1234"]    # 1234 = Data Explorer UI

  ollama:
    image: ollama/ollama
    profiles: ["local-llm"]
    ports: ["11434:11434"]
    volumes: [ollama:/root/.ollama]

  qdrant:
    image: qdrant/qdrant
    profiles: ["qdrant"]
    ports: ["6333:6333"]

  oracle:
    image: gvenzl/oracle-free:slim       # community image of Oracle Database Free
    profiles: ["oracle"]
    environment:
      ORACLE_PASSWORD: taxlens
      APP_USER: taxlens
      APP_USER_PASSWORD: taxlens
    ports: ["1521:1521"]

volumes:
  appdata: {}
  ollama: {}
```

Why SQLite works with two processes: the repository turns on WAL mode and a busy timeout for every
connection, so the API and the worker can write to the same file safely. In Azure, both talk to
Cosmos DB instead, so this only matters locally.

To run against the **Cosmos emulator** instead of SQLite:
`docker compose --profile cosmos up -d cosmos`, then set `DB_BACKEND=cosmos`,
`COSMOS_ENDPOINT=http://localhost:8081/` and `COSMOS_KEY` to the emulator's well-known key (from the
emulator docs), and run `uv run python scripts/init_cosmos.py`. Check the emulator docs for whether
your image version serves HTTP or HTTPS by default.

## 3.6 Daily commands

```powershell
docker compose up -d azurite phoenix             # infra only; run the API from your IDE
uv run python scripts/init_pinecone.py           # once: creates the taxlens-dev index
uv run alembic upgrade head                      # creates var/taxlens.db
uv run python scripts/create_user.py --email you@example.com --admin
uv run python scripts/download_sources.py         # public corpus -> data/raw
uv run python scripts/seed_index.py data/raw      # synchronous ingest (no queue)
uv run uvicorn taxlens.main:app --reload          # http://localhost:8000/docs
uv run python -m taxlens.worker.indexer --loop    # in a second terminal

uv run ruff check . ; uv run ruff format .
uv run mypy taxlens
uv run pytest                                     # unit + integration, no LLM calls
uv run pytest -m eval                             # real eval (costs API calls)

docker compose up --build                         # everything in containers
docker compose --profile local-llm up -d ollama   # local Llama
```

Phoenix UI: <http://localhost:6006>. Swagger: <http://localhost:8000/docs>.

## 3.7 Getting the source documents

Pick a curated set of about 60–120 documents. Don't crawl. List each one in `data/sources.yaml`
so ingestion is reproducible and every citation can link back to its source.

```yaml
# data/sources.yaml
- id: ita-2025
  title: Income-tax Act, 2025
  doc_type: act
  act: Income-tax Act, 2025
  url: <link from incometaxindia.gov.in or indiacode.nic.in>
  sha256: <filled in by download_sources.py --update-hashes>
- id: ita-1961
  title: Income-tax Act, 1961 (as amended)
  doc_type: act
  act: Income-tax Act, 1961
  url: ...
- id: cbdt-circular-2025-03
  title: CBDT Circular No. 3/2025
  doc_type: circular
  circular_no: 3/2025
  issued_on: 2025-xx-xx
  url: ...
- id: cgst-notif-12-2017-rate
  title: Notification No. 12/2017-Central Tax (Rate)
  doc_type: notification
  notification_no: 12/2017-Central Tax (Rate)
  url: ...
```

Where to find them:

| Source | What |
|--------|------|
| incometaxindia.gov.in | Income-tax Acts, Rules, CBDT circulars and notifications |
| indiacode.nic.in | Official Act texts |
| cbic-gst.gov.in / cbic.gov.in | CGST/IGST Acts, Central Tax and Central Tax (Rate) notifications, GST circulars |
| egazette.gov.in | Gazette copies (often bilingual) |

Suggested mix: both Income-tax Acts, Income-tax Rules, 30–40 CBDT circulars (TDS, 80C/80D, capital
gains, the new regime), CGST Act, and 20–30 GST notifications (rates, e-invoicing thresholds, RCM).
These are public government documents, so it's fine to cite and quote them. Still keep only
what you need.
