# TaxLens

**An AI tax & legal document assistant for Indian tax law.** It answers questions about the
Income-tax Acts (1961 and 2025), CBDT circulars and GST notifications, and every answer cites its
sources. It reviews uploaded client documents, and it masks personal data (PAN, Aadhaar, GSTIN,
names, phone numbers) **before** any text reaches an LLM.

> **Status:** 🚧 Sprint 0. The project skeleton and design docs are in place, and implementation
> starts with [Week 1](docs/04-build-guide.md#week-1--ingestion-hybrid-retrieval-ask).

## Why it's interesting

- **Citations are enforced, not just requested.** Every `[S1]` reference is checked against the
  retrieved sources. An answer whose citations don't check out is never returned as a sourced answer.
- **An eval gate in CI.** A golden set of about 100 questions runs on every PR, and the PR fails if
  retrieval or answer accuracy drops below a threshold.
- **PII masking first.** Presidio plus custom Indian recognizers mask PII before any LLM call and
  before anything is traced.
- **Hybrid retrieval.** Pinecone combines dense (BGE) and sparse (SPLADE) vectors, with
  a boost for exact section matches such as `80C`, `194J` and `Circular 3/2025`.
- **Multi-tenant.** Each client workspace gets its own Pinecone namespace, and every repository
  call is scoped to a workspace. There is also an append-only audit log.

## Stack (free tiers to start)

| Concern | Choice |
|---------|--------|
| API | FastAPI, OAuth2 + JWT, OpenAPI docs |
| Orchestration | LangGraph router: Q&A / document review / tax calculator |
| Ingestion | LlamaIndex, PyMuPDF, section-aware chunking |
| LLMs | Gemini API (primary), Groq (fallback), Ollama (local Llama) |
| Embeddings | FastEmbed, run locally: `bge-small-en-v1.5` (dense) + SPLADE (sparse) |
| Vector DB | Pinecone serverless (Starter plan), one namespace per workspace |
| App data | SQLite (local/CI) and Azure Cosmos DB (free tier), behind a `Repository` interface |
| Async indexing | Azure Storage Queue → Container Apps Job |
| Hosting / secrets | Azure Container Apps, Key Vault, managed identity |
| Evals / tracing | Golden set + LangSmith experiments, Arize Phoenix |
| CI/CD | GitHub Actions: lint → test → eval gate → Docker build → deploy |

## Quick start

```powershell
# prerequisites: uv (https://docs.astral.sh/uv/), Docker Desktop
uv sync                                   # creates .venv with Python 3.12 and installs deps
Copy-Item .env.example .env               # then fill in your free-tier API keys
uv run pytest                             # sanity check
uv run uvicorn taxlens.main:app --reload  # http://localhost:8000/docs
```

Full setup: [docs/03-local-setup.md](docs/03-local-setup.md).

## Project layout

```text
taxlens/        application package (api, agent, ingestion, retrieval, llm, tools, security, storage, worker, ...)
tests/          unit / integration / eval tests
eval/           golden set, evaluators, thresholds for the CI gate
data/           source manifest (PDFs are downloaded, not committed)
scripts/        download, seed, init Pinecone, create user
infra/          Azure provisioning
docs/           architecture and build guide
```

Details: [docs/02-project-structure.md](docs/02-project-structure.md).

## Documentation

| | |
|---|---|
| [Architecture](docs/01-architecture.md) | System design, Pinecone hybrid search, data model, decisions |
| [Build guide](docs/04-build-guide.md) | 5-week plan with code skeletons |
| [API reference](docs/05-api-reference.md) | `/ask`, `/review`, `/redact`, `/ingest` contracts |
| [Evaluation](docs/06-evaluation.md) | Golden set, metrics, CI eval gate |
| [Security & compliance](docs/07-security-compliance.md) | PII masking, isolation, audit, prompt injection |
| [Deployment & CI/CD](docs/08-deployment-cicd.md) | Azure (free tiers) + GitHub Actions |

## Eval results

_Coming in week 2._

## Data handling

- The public corpus is Government of India publications, cited with source URLs.
- Client documents are masked before they are chunked, embedded or sent to an LLM.
- Free tiers of LLM providers may retain prompts. Use paid tiers or the local Ollama model for
  real client data. On the Pinecone Starter plan, data is hosted in the US.
- Outputs are informational and **not tax or legal advice**.
