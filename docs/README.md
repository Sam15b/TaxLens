# TaxLens Documentation

TaxLens is a tax and legal document assistant for Indian tax law. It answers questions about the
Income-tax Act, CBDT circulars and GST notifications, and every answer cites its sources. It also
reviews uploaded client documents. Personal data (PAN, Aadhaar, GSTIN, names, phone numbers) is
masked **before** any text reaches an LLM.

> These docs are the build guide. They describe what to build, where each piece goes in the
> folder structure, and the order to build it in over about 5 weeks.

## Read in this order

| # | Document | What it covers |
|---|----------|----------------|
| 1 | [Architecture](01-architecture.md) | System design, the replacements for the three dropped Azure services, Pinecone hybrid design, data flow, LangGraph design, Repository (SQLite / Cosmos DB) data model |
| 2 | [Project structure](02-project-structure.md) | The folder layout, what each file does, and the dependency rules between modules |
| 3 | [Local setup](03-local-setup.md) | Tooling, `pyproject.toml`, `.env`, Dockerfile, docker-compose, and getting the source PDFs |
| 4 | [Build guide](04-build-guide.md) | Week-by-week implementation with code skeletons and a definition of done for each week |
| 5 | [API reference](05-api-reference.md) | `/auth/token`, `/ingest`, `/ask`, `/review`, `/redact` request and response contracts |
| 6 | [Evaluation](06-evaluation.md) | Golden set format, metrics, LangSmith experiments, Phoenix tracing, the CI eval gate |
| 7 | [Security & compliance](07-security-compliance.md) | PII masking, workspace isolation, auth, audit logs, prompt-injection defences |
| 8 | [Deployment & CI/CD](08-deployment-cicd.md) | Azure resources, `az` CLI scripts, GitHub Actions pipeline |
| 9 | [Extensions](09-extensions.md) | Multiple vector-store backends (Pinecone, Qdrant, Oracle), the MCP server |
| 10 | [Agile workflow](10-agile-workflow.md) | GitHub Projects board, sprints, issue and PR templates |

## Tech stack at a glance

| Concern | Choice |
|---------|--------|
| Language / packaging | Python 3.12, `uv` |
| Document loading & chunking | LlamaIndex (`PyMuPDFReader`, `SentenceSplitter`), section-aware chunking |
| Embeddings | Dense: `BAAI/bge-small-en-v1.5` (384-dim). Sparse: SPLADE (`prithivida/Splade_PP_en_v1`). Both run locally with FastEmbed |
| Orchestration | LangGraph router: Q&A, document review, tax calculator |
| LLMs | **Google Gemini API** (primary), **Groq** (fallback), **Ollama** local Llama (offline/dev) |
| Retrieval | **Pinecone serverless** hybrid index (dense + sparse in one query), one namespace per workspace |
| API | FastAPI, OAuth2 password flow + JWT, OpenAPI docs at `/docs` |
| Async indexing | **Azure Storage Queue + Container Apps Job** (event-driven, scales to zero) |
| Storage | Azure Blob Storage (Azurite locally) |
| Hosting | Azure Container Apps |
| Secrets | Azure Key Vault + managed identity |
| App data (users, workspaces, documents, audit) | **SQLite** locally and in CI, **Azure Cosmos DB for NoSQL** in the cloud, behind one `Repository` interface |
| Evaluation | Golden set of about 100 questions, LangSmith experiments, CI gate |
| Tracing | Arize Phoenix (OpenTelemetry / OpenInference) |
| PII | Microsoft Presidio, plus your Redact rules ported in as custom recognizers |
| Tests / CI | pytest, ruff, mypy, GitHub Actions: lint → test → eval → build → deploy |
| Extensions | Qdrant and Oracle Database Free vector backends, MCP server |

## What changed from the original plan

The original plan used three Azure services that this version leaves out. Their replacements are
explained in detail in [Architecture § 1.3](01-architecture.md#13-replacing-the-three-azure-services).

| Removed | Replacement | One-line reason |
|---------|-------------|-----------------|
| Azure OpenAI | Gemini API + Groq + Ollama behind one provider factory | Free tiers for development, no approval process, and switching provider is a config change |
| Azure AI Search (hybrid) | Pinecone serverless (dense + sparse hybrid, namespaces per client) | Managed vector DB with a free Starter plan; namespaces give clean tenant isolation; hybrid weighting is tunable and measurable |
| Azure Function (index on upload) | Storage Queue → Container Apps Job (KEDA queue trigger) | Same Docker image as the API, retries and a poison queue, scales to zero, no Functions packaging problems with heavy ML dependencies |

The remaining Azure services are Blob Storage, Storage Queue, Event Grid, Container Apps (app + job),
Key Vault and Azure Cosmos DB. PostgreSQL is not used anywhere.

| Store | Holds | Local / CI | Cloud |
|-------|-------|-----------|-------|
| Pinecone | Chunk vectors (dense + sparse) + chunk metadata and text | Pinecone Starter project (`taxlens-dev` index) | Pinecone (`taxlens-prod` index) |
| Repository | Users, workspaces, memberships, documents, audit log | SQLite file (`var/taxlens.db`) | Azure Cosmos DB for NoSQL (free tier) |
| Blob Storage | Original uploaded files | Azurite | Azure Blob Storage |
