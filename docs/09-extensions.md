# 9. Extensions (week 5)

## 9.1 One `VectorStore` interface, three backends

| Backend | Hybrid strategy | Why it's worth having | Run locally |
|---------|-----------------|-----------------------|-------------|
| **Pinecone** (primary) | Dense + SPLADE sparse in one dotproduct index, `alpha` weighting, section boost with RRF | Managed, namespaces per tenant | Cloud only (Starter plan); tests use their own namespace |
| **Qdrant** | Dense + sparse BM25 vectors, server-side RRF via the Query API | Purpose-built vector DB, free cloud tier, shows a second design | `docker compose --profile qdrant up -d` |
| **Oracle Database Free (AI Vector Search)** | `VECTOR_DISTANCE` + Oracle Text `CONTAINS`, RRF in Python | Relevant for Oracle interviews; enterprise DB with native vectors | `docker compose --profile oracle up -d` |

The factory picks one from config:

```python
# taxlens/retrieval/factory.py
from functools import lru_cache
from taxlens.config import settings
from taxlens.retrieval.base import VectorStore

@lru_cache
def get_vector_store() -> VectorStore:
    match settings.vector_backend:
        case "pinecone":
            from taxlens.retrieval.pinecone_store import PineconeStore
            return PineconeStore(settings.pinecone_index, alpha=settings.hybrid_alpha)
        case "qdrant":
            from taxlens.retrieval.qdrant_store import QdrantStore
            return QdrantStore(settings.qdrant_url)
        case "oracle":
            from taxlens.retrieval.oracle_store import OracleStore
            return OracleStore(settings.oracle_dsn, settings.oracle_user,
                               settings.oracle_password.get_secret_value())
    raise ValueError(settings.vector_backend)
```

### Contract tests: the same tests for every backend

```python
# tests/integration/test_vector_store_contract.py
import pytest
from taxlens.retrieval.base import SearchFilters

BACKENDS = ["pinecone", "qdrant", "oracle"]

@pytest.fixture(params=BACKENDS)
def store(request, backend_factory):          # backend_factory skips if the service isn't up
    return backend_factory(request.param)

def test_workspace_isolation(store, make_chunk):
    store.upsert([make_chunk(workspace="A", text="Section 80C deduction limit")])
    hits = store.hybrid_search(query_text="80C", query_vector=vec("80C"), workspace_ids=["B"])
    assert hits == []

def test_keyword_beats_vector_on_section_ids(store, make_chunk):
    store.upsert([make_chunk(workspace="P", text="194J fees for professional services", section="194J"),
                  make_chunk(workspace="P", text="payments to contractors", section="194C")])
    hits = store.hybrid_search(query_text="194J", query_vector=vec("194J"), workspace_ids=["P"], k=1)
    assert hits[0].metadata["section"] == "194J"

def test_suspicious_chunks_excluded_by_default(store, make_chunk): ...
def test_delete_document_removes_all_chunks(store, make_chunk): ...
def test_upsert_is_idempotent(store, make_chunk): ...
def test_filters_doc_type_and_issued_after(store, make_chunk): ...
```

This gives you a concrete interview story: "I defined the contract once, and three very different
databases pass the same tests."

### Qdrant sketch

```python
# taxlens/retrieval/qdrant_store.py (key parts)
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

class QdrantStore:
    COLLECTION = "chunks"

    def __init__(self, url: str):
        self.client = QdrantClient(url=url)
        self.bm25 = SparseTextEmbedding("Qdrant/bm25")
        if not self.client.collection_exists(self.COLLECTION):
            self.client.create_collection(
                self.COLLECTION,
                vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            self.client.create_payload_index(self.COLLECTION, "workspace_id",
                                             models.PayloadSchemaType.KEYWORD)

    def hybrid_search(self, *, query_text, query_vector, workspace_ids, k=8, filters=None):
        ws = models.Filter(must=[models.FieldCondition(
            key="workspace_id", match=models.MatchAny(any=list(workspace_ids)))])
        sparse = next(self.bm25.query_embed(query_text))
        res = self.client.query_points(
            self.COLLECTION,
            prefetch=[
                models.Prefetch(query=query_vector, using="dense", limit=40, filter=ws),
                models.Prefetch(query=models.SparseVector(indices=sparse.indices.tolist(),
                                                          values=sparse.values.tolist()),
                                using="bm25", limit=40, filter=ws),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k, with_payload=True,
        )
        return [to_hit(p) for p in res.points]
```

Add `qdrant-client` with `uv add qdrant-client`. Qdrant point ids must be UUIDs or integers, but
chunk ids are `{document_id}#{chunk_no}` (the Pinecone format). Use `uuid5(NAMESPACE_URL, chunk_id)`
as the point id and keep the original `chunk_id` in the payload.

### Oracle sketch

```sql
-- run once as the app user
CREATE TABLE chunks (
  id            VARCHAR2(36) PRIMARY KEY,
  workspace_id  VARCHAR2(36) NOT NULL,
  document_id   VARCHAR2(36) NOT NULL,
  content       CLOB NOT NULL,
  embedding     VECTOR(384, FLOAT32) NOT NULL,
  doc_type      VARCHAR2(30),
  act           VARCHAR2(100),
  section       VARCHAR2(30),
  circular_no   VARCHAR2(30),
  issued_on     DATE,
  page          NUMBER,
  suspicious    NUMBER(1) DEFAULT 0
);
CREATE VECTOR INDEX chunks_vec_idx ON chunks (embedding)
  ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE;          -- IVF; HNSW needs VECTOR_MEMORY_SIZE
CREATE INDEX chunks_txt_idx ON chunks (content) INDEXTYPE IS CTXSYS.CONTEXT
  PARAMETERS ('SYNC (ON COMMIT)');
```

```python
# taxlens/retrieval/oracle_store.py (key parts)
import array
import oracledb
from taxlens.retrieval.fusion import rrf_fuse

SEMANTIC = """
SELECT id FROM chunks
WHERE workspace_id IN (SELECT column_value FROM TABLE(:ws)) AND suspicious = 0
ORDER BY VECTOR_DISTANCE(embedding, :qvec, COSINE)
FETCH APPROX FIRST 40 ROWS ONLY"""

KEYWORD = """
SELECT id FROM chunks
WHERE workspace_id IN (SELECT column_value FROM TABLE(:ws)) AND suspicious = 0
  AND CONTAINS(content, :q, 1) > 0
ORDER BY SCORE(1) DESC
FETCH FIRST 40 ROWS ONLY"""

class OracleStore:
    def __init__(self, dsn, user, password):
        self.pool = oracledb.create_pool(user=user, password=password, dsn=dsn, min=1, max=4)

    def hybrid_search(self, *, query_text, query_vector, workspace_ids, k=8, filters=None):
        qvec = array.array("f", query_vector)            # python-oracledb binds this to VECTOR
        with self.pool.acquire() as conn, conn.cursor() as cur:
            ws = conn.gettype("SYS.ODCIVARCHAR2LIST").newobject(list(workspace_ids))
            sem = [r[0] for r in cur.execute(SEMANTIC, ws=ws, qvec=qvec)]
            kw = [r[0] for r in cur.execute(KEYWORD, ws=ws, q=to_oracle_text_query(query_text))]
            fused = rrf_fuse(sem, kw)[:k]
            return self._load_hits(cur, fused)
```

`to_oracle_text_query` must escape Oracle Text operators. Wrap each term in `{}` and join them
with `ACCUM`. Newer Oracle releases (23ai / 26ai) also offer a built-in **hybrid vector index**
(`DBMS_HYBRID_VECTOR`). Compare it with your Python RRF in the benchmark. Add the driver with
`uv add oracledb` (thin mode, so no Oracle client install is needed).

### Benchmark table for the README

| Backend | hit@5 | citation precision | answer correctness | p50 search ms | p95 search ms | index time (100 docs) |
|---------|------:|-------------------:|-------------------:|--------------:|--------------:|----------------------:|
| Pinecone | | | | | | |
| Qdrant | | | | | | |
| Oracle | | | | | | |

Fill it from `uv run python -m eval.run_eval --subset all` with `VECTOR_BACKEND` switched each run.
Any bug you find while building these backends (in a client library, LlamaIndex or LangChain
integration) is a good candidate for an upstream issue or PR.

## 9.2 Alternative LLMs

Gemini, Groq-Llama and Ollama-Llama are already wired up from week 1. For the extension, run
**E6** from [Evaluation § 6.4](06-evaluation.md) and publish the quality/latency/cost table.
Optionally add one more provider branch in `llm/providers.py` (OpenAI or Anthropic Claude) to show
that the factory really is swappable.

## 9.3 MCP server

Expose TaxLens to any MCP client, including your Hermes agent.

**Design rule:** the MCP server is a **thin client of the REST API**. It authenticates with a
service account JWT, so auth, workspace checks, PII masking and audit logging all still apply.
There is no second path into the data.

```python
# taxlens/mcp_server/server.py
import os
import httpx
from mcp.server.fastmcp import FastMCP

API = os.environ.get("TAXLENS_API_URL", "http://localhost:8000")
TOKEN = os.environ["TAXLENS_SERVICE_TOKEN"]
WORKSPACE = os.environ["TAXLENS_WORKSPACE_ID"]

mcp = FastMCP("taxlens")

def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=API, timeout=60, headers={
        "Authorization": f"Bearer {TOKEN}", "X-Workspace-Id": WORKSPACE})

@mcp.tool()
async def ask_tax_question(question: str, act: str | None = None) -> dict:
    """Answer a question about Indian tax law (Income-tax Acts 1961/2025, CBDT circulars,
    GST notifications). Returns an answer with citations [S1].. and their sources."""
    body = {"question": question, "filters": {"act": act} if act else None}
    async with _client() as c:
        r = await c.post("/ask", json=body)
        r.raise_for_status()
        return r.json()

@mcp.tool()
async def redact_pii(text: str) -> dict:
    """Mask Indian PII (PAN, Aadhaar, GSTIN, phone, names) in text. No LLM involved."""
    async with _client() as c:
        r = await c.post("/redact", json={"text": text})
        r.raise_for_status()
        return r.json()

@mcp.tool()
async def compute_income_tax(gross_income: float, regime: str = "new",
                             tax_year: str = "2025-26", is_salaried: bool = True) -> dict:
    """Compute Indian income tax deterministically from versioned slab rules."""
    q = f"Compute tax: gross income {gross_income}, {regime} regime, tax year {tax_year}, " \
        f"{'salaried' if is_salaried else 'not salaried'}"
    async with _client() as c:
        r = await c.post("/ask", json={"question": q})
        r.raise_for_status()
        return r.json()

@mcp.tool()
async def review_document(document_id: str, focus: str | None = None) -> dict:
    """Review an already-ingested client document and return findings with citations."""
    async with _client() as c:
        r = await c.post("/review", json={"document_id": document_id, "focus": focus})
        r.raise_for_status()
        return r.json()

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio")      # stdio | streamable-http
    mcp.run(transport=transport)
```

`compute_income_tax` currently goes through `/ask` routing. A cleaner option is a dedicated
`POST /calculate` endpoint that takes `TaxInput` directly; add it if Hermes uses the tool often.

Running it:

```powershell
# local, stdio (most MCP clients launch it as a subprocess)
uv run python -m taxlens.mcp_server.server

# inspect / debug the tools interactively
uv run mcp dev taxlens/mcp_server/server.py
```

Generic MCP client config (adapt it to your Hermes agent's config format):

```json
{
  "mcpServers": {
    "taxlens": {
      "command": "uv",
      "args": ["--directory", "E:/TaxLens", "run", "python", "-m", "taxlens.mcp_server.server"],
      "env": {
        "TAXLENS_API_URL": "http://localhost:8000",
        "TAXLENS_SERVICE_TOKEN": "<token from scripts/create_user.py --service>",
        "TAXLENS_WORKSPACE_ID": "<workspace uuid>"
      }
    }
  }
}
```

For remote use, run it with `MCP_TRANSPORT=streamable-http` as another Container App behind
authentication. Never expose an unauthenticated MCP endpoint.

Tests: unit-test each tool with `httpx.MockTransport`. Then add one end-to-end test that starts the
API (TestClient) and calls the tools through an MCP client session.
