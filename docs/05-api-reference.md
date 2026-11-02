# 5. API reference

FastAPI generates the OpenAPI spec automatically. In a running app it is at `/docs` (Swagger UI),
`/redoc` and `/openapi.json`. This page is the design contract to build against.

## 5.1 Conventions

| Item | Rule |
|------|------|
| Auth | `Authorization: Bearer <JWT>` on everything except `/health`, `/ready`, `/auth/token` |
| Workspace | `X-Workspace-Id: <uuid>` header. The caller must be a member. The public corpus is always searched in addition |
| Scopes | `ask`, `review`, `ingest`, `redact`, `admin` |
| IDs | UUIDv4 strings |
| Errors | `{"error": {"code": "string", "message": "string", "request_id": "string"}}` |
| Tracing | Every response has an `X-Request-Id` header, and AI responses include `trace_id` |
| Disclaimer | AI answers include `"disclaimer": "Informational only. Not legal or tax advice."` |

| Status | When |
|--------|------|
| 400 | Validation passed but the request makes no sense (e.g. an empty PDF) |
| 401 | Missing, invalid or expired token |
| 403 | Missing scope, or not a member of the workspace |
| 413 | Upload larger than `MAX_UPLOAD_MB` |
| 415 | Not a PDF (checked by magic bytes, not just the extension) |
| 422 | Pydantic validation error |
| 429 | Rate limit (optional, via `slowapi`) |
| 503 | LLM provider and fallback both unavailable |

## 5.2 `POST /auth/token`

OAuth2 password flow (form-encoded), so the Swagger "Authorize" button works.

```http
POST /auth/token
Content-Type: application/x-www-form-urlencoded

username=analyst@example.com&password=...&scope=ask review ingest
```

```json
{ "access_token": "eyJhbGciOi...", "token_type": "bearer", "expires_in": 3600 }
```

JWT claims: `sub` (user id), `scopes`, `ws` (list of workspace ids), `exp`, `iat`, `jti`.

## 5.3 `POST /ingest`. Scope: `ingest`

Upload a document. Indexing happens asynchronously.

```http
POST /ingest
X-Workspace-Id: 7d0c...
Content-Type: multipart/form-data

file=@notice_143_1.pdf
doc_type=client_doc            # act | rules | circular | notification | client_doc
title=AY 2024-25 intimation    # optional
```

`202 Accepted`

```json
{ "document_id": "b1f2...", "status": "queued", "duplicate": false }
```

Uploading the same file to the same workspace again returns the existing `document_id` with
`"duplicate": true`, because documents are deduplicated by sha256.

### `GET /documents/{document_id}`

```json
{
  "document_id": "b1f2...",
  "status": "indexed",            // queued | processing | indexed | failed | needs_ocr
  "doc_type": "client_doc",
  "chunk_count": 37,
  "pii_entities_masked": 12,
  "suspicious_chunks": 0,
  "error": null,
  "created_at": "2026-11-02T09:15:00Z"
}
```

## 5.4 `POST /ask`. Scope: `ask`

```json
{
  "question": "What is the TDS rate on professional fees and from what threshold?",
  "filters": {
    "doc_types": ["act", "circular"],
    "act": "Income-tax Act, 2025",
    "issued_after": "2024-04-01"
  },
  "top_k": 8
}
```

`200 OK`

```json
{
  "answer": "Under section 194J of the Income-tax Act, 1961, ... [S1] ... [S3]",
  "route": "qa",
  "citations": [
    {
      "ref": "S1",
      "chunk_id": "5a9e...",
      "title": "Income-tax Act, 1961 — s. 194J",
      "doc_type": "act",
      "section": "194J",
      "page": 412,
      "issued_on": null,
      "source_url": "https://..."
    }
  ],
  "found": true,
  "disclaimer": "Informational only. Not legal or tax advice.",
  "trace_id": "a3c1...",
  "prompt_version": "qa-v1",
  "model": "gemini-2.5-flash"
}
```

When nothing relevant is found: `"found": false`, the answer is the fixed NOT_FOUND string, and
`citations` is `[]`.

For `route: "calc"` the response includes a `calculation` object (the `TaxResult` from the
calculator), so the client can show the slab breakdown.

## 5.5 `POST /review`. Scope: `review`

Review a client document that was already ingested, or upload one inline.

```json
{
  "document_id": "b1f2...",
  "focus": "Check whether the TDS deducted matches the applicable rate"   // optional
}
```

`200 OK`

```json
{
  "document_type": "tds_certificate",
  "summary": "Form 16A issued by <ORG_1> for <PERSON_1> (PAN <IN_PAN_1>) ...",
  "findings": [
    {
      "id": "F1",
      "severity": "high",                       // high | medium | low | info
      "title": "TDS deducted at 2% where 10% appears applicable",
      "explanation": "The payment is described as professional fees ... [S2]",
      "citations": ["S2"],
      "evidence": "page 1: 'Nature of payment: Professional fees'"
    }
  ],
  "citations": [ { "ref": "S2", "title": "Income-tax Act, 1961 — s. 194J", "...": "..." } ],
  "pii_masked": true,
  "disclaimer": "Informational only. Not legal or tax advice.",
  "trace_id": "..."
}
```

The review output stays **masked**: placeholders are not reversed. That makes it safe to share
or store. A user with the `admin` scope can request `?unmask=true`, and that request is logged in
the audit log.

## 5.6 `POST /redact`. Scope: `redact`

Masks text without any LLM call. It is useful on its own and makes PII handling easy to demo.

```json
{ "text": "Mr Ravi Kumar, PAN ABCPK1234F, Aadhaar 2345 6789 0123, ph +91 98xxxxxx10", "mode": "placeholder" }
```

`mode`: `placeholder` (`<IN_PAN_1>`), `mask` (`ABCPK****F`), or `remove`.

```json
{
  "redacted_text": "Mr <PERSON_1>, PAN <IN_PAN_1>, Aadhaar <IN_AADHAAR_1>, ph <PHONE_NUMBER_1>",
  "entities": [
    { "type": "PERSON", "start": 3, "end": 13, "score": 0.85 },
    { "type": "IN_PAN", "start": 19, "end": 29, "score": 0.95 }
  ]
}
```

The response **never** contains the original values. The service doesn't store the request text,
and the audit log keeps only a hash of it.

## 5.7 Health

| Endpoint | Returns |
|----------|---------|
| `GET /health` | `{"status": "ok"}`. The process is alive (liveness probe) |
| `GET /ready` | Checks the repository (SQLite / Cosmos), Pinecone (`describe_index_stats`), Blob, and that the dense + sparse models are loaded; 503 if not (readiness probe) |

## 5.8 Pydantic schemas (starting point)

```python
# taxlens/api/schemas.py
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field

class AskFilters(BaseModel):
    doc_types: list[Literal["act", "rules", "circular", "notification", "client_doc"]] | None = None
    act: str | None = None
    issued_after: date | None = None

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    filters: AskFilters | None = None
    top_k: int = Field(default=8, ge=1, le=20)

class CitationOut(BaseModel):
    ref: str
    chunk_id: str
    title: str
    doc_type: str
    section: str | None = None
    page: int | None = None
    issued_on: date | None = None
    source_url: str | None = None

class AskResponse(BaseModel):
    answer: str
    route: Literal["qa", "calc", "refuse"]
    citations: list[CitationOut]
    found: bool
    disclaimer: str = "Informational only. Not legal or tax advice."
    trace_id: str | None = None
    prompt_version: str
    model: str
    calculation: dict | None = None

class RedactRequest(BaseModel):
    text: str = Field(max_length=100_000)
    mode: Literal["placeholder", "mask", "remove"] = "placeholder"
```
