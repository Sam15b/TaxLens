# 7. Security and compliance

## 7.1 Threat model

| # | Threat | Where | Control |
|---|--------|-------|---------|
| T1 | Client PII sent to a third-party LLM | `/ask`, `/review`, ingestion | Mask with Presidio + Redact rules before any LLM call (§ 7.2) |
| T2 | PII in logs or traces | logging, Phoenix | Only masked text reaches instrumented code; logs store hashes (§ 7.2) |
| T3 | One client sees another client's documents | retrieval, `/documents` | Membership check + Pinecone namespace per workspace + repository scoped by workspace / Cosmos partition key (§ 7.3) |
| T4 | Prompt injection in an uploaded document or question | ingest, `/ask` | Delimited sources, scanner, chunk quarantine, output validation (§ 7.5) |
| T5 | Stolen or forged token | API | Short-lived JWT, argon2 passwords, scopes (§ 7.4) |
| T6 | Leaked secrets | repo, images | Key Vault + managed identity; no secrets in code, images or GitHub (§ 7.7) |
| T7 | Malicious file (zip bomb, non-PDF) | `/ingest` | Size limit, magic-byte check, page limit, parse in the worker, not the API |
| T8 | Actions can't be traced afterwards | everything | Append-only audit log (§ 7.6) |
| T9 | Wrong answer presented as fact | `/ask` | Mandatory valid citations, NOT_FOUND fallback, disclaimer |

## 7.2 PII masking

### What is detected

| Entity | How |
|--------|-----|
| `IN_PAN` | Presidio's built-in Indian PAN recognizer (enable it) or regex `[A-Z]{5}[0-9]{4}[A-Z]` with a context boost |
| `IN_AADHAAR` | Presidio's built-in recognizer (12 digits + Verhoeff checksum, so random 12-digit numbers are rejected) |
| `IN_GSTIN` | Custom: `\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]` (the embedded PAN can also be checked) |
| `IN_IFSC` | Custom: `[A-Z]{4}0[A-Z0-9]{6}` |
| `BANK_ACCOUNT` | Custom: 9–18 digits **with** context words ("A/c", "account no") |
| `PHONE_NUMBER` | Presidio + an Indian `+91` / 10-digit pattern |
| `EMAIL_ADDRESS`, `PERSON`, `LOCATION` | Presidio (spaCy NER) |
| Your SIH Redact rules | Port each one as a `PatternRecognizer` in `security/recognizers.py` |

Check which Indian recognizers your Presidio version ships and enables by default, and add any
that are missing to the registry explicitly.

### Masker

```python
# taxlens/security/pii.py
from dataclasses import dataclass, field
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from taxlens.security.recognizers import custom_recognizers

ENTITIES = ["IN_PAN", "IN_AADHAAR", "IN_GSTIN", "IN_IFSC", "BANK_ACCOUNT",
            "PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON"]

@dataclass
class MaskResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)   # placeholder -> original
    entities: list[dict] = field(default_factory=list)

class PIIMasker:
    def __init__(self) -> None:
        nlp = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }).create_engine()
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp)
        for r in custom_recognizers():
            registry.add_recognizer(r)
        self.analyzer = AnalyzerEngine(nlp_engine=nlp, registry=registry)

    def mask(self, text: str, score_threshold: float = 0.5) -> MaskResult:
        results = self.analyzer.analyze(text=text, language="en", entities=ENTITIES,
                                        score_threshold=score_threshold)
        results = _drop_overlaps(results)                  # keep the higher-score span
        mapping, reverse, counters = {}, {}, {}
        out = text
        for r in sorted(results, key=lambda r: r.start, reverse=True):   # replace right-to-left
            original = text[r.start:r.end]
            if original not in reverse:                    # same value -> same placeholder
                counters[r.entity_type] = counters.get(r.entity_type, 0) + 1
                ph = f"<{r.entity_type}_{counters[r.entity_type]}>"
                reverse[original], mapping[ph] = ph, original
            out = out[:r.start] + reverse[original] + out[r.end:]
        entities = [{"type": r.entity_type, "start": r.start, "end": r.end, "score": r.score}
                    for r in results]
        return MaskResult(out, mapping, entities)

    @staticmethod
    def unmask(text: str, mapping: dict[str, str]) -> str:
        for ph, original in mapping.items():
            text = text.replace(ph, original)
        return text
```

Placeholder numbering above runs right to left, so `<PERSON_1>` may be the *last* person in the
text. If you want left-to-right numbering, assign placeholders in a first pass sorted by `start`
ascending, then replace in a second pass. Add a test either way.

### Rules

- **Where masking happens:** in `input_guard` for questions, in the ingestion pipeline for
  `client_doc` documents (chunks are **stored masked**), and in `/redact`. Public government
  documents are not masked, since they contain no personal data and masking them could hide
  officials' names, which are fine to keep.
- **The mapping never leaves the request.** It is held in graph state, isn't persisted, and isn't
  traced. Mark it as excluded from tracing, or keep it outside the state in a request-scoped
  `contextvar`. The second option is safer.
- **Unmasking** happens in `output_guard`, only for `/ask` responses to the user who asked.
  `/review` output stays masked by default.
- **Tests** (`tests/unit/test_pii.py`): synthetic PAN/Aadhaar/GSTIN in sentences, tables and OCR-ish
  text; no false positive on section numbers like `194J` or on amounts; the round trip
  `unmask(mask(x)) == x`; and a **trace leak test** that runs `/ask` with an in-memory span exporter
  and asserts that no original value appears in any span attribute.

## 7.3 Workspace isolation

There is no PostgreSQL row-level security in this design, so isolation comes from four layers.
Each one is tested on its own.

1. **API:** the `get_workspace` dependency checks that `X-Workspace-Id` is in the token's `ws`
   claim **and** in the repository's memberships. Otherwise it returns 403.
2. **Pinecone namespaces (physical separation):** each client's chunks live in their own namespace
   `ws-<uuid>`. A Pinecone query runs against exactly one namespace, so a filter mistake can't leak
   another client's chunks. A query can only reach namespaces the code explicitly passes.
   `hybrid_search(workspace_ids=...)` is a required argument with no default, and the store maps
   each id to its namespace.
3. **Repository scoping:** every document and audit method takes `workspace_id`. In Cosmos DB that
   is the **partition key**, so each read is a single-partition operation inside that tenant's
   partition, and the repository never enables cross-partition queries. In SQLite every query has
   `WHERE workspace_id = :ws`.
4. **Credentials (cloud):** the managed identity only has the Cosmos data-plane roles it needs (see
   [Deployment](08-deployment-cicd.md)), and the Pinecone API key lives only in Key Vault.

```python
# taxlens/storage/cosmos_repo.py (excerpt)
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

class CosmosRepository:
    def __init__(self, endpoint: str, database: str, key: str | None = None):
        credential = key or DefaultAzureCredential()   # key only for the local emulator
        db = CosmosClient(endpoint, credential=credential).get_database_client(database)
        self.documents = db.get_container_client("documents")
        self.audit = db.get_container_client("audit_log")

    def get_document(self, workspace_id: str, document_id: str) -> Document | None:
        try:
            item = self.documents.read_item(item=document_id, partition_key=workspace_id)
        except CosmosResourceNotFoundError:
            return None
        return Document.model_validate(item)

    def list_documents(self, workspace_id: str, limit: int = 50) -> list[Document]:
        items = self.documents.query_items(
            "SELECT TOP @n * FROM c ORDER BY c.created_at DESC",
            parameters=[{"name": "@n", "value": limit}],
            partition_key=workspace_id,                  # never cross-partition
        )
        return [Document.model_validate(i) for i in items]
```

**Cross-tenant tests** (`tests/integration/test_isolation.py`):

- Index a document into workspace A, query as a user of B, and expect zero hits and a 403 on
  `GET /documents/{A's doc}`.
- Call `PineconeStore.hybrid_search(workspace_ids=["B"])` directly with a question that matches A's
  text exactly, and expect no hits from `ws-A`.
- Call `repo.get_document("B", A_doc_id)`, and expect `None` in both SQLite and Cosmos.
- Deleting workspace A removes namespace `ws-A`, A's blobs, and A's document items. Audit items stay.

## 7.4 Authentication and authorisation

```python
# taxlens/security/auth.py
from datetime import datetime, timedelta, timezone
import uuid, jwt
from pwdlib import PasswordHash
from taxlens.config import settings

pwd = PasswordHash.recommended()            # argon2

def hash_password(p: str) -> str: return pwd.hash(p)
def verify_password(p: str, h: str) -> bool: return pwd.verify(p, h)

def create_access_token(user_id: str, scopes: list[str], workspaces: list[str]) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": user_id, "scopes": scopes, "ws": workspaces, "iat": now,
              "exp": now + timedelta(minutes=settings.jwt_expire_minutes), "jti": str(uuid.uuid4())}
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm)

def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret.get_secret_value(),
                      algorithms=[settings.jwt_algorithm])
```

```python
# taxlens/api/deps.py (excerpt)
from fastapi import Depends, Header, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes

oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token", scopes={
    "ask": "Ask questions", "review": "Review documents", "ingest": "Upload documents",
    "redact": "Redact text", "admin": "Admin"})

def get_current_user(security_scopes: SecurityScopes, token: str = Depends(oauth2)) -> dict:
    try:
        claims = decode_token(token)
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token", headers={"WWW-Authenticate": "Bearer"})
    missing = set(security_scopes.scopes) - set(claims["scopes"])
    if missing:
        raise HTTPException(403, f"Missing scope(s): {', '.join(sorted(missing))}")
    return claims

def get_workspace(x_workspace_id: str = Header(...),
                  user: dict = Security(get_current_user)) -> str:
    if x_workspace_id not in user["ws"]:
        raise HTTPException(403, "Not a member of this workspace")
    return x_workspace_id

# usage: user = Security(get_current_user, scopes=["ask"])
```

Also check membership in the repository, not just the token claims, so that removing a member takes
effect before their token expires. For "Login with Microsoft/Google" later, swap the token issuer
for Entra ID / Google OIDC and keep the scope checks unchanged.

## 7.5 Prompt-injection defences

| Layer | What |
|-------|------|
| Ingestion scan | `injection.score(chunk)` using heuristics (phrases like "ignore previous instructions", "system prompt", role tags, hidden-text tricks, long base64 blobs). Optionally add an open-source classifier such as a DeBERTa prompt-injection model from Hugging Face. Chunks with a score above 0.8 are stored with `suspicious = true` and excluded from retrieval by default |
| Input scan | Same scorer on the question. Above the threshold → `route = "refuse"` + audit event |
| Prompt structure | Sources go inside `<source>` tags, the system prompt says they are data, and the user question comes after the sources |
| Least privilege | The LLM has no tools that write, send or fetch URLs. The calculator is pure. Review can only read the current document |
| Output validation | Citation validation; the answer must not contain the system prompt (canary string check); placeholders must be unchanged |
| Test set | `tests/fixtures/injection_cases.jsonl` with 20–30 attacks (direct, indirect via an uploaded PDF, and Hindi/Hinglish variants). Each must be refused, or answered without obeying the injected instruction |

```python
# taxlens/security/injection.py (heuristic part)
import re
PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now", r"act as (an? )?(unfiltered|jailbroken)",
    r"reveal (your|the) (system )?prompt", r"</?(system|assistant)>",
    r"print (all|every) (pan|aadhaar|document)",
]
RX = [re.compile(p, re.I) for p in PATTERNS]

def score(text: str) -> float:
    hits = sum(1 for r in RX if r.search(text))
    return min(1.0, hits * 0.5)
```

## 7.6 Audit log

Append-only, enforced below the application code in both backends:

- **SQLite:** `BEFORE UPDATE` / `BEFORE DELETE` triggers raise an error (see [Architecture § 1.6](01-architecture.md#16-data-model)).
- **Cosmos DB:** the managed identity gets a **custom data-plane role** on the `audit_log`
  container that allows only `create`, `read`, `executeQuery` and `readChangeFeed`. There is no
  `replace`, `upsert` or `delete`, so even a bug or a compromised app can't rewrite history
  (see [Deployment § 8.2](08-deployment-cicd.md)).
- Optional: stream the Cosmos change feed of `audit_log` to a Blob container with an immutability
  (WORM) policy for long-term retention.

| Column | Example |
|--------|---------|
| `at` | `2026-11-02T09:15:03Z` |
| `request_id` / `trace_id` | links to Phoenix |
| `user_id`, `workspace_id` | |
| `action` | `ask`, `review`, `ingest`, `redact`, `login`, `login_failed`, `unmask`, `refused_injection` |
| `resource_id` | document id |
| `request_hash` | sha256 of the masked question, never raw text |
| `pii_entity_counts` | `{"IN_PAN": 1, "PERSON": 2}` |
| `route`, `model`, `prompt_version` | |
| `prompt_tokens`, `completion_tokens`, `latency_ms` | |
| `outcome` | `ok`, `not_found`, `refused`, `error` |

Write it with `repo.append_audit(...)` from a FastAPI dependency or middleware, so no endpoint can skip it. Add
`GET /admin/audit?workspace_id=...` (admin scope) for review.

## 7.7 Secrets and configuration

- **Local:** `.env` (gitignored). **Cloud:** Key Vault. Container Apps reference secrets with
  `keyvaultref:` and read them using a user-assigned managed identity, so no secret values appear
  in the app config.
- Blob/Queue access in Azure uses the **managed identity** (`DefaultAzureCredential`) with
  `Storage Blob Data Contributor` and `Storage Queue Data Contributor` roles, not account keys.
- GitHub → Azure uses **OIDC federated credentials**, so there are no long-lived Azure secrets in GitHub.
- `ruff` rule set `S` (bandit) runs in CI. Add `gitleaks` as a pre-commit hook or CI step.

## 7.8 Data handling statement (put this in the README)

- Public corpus: Government of India publications, cited with a source URL.
- Client documents: stored in a private Blob container per workspace; chunks are stored masked;
  originals can be deleted with `DELETE /documents/{id}` (removes the blob, the document's
  Pinecone records by id prefix, and the document item, then audits the deletion).
- Vector database: Pinecone stores chunk vectors and the **masked** chunk text. On the free Starter
  plan this data is hosted in the US (AWS `us-east-1`); a paid plan can use other regions.
- App data (users, documents, audit) is stored in Azure Cosmos DB in the chosen Azure region.
- LLM providers receive only masked text. On free tiers, providers may keep or use prompts. For
  real client data use a paid tier with no-training terms, or the local Ollama model.
- Outputs are informational and not tax or legal advice.
