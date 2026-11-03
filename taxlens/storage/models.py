"""Domain models shared by every Repository backend (SQLite locally, Cosmos DB in Azure).

These are plain Pydantic models, not ORM classes, so both backends store and return the same shapes.
See docs/01-architecture.md § 1.6.
"""

from datetime import UTC, datetime
from typing import Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

DocType = Literal["act", "rules", "circular", "notification", "client_doc"]
DocStatus = Literal["queued", "processing", "indexed", "failed", "needs_ocr"]
Role = Literal["owner", "member", "viewer"]
Scope = Literal["ask", "review", "ingest", "redact", "admin"]
AuditAction = Literal[
    "ask",
    "review",
    "ingest",
    "redact",
    "login",
    "login_failed",
    "unmask",
    "refused_injection",
    "delete_document",
]
AuditOutcome = Literal["ok", "not_found", "refused", "error"]

PUBLIC_WORKSPACE_ID = "public"
"""Workspace that holds the public government corpus; every user can read it."""

_DOCUMENT_NAMESPACE = uuid5(NAMESPACE_URL, "https://github.com/Sam15b/TaxLens/documents")


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid4())


def document_id_for(workspace_id: str, sha256: str) -> str:
    """Deterministic document id: the same file uploaded to the same workspace gets the same id.

    This turns de-duplication into a point read (no query) in both SQLite and Cosmos DB.
    """
    return str(uuid5(_DOCUMENT_NAMESPACE, f"{workspace_id}:{sha256.lower()}"))


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    schema_version: int = 1
    """Cosmos items have no schema; versioning them allows upgrade-on-read later."""


class User(_Model):
    id: str = Field(default_factory=new_id)
    email: str
    password_hash: str
    scopes: list[Scope] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        value = value.strip().lower()
        local, sep, domain = value.partition("@")
        if not (local and sep and "." in domain):
            raise ValueError("invalid email address")
        return value


class Workspace(_Model):
    id: str = Field(default_factory=new_id)
    name: str = Field(min_length=1, max_length=200)
    is_public: bool = False
    created_at: datetime = Field(default_factory=utcnow)


class Membership(_Model):
    user_id: str
    workspace_id: str
    role: Role = "member"
    created_at: datetime = Field(default_factory=utcnow)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def id(self) -> str:
        """Item id used by Cosmos DB (partitioned by user_id)."""
        return f"{self.user_id}:{self.workspace_id}"


class Document(_Model):
    id: str
    workspace_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    blob_path: str
    doc_type: DocType
    title: str | None = None
    status: DocStatus = "queued"
    dense_model: str | None = None
    sparse_model: str | None = None
    chunk_count: int = Field(default=0, ge=0)
    suspicious_chunks: int = Field(default=0, ge=0)
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @classmethod
    def new(
        cls,
        *,
        workspace_id: str,
        sha256: str,
        blob_path: str,
        doc_type: DocType,
        title: str | None = None,
    ) -> "Document":
        sha256 = sha256.lower()
        return cls(
            id=document_id_for(workspace_id, sha256),
            workspace_id=workspace_id,
            sha256=sha256,
            blob_path=blob_path,
            doc_type=doc_type,
            title=title,
        )


class AuditEvent(_Model):
    """One append-only audit record. Never holds raw request text, only hashes and counts."""

    id: str = Field(default_factory=new_id)
    at: datetime = Field(default_factory=utcnow)
    workspace_id: str
    user_id: str | None = None
    action: AuditAction
    outcome: AuditOutcome = "ok"
    resource_id: str | None = None
    request_hash: str | None = None
    route: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    trace_id: str | None = None
    pii_entity_counts: dict[str, int] = Field(default_factory=dict)
