"""SQLAlchemy Core schema for the SQLite repository (local dev, CI, tests).

Alembic migrations in ``migrations/versions`` create these tables explicitly; this module is the
in-code description used by the repository and by ``alembic revision --autogenerate``.
Datetimes are stored as ISO-8601 strings with timezone so they round-trip exactly.
"""

from pathlib import Path

import sqlalchemy as sa

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = sa.MetaData(naming_convention=NAMING_CONVENTION)

users = sa.Table(
    "users",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("email", sa.String(320), nullable=False, unique=True),
    sa.Column("password_hash", sa.Text, nullable=False),
    sa.Column("scopes", sa.JSON, nullable=False),
    sa.Column("is_active", sa.Boolean, nullable=False),
    sa.Column("created_at", sa.String(40), nullable=False),
    sa.Column("schema_version", sa.Integer, nullable=False),
)

workspaces = sa.Table(
    "workspaces",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("name", sa.String(200), nullable=False),
    sa.Column("is_public", sa.Boolean, nullable=False),
    sa.Column("created_at", sa.String(40), nullable=False),
    sa.Column("schema_version", sa.Integer, nullable=False),
)

memberships = sa.Table(
    "memberships",
    metadata,
    sa.Column(
        "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    ),
    sa.Column(
        "workspace_id",
        sa.String(36),
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("role", sa.String(20), nullable=False),
    sa.Column("created_at", sa.String(40), nullable=False),
    sa.Column("schema_version", sa.Integer, nullable=False),
)

documents = sa.Table(
    "documents",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column(
        "workspace_id",
        sa.String(36),
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    ),
    sa.Column("sha256", sa.String(64), nullable=False),
    sa.Column("blob_path", sa.Text, nullable=False),
    sa.Column("doc_type", sa.String(20), nullable=False),
    sa.Column("title", sa.Text),
    sa.Column("status", sa.String(20), nullable=False),
    sa.Column("dense_model", sa.String(200)),
    sa.Column("sparse_model", sa.String(200)),
    sa.Column("chunk_count", sa.Integer, nullable=False),
    sa.Column("suspicious_chunks", sa.Integer, nullable=False),
    sa.Column("error", sa.Text),
    sa.Column("created_at", sa.String(40), nullable=False),
    sa.Column("updated_at", sa.String(40), nullable=False),
    sa.Column("schema_version", sa.Integer, nullable=False),
    sa.UniqueConstraint("workspace_id", "sha256", name="uq_documents_workspace_sha256"),
    sa.Index("ix_documents_workspace_created", "workspace_id", "created_at"),
)

# No foreign key to workspaces: audit history must outlive a deleted workspace.
audit_log = sa.Table(
    "audit_log",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("at", sa.String(40), nullable=False),
    sa.Column("workspace_id", sa.String(36), nullable=False),
    sa.Column("user_id", sa.String(36)),
    sa.Column("action", sa.String(40), nullable=False),
    sa.Column("outcome", sa.String(20), nullable=False),
    sa.Column("resource_id", sa.String(36)),
    sa.Column("request_hash", sa.String(64)),
    sa.Column("route", sa.String(20)),
    sa.Column("model", sa.String(200)),
    sa.Column("prompt_version", sa.String(40)),
    sa.Column("prompt_tokens", sa.Integer),
    sa.Column("completion_tokens", sa.Integer),
    sa.Column("latency_ms", sa.Integer),
    sa.Column("trace_id", sa.String(64)),
    sa.Column("pii_entity_counts", sa.JSON, nullable=False),
    sa.Column("schema_version", sa.Integer, nullable=False),
    sa.Index("ix_audit_log_workspace_at", "workspace_id", "at"),
)


def sqlite_url(path: str | Path) -> str:
    """SQLAlchemy URL for a SQLite file (relative paths are relative to the working directory)."""
    return f"sqlite:///{Path(path).as_posix()}"
