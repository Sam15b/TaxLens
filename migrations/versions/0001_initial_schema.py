"""Initial schema: users, workspaces, memberships, documents, append-only audit_log.

Revision ID: 0001
Revises:
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

AUDIT_TRIGGERS = {
    "audit_log_no_update": "UPDATE",
    "audit_log_no_delete": "DELETE",
}


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("scopes", sa.JSON(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_table(
        "workspaces",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("is_public", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_workspaces"),
    )
    op.create_table(
        "memberships",
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("workspace_id", sa.String(36), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_memberships_user_id_users", ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_memberships_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("user_id", "workspace_id", name="pk_memberships"),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("workspace_id", sa.String(36), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("blob_path", sa.Text(), nullable=False),
        sa.Column("doc_type", sa.String(20), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("dense_model", sa.String(200), nullable=True),
        sa.Column("sparse_model", sa.String(200), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("suspicious_chunks", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("updated_at", sa.String(40), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name="fk_documents_workspace_id_workspaces",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("workspace_id", "sha256", name="uq_documents_workspace_sha256"),
    )
    op.create_index("ix_documents_workspace_created", "documents", ["workspace_id", "created_at"])
    op.create_table(
        "audit_log",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("at", sa.String(40), nullable=False),
        sa.Column("workspace_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=True),
        sa.Column("action", sa.String(40), nullable=False),
        sa.Column("outcome", sa.String(20), nullable=False),
        sa.Column("resource_id", sa.String(36), nullable=True),
        sa.Column("request_hash", sa.String(64), nullable=True),
        sa.Column("route", sa.String(20), nullable=True),
        sa.Column("model", sa.String(200), nullable=True),
        sa.Column("prompt_version", sa.String(40), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("pii_entity_counts", sa.JSON(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_audit_log"),
    )
    op.create_index("ix_audit_log_workspace_at", "audit_log", ["workspace_id", "at"])

    # Append-only audit log, enforced by the database rather than by application code.
    for name, operation in AUDIT_TRIGGERS.items():
        op.execute(
            f"CREATE TRIGGER {name} BEFORE {operation} ON audit_log "
            "BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;"
        )


def downgrade() -> None:
    for name in AUDIT_TRIGGERS:
        op.execute(f"DROP TRIGGER IF EXISTS {name}")
    op.drop_index("ix_audit_log_workspace_at", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_documents_workspace_created", table_name="documents")
    op.drop_table("documents")
    op.drop_table("memberships")
    op.drop_table("workspaces")
    op.drop_table("users")
