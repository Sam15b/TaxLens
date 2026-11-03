"""Repository protocol: the only way application code reads or writes relational app data.

Implementations: SqliteRepository (local dev, CI, tests) and CosmosRepository (Azure).
Both must pass tests/integration/test_repository_contract.py.

Isolation rule: every document/audit method takes ``workspace_id`` explicitly. In Cosmos DB it is
the partition key, so a call can never read another tenant's partition.
"""

from typing import Any, Protocol

from taxlens.storage.models import AuditEvent, Document, Membership, User, Workspace


class RepositoryError(Exception):
    """Base class for repository errors."""


class AlreadyExistsError(RepositoryError):
    """An item with the same id or unique key already exists."""


class NotFoundError(RepositoryError):
    """The item to update does not exist (in the given workspace)."""


UPDATABLE_DOCUMENT_FIELDS = frozenset(
    {
        "title",
        "status",
        "dense_model",
        "sparse_model",
        "chunk_count",
        "suspicious_chunks",
        "error",
    }
)
"""Fields ``update_document`` may change. Identity fields (id, workspace, sha256) never change."""


class Repository(Protocol):
    # ── users ────────────────────────────────────────────────────────
    def create_user(self, user: User) -> None:
        """Raises AlreadyExistsError if the email is taken."""
        ...

    def get_user(self, user_id: str) -> User | None: ...

    def get_user_by_email(self, email: str) -> User | None: ...

    # ── workspaces & memberships ─────────────────────────────────────
    def create_workspace(self, workspace: Workspace) -> None: ...

    def get_workspace(self, workspace_id: str) -> Workspace | None: ...

    def add_member(self, membership: Membership) -> None:
        """Raises AlreadyExistsError if the user is already a member."""
        ...

    def remove_member(self, user_id: str, workspace_id: str) -> bool:
        """Returns True if a membership was removed."""
        ...

    def memberships_for(self, user_id: str) -> list[Membership]: ...

    # ── documents (always scoped by workspace_id) ────────────────────
    def create_document(self, document: Document) -> None:
        """Raises AlreadyExistsError for the same id or (workspace_id, sha256)."""
        ...

    def get_document(self, workspace_id: str, document_id: str) -> Document | None: ...

    def update_document(self, workspace_id: str, document_id: str, **fields: Any) -> Document:
        """Updates only UPDATABLE_DOCUMENT_FIELDS, bumps updated_at, returns the new document.

        Raises NotFoundError if the document is not in this workspace, ValueError for other fields.
        """
        ...

    def list_documents(self, workspace_id: str, limit: int = 50) -> list[Document]:
        """Newest first."""
        ...

    def delete_document(self, workspace_id: str, document_id: str) -> bool:
        """Returns True if a document was deleted."""
        ...

    # ── audit log (append-only: deliberately no update/delete) ───────
    def append_audit(self, event: AuditEvent) -> None: ...

    def list_audit(self, workspace_id: str, limit: int = 100) -> list[AuditEvent]:
        """Newest first."""
        ...
