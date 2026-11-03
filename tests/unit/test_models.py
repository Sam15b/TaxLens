import hashlib

import pytest
from pydantic import ValidationError

from taxlens.storage.models import (
    AuditEvent,
    Document,
    Membership,
    User,
    Workspace,
    document_id_for,
)

SHA = hashlib.sha256(b"sample circular").hexdigest()


def test_document_id_is_deterministic_per_workspace() -> None:
    assert document_id_for("ws-a", SHA) == document_id_for("ws-a", SHA)
    assert document_id_for("ws-a", SHA) == document_id_for("ws-a", SHA.upper())
    assert document_id_for("ws-a", SHA) != document_id_for("ws-b", SHA)


def test_document_new_sets_id_and_defaults() -> None:
    doc = Document.new(
        workspace_id="ws-a", sha256=SHA, blob_path="workspaces/ws-a/x.pdf", doc_type="circular"
    )
    assert doc.id == document_id_for("ws-a", SHA)
    assert doc.status == "queued"
    assert doc.chunk_count == 0
    assert doc.created_at.tzinfo is not None


def test_document_rejects_invalid_status_and_hash() -> None:
    doc = Document.new(workspace_id="ws", sha256=SHA, blob_path="p", doc_type="act")
    with pytest.raises(ValidationError):
        doc.status = "done"  # type: ignore[assignment]
    with pytest.raises(ValidationError):
        Document.new(workspace_id="ws", sha256="not-a-hash", blob_path="p", doc_type="act")


def test_user_email_is_normalised_and_validated() -> None:
    user = User(email="  Analyst@Example.COM ", password_hash="h")
    assert user.email == "analyst@example.com"
    with pytest.raises(ValidationError):
        User(email="no-at-sign", password_hash="h")


def test_user_scopes_are_restricted() -> None:
    with pytest.raises(ValidationError):
        User(email="a@b.co", password_hash="h", scopes=["superuser"])  # type: ignore[list-item]


def test_membership_id_combines_user_and_workspace() -> None:
    m = Membership(user_id="u1", workspace_id="w1")
    assert m.id == "u1:w1"
    assert m.model_dump()["id"] == "u1:w1"


def test_workspace_name_required() -> None:
    with pytest.raises(ValidationError):
        Workspace(name="")


def test_audit_event_defaults() -> None:
    event = AuditEvent(workspace_id="w1", action="ask")
    assert event.outcome == "ok"
    assert event.pii_entity_counts == {}
    assert event.schema_version == 1
