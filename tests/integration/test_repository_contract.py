"""Behaviour every Repository backend must have (SQLite today, Cosmos DB later)."""

import hashlib
from datetime import timedelta

import pytest
from pydantic import ValidationError

from taxlens.storage.base import AlreadyExistsError, NotFoundError, Repository
from taxlens.storage.models import AuditEvent, Document, Membership, User, Workspace, utcnow


def sha(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def make_workspace(repo: Repository, name: str = "Client") -> Workspace:
    ws = Workspace(name=name)
    repo.create_workspace(ws)
    return ws


def make_document(repo: Repository, ws: Workspace, content: str = "circular") -> Document:
    doc = Document.new(
        workspace_id=ws.id,
        sha256=sha(content),
        blob_path=f"workspaces/{ws.id}/{content}.pdf",
        doc_type="circular",
    )
    repo.create_document(doc)
    return doc


# ── users ────────────────────────────────────────────────────────────
def test_create_and_get_user(repo: Repository) -> None:
    user = User(email="Analyst@Example.com", password_hash="h", scopes=["ask", "review"])
    repo.create_user(user)

    by_id = repo.get_user(user.id)
    by_email = repo.get_user_by_email("  ANALYST@example.com ")
    assert by_id == user
    assert by_email == user
    assert repo.get_user("missing") is None
    assert repo.get_user_by_email("nobody@example.com") is None


def test_duplicate_email_is_rejected(repo: Repository) -> None:
    repo.create_user(User(email="a@example.com", password_hash="h"))
    with pytest.raises(AlreadyExistsError):
        repo.create_user(User(email="A@example.com", password_hash="other"))


# ── workspaces & memberships ─────────────────────────────────────────
def test_memberships(repo: Repository) -> None:
    user = User(email="a@example.com", password_hash="h")
    repo.create_user(user)
    ws_a, ws_b = make_workspace(repo, "A"), make_workspace(repo, "B")

    repo.add_member(Membership(user_id=user.id, workspace_id=ws_a.id, role="owner"))
    repo.add_member(Membership(user_id=user.id, workspace_id=ws_b.id))

    assert repo.get_workspace(ws_a.id) == ws_a
    assert {m.workspace_id for m in repo.memberships_for(user.id)} == {ws_a.id, ws_b.id}

    with pytest.raises(AlreadyExistsError):
        repo.add_member(Membership(user_id=user.id, workspace_id=ws_a.id))

    assert repo.remove_member(user.id, ws_b.id) is True
    assert repo.remove_member(user.id, ws_b.id) is False
    assert [m.workspace_id for m in repo.memberships_for(user.id)] == [ws_a.id]


def test_membership_for_unknown_user_is_rejected(repo: Repository) -> None:
    ws = make_workspace(repo)
    with pytest.raises(NotFoundError):
        repo.add_member(Membership(user_id="missing-user", workspace_id=ws.id))


# ── documents ────────────────────────────────────────────────────────
def test_create_and_get_document(repo: Repository) -> None:
    ws = make_workspace(repo)
    doc = make_document(repo, ws)
    assert repo.get_document(ws.id, doc.id) == doc


def test_documents_are_invisible_across_workspaces(repo: Repository) -> None:
    ws_a, ws_b = make_workspace(repo, "A"), make_workspace(repo, "B")
    doc = make_document(repo, ws_a)

    assert repo.get_document(ws_b.id, doc.id) is None
    assert repo.list_documents(ws_b.id) == []
    assert repo.delete_document(ws_b.id, doc.id) is False
    with pytest.raises(NotFoundError):
        repo.update_document(ws_b.id, doc.id, status="failed")
    # ...and the document in A is untouched
    stored = repo.get_document(ws_a.id, doc.id)
    assert stored is not None and stored.status == "queued"


def test_same_file_is_deduplicated_per_workspace(repo: Repository) -> None:
    ws_a, ws_b = make_workspace(repo, "A"), make_workspace(repo, "B")
    doc_a = make_document(repo, ws_a, "same-bytes")

    with pytest.raises(AlreadyExistsError):
        make_document(repo, ws_a, "same-bytes")

    doc_b = make_document(repo, ws_b, "same-bytes")  # other tenant may upload the same file
    assert doc_b.id != doc_a.id


def test_update_document_changes_only_allowed_fields(repo: Repository) -> None:
    ws = make_workspace(repo)
    doc = make_document(repo, ws)

    updated = repo.update_document(
        ws.id, doc.id, status="indexed", chunk_count=42, dense_model="BAAI/bge-small-en-v1.5"
    )
    assert updated.status == "indexed"
    assert updated.updated_at > doc.updated_at
    assert repo.get_document(ws.id, doc.id) == updated

    with pytest.raises(ValueError, match="cannot be updated"):
        repo.update_document(ws.id, doc.id, sha256=sha("tampered"))
    with pytest.raises(ValidationError):
        repo.update_document(ws.id, doc.id, status="done")


def test_list_documents_newest_first_with_limit(repo: Repository) -> None:
    ws = make_workspace(repo)
    base = utcnow()
    for i in range(3):
        doc = Document.new(
            workspace_id=ws.id, sha256=sha(f"doc-{i}"), blob_path=f"p{i}", doc_type="act"
        )
        doc.created_at = base + timedelta(seconds=i)
        repo.create_document(doc)

    listed = repo.list_documents(ws.id, limit=2)
    assert [d.blob_path for d in listed] == ["p2", "p1"]


def test_delete_document(repo: Repository) -> None:
    ws = make_workspace(repo)
    doc = make_document(repo, ws)
    assert repo.delete_document(ws.id, doc.id) is True
    assert repo.get_document(ws.id, doc.id) is None


# ── audit log ────────────────────────────────────────────────────────
def test_audit_is_scoped_and_newest_first(repo: Repository) -> None:
    base = utcnow()
    repo.append_audit(AuditEvent(workspace_id="w1", action="login", at=base))
    repo.append_audit(
        AuditEvent(
            workspace_id="w1",
            action="ask",
            at=base + timedelta(seconds=1),
            pii_entity_counts={"IN_PAN": 2},
        )
    )
    repo.append_audit(AuditEvent(workspace_id="w2", action="ask", at=base))

    events = repo.list_audit("w1")
    assert [e.action for e in events] == ["ask", "login"]
    assert events[0].pii_entity_counts == {"IN_PAN": 2}
    assert len(repo.list_audit("w2")) == 1


def test_audit_event_ids_are_unique(repo: Repository) -> None:
    event = AuditEvent(workspace_id="w1", action="ask")
    repo.append_audit(event)
    with pytest.raises(AlreadyExistsError):
        repo.append_audit(event)


def test_repository_exposes_no_audit_mutation(repo: Repository) -> None:
    for name in ("update_audit", "delete_audit", "clear_audit"):
        assert not hasattr(repo, name)
