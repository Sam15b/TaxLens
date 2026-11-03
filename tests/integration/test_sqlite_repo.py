"""SQLite-specific guarantees on top of the repository contract."""

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from taxlens.storage.models import AuditEvent
from taxlens.storage.sqlite_repo import SqliteRepository


@pytest.fixture
def sqlite_repo(tmp_path: Path) -> SqliteRepository:
    return SqliteRepository(tmp_path / "taxlens.db", migrate=True)


@pytest.mark.parametrize(
    "statement", ["UPDATE audit_log SET outcome = 'error'", "DELETE FROM audit_log"]
)
def test_database_rejects_audit_tampering(sqlite_repo: SqliteRepository, statement: str) -> None:
    sqlite_repo.append_audit(AuditEvent(workspace_id="w1", action="ask"))
    with pytest.raises(IntegrityError, match="append-only"):
        with sqlite_repo.engine.begin() as conn:
            conn.execute(sa.text(statement))
    assert len(sqlite_repo.list_audit("w1")) == 1


def test_connections_use_wal_and_foreign_keys(sqlite_repo: SqliteRepository) -> None:
    with sqlite_repo.engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
        assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert conn.exec_driver_sql("PRAGMA busy_timeout").scalar() == 5000
