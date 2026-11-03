from pathlib import Path

import sqlalchemy as sa

from taxlens.storage.migrations import downgrade_to_base, upgrade_to_head
from taxlens.storage.tables import sqlite_url

EXPECTED_TABLES = {"users", "workspaces", "memberships", "documents", "audit_log"}


def _tables_and_triggers(db: Path) -> tuple[set[str], set[str]]:
    engine = sa.create_engine(sqlite_url(db))
    with engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT type, name FROM sqlite_master")).all()
    engine.dispose()
    tables = {name for kind, name in rows if kind == "table" and name != "alembic_version"}
    triggers = {name for kind, name in rows if kind == "trigger"}
    return tables, triggers


def test_upgrade_creates_schema_and_audit_triggers(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "taxlens.db"
    upgrade_to_head(db)
    tables, triggers = _tables_and_triggers(db)
    assert EXPECTED_TABLES <= tables
    assert triggers == {"audit_log_no_update", "audit_log_no_delete"}


def test_downgrade_then_upgrade_round_trips(tmp_path: Path) -> None:
    db = tmp_path / "taxlens.db"
    upgrade_to_head(db)
    downgrade_to_base(db)
    tables, triggers = _tables_and_triggers(db)
    assert tables.isdisjoint(EXPECTED_TABLES)
    assert triggers == set()
    upgrade_to_head(db)
    tables, _ = _tables_and_triggers(db)
    assert EXPECTED_TABLES <= tables
