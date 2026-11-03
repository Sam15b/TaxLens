"""Repository backed by SQLite (local development, CI and tests).

- SQLAlchemy Core (no ORM): the domain models in ``storage.models`` are the only data classes.
- Every connection enables WAL, a busy timeout and foreign keys, so the API and the worker can
  share one database file safely.
- Schema changes go through Alembic (``storage.migrations``); this module never creates tables.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from taxlens.storage.base import AlreadyExistsError, NotFoundError, RepositoryError
from taxlens.storage.migrations import upgrade_to_head
from taxlens.storage.models import Membership, User, Workspace
from taxlens.storage.tables import memberships, sqlite_url, users, workspaces


def _set_sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _timestamp(value: datetime) -> str:
    """Fixed-width UTC ISO-8601 so string order equals time order."""
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def _to_row(model: BaseModel, table: sa.Table) -> dict[str, Any]:
    columns = set(table.columns.keys())
    row: dict[str, Any] = {}
    for key, value in model.model_dump().items():
        if key in columns:
            row[key] = _timestamp(value) if isinstance(value, datetime) else value
    return row


def _translate_integrity_error(exc: IntegrityError, what: str) -> RepositoryError:
    message = str(exc.orig).upper()
    if "FOREIGN KEY" in message:
        return NotFoundError(f"{what}: referenced item does not exist")
    return AlreadyExistsError(f"{what} already exists")


class SqliteRepository:
    def __init__(self, path: str | Path, *, migrate: bool = False) -> None:
        self.path = Path(path)
        if migrate:
            upgrade_to_head(self.path)
        self.engine: Engine = sa.create_engine(sqlite_url(self.path))
        sa.event.listen(self.engine, "connect", _set_sqlite_pragmas)

    def dispose(self) -> None:
        self.engine.dispose()

    # ── helpers ──────────────────────────────────────────────────────
    def _insert(self, table: sa.Table, model: BaseModel, what: str) -> None:
        try:
            with self.engine.begin() as conn:
                conn.execute(table.insert().values(**_to_row(model, table)))
        except IntegrityError as exc:
            raise _translate_integrity_error(exc, what) from exc

    def _one(self, statement: sa.Select[Any]) -> Mapping[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(statement).mappings().first()
        return dict(row) if row is not None else None

    def _many(self, statement: sa.Select[Any]) -> list[Mapping[str, Any]]:
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(statement).mappings().all()]

    # ── users ────────────────────────────────────────────────────────
    def create_user(self, user: User) -> None:
        self._insert(users, user, f"user {user.email}")

    def get_user(self, user_id: str) -> User | None:
        row = self._one(sa.select(users).where(users.c.id == user_id))
        return User.model_validate(row) if row else None

    def get_user_by_email(self, email: str) -> User | None:
        row = self._one(sa.select(users).where(users.c.email == email.strip().lower()))
        return User.model_validate(row) if row else None

    # ── workspaces & memberships ─────────────────────────────────────
    def create_workspace(self, workspace: Workspace) -> None:
        self._insert(workspaces, workspace, f"workspace {workspace.id}")

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        row = self._one(sa.select(workspaces).where(workspaces.c.id == workspace_id))
        return Workspace.model_validate(row) if row else None

    def add_member(self, membership: Membership) -> None:
        self._insert(memberships, membership, f"membership {membership.id}")

    def remove_member(self, user_id: str, workspace_id: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(
                memberships.delete().where(
                    memberships.c.user_id == user_id,
                    memberships.c.workspace_id == workspace_id,
                )
            )
        return result.rowcount > 0

    def memberships_for(self, user_id: str) -> list[Membership]:
        rows = self._many(
            sa.select(memberships)
            .where(memberships.c.user_id == user_id)
            .order_by(memberships.c.created_at)
        )
        return [Membership.model_validate(r) for r in rows]
