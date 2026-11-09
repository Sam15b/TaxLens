"""Shared fixtures.

``repo`` is parametrised over Repository backends so the same contract tests run on each one.
SQLite runs everywhere; the Cosmos DB backend joins the list once it exists.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from taxlens.storage.base import Repository
from taxlens.storage.sqlite_repo import SqliteRepository
from tests.fakes import FakeVectorStore

REPOSITORY_BACKENDS = ["sqlite"]


@pytest.fixture(params=REPOSITORY_BACKENDS)
def repo(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Repository]:
    if request.param == "sqlite":
        sqlite_repo = SqliteRepository(tmp_path / "taxlens.db", migrate=True)
        yield sqlite_repo
        sqlite_repo.dispose()
    else:  # pragma: no cover - future backends
        pytest.skip(f"backend {request.param} not available")


@pytest.fixture
def vector_store() -> FakeVectorStore:
    return FakeVectorStore()
