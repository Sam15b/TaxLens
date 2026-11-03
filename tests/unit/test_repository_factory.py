from collections.abc import Iterator
from pathlib import Path

import pytest

from taxlens.config import settings
from taxlens.storage.factory import get_repository
from taxlens.storage.sqlite_repo import SqliteRepository


@pytest.fixture(autouse=True)
def fresh_factory() -> Iterator[None]:
    get_repository.cache_clear()
    yield
    get_repository.cache_clear()


def test_sqlite_backend_is_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "db_backend", "sqlite")
    monkeypatch.setattr(settings, "sqlite_path", str(tmp_path / "factory.db"))
    repo = get_repository()
    assert isinstance(repo, SqliteRepository)
    assert get_repository() is repo  # cached: one engine per process


def test_cosmos_backend_not_available_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "db_backend", "cosmos")
    with pytest.raises(NotImplementedError):
        get_repository()
