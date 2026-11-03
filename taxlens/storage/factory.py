"""Select the Repository implementation from settings (DB_BACKEND=sqlite|cosmos)."""

from functools import lru_cache

from taxlens.config import settings
from taxlens.storage.base import Repository


@lru_cache
def get_repository() -> Repository:
    if settings.db_backend == "sqlite":
        from taxlens.storage.sqlite_repo import SqliteRepository

        return SqliteRepository(settings.sqlite_path)
    # Cosmos DB backend is added together with the Azure deployment.
    raise NotImplementedError("DB_BACKEND=cosmos is not implemented yet; use sqlite")
