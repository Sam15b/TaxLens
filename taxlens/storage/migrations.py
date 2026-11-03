"""Run Alembic migrations for the SQLite repository from code (tests, scripts, app start-up).

CLI equivalent: ``uv run alembic upgrade head`` (uses SQLITE_PATH from settings / .env).
"""

from pathlib import Path

from alembic import command
from alembic.config import Config

from taxlens.storage.tables import sqlite_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def alembic_config(sqlite_path: str | Path) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sqlite_url(sqlite_path))
    cfg.attributes["configure_logger"] = False  # don't let alembic.ini reset app logging
    return cfg


def upgrade_to_head(sqlite_path: str | Path) -> None:
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    command.upgrade(alembic_config(sqlite_path), "head")


def downgrade_to_base(sqlite_path: str | Path) -> None:
    command.downgrade(alembic_config(sqlite_path), "base")
