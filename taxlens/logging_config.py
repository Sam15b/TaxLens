"""Structured logging setup.

Rule: never log raw request text or document content. Log ids, hashes and counts instead.
"""

import logging

from taxlens.config import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )
