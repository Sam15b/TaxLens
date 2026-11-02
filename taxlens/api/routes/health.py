"""Liveness and readiness probes."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """The process is alive (liveness probe)."""
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict[str, str]:
    """Dependencies are reachable (readiness probe).

    TODO (week 1-3): check the repository (SQLite / Cosmos), Pinecone (describe_index_stats),
    Blob, and that the embedding models are loaded; return 503 if any fails.
    """
    return {"status": "ready"}
