"""The configured vector store (``VECTOR_BACKEND``). Qdrant and Oracle arrive in week 5."""

from functools import lru_cache

from taxlens.config import settings
from taxlens.retrieval.base import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    if settings.vector_backend == "pinecone":
        from taxlens.retrieval.pinecone_store import PineconeStore

        return PineconeStore.from_settings()
    raise NotImplementedError(f"vector backend {settings.vector_backend!r} is not built yet")
