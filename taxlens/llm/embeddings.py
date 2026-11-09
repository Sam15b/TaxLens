"""Local dense embeddings with FastEmbed (ONNX on the CPU, no API calls).

``BAAI/bge-small-en-v1.5`` gives 384-dimensional vectors. They are L2-normalised here, because the
Pinecone index uses ``dotproduct`` (needed for hybrid dense + sparse search), and dot products of
unit vectors are cosine similarities. See docs/01-architecture.md § 1.3 A.

Queries go through ``query_embed``, which applies the model's query instruction if it has one.
Passages go through ``passage_embed``.
"""

import math
from collections.abc import Iterable, Sequence
from functools import lru_cache

from fastembed import TextEmbedding

from taxlens.config import settings

BATCH_SIZE = 32


@lru_cache
def get_embedder() -> TextEmbedding:
    """The dense model, loaded once per process (downloaded to ``FASTEMBED_CACHE_PATH``)."""
    return TextEmbedding(settings.embedding_model, cache_dir=settings.fastembed_cache_path)


def _unit(vector: Iterable[float]) -> list[float]:
    values = [float(v) for v in vector]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values] if norm else values


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """One unit vector per passage, in order."""
    if not texts:
        return []
    vectors = get_embedder().passage_embed(list(texts), batch_size=BATCH_SIZE)
    return [_unit(v) for v in vectors]


def embed_query(text: str) -> list[float]:
    """The unit vector for a search query."""
    return _unit(next(iter(get_embedder().query_embed(text))))
