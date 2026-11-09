"""Create the Pinecone index if it is missing, and wait until it is ready.

    uv run python scripts/init_pinecone.py                 # index from PINECONE_INDEX
    uv run python scripts/init_pinecone.py --index taxlens-ci

Serverless, dimension 384 (BGE-small), metric ``dotproduct`` (needed to store dense and sparse
vectors in one index), on the Starter plan's cloud/region. Running it again is a no-op.
See docs/01-architecture.md § 1.3 B.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from taxlens.config import settings

POLL_SECONDS = 2.0
READY_TIMEOUT_SECONDS = 300.0


def ensure_index(
    pc: Any,
    name: str,
    *,
    dimension: int,
    cloud: str,
    region: str,
    poll: float = POLL_SECONDS,
    timeout: float = READY_TIMEOUT_SECONDS,
) -> str:
    """``"exists"`` or ``"created"``; returns once the index reports ready."""
    from pinecone import ServerlessSpec

    if pc.has_index(name):
        outcome = "exists"
    else:
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        outcome = "created"
    deadline = time.monotonic() + timeout
    while not pc.describe_index(name).status.ready:
        if time.monotonic() > deadline:
            raise TimeoutError(f"index {name} not ready after {timeout:.0f}s")
        time.sleep(poll)
    return outcome


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", default=settings.pinecone_index)
    a = p.parse_args(argv)

    if settings.pinecone_api_key is None:
        print("PINECONE_API_KEY is not set (add it to .env)", file=sys.stderr)
        return 2
    from pinecone import Pinecone

    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    outcome = ensure_index(
        pc,
        a.index,
        dimension=settings.embedding_dim,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )
    print(f"index {a.index}: {'already exists' if outcome == 'exists' else 'created'}, ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
