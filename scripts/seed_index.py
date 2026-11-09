"""Register and index the public corpus synchronously (no queue, no worker).

    uv run python scripts/seed_index.py data/raw                 # every PDF listed in sources.yaml
    uv run python scripts/seed_index.py data/raw --only ita-1961,cgst-act-2017
    uv run python scripts/seed_index.py data/raw --wait          # poll Pinecone until searchable
    uv run python scripts/seed_index.py data/raw --force         # re-index indexed documents too

Each PDF named ``<id>.pdf`` with an entry in data/sources.yaml becomes a document in workspace
``public`` (created if missing) and is run through ``ingest_document``. Already-indexed documents
are skipped, so a re-run only does what is left. Free tier: every run writes to Pinecone, so
don't re-seed the whole corpus in a loop.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

from taxlens.config import settings
from taxlens.ingestion.pipeline import ROOT, ingest_document, load_sources
from taxlens.retrieval.base import VectorStore
from taxlens.storage.base import Repository
from taxlens.storage.models import PUBLIC_WORKSPACE_ID, Document, Workspace

POLL_SECONDS, POLL_TIMEOUT = 3.0, 300.0


def ensure_public_workspace(repo: Repository) -> None:
    if repo.get_workspace(PUBLIC_WORKSPACE_ID) is None:
        repo.create_workspace(
            Workspace(id=PUBLIC_WORKSPACE_ID, name="Public corpus", is_public=True)
        )


def register(repo: Repository, pdf: Path, source: dict[str, object]) -> Document:
    """The document row for ``pdf`` (created on first sight; the id is derived from the bytes)."""
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    try:
        blob_path = pdf.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        blob_path = str(pdf.resolve())
    doc = Document.new(
        workspace_id=PUBLIC_WORKSPACE_ID,
        sha256=sha,
        blob_path=blob_path,
        doc_type=str(source["doc_type"]),  # type: ignore[arg-type]
        title=str(source.get("title") or pdf.stem),
    )
    existing = repo.get_document(PUBLIC_WORKSPACE_ID, doc.id)
    if existing is not None:
        return existing
    repo.create_document(doc)
    return doc


def seed(
    repo: Repository,
    store: VectorStore,
    folder: Path,
    *,
    only: set[str] | None = None,
    force: bool = False,
) -> dict[str, int]:
    """``{source_id: chunks upserted}`` for every listed PDF found in ``folder``."""
    ensure_public_workspace(repo)
    done: dict[str, int] = {}
    for sid, source in load_sources().items():
        pdf = folder / f"{sid}.pdf"
        if (only and sid not in only) or not pdf.is_file():
            continue
        doc = register(repo, pdf, source)
        started = time.monotonic()
        n = ingest_document(repo, store, PUBLIC_WORKSPACE_ID, doc.id, source=source, force=force)
        status = repo.get_document(PUBLIC_WORKSPACE_ID, doc.id)
        state = status.status if status else "?"
        print(f"{state:10} {sid:28} {n:5} chunks  {time.monotonic() - started:6.1f}s", flush=True)
        done[sid] = n
    return done


def wait_until_searchable(store: VectorStore, expected: int) -> None:
    count = getattr(store, "count", None)
    if count is None or expected == 0:
        return
    deadline = time.monotonic() + POLL_TIMEOUT
    while (seen := count(PUBLIC_WORKSPACE_ID)) < expected:
        if time.monotonic() > deadline:
            raise TimeoutError(f"only {seen} of {expected} vectors visible after {POLL_TIMEOUT}s")
        time.sleep(POLL_SECONDS)
    print(f"{seen} vectors searchable in namespace {PUBLIC_WORKSPACE_ID}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("folder", type=Path)
    p.add_argument("--only", help="comma-separated source ids")
    p.add_argument(
        "--wait", action="store_true", help="poll Pinecone until the vectors are visible"
    )
    p.add_argument("--force", action="store_true", help="re-index documents already indexed")
    a = p.parse_args(argv)

    from taxlens.retrieval.factory import get_vector_store
    from taxlens.storage.factory import get_repository
    from taxlens.storage.migrations import upgrade_to_head

    if settings.db_backend == "sqlite":
        upgrade_to_head(Path(settings.sqlite_path))
    repo, store = get_repository(), get_vector_store()
    only = set(a.only.split(",")) if a.only else None
    done = seed(repo, store, a.folder, only=only, force=a.force)
    print(f"{len(done)} documents, {sum(done.values())} chunks upserted")
    if a.wait:
        indexed = [d for d in repo.list_documents(PUBLIC_WORKSPACE_ID, limit=1000)]
        wait_until_searchable(store, sum(d.chunk_count for d in indexed if d.status == "indexed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
