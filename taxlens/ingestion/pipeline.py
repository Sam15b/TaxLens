"""Ingest one document: load → clean → metadata → chunk → embed → upsert, with status updates.

``ingest_document`` is what the worker runs for each queued document (week 3), and what
``scripts/seed_index.py`` runs synchronously for the public corpus. It is idempotent: an
already-indexed document is skipped, and a re-run deletes the document's old chunks (by id
prefix) before upserting, so no stale chunks survive. See docs/04-build-guide.md → "Ingestion
pipeline".
"""

import logging
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from taxlens.config import settings
from taxlens.ingestion.chunking import chunk_pages
from taxlens.ingestion.cleaning import clean_pages
from taxlens.ingestion.loaders import load_pdf, needs_ocr
from taxlens.ingestion.metadata import extract_metadata
from taxlens.llm.embeddings import embed_texts
from taxlens.retrieval.base import ChunkRecord, VectorStore, chunk_id
from taxlens.storage.base import Repository
from taxlens.storage.models import Document

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "data" / "sources.yaml"

# sources.yaml keys copied onto every chunk (``url`` becomes ``source_url``)
_SOURCE_KEYS = ("title", "doc_type", "act", "circular_no", "notification_no", "issued_on", "url")


@cache
def load_sources(path: Path = SOURCES) -> dict[str, dict[str, Any]]:
    """data/sources.yaml as ``{id: entry}``."""
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return {e["id"]: e for e in entries}


def source_metadata(source: dict[str, Any] | None, text: str) -> dict[str, Any]:
    """Chunk metadata: what the text reveals, overridden by the curated sources.yaml entry."""
    meta: dict[str, Any] = dict(extract_metadata(text))
    for key in _SOURCE_KEYS:
        value = (source or {}).get(key)
        if value is not None:
            meta["source_url" if key == "url" else key] = value
    if meta.get("doc_type") == "act" and not (source or {}).get("issued_on"):
        meta.pop("issued_on", None)  # a date found inside an Act's text is not its date
    if isinstance(meta.get("issued_on"), str):
        meta["issued_on"] = date.fromisoformat(meta["issued_on"])
    return meta


def _record(doc: Document, text: str, meta: dict[str, Any], vector: list[float]) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id(doc.id, meta["chunk_no"]),
        workspace_id=doc.workspace_id,
        document_id=doc.id,
        content=text,
        embedding=vector,
        doc_type=meta.get("doc_type") or doc.doc_type,
        act=meta.get("act"),
        section=meta.get("section"),
        heading=meta.get("heading") or None,
        title=meta.get("title") or doc.title,
        circular_no=meta.get("circular_no"),
        notification_no=meta.get("notification_no"),
        issued_on=meta.get("issued_on"),
        page=meta.get("page"),
        source_url=meta.get("source_url"),
        suspicious=False,  # the injection detector arrives in week 3
    )


def local_path(doc: Document) -> Path:
    """The PDF behind ``doc.blob_path``. Until Blob storage exists (week 3), that is a local path,
    absolute or relative to the repository root."""
    path = Path(doc.blob_path)
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"{doc.blob_path} (blob download arrives with the worker)")
    return path


def ingest_document(
    repo: Repository,
    store: VectorStore,
    workspace_id: str,
    document_id: str,
    *,
    source: dict[str, Any] | None = None,
    force: bool = False,
) -> int:
    """Index one document; returns the number of chunks upserted (0 when skipped).

    ``source`` is the document's sources.yaml entry, for the public corpus. The document ends
    ``indexed``, ``needs_ocr`` (a scanned PDF, skipped until OCR exists) or ``failed``.
    """
    doc = repo.get_document(workspace_id, document_id)
    if doc is None:
        raise LookupError(f"document {document_id} not found in workspace {workspace_id}")
    if doc.status == "indexed" and not force:
        return 0
    repo.update_document(workspace_id, document_id, status="processing", error=None)
    try:
        pages = load_pdf(local_path(doc))
        if needs_ocr(pages):
            repo.update_document(workspace_id, document_id, status="needs_ocr", chunk_count=0)
            return 0
        # Low-text pages stay in as blanks, so page numbers in citations stay right.
        cleaned = clean_pages(["" if p.needs_ocr else p.text for p in pages])
        meta = source_metadata(source, "\n".join(cleaned))
        meta.setdefault("doc_type", doc.doc_type)
        meta.setdefault("title", doc.title)
        pieces = chunk_pages(cleaned, meta)
        vectors = embed_texts([text for text, _ in pieces])
        records = [_record(doc, t, m, v) for (t, m), v in zip(pieces, vectors, strict=True)]
        store.delete_document(document_id=document_id, workspace_id=workspace_id)
        store.upsert(records)
    except Exception as exc:
        repo.update_document(workspace_id, document_id, status="failed", error=str(exc)[:500])
        raise
    repo.update_document(
        workspace_id,
        document_id,
        status="indexed",
        chunk_count=len(records),
        dense_model=settings.embedding_model,
        sparse_model=settings.sparse_model,
    )
    log.info("indexed %s: %d chunks", document_id, len(records))
    return len(records)
