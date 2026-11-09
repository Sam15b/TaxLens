"""Translate between TaxLens records/filters and Pinecone metadata.

Pinecone metadata can't hold nulls, so ``None`` values are dropped. Dates are stored as integers
(``20250315``) so ``$gte`` / ``$lte`` work on them. ``suspicious`` is always stored, and the
default filter excludes suspicious chunks (possible prompt injection; flagged in week 3).
"""

from dataclasses import asdict
from datetime import date
from typing import Any

from taxlens.retrieval.base import ChunkRecord, SearchFilters

_NOT_METADATA = ("id", "embedding", "content")


def date_to_int(d: date) -> int:
    return d.year * 10_000 + d.month * 100 + d.day


def int_to_date(n: int) -> date:
    return date(n // 10_000, n // 100 % 100, n % 100)


def to_metadata(c: ChunkRecord) -> dict[str, Any]:
    """Metadata for Pinecone: no ``None`` values, dates as ints, the chunk text as ``text``."""
    meta: dict[str, Any] = {}
    for key, value in asdict(c).items():
        if key in _NOT_METADATA or value is None:
            continue
        meta[key] = date_to_int(value) if isinstance(value, date) else value
    meta["text"] = c.content
    return meta


def from_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Pinecone metadata back into hit metadata: the text removed, ``issued_on`` an ISO date."""
    out = {k: v for k, v in meta.items() if k != "text"}
    if isinstance(out.get("issued_on"), int | float):
        out["issued_on"] = int_to_date(int(out["issued_on"])).isoformat()
    if isinstance(out.get("page"), float):  # Pinecone returns numbers as floats
        out["page"] = int(out["page"])
    return out


def to_pinecone_filter(filters: SearchFilters | None) -> dict[str, Any]:
    f = filters or SearchFilters()
    out: dict[str, Any] = {}
    if not f.include_suspicious:
        out["suspicious"] = {"$eq": False}
    if f.doc_types is not None:
        out["doc_type"] = {"$in": list(f.doc_types)}
    if f.act is not None:
        out["act"] = {"$eq": f.act}
    if f.sections is not None:
        out["section"] = {"$in": list(f.sections)}
    if f.issued_after is not None:
        out["issued_on"] = {"$gte": date_to_int(f.issued_after)}
    return out
