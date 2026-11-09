"""PineconeStore against the real index, in a throwaway namespace.

    uv run pytest -m integration tests/integration/test_pinecone_store.py -q

Needs PINECONE_API_KEY and an index (scripts/init_pinecone.py). Upserts are eventually
consistent, so the fixture polls the namespace's vector count before searching, and deletes the
namespace afterwards.
"""

import time
from collections.abc import Iterator
from uuid import uuid4

import pytest

from taxlens.config import settings
from taxlens.llm.embeddings import embed_query, embed_texts
from taxlens.retrieval import pinecone_store as ps
from taxlens.retrieval.base import ChunkRecord, SearchFilters, chunk_id

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(settings.pinecone_api_key is None, reason="PINECONE_API_KEY not set"),
]

POLL_SECONDS, POLL_TIMEOUT = 2.0, 90.0

TEXTS = {
    ("ita", "194J"): "Section 194J: Fees for professional or technical services. Any person paying "
    "fees for professional services to a resident shall deduct tax at source.",
    ("ita", "80C"): "Section 80C: Deduction in respect of life insurance premia, contributions to "
    "provident fund and subscription to certain equity shares.",
    ("gst", None): "Notification No. 12/2017-Central Tax (Rate): exemption for services by way of "
    "pure labour contracts and services provided by a zoo.",
}


@pytest.fixture(scope="module")
def store() -> Iterator[ps.PineconeStore]:
    old_prefix = settings.pinecone_namespace_prefix
    settings.pinecone_namespace_prefix = f"test-{uuid4().hex[:8]}-"
    s = ps.PineconeStore.from_settings()
    try:
        vectors = embed_texts(list(TEXTS.values()))
        s.upsert(
            [
                ChunkRecord(
                    id=chunk_id(doc, n),
                    workspace_id="public",
                    document_id=doc,
                    content=text,
                    embedding=vec,
                    doc_type="act" if doc == "ita" else "notification",
                    section=section,
                )
                for n, ((doc, section), text), vec in zip(
                    range(len(TEXTS)), TEXTS.items(), vectors, strict=True
                )
            ]
        )
        deadline = time.monotonic() + POLL_TIMEOUT
        while s.count("public") < len(TEXTS):
            assert time.monotonic() < deadline, "upserted vectors never became visible"
            time.sleep(POLL_SECONDS)
        yield s
    finally:
        s.delete_namespace("public")
        settings.pinecone_namespace_prefix = old_prefix


def search(store: ps.PineconeStore, q: str, **kw: object) -> list[str]:
    hits = store.hybrid_search(
        query_text=q,
        query_vector=embed_query(q),
        workspace_ids=["public"],
        **kw,  # type: ignore[arg-type]
    )
    return [h.chunk_id for h in hits]


def test_hybrid_search_ranks_the_matching_section_first(store: ps.PineconeStore) -> None:
    assert search(store, "TDS on professional fees")[0] == chunk_id("ita", 0)
    assert search(store, "deduction for life insurance premium u/s 80C")[0] == chunk_id("ita", 1)


def test_filters_and_other_workspaces(store: ps.PineconeStore) -> None:
    only_gst = SearchFilters(doc_types=["notification"])
    assert search(store, "professional fees", filters=only_gst) == [chunk_id("gst", 2)]
    assert (
        store.hybrid_search(
            query_text="fees", query_vector=embed_query("fees"), workspace_ids=["someone-else"]
        )
        == []
    )


def test_delete_document_by_prefix(store: ps.PineconeStore) -> None:
    assert store.delete_document(document_id="gst", workspace_id="public") == 1
