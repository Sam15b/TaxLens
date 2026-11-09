"""The FakeVectorStore honours the VectorStore contract that unit tests rely on."""

from datetime import date

from taxlens.retrieval.base import ChunkRecord, SearchFilters, VectorStore, chunk_id
from tests.fakes import FakeVectorStore


def record(doc: str, n: int, ws: str, vec: list[float], **kw: object) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id(doc, n),
        workspace_id=ws,
        document_id=doc,
        content=f"{doc} chunk {n}",
        embedding=vec,
        doc_type=str(kw.pop("doc_type", "act")),
        **kw,  # type: ignore[arg-type]
    )


def test_chunk_id_is_zero_padded_with_a_document_prefix() -> None:
    assert chunk_id("doc-1", 7) == "doc-1#00007"


def test_search_only_sees_the_given_workspaces(vector_store: FakeVectorStore) -> None:
    store: VectorStore = vector_store
    store.upsert(
        [
            record("pub", 0, "public", [1.0, 0.0]),
            record("cli", 0, "ws-a", [1.0, 0.0]),
            record("other", 0, "ws-b", [1.0, 0.0]),
        ]
    )
    hits = store.hybrid_search(
        query_text="q", query_vector=[1.0, 0.0], workspace_ids=["public", "ws-a"]
    )
    assert sorted(h.document_id for h in hits) == ["cli", "pub"]
    assert store.hybrid_search(query_text="q", query_vector=[1.0, 0.0], workspace_ids=[]) == []


def test_ranking_filters_and_suspicious_chunks(vector_store: FakeVectorStore) -> None:
    vector_store.upsert(
        [
            record("a", 0, "public", [1.0, 0.0], section="194J", issued_on=date(2025, 3, 1)),
            record("a", 1, "public", [0.6, 0.8], section="80C"),
            record("a", 2, "public", [1.0, 0.0], suspicious=True),
            record("c", 0, "public", [0.0, 1.0], doc_type="circular"),
        ]
    )
    search = vector_store.hybrid_search
    hits = search(query_text="q", query_vector=[1.0, 0.0], workspace_ids=["public"], k=2)
    assert [h.chunk_id for h in hits] == ["a#00000", "a#00001"]  # suspicious one left out
    assert hits[0].metadata["section"] == "194J" and "embedding" not in hits[0].metadata

    only = SearchFilters(sections=["80C"])
    assert [
        h.chunk_id
        for h in search(
            query_text="q", query_vector=[1.0, 0.0], workspace_ids=["public"], filters=only
        )
    ] == ["a#00001"]
    circ = SearchFilters(doc_types=["circular"])
    assert [
        h.document_id
        for h in search(
            query_text="q", query_vector=[1.0, 0.0], workspace_ids=["public"], filters=circ
        )
    ] == ["c"]
    after = SearchFilters(issued_after=date(2025, 1, 1))
    assert [
        h.chunk_id
        for h in search(
            query_text="q", query_vector=[1.0, 0.0], workspace_ids=["public"], filters=after
        )
    ] == ["a#00000"]


def test_delete_document_removes_its_chunks_only(vector_store: FakeVectorStore) -> None:
    vector_store.upsert(
        [
            record("a", 0, "public", [1.0]),
            record("a", 1, "public", [1.0]),
            record("ab", 0, "public", [1.0]),
        ]
    )
    assert vector_store.delete_document(document_id="a", workspace_id="public") == 2
    assert vector_store.delete_document(document_id="a", workspace_id="ws-x") == 0
    assert list(vector_store.by_workspace["public"]) == ["ab#00000"]
