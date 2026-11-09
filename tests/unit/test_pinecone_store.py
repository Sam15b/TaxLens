"""PineconeStore and its metadata/filter helpers against a fake index: no network, no SPLADE."""

from collections.abc import Sequence
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from taxlens.retrieval import pinecone_store as ps
from taxlens.retrieval.base import ChunkRecord, SearchFilters, chunk_id
from taxlens.retrieval.pinecone_filters import (
    from_metadata,
    int_to_date,
    to_metadata,
    to_pinecone_filter,
)


def test_scale_weights_dense_by_alpha_and_sparse_by_one_minus_alpha() -> None:
    dense, sparse = ps.scale([1.0, 2.0], {"indices": [3, 7], "values": [1.0, 0.5]}, 0.25)
    assert dense == [0.25, 0.5]
    assert sparse == {"indices": [3, 7], "values": [0.75, 0.375]}
    assert ps.scale([1.0], {"indices": [1], "values": [1.0]}, 1.0)[1]["values"] == [0.0]
    with pytest.raises(ValueError):
        ps.scale([1.0], {"indices": [], "values": []}, 1.5)


def test_default_filter_only_excludes_suspicious_chunks() -> None:
    assert to_pinecone_filter(None) == {"suspicious": {"$eq": False}}
    assert to_pinecone_filter(SearchFilters(include_suspicious=True)) == {}


def test_filter_translation() -> None:
    f = SearchFilters(
        doc_types=["act", "circular"],
        act="Income-tax Act, 1961",
        sections=["80C"],
        issued_after=date(2025, 3, 15),
    )
    assert to_pinecone_filter(f) == {
        "suspicious": {"$eq": False},
        "doc_type": {"$in": ["act", "circular"]},
        "act": {"$eq": "Income-tax Act, 1961"},
        "section": {"$in": ["80C"]},
        "issued_on": {"$gte": 20250315},
    }


def make_record(doc: str = "doc", n: int = 0, ws: str = "public", **kw: Any) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id(doc, n),
        workspace_id=ws,
        document_id=doc,
        content=f"text of {doc} {n}",
        embedding=[1.0, 0.0],
        doc_type=kw.pop("doc_type", "act"),
        **kw,
    )


def test_metadata_has_no_nulls_int_dates_and_the_text() -> None:
    meta = to_metadata(make_record(section="80C", issued_on=date(2025, 3, 15), page=4))
    assert meta == {
        "workspace_id": "public",
        "document_id": "doc",
        "doc_type": "act",
        "section": "80C",
        "issued_on": 20250315,
        "page": 4,
        "suspicious": False,
        "text": "text of doc 0",
    }
    assert int_to_date(20250315) == date(2025, 3, 15)
    back = from_metadata({**meta, "page": 4.0, "issued_on": 20250315.0})
    assert back["issued_on"] == "2025-03-15" and back["page"] == 4 and "text" not in back


def test_namespaces_and_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ps.namespace_for("public") == "public"
    assert ps.namespace_for("abc") == "ws-abc"
    monkeypatch.setattr(ps.settings, "pinecone_namespace_prefix", "ci-42-")
    assert ps.namespace_for("public") == "ci-42-public"
    assert ps.namespace_for("abc") == "ci-42-ws-abc"


# ── the store, with a fake index ─────────────────────────────────────────────


class FakeIndex:
    """Stores upserted records per namespace; ``query`` returns preset rankings."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, dict[str, Any]]] = {}
        self.upsert_calls: list[int] = []
        self.queries: list[dict[str, Any]] = []
        self.results: dict[tuple[str, bool], list[tuple[str, float]]] = {}

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str, **_: Any) -> None:
        self.upsert_calls.append(len(vectors))
        for v in vectors:
            self.records.setdefault(namespace, {})[v["id"]] = v

    def query(self, *, namespace: str, filter: dict[str, Any] | None, **kw: Any) -> Any:
        self.queries.append({"namespace": namespace, "filter": filter, **kw})
        boosted = bool(filter and "section" in filter)
        ranked = self.results.get((namespace, boosted), [])
        return SimpleNamespace(
            matches=[
                SimpleNamespace(id=i, score=s, metadata={"document_id": i.split("#")[0], "text": i})
                for i, s in ranked[: kw["top_k"]]
            ]
        )

    def list(self, *, prefix: str, namespace: str) -> Any:
        ids = sorted(i for i in self.records.get(namespace, {}) if i.startswith(prefix))
        for start in range(0, len(ids), 2):  # pages of 2
            yield SimpleNamespace(vectors=[SimpleNamespace(id=i) for i in ids[start : start + 2]])

    def delete(self, *, ids: Sequence[str] = (), namespace: str, delete_all: bool = False) -> None:
        if delete_all:
            self.records.pop(namespace, None)
        for i in ids:
            del self.records[namespace][i]

    def describe_index_stats(self) -> Any:
        counts = {ns: SimpleNamespace(vector_count=len(r)) for ns, r in self.records.items()}
        return SimpleNamespace(namespaces=counts)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> ps.PineconeStore:
    s = ps.PineconeStore(FakeIndex(), alpha=0.5)
    monkeypatch.setattr(
        s, "sparse", lambda texts: [{"indices": [1], "values": [1.0]} for _ in texts]
    )
    return s


def fake(store: ps.PineconeStore) -> FakeIndex:
    assert isinstance(store.index, FakeIndex)
    return store.index


def test_upsert_groups_by_namespace_and_batches(store: ps.PineconeStore) -> None:
    store.upsert([make_record(n=i) for i in range(250)] + [make_record("cli", ws="w1")])
    idx = fake(store)
    assert sorted(idx.upsert_calls) == [1, 50, 100, 100]
    assert len(idx.records["public"]) == 250 and list(idx.records["ws-w1"]) == ["cli#00000"]
    rec = idx.records["public"]["doc#00007"]
    assert rec["sparse_values"] == {"indices": [1], "values": [1.0]}
    assert rec["metadata"]["text"] == "text of doc 7" and rec["values"] == [1.0, 0.0]
    assert store.count("public") == 250 and store.count("nobody") == 0


def test_delete_document_by_id_prefix(store: ps.PineconeStore) -> None:
    store.upsert([make_record("a", i) for i in range(5)] + [make_record("ab", 0)])
    assert store.delete_document(document_id="a", workspace_id="public") == 5
    assert list(fake(store).records["public"]) == ["ab#00000"]
    store.delete_namespace("public")
    assert store.count("public") == 0


def test_search_merges_namespaces_by_score_and_scales_the_query(store: ps.PineconeStore) -> None:
    idx = fake(store)
    idx.results[("public", False)] = [("p#00000", 0.9), ("p#00001", 0.5)]
    idx.results[("ws-w1", False)] = [("c#00000", 0.7)]
    hits = store.hybrid_search(
        query_text="what is a deduction",
        query_vector=[1.0, 1.0],
        workspace_ids=["w1", "public"],
        k=3,
    )
    assert [h.chunk_id for h in hits] == ["p#00000", "c#00000", "p#00001"]
    assert hits[0].document_id == "p" and hits[0].content == "p#00000"
    q = idx.queries[0]
    assert q["vector"] == [0.5, 0.5] and q["sparse_vector"] == {"indices": [1], "values": [0.5]}
    assert q["filter"] == {"suspicious": {"$eq": False}} and q["include_metadata"] is True
    assert len(idx.queries) == 2  # one per namespace, no section boost


def test_section_boost_query_is_fused_with_rrf(store: ps.PineconeStore) -> None:
    idx = fake(store)
    idx.results[("public", False)] = [(f"x#{i:05d}", 1 - i / 100) for i in range(30)]
    idx.results[("public", True)] = [("s#00000", 0.2)]  # missed by the main query
    hits = store.hybrid_search(
        query_text="TDS under section 194J", query_vector=[1.0], workspace_ids=["public"], k=3
    )
    assert [h.chunk_id for h in hits] == ["x#00000", "s#00000", "x#00001"]
    boost = idx.queries[-1]
    assert boost["filter"]["section"] == {"$in": ["194J"]} and boost["top_k"] == ps.BOOST_TOP_K


def test_no_workspaces_means_no_query(store: ps.PineconeStore) -> None:
    assert store.hybrid_search(query_text="q", query_vector=[1.0], workspace_ids=[]) == []
    assert fake(store).queries == []


def test_from_settings_needs_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps.settings, "pinecone_api_key", None)
    with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
        ps.PineconeStore.from_settings()
