"""ingest_document and the seed script, with SQLite, FakeVectorStore and stub embeddings."""

import shutil
from datetime import date
from pathlib import Path

import pytest

from scripts import seed_index
from taxlens.ingestion import pipeline
from taxlens.storage.base import Repository
from taxlens.storage.models import PUBLIC_WORKSPACE_ID, Document
from tests.fakes import FakeVectorStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ACT = {"id": "sample-act", "title": "Sample Act", "doc_type": "act", "act": "Sample Act, 2025",
       "url": "https://example.gov.in/act.pdf"}  # fmt: skip


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts: [[1.0, 0.0] for _ in texts])


def add_doc(repo: Repository, pdf: Path, doc_type: str = "act") -> Document:
    seed_index.ensure_public_workspace(repo)
    return seed_index.register(repo, pdf, {"doc_type": doc_type, "title": pdf.stem})


def test_indexes_an_act_with_section_and_page_metadata(
    repo: Repository, vector_store: FakeVectorStore
) -> None:
    doc = add_doc(repo, FIXTURES / "sample_act.pdf")
    n = pipeline.ingest_document(repo, vector_store, PUBLIC_WORKSPACE_ID, doc.id, source=ACT)
    assert n > 0
    stored = repo.get_document(PUBLIC_WORKSPACE_ID, doc.id)
    assert stored is not None
    assert (stored.status, stored.chunk_count) == ("indexed", n)
    assert stored.dense_model and stored.sparse_model

    chunks = list(vector_store.by_workspace[PUBLIC_WORKSPACE_ID].values())
    assert len(chunks) == n
    assert all(c.id.startswith(f"{doc.id}#") and c.document_id == doc.id for c in chunks)
    by_section = {c.section: c for c in chunks}
    assert {"80C", "194J"} <= set(by_section)
    c80 = by_section["80C"]
    assert c80.act == "Sample Act, 2025" and c80.source_url == ACT["url"]
    assert (
        c80.page == 2 and c80.heading and c80.content.startswith("Sample Act, 2025 | Section 80C")
    )
    assert c80.issued_on is None  # Acts carry no date unless sources.yaml gives one


def test_is_idempotent_and_force_replaces_old_chunks(
    repo: Repository, vector_store: FakeVectorStore
) -> None:
    doc = add_doc(repo, FIXTURES / "sample_act.pdf")
    first = pipeline.ingest_document(repo, vector_store, PUBLIC_WORKSPACE_ID, doc.id, source=ACT)
    assert pipeline.ingest_document(repo, vector_store, PUBLIC_WORKSPACE_ID, doc.id) == 0
    stale = f"{doc.id}#99999"
    vector_store.by_workspace[PUBLIC_WORKSPACE_ID][stale] = next(
        iter(vector_store.by_workspace[PUBLIC_WORKSPACE_ID].values())
    )
    again = pipeline.ingest_document(
        repo, vector_store, PUBLIC_WORKSPACE_ID, doc.id, source=ACT, force=True
    )
    assert again == first
    assert stale not in vector_store.by_workspace[PUBLIC_WORKSPACE_ID]


def test_scanned_pdf_ends_needs_ocr(repo: Repository, vector_store: FakeVectorStore) -> None:
    doc = add_doc(repo, FIXTURES / "sample_circular.pdf", "circular")  # 1 of 3 pages blank
    assert pipeline.ingest_document(repo, vector_store, PUBLIC_WORKSPACE_ID, doc.id) == 0
    stored = repo.get_document(PUBLIC_WORKSPACE_ID, doc.id)
    assert stored is not None and stored.status == "needs_ocr"


def test_failure_marks_the_document_failed(
    repo: Repository, vector_store: FakeVectorStore, tmp_path: Path
) -> None:
    pdf = tmp_path / "gone.pdf"
    shutil.copy(FIXTURES / "sample_act.pdf", pdf)
    doc = add_doc(repo, pdf)
    pdf.unlink()
    with pytest.raises(FileNotFoundError):
        pipeline.ingest_document(repo, vector_store, PUBLIC_WORKSPACE_ID, doc.id)
    stored = repo.get_document(PUBLIC_WORKSPACE_ID, doc.id)
    assert stored is not None and stored.status == "failed" and "gone.pdf" in (stored.error or "")


def test_missing_document_raises(repo: Repository, vector_store: FakeVectorStore) -> None:
    with pytest.raises(LookupError):
        pipeline.ingest_document(repo, vector_store, PUBLIC_WORKSPACE_ID, "nope")


def test_source_metadata_prefers_the_curated_entry() -> None:
    text = "Circular No. 9/2025\nNew Delhi, dated 1st January, 2024"
    meta = pipeline.source_metadata(
        {"doc_type": "circular", "circular_no": "9/2025", "issued_on": date(2025, 7, 21)}, text
    )
    assert meta["circular_no"] == "9/2025" and meta["issued_on"] == date(2025, 7, 21)
    found = pipeline.source_metadata({"doc_type": "circular"}, text)
    assert found["issued_on"] == date(2024, 1, 1)  # from the text, as a date
    assert "issued_on" not in pipeline.source_metadata({"doc_type": "act"}, text)


def test_every_source_entry_has_the_fields_the_pipeline_needs() -> None:
    for sid, entry in pipeline.load_sources().items():
        assert entry["doc_type"] in ("act", "rules", "circular", "notification"), sid
        assert entry.get("title"), sid


def test_seed_registers_and_indexes_listed_pdfs_once(
    repo: Repository,
    vector_store: FakeVectorStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shutil.copy(FIXTURES / "sample_act.pdf", tmp_path / "sample-act.pdf")
    shutil.copy(FIXTURES / "sample_notification.pdf", tmp_path / "unlisted.pdf")
    monkeypatch.setattr(seed_index, "load_sources", lambda: {"sample-act": ACT, "absent": ACT})

    done = seed_index.seed(repo, vector_store, tmp_path)
    assert list(done) == ["sample-act"] and done["sample-act"] > 0
    assert "indexed" in capsys.readouterr().out
    assert seed_index.seed(repo, vector_store, tmp_path) == {"sample-act": 0}  # skipped
    docs = repo.list_documents(PUBLIC_WORKSPACE_ID)
    assert len(docs) == 1 and docs[0].blob_path.endswith("sample-act.pdf")
    seed_index.wait_until_searchable(vector_store, 0)  # FakeVectorStore has no count: no-op
