"""scripts/download_sources.py against a mock transport: no network."""

import hashlib
from pathlib import Path

import httpx
import pytest

from scripts import download_sources as ds

PDF = b"%PDF-1.7\nfake body\n%%EOF\n"
PDF_SHA = hashlib.sha256(PDF).hexdigest()


def no_sleep(_: float) -> None:
    pass


def client_for(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def serve(body: bytes = PDF, status: int = 200) -> tuple[httpx.MockTransport, list[str]]:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(status, content=body)

    return httpx.MockTransport(handler), calls


def source(**over: object) -> dict[str, object]:
    return {"id": "doc", "url": "https://example.test/doc.pdf", "sha256": None, **over}


def test_downloads_then_skips_on_second_run(tmp_path: Path) -> None:
    transport, calls = serve()
    with client_for(transport) as c:
        first = ds.process(source(sha256=PDF_SHA), tmp_path, c, update_hashes=False)
        second = ds.process(source(sha256=PDF_SHA), tmp_path, c, update_hashes=False)
    assert (first.status, second.status) == ("downloaded", "skipped")
    assert len(calls) == 1
    assert (tmp_path / "doc.pdf").read_bytes() == PDF


def test_hash_mismatch_is_an_error_and_leaves_no_file(tmp_path: Path) -> None:
    transport, _ = serve()
    with client_for(transport) as c:
        r = ds.process(source(sha256="0" * 64), tmp_path, c, update_hashes=False)
    assert r.status == "mismatch"
    assert not (tmp_path / "doc.pdf").exists()


def test_update_hashes_accepts_new_content(tmp_path: Path) -> None:
    transport, _ = serve()
    with client_for(transport) as c:
        r = ds.process(source(sha256="0" * 64), tmp_path, c, update_hashes=True)
    assert (r.status, r.sha256) == ("downloaded", PDF_SHA)


def test_html_error_page_is_rejected(tmp_path: Path) -> None:
    transport, _ = serve(body=b"<html>blocked</html>")
    with client_for(transport) as c:
        r = ds.process(source(), tmp_path, c, update_hashes=False)
    assert r.status == "failed"
    assert "not a PDF" in r.detail
    assert list(tmp_path.iterdir()) == []


def test_403_is_not_retried(tmp_path: Path) -> None:
    transport, calls = serve(status=403)
    with client_for(transport) as c:
        r = ds.process(source(), tmp_path, c, update_hashes=False, sleep=no_sleep)
    assert r.status == "failed"
    assert len(calls) == 1


def test_5xx_is_retried_with_backoff(tmp_path: Path) -> None:
    transport, calls = serve(status=503)
    with client_for(transport) as c:
        r = ds.process(source(), tmp_path, c, update_hashes=False, sleep=no_sleep)
    assert r.status == "failed"
    assert len(calls) == ds.RETRIES


def test_manual_entries_are_never_fetched(tmp_path: Path) -> None:
    transport, calls = serve()
    with client_for(transport) as c:
        missing = ds.process(source(manual=True), tmp_path, c, update_hashes=False)
        (tmp_path / "doc.pdf").write_bytes(PDF)
        present = ds.process(source(manual=True, sha256=PDF_SHA), tmp_path, c, update_hashes=False)
        wrong = ds.process(source(manual=True, sha256="0" * 64), tmp_path, c, update_hashes=False)
    assert (missing.status, present.status, wrong.status) == ("manual", "skipped", "mismatch")
    assert calls == []
    assert (tmp_path / "doc.pdf").exists()  # a hand-saved file is never deleted


def test_write_hashes_keeps_comments_and_other_entries(tmp_path: Path) -> None:
    y = tmp_path / "sources.yaml"
    y.write_text(
        "# keep me\n- id: a\n  url: u\n  sha256: null\n\n- id: b\n  url: u\n  sha256: null\n",
        encoding="utf-8",
    )
    assert ds.write_hashes(y, {"b": PDF_SHA}) == 1
    text = y.read_bytes().decode()
    assert text.startswith("# keep me\n")
    assert "- id: a\n  url: u\n  sha256: null\n" in text
    assert f"- id: b\n  url: u\n  sha256: {PDF_SHA}\n" in text
    assert "\r\n" not in text


def test_subset_ids_reads_sources_section(tmp_path: Path) -> None:
    y = tmp_path / "subsets.yaml"
    y.write_text("ci: [q001]\nsources:\n  ci: [a, b]\n", encoding="utf-8")
    assert ds.subset_ids("ci", y) == {"a", "b"}
    with pytest.raises(SystemExit):
        ds.subset_ids("nightly", y)


def test_repo_manifest_is_consistent() -> None:
    sources = ds.load_sources()
    ids = {s["id"] for s in sources}
    assert ds.subset_ids("ci") <= ids
    for s in sources:
        assert s["doc_type"] in {"act", "rules", "circular", "notification"}
        assert s["url"].startswith("https://")
        assert s["sha256"] is None or len(s["sha256"]) == 64
    ci = [s for s in sources if s["id"] in ds.subset_ids("ci")]
    assert all(not s.get("manual") and s["sha256"] for s in ci)
