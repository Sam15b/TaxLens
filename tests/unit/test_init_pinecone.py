"""scripts/init_pinecone.py against a fake control-plane client: no network."""

from types import SimpleNamespace
from typing import Any

import pytest

from scripts import init_pinecone


class FakePinecone:
    def __init__(self, existing: bool, ready_after: int = 0) -> None:
        self.indexes: dict[str, dict[str, Any]] = {"taxlens-dev": {}} if existing else {}
        self.polls_until_ready = ready_after

    def has_index(self, name: str) -> bool:
        return name in self.indexes

    def create_index(self, *, name: str, **kwargs: Any) -> None:
        self.indexes[name] = kwargs

    def describe_index(self, name: str) -> SimpleNamespace:
        self.polls_until_ready -= 1
        return SimpleNamespace(status=SimpleNamespace(ready=self.polls_until_ready < 0))


def ensure(pc: FakePinecone, **kw: Any) -> str:
    return init_pinecone.ensure_index(
        pc, "taxlens-dev", dimension=384, cloud="aws", region="us-east-1", poll=0, **kw
    )


def test_creates_a_dotproduct_serverless_index_then_waits_until_ready() -> None:
    pc = FakePinecone(existing=False, ready_after=2)
    assert ensure(pc) == "created"
    spec = pc.indexes["taxlens-dev"]
    assert (spec["dimension"], spec["metric"]) == (384, "dotproduct")
    assert (spec["spec"].cloud, spec["spec"].region) == ("aws", "us-east-1")
    assert pc.polls_until_ready < 0


def test_second_run_reports_already_exists() -> None:
    pc = FakePinecone(existing=False)
    assert ensure(pc) == "created"
    assert ensure(pc) == "exists"


def test_times_out_when_never_ready() -> None:
    with pytest.raises(TimeoutError):
        ensure(FakePinecone(existing=True, ready_after=10**9), timeout=0)


def test_missing_api_key_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(init_pinecone.settings, "pinecone_api_key", None)
    assert init_pinecone.main([]) == 2
    assert "PINECONE_API_KEY" in capsys.readouterr().err
