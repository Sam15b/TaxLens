import pytest
from pydantic import ValidationError

from taxlens.config import Settings


def make_settings() -> Settings:
    # Ignore any developer .env so tests only see defaults + monkeypatched env vars.
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_PROVIDER", "VECTOR_BACKEND", "DB_BACKEND", "HYBRID_ALPHA", "PINECONE_INDEX"):
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_free_tier_choices() -> None:
    s = make_settings()
    assert s.llm_provider == "gemini"
    assert s.llm_fallback_provider == "groq"
    assert s.vector_backend == "pinecone"
    assert s.pinecone_index == "taxlens-dev"
    assert s.db_backend == "sqlite"
    assert s.embedding_dim == 384


def test_judge_model_differs_from_answer_model() -> None:
    s = make_settings()
    assert (s.judge_provider, s.judge_model) != (s.llm_provider, s.llm_model)


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYBRID_ALPHA", "0.3")
    monkeypatch.setenv("DB_BACKEND", "cosmos")
    s = make_settings()
    assert s.hybrid_alpha == pytest.approx(0.3)
    assert s.db_backend == "cosmos"


def test_invalid_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_BACKEND", "postgres")
    with pytest.raises(ValidationError):
        make_settings()


def test_secrets_are_not_exposed_in_repr() -> None:
    s = make_settings()
    assert "change-me" not in repr(s)
