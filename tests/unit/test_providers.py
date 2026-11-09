"""The provider factory, with fake models: no API calls."""

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, SecretStr

from taxlens.llm import providers


class Broken(FakeListChatModel):
    """A chat model that always fails, like a provider returning 429."""

    def _call(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("429 Too Many Requests")


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers.settings, "google_api_key", SecretStr("g-test"))
    monkeypatch.setattr(providers.settings, "groq_api_key", SecretStr("q-test"))
    monkeypatch.setattr(providers.settings, "llm_provider", "gemini")
    monkeypatch.setattr(providers.settings, "llm_fallback_provider", "groq")


def fake_builds(monkeypatch: pytest.MonkeyPatch, models: dict[str, Any]) -> None:
    def build(provider: str, model: str) -> Any:
        if models.get(provider) is None:
            raise providers.ProviderUnavailableError(f"{provider} key missing")
        return models[provider]

    monkeypatch.setattr(providers, "_build", build)


@pytest.mark.usefixtures("keys")
def test_fallback_answers_when_the_primary_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_builds(
        monkeypatch,
        {"gemini": Broken(responses=["never"]), "groq": FakeListChatModel(responses=["from groq"])},
    )
    reply = providers.get_chat_model().invoke("hi")
    assert isinstance(reply, AIMessage) and reply.content == "from groq"


@pytest.mark.usefixtures("keys")
def test_primary_answers_when_it_works(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_builds(
        monkeypatch,
        {"gemini": FakeListChatModel(responses=["gemini"]), "groq": Broken(responses=["x"])},
    )
    assert providers.get_chat_model().invoke("hi").content == "gemini"


@pytest.mark.usefixtures("keys")
def test_a_provider_without_a_key_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_builds(monkeypatch, {"gemini": None, "groq": FakeListChatModel(responses=["only groq"])})
    assert providers.get_chat_model().invoke("hi").content == "only groq"
    fake_builds(monkeypatch, {})
    with pytest.raises(providers.ProviderUnavailableError, match="gemini.*groq"):
        providers.get_chat_model()


class Answer(BaseModel):
    value: int


class Structured:
    """Just enough of a chat model for with_structured_output."""

    def __init__(self, value: int | None) -> None:
        self.value = value

    def with_structured_output(self, schema: type[BaseModel]) -> RunnableLambda[Any, Any]:
        def run(_: Any) -> BaseModel:
            if self.value is None:
                raise RuntimeError("503")
            return schema(value=self.value)

        return RunnableLambda(run)


@pytest.mark.usefixtures("keys")
def test_structured_output_is_applied_per_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_builds(monkeypatch, {"gemini": Structured(None), "groq": Structured(7)})
    assert providers.get_structured_model(Answer).invoke("q") == Answer(value=7)


@pytest.mark.usefixtures("keys")
def test_build_passes_keys_and_temperature_zero() -> None:
    gemini = providers._build("gemini", "gemini-2.5-flash")
    groq = providers._build("groq", "llama-3.3-70b-versatile")
    assert type(gemini).__name__ == "ChatGoogleGenerativeAI" and gemini.temperature == 0
    assert type(groq).__name__ == "ChatGroq" and groq.temperature < 1e-6  # Groq stores 0 as 1e-8
    assert groq.groq_api_key is not None and groq.groq_api_key.get_secret_value() == "q-test"
    with pytest.raises(ValueError):
        providers._build("openai", "gpt")  # type: ignore[arg-type]


def test_missing_keys_make_the_provider_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers.settings, "google_api_key", None)
    monkeypatch.setattr(providers.settings, "groq_api_key", None)
    for provider in ("gemini", "groq"):
        with pytest.raises(providers.ProviderUnavailableError):
            providers._build(provider, "m")  # type: ignore[arg-type]
