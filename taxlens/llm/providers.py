"""The one place that builds chat models: Gemini (primary), Groq (fallback), Ollama (local).

The rest of the code never imports a provider SDK. Provider SDKs are imported lazily, and API
keys are passed explicitly from Settings (values in .env are not exported to the environment).
A provider whose key is missing is skipped, so with only a Groq key the fallback serves alone.
See docs/01-architecture.md § 1.3 A.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import SecretStr

from taxlens.config import Provider, settings


class ProviderUnavailableError(RuntimeError):
    """No configured LLM provider can be built (missing keys or packages)."""


def _key(provider: Provider) -> str | None:
    secret = {"gemini": settings.google_api_key, "groq": settings.groq_api_key}.get(provider)
    return secret.get_secret_value() if secret is not None else None


def _build(provider: Provider, model: str) -> BaseChatModel:
    common: dict[str, Any] = {"temperature": 0, "max_retries": settings.llm_max_retries}
    if provider == "gemini":
        if not (key := _key("gemini")):
            raise ProviderUnavailableError("GOOGLE_API_KEY is not set")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, google_api_key=key, timeout=settings.llm_timeout_seconds, **common
        )
    if provider == "groq":
        if not (key := _key("groq")):
            raise ProviderUnavailableError("GROQ_API_KEY is not set")
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model, api_key=SecretStr(key), timeout=settings.llm_timeout_seconds, **common
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=0)
    raise ValueError(f"Unknown LLM provider: {provider}")


def _candidates(
    primary: tuple[Provider, str], fallback: tuple[Provider, str] | None
) -> list[BaseChatModel]:
    built, problems = [], []
    for provider, model in [primary, *([fallback] if fallback else [])]:
        try:
            built.append(_build(provider, model))
        except ProviderUnavailableError as exc:
            problems.append(f"{provider}: {exc}")
    if not built:
        raise ProviderUnavailableError("no LLM provider available (" + "; ".join(problems) + ")")
    return built


def _configured() -> tuple[tuple[Provider, str], tuple[Provider, str] | None]:
    fallback = settings.llm_fallback_provider
    return (
        (settings.llm_provider, settings.llm_model),
        (fallback, settings.llm_fallback_model) if fallback else None,
    )


def get_chat_model() -> Runnable[Any, Any]:
    """The answer model, with the fallback tried when the primary raises (429, timeout, ...)."""
    first, *rest = _candidates(*_configured())
    return first.with_fallbacks(rest) if rest else first


def get_structured_model(schema: type) -> Runnable[Any, Any]:
    """Like ``get_chat_model`` but returning ``schema`` instances. Structured output is applied to
    each provider before the fallback is chained (a fallback wrapper can't do it afterwards)."""
    first, *rest = (m.with_structured_output(schema) for m in _candidates(*_configured()))
    return first.with_fallbacks(rest) if rest else first


def get_judge_model() -> BaseChatModel:
    """The eval judge: a different provider/model from the answer model, no fallback."""
    return _candidates((settings.judge_provider, settings.judge_model), None)[0]


def model_name() -> str:
    """The primary model's name, reported with answers."""
    return settings.llm_model
