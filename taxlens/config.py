"""Application settings. This is the ONLY module that reads environment variables.

See docs/02-project-structure.md § 2.3.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["gemini", "groq", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: Literal["local", "ci", "prod"] = "local"
    log_level: str = "INFO"

    # ── LLM (free tiers: Gemini API, Groq; Ollama is local) ──────────
    llm_provider: Provider = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_fallback_provider: Provider | None = "groq"
    llm_fallback_model: str = "llama-3.3-70b-versatile"
    judge_provider: Provider = "groq"  # eval judge must differ from the answer model
    judge_model: str = "llama-3.3-70b-versatile"
    ollama_base_url: str = "http://localhost:11434"
    google_api_key: SecretStr | None = None  # Gemini API
    groq_api_key: SecretStr | None = None
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2  # per provider, before the fallback is tried

    # ── Embeddings (local, FastEmbed) ────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    sparse_model: str = "prithivida/Splade_PP_en_v1"
    fastembed_cache_path: str = ".models"  # ONNX model files, downloaded on first use

    # ── Vector store (Pinecone Starter plan) ─────────────────────────
    vector_backend: Literal["pinecone", "qdrant", "oracle"] = "pinecone"
    # Passed to the SDK explicitly: values in .env are not exported to the process environment.
    pinecone_api_key: SecretStr | None = None
    pinecone_index: str = "taxlens-dev"
    pinecone_cloud: str = "aws"  # the Starter plan offers one cloud/region
    pinecone_region: str = "us-east-1"
    pinecone_namespace_prefix: str = ""  # CI sets "ci-<run_id>-"
    hybrid_alpha: float = 0.5  # 1 = dense only, 0 = sparse only
    retrieval_k: int = 8
    qdrant_url: str = "http://localhost:6333"
    oracle_dsn: str = "localhost:1521/FREEPDB1"
    oracle_user: str = "taxlens"
    oracle_password: SecretStr = SecretStr("taxlens")

    # ── App data: SQLite locally / CI, Cosmos DB (free tier) in Azure ─
    db_backend: Literal["sqlite", "cosmos"] = "sqlite"
    sqlite_path: str = "var/taxlens.db"
    cosmos_endpoint: str | None = None
    cosmos_key: SecretStr | None = None  # emulator only; Azure uses managed identity
    cosmos_database: str = "taxlens"

    # ── Blob + Queue (Azurite locally) ───────────────────────────────
    azure_storage_connection_string: SecretStr | None = None
    azure_storage_account_url: str | None = None
    azure_storage_queue_url: str | None = None
    blob_container: str = "documents"
    ingest_queue: str = "ingest-jobs"
    poison_queue: str = "ingest-poison"
    max_upload_mb: int = 25

    # ── Auth ─────────────────────────────────────────────────────────
    jwt_secret: SecretStr = SecretStr("change-me")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # ── Observability ────────────────────────────────────────────────
    phoenix_collector_endpoint: str | None = "http://localhost:6006/v1/traces"
    langsmith_project: str = "taxlens"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
