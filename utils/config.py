"""
app/utils/config.py
Centralised configuration via pydantic-settings.
All values come from environment variables or .env file.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    openai_chat_model: str = Field("gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_max_tokens: int = Field(1500, alias="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(0.2, alias="OPENAI_TEMPERATURE")

    # ── AWS S3 ────────────────────────────────────────────────────────────
    aws_access_key_id: str = Field("", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field("", alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field("us-east-1", alias="AWS_REGION")
    s3_bucket_name: str = Field("rag-document-assistant", alias="S3_BUCKET_NAME")
    s3_prefix: str = Field("documents/", alias="S3_PREFIX")

    # ── FAISS / Chunking ──────────────────────────────────────────────────
    faiss_index_path: str = Field("./data/faiss_index", alias="FAISS_INDEX_PATH")
    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP")

    # ── App ───────────────────────────────────────────────────────────────
    app_name: str = Field("RAG Document Assistant", alias="APP_NAME")
    app_env: str = Field("development", alias="APP_ENV")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    max_upload_size_mb: int = Field(50, alias="MAX_UPLOAD_SIZE_MB")
    max_history_messages: int = Field(10, alias="MAX_HISTORY_MESSAGES")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "populate_by_name": True,
    }

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
