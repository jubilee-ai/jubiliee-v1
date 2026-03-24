from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Optional

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=True)


class Settings(BaseSettings):
    """
    Centralized app settings.
    Use this as the single env/config entrypoint instead of scattered os.getenv.
    """

    # Core env vars
    OPENAI_API_KEY: Optional[SecretStr] = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY"),
    )

    # POST /api/chat — LLM classifies training graph vs orchestrator when mode is unset
    CHAT_INTENT_ROUTER_MODEL: str = Field(
        default="openai:gpt-5.4-mini",
        description="LangChain init_chat_model id for HTTP-layer chat intent (override if unavailable)",
    )

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///jubilee.db",
        description="PostgreSQL (or SQLite fallback) connection URL",
    )

    # Redis / Celery
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for Celery broker and result backend",
    )

    # R2 / S3-compatible object storage (model weights)
    R2_ENDPOINT_URL: Optional[str] = Field(
        default=None,
        description="Cloudflare R2 S3-compatible endpoint URL",
    )
    R2_ACCESS_KEY_ID: Optional[str] = Field(
        default=None,
        description="R2 API token access key ID",
    )
    R2_SECRET_ACCESS_KEY: Optional[SecretStr] = Field(
        default=None,
        description="R2 API token secret access key",
    )
    R2_BUCKET_NAME: str = Field(
        default="jubilee-models",
        description="R2 bucket name for model weights",
    )

    # App config
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:5175",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://127.0.0.1:5175",
        ]
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        str_strip_whitespace=True,
        case_sensitive=False,
    )

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def env_file(self) -> Path:
        return self.project_root / ".env"

    @property
    def data_tools_dir(self) -> Path:
        return self.project_root / "tools" / "data-tools"

    @property
    def datasets_dir(self) -> Path:
        return self.project_root / "datasets"

    @property
    def datasets_catalog_path(self) -> Path:
        return self.datasets_dir / "catalog.json"

    @property
    def models_registry_path(self) -> Path:
        return self.project_root / "trained_models" / "registry.json"

    @property
    def cors_origins(self) -> list[str]:
        return self.CORS_ORIGINS

    @property
    def openai_api_key(self) -> Optional[str]:
        return (
            self.OPENAI_API_KEY.get_secret_value()
            if self.OPENAI_API_KEY is not None
            else None
        )

    @property
    def r2_enabled(self) -> bool:
        return bool(
            self.R2_ENDPOINT_URL
            and self.R2_ACCESS_KEY_ID
            and self.R2_SECRET_ACCESS_KEY
        )

    @property
    def trained_models_dir(self) -> Path:
        return self.project_root / "trained_models"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def bootstrap_paths() -> None:
    settings = get_settings()
    project_root = str(settings.project_root)
    data_tools_dir = str(settings.data_tools_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if data_tools_dir not in sys.path:
        sys.path.insert(0, data_tools_dir)


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Centralized env access.
    Prefer this over scattered os.getenv calls in backend code.
    """
    # Ensure .env is loaded before env reads.
    get_settings()
    return os.environ.get(name, default)


def require_env(name: str) -> str:
    """
    Return required env var or raise a helpful error.
    """
    value = get_env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
