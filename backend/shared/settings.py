import os
import sys
from functools import lru_cache
from pathlib import Path
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
        default="openai:gpt-5.4",
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

    # Training / scaling (Railway and production workers)
    TRAINING_REQUIRE_CELERY: bool = Field(
        default=False,
        description="When True, /api/train requires Redis+Celery; no in-process thread fallback.",
    )
    FEATURE_EXPERIMENT_ENABLED: bool = Field(
        default=True,
        description="Enable parallel feature scout grid; disable under memory pressure.",
    )
    FEATURE_SCOUT_MAX_WORKERS: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Max process pool workers for feature scout jobs.",
    )
    FEATURE_SCOUT_MAX_VARIANTS: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Cap feature variants in the scout grid (reduces memory).",
    )
    FEATURE_SCOUT_RF_N_JOBS: int = Field(
        default=1,
        ge=1,
        le=64,
        description="RandomForest n_jobs inside scout workers (use 1 to avoid nested oversubscription).",
    )
    FEATURE_EXPERIMENT_SKIP_ABOVE_ROWS: int = Field(
        default=200_000,
        ge=0,
        description="Skip scout grid when raw training rows exceed this (0 disables this skip).",
    )
    MAX_PARALLEL_BATCH_TRAIN: int = Field(
        default=1,
        ge=1,
        le=4,
        description="ThreadPool max workers for batch_train_with_skill.",
    )
    MAX_PARALLEL_MODEL_EVAL: int = Field(
        default=1,
        ge=1,
        le=4,
        description="ThreadPool max workers for tool_evaluate_models in the simple agent.",
    )
    TRAINING_STATE_BLOB_MIN_BYTES: int = Field(
        default=65536,
        ge=4096,
        description="When object storage is enabled, persist full training state snapshots "
        "larger than this threshold and keep only compact data plus a storage key in Postgres.",
    )

    # Clerk authentication
    CLERK_JWKS_URL: Optional[str] = Field(
        default=None,
        description="Clerk JWKS endpoint URL, e.g. https://<frontend-api>.clerk.accounts.dev/.well-known/jwks.json",
    )
    CLERK_SECRET_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Clerk secret key for backend SDK calls (member lookups, etc.)",
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

    @property
    def training_require_celery_effective(self) -> bool:
        """True if thread fallback for training must not be used."""
        if self.TRAINING_REQUIRE_CELERY:
            return True
        return bool(
            os.environ.get("RAILWAY_ENVIRONMENT")
            or os.environ.get("RAILWAY_PROJECT_ID")
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def bootstrap_paths() -> None:
    settings = get_settings()
    project_root = str(settings.project_root)
    data_tools_dir = str(settings.data_tools_dir)
    models_training_dir = str(
        settings.project_root / "tools" / "models-tools" / "training"
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if data_tools_dir not in sys.path:
        sys.path.insert(0, data_tools_dir)
    if models_training_dir not in sys.path:
        sys.path.insert(0, models_training_dir)


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
