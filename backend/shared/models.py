import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_pk():
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )


# =============================================================================
# Existing tables (renamed for clarity)
# =============================================================================


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    chat_thread_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    chat_history: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    training_state: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    training_context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    linked_datasets: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    org_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    training_jobs: Mapped[list["TrainingJob"]] = relationship(
        back_populates="experiment", passive_deletes=True,
    )


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("experiments.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    current_step: Mapped[str | None] = mapped_column(String(128), nullable=True)
    goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    linked_datasets: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    model_preference: Mapped[str | None] = mapped_column(String(128), nullable=True)
    state: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    org_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    experiment: Mapped["Experiment | None"] = relationship(
        back_populates="training_jobs",
    )
    model_versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="training_run", passive_deletes=True,
    )
    dataset_links: Mapped[list["RunDatasetLink"]] = relationship(
        back_populates="training_run", passive_deletes=True,
    )


class TrainingSummary(Base):
    """Post-training snapshot surfaced to the chat interface."""
    __tablename__ = "training_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    model_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_column: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ChatThread(Base):
    __tablename__ = "chat_threads"

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    has_training_context: Mapped[bool] = mapped_column(default=False)
    training_context_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AgentCheckpoint(Base):
    """Persists LangGraph agent thread state for resume/interrupt flows."""
    __tablename__ = "agent_checkpoints"

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[dict] = mapped_column(JSONB, nullable=False)
    interrupt_ids: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# =============================================================================
# New tables -- MLOps registry
# =============================================================================


class Dataset(Base):
    """Dataset registry. Metadata stored here; data files persisted to R2."""
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    properties: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    org_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    run_links: Mapped[list["RunDatasetLink"]] = relationship(
        back_populates="dataset", passive_deletes=True,
    )


class Model(Base):
    """Stable model identity. One row per model concept, versions tracked separately."""
    __tablename__ = "models"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    properties: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="model", passive_deletes=True,
        order_by="ModelVersion.version.desc()",
    )


class ModelVersion(Base):
    """Per-training snapshot. Holds R2 storage key for model weights."""
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_id", "version", name="uq_model_version"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("models.id", ondelete="CASCADE"), nullable=False,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    training_run_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("training_jobs.id", ondelete="SET NULL"), nullable=True,
    )
    storage_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    properties: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    model: Mapped["Model"] = relationship(back_populates="versions")
    training_run: Mapped["TrainingJob | None"] = relationship(
        back_populates="model_versions",
    )


class RunDatasetLink(Base):
    """Bidirectional link: which datasets participated in which training run."""
    __tablename__ = "run_dataset_links"
    __table_args__ = (
        UniqueConstraint(
            "training_run_id", "dataset_id", "role",
            name="uq_run_dataset_role",
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    training_run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False,
    )
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    training_run: Mapped["TrainingJob"] = relationship(back_populates="dataset_links")
    dataset: Mapped["Dataset"] = relationship(back_populates="run_links")


# Backward-compat aliases for imports that reference old names.
# These keep existing code working until we update all call sites.
TrainingContext = TrainingSummary
AgentStore = AgentCheckpoint
