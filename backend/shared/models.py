import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    BigInteger,
    DateTime,
    Float,
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
    mrm_profile: Mapped["ModelRiskProfile | None"] = relationship(
        back_populates="model", uselist=False, passive_deletes=True,
    )
    mrm_lifecycle_stages: Mapped[list["ModelLifecycleStage"]] = relationship(
        back_populates="model", passive_deletes=True,
    )
    mrm_artifacts: Mapped[list["ModelArtifact"]] = relationship(
        back_populates="model", passive_deletes=True,
    )
    mrm_approvals: Mapped[list["ModelApproval"]] = relationship(
        back_populates="model", passive_deletes=True,
    )
    mrm_monitoring_snapshots: Mapped[list["ModelMonitoringSnapshot"]] = relationship(
        back_populates="model", passive_deletes=True,
    )
    mrm_review_schedules: Mapped[list["ModelReviewSchedule"]] = relationship(
        back_populates="model", passive_deletes=True,
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
    mrm_monitoring_snapshots: Mapped[list["ModelMonitoringSnapshot"]] = relationship(
        back_populates="model_version",
        passive_deletes=False,
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


# =============================================================================
# Model Risk Management (MRM) demo registry
# =============================================================================


class ModelRiskProfile(Base):
    """One-to-one governance profile per catalog model."""
    __tablename__ = "model_risk_profiles"

    id: Mapped[uuid.UUID] = _uuid_pk()
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    business_owner: Mapped[str] = mapped_column(String(256), nullable=False)
    business_line: Mapped[str] = mapped_column(String(256), nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(64), nullable=False)
    governance_profile: Mapped[str] = mapped_column(String(64), nullable=False)
    model_origin: Mapped[str] = mapped_column(String(32), nullable=False)
    business_purpose: Mapped[str] = mapped_column(Text, nullable=False)
    approved_use: Mapped[str] = mapped_column(Text, nullable=False)
    prohibited_use: Mapped[str] = mapped_column(Text, nullable=False)
    limitations: Mapped[str | None] = mapped_column(Text, nullable=True)
    board_visible: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    current_lifecycle_stage: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    model: Mapped["Model"] = relationship(back_populates="mrm_profile")


class ModelLifecycleStage(Base):
    __tablename__ = "model_lifecycle_stages"
    __table_args__ = (
        UniqueConstraint("model_id", "stage_name", name="uq_mrm_lifecycle_model_stage"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage_name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    owner_role: Mapped[str | None] = mapped_column(String(128), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    model: Mapped["Model"] = relationship(back_populates="mrm_lifecycle_stages")


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[uuid.UUID] = _uuid_pk()
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage_name: Mapped[str] = mapped_column(String(256), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    uploaded_by: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    model: Mapped["Model"] = relationship(back_populates="mrm_artifacts")


class ModelApproval(Base):
    __tablename__ = "model_approvals"

    id: Mapped[uuid.UUID] = _uuid_pk()
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage_name: Mapped[str] = mapped_column(String(256), nullable=False)
    approval_role: Mapped[str] = mapped_column(String(128), nullable=False)
    approver_name: Mapped[str] = mapped_column(String(256), nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    model: Mapped["Model"] = relationship(back_populates="mrm_approvals")


class ModelMonitoringSnapshot(Base):
    __tablename__ = "model_monitoring_snapshots"

    id: Mapped[uuid.UUID] = _uuid_pk()
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("model_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    psi_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    backtest_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    top_csi_features: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    model: Mapped["Model"] = relationship(back_populates="mrm_monitoring_snapshots")
    model_version: Mapped["ModelVersion | None"] = relationship(
        back_populates="mrm_monitoring_snapshots",
    )


class ModelReviewSchedule(Base):
    __tablename__ = "model_review_schedules"

    id: Mapped[uuid.UUID] = _uuid_pk()
    model_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    review_type: Mapped[str] = mapped_column(String(128), nullable=False)
    cadence: Mapped[str] = mapped_column(String(64), nullable=False)
    next_due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(String(64), nullable=False)

    model: Mapped["Model"] = relationship(back_populates="mrm_review_schedules")


# Backward-compat aliases for imports that reference old names.
# These keep existing code working until we update all call sites.
TrainingContext = TrainingSummary
AgentStore = AgentCheckpoint
