from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _json_col():
    """JSONB on Postgres, JSON on SQLite."""
    return JSONB().with_variant(Text(), "sqlite")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    current_step: Mapped[str | None] = mapped_column(String(128), nullable=True)
    goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    linked_datasets: Mapped[dict | None] = mapped_column(_json_col(), nullable=True)
    model_preference: Mapped[str | None] = mapped_column(String(128), nullable=True)
    state: Mapped[dict | None] = mapped_column(_json_col(), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class TrainingContext(Base):
    """Stores the latest training context so chat can reference it across restarts."""
    __tablename__ = "training_contexts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    training_job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    model_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_column: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(_json_col(), nullable=True)
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


class AgentStore(Base):
    """Persists LangGraph agent thread state for resume/interrupt flows."""
    __tablename__ = "agent_store"

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[dict] = mapped_column(_json_col(), nullable=False)
    interrupt_ids: Mapped[dict | None] = mapped_column(_json_col(), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
