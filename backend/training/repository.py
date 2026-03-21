import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select

from backend.shared.database import get_db_session
from backend.shared.models import AgentStore, ChatThread, TrainingContext, TrainingJob

# Also re-export the in-memory fallbacks so existing `from repository import` still works.
from backend.shared.state import (
    chat_threads_with_context,
    last_training_context,
    simple_agent_store,
    training_jobs,
)


def _job_to_dict(job: TrainingJob) -> dict[str, object]:
    return {
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "goal": job.goal,
        "linked_datasets": job.linked_datasets,
        "model_preference": job.model_preference,
        "state": job.state,
        "error": job.error,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


def start_training(job_id: str, request_payload: dict[str, object]) -> None:
    with get_db_session() as session:
        job = TrainingJob(
            id=job_id,
            status="pending",
            progress=0,
            goal=request_payload.get("goal"),
            linked_datasets=request_payload.get("linked_datasets"),
            model_preference=request_payload.get("model_preference"),
        )
        session.add(job)

    # Keep in-memory mirror for SSE streaming within the same process
    training_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "current_step": None,
        "state": None,
        "error": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        **request_payload,
    }


def get_training_status(job_id: str) -> Optional[dict[str, object]]:
    # Check in-memory first (hot path for SSE polling)
    if job_id in training_jobs:
        return training_jobs[job_id]

    with get_db_session() as session:
        job = session.get(TrainingJob, job_id)
        if job is None:
            return None
        return _job_to_dict(job)


def update_training_status(job_id: str, updates: dict[str, object]) -> None:
    # Update in-memory mirror
    if job_id in training_jobs:
        training_jobs[job_id].update(updates)

    # Persist to DB
    with get_db_session() as session:
        job = session.get(TrainingJob, job_id)
        if job is None:
            return
        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)


def cancel_training(job_id: str) -> bool:
    training_jobs.pop(job_id, None)

    with get_db_session() as session:
        job = session.get(TrainingJob, job_id)
        if job is None:
            return False
        job.status = "cancelled"
    return True


# --- Agent store (for LangGraph interrupt/resume) ---

def put_simple_agent_store(thread_id: str, value: dict[str, object]) -> None:
    simple_agent_store[thread_id] = value

    serializable = _make_serializable(value)
    with get_db_session() as session:
        existing = session.get(AgentStore, thread_id)
        if existing:
            existing.state = serializable
        else:
            session.add(AgentStore(thread_id=thread_id, state=serializable))


def get_simple_agent_store(thread_id: str) -> Optional[dict[str, object]]:
    if thread_id in simple_agent_store:
        return simple_agent_store[thread_id]

    with get_db_session() as session:
        row = session.get(AgentStore, thread_id)
        if row is None:
            return None
        return row.state


def thread_exists(thread_id: str) -> bool:
    if thread_id in simple_agent_store:
        return True

    with get_db_session() as session:
        return session.get(AgentStore, thread_id) is not None


def save_interrupt_ids(thread_id: str, interrupt_ids: list[str]) -> None:
    if thread_id in simple_agent_store:
        simple_agent_store[thread_id]["interrupt_ids"] = interrupt_ids

    with get_db_session() as session:
        row = session.get(AgentStore, thread_id)
        if row:
            row.interrupt_ids = interrupt_ids


# --- Training context (for chat) ---

def save_training_context(shared_state: dict[str, object]) -> None:
    metrics = shared_state.get("training_metrics", {})
    label_def = shared_state.get("label_definition") or {}

    ctx = {
        "model_name": metrics.get("model_name") or shared_state.get("model_weights_path"),
        "model_type": metrics.get("model_type") or shared_state.get("selected_model"),
        "goal": shared_state.get("goal"),
        "target_column": label_def.get("target_column"),
        "val_accuracy": metrics.get("val_accuracy"),
        "val_roc_auc": metrics.get("val_roc_auc"),
        "test_accuracy": metrics.get("test_accuracy"),
        "test_roc_auc": metrics.get("test_roc_auc"),
        "val_r2": metrics.get("val_r2"),
        "test_r2": metrics.get("test_r2"),
        "test_rmse": metrics.get("test_rmse"),
        "test_mae": metrics.get("test_mae"),
        "summary": metrics.get("summary"),
        "num_iterations": metrics.get("num_iterations"),
        "report_path": shared_state.get("report_path"),
    }

    # In-memory mirror
    last_training_context.clear()
    last_training_context.update(ctx)
    chat_threads_with_context.clear()

    # Persist
    metric_fields = {
        k: ctx[k]
        for k in [
            "val_accuracy", "val_roc_auc", "test_accuracy", "test_roc_auc",
            "val_r2", "test_r2", "test_rmse", "test_mae",
            "summary", "num_iterations",
        ]
        if ctx.get(k) is not None
    }
    with get_db_session() as session:
        row = TrainingContext(
            model_name=ctx.get("model_name"),
            model_type=ctx.get("model_type"),
            goal=ctx.get("goal"),
            target_column=ctx.get("target_column"),
            metrics=metric_fields or None,
            report_path=ctx.get("report_path"),
        )
        session.add(row)


def get_latest_training_context() -> Optional[dict[str, object]]:
    """Retrieve the most recent training context from DB."""
    if last_training_context:
        return dict(last_training_context)

    with get_db_session() as session:
        row = session.execute(
            select(TrainingContext).order_by(TrainingContext.id.desc()).limit(1)
        ).scalar_one_or_none()
        if row is None:
            return None
        ctx = {
            "model_name": row.model_name,
            "model_type": row.model_type,
            "goal": row.goal,
            "target_column": row.target_column,
            "report_path": row.report_path,
        }
        if row.metrics:
            ctx.update(row.metrics)
        return ctx


def mark_chat_thread_has_context(thread_id: str) -> None:
    chat_threads_with_context.add(thread_id)

    with get_db_session() as session:
        existing = session.get(ChatThread, thread_id)
        if existing:
            existing.has_training_context = True
        else:
            session.add(ChatThread(thread_id=thread_id, has_training_context=True))


def chat_thread_has_context(thread_id: str) -> bool:
    if thread_id in chat_threads_with_context:
        return True

    with get_db_session() as session:
        row = session.get(ChatThread, thread_id)
        return row is not None and row.has_training_context


def _make_serializable(obj: object) -> object:
    """Best-effort JSON-safe conversion for agent state."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(v) for v in obj]
        return str(obj)
