import json
import uuid as _uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import or_, and_, select

from backend.shared.database import get_db_session
from backend.shared.models import (
    AgentCheckpoint, ChatThread, Experiment, TrainingSummary, TrainingJob, User,
)
from backend.shared.state import (
    chat_threads_with_context,
    last_training_context,
    simple_agent_store,
    training_contexts,
    training_jobs,
    training_states,
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


# --- Agent checkpoints (for LangGraph interrupt/resume) ---

def put_simple_agent_store(thread_id: str, value: dict[str, object]) -> None:
    simple_agent_store[thread_id] = value

    serializable = _make_serializable(value)
    with get_db_session() as session:
        existing = session.get(AgentCheckpoint, thread_id)
        if existing:
            existing.state = serializable
        else:
            session.add(AgentCheckpoint(thread_id=thread_id, state=serializable))


def get_simple_agent_store(thread_id: str) -> Optional[dict[str, object]]:
    if thread_id in simple_agent_store:
        return simple_agent_store[thread_id]

    with get_db_session() as session:
        row = session.get(AgentCheckpoint, thread_id)
        if row is None:
            return None
        return row.state


def thread_exists(thread_id: str) -> bool:
    if thread_id in simple_agent_store:
        return True

    with get_db_session() as session:
        return session.get(AgentCheckpoint, thread_id) is not None


def save_interrupt_ids(thread_id: str, interrupt_ids: list[str]) -> None:
    if thread_id in simple_agent_store:
        simple_agent_store[thread_id]["interrupt_ids"] = interrupt_ids

    with get_db_session() as session:
        row = session.get(AgentCheckpoint, thread_id)
        if row:
            row.interrupt_ids = interrupt_ids


# --- Training summaries (for chat) ---

def _build_context_dict(shared_state: dict[str, object]) -> dict[str, object]:
    raw_metrics = shared_state.get("training_metrics")
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    label_def = shared_state.get("label_definition") or {}
    return {
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


def save_training_context(
    shared_state: dict[str, object],
    experiment_id: Optional[str] = None,
) -> None:
    ctx = _build_context_dict(shared_state)

    # Experiment-keyed in-memory store
    if experiment_id:
        training_contexts[experiment_id] = dict(ctx)
    # Backward compat: also write to the old singleton
    last_training_context.clear()
    last_training_context.update(ctx)
    chat_threads_with_context.clear()

    # Persist to training_summaries table
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
        row = TrainingSummary(
            model_name=ctx.get("model_name"),
            model_type=ctx.get("model_type"),
            goal=ctx.get("goal"),
            target_column=ctx.get("target_column"),
            metrics=metric_fields or None,
            report_path=ctx.get("report_path"),
        )
        session.add(row)

        # Also persist context on the experiment row
        if experiment_id:
            exp = session.get(Experiment, experiment_id)
            if exp:
                exp.training_context = ctx


def get_latest_training_context(
    experiment_id: Optional[str] = None,
) -> Optional[dict[str, object]]:
    """Retrieve the most recent training summary, scoped to an experiment if given."""
    if experiment_id:
        if experiment_id in training_contexts:
            return dict(training_contexts[experiment_id])
        with get_db_session() as session:
            exp = session.get(Experiment, experiment_id)
            if exp and exp.training_context:
                return dict(exp.training_context)

    # Fallback: global most-recent (backward compat)
    if last_training_context:
        return dict(last_training_context)

    with get_db_session() as session:
        row = session.execute(
            select(TrainingSummary).order_by(TrainingSummary.id.desc()).limit(1)
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


def save_run_dataset_links(job_id: str, agent_state: dict[str, object]) -> None:
    """Extract dataset refs from agent state and persist bidirectional links."""
    try:
        from backend.shared.models import Dataset, RunDatasetLink
    except ImportError:
        return

    ref_role_pairs = [
        (agent_state.get("collected_dataset_ref"), "source"),
        (agent_state.get("train_dataset_ref"), "train"),
        (agent_state.get("val_dataset_ref"), "validation"),
        (agent_state.get("test_dataset_ref"), "test"),
    ]

    refs_to_link = [(ref, role) for ref, role in ref_role_pairs if ref]
    if not refs_to_link:
        return

    try:
        with get_db_session() as session:
            for ref, role in refs_to_link:
                ds = session.query(Dataset).filter(Dataset.name == ref).first()
                if ds is None:
                    continue
                exists = (
                    session.query(RunDatasetLink)
                    .filter(
                        RunDatasetLink.training_run_id == job_id,
                        RunDatasetLink.dataset_id == ds.id,
                        RunDatasetLink.role == role,
                    )
                    .first()
                )
                if not exists:
                    session.add(RunDatasetLink(
                        training_run_id=job_id,
                        dataset_id=ds.id,
                        role=role,
                    ))
    except Exception as e:
        print(f"Warning: Failed to save run-dataset links for {job_id}: {e}")


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


# =========================================================================
# Experiment CRUD
# =========================================================================

def create_experiment(
    experiment_id: str,
    name: str,
    chat_thread_id: str,
    linked_datasets: Optional[list[str]] = None,
    user_id: Optional[_uuid.UUID] = None,
) -> dict[str, object]:
    with get_db_session() as session:
        exp = Experiment(
            id=experiment_id,
            name=name,
            chat_thread_id=chat_thread_id,
            linked_datasets=linked_datasets or [],
            user_id=user_id,
        )
        session.add(exp)
    return {
        "id": experiment_id,
        "name": name,
        "status": "created",
        "chat_thread_id": chat_thread_id,
        "linked_datasets": linked_datasets or [],
        "chat_history": [],
        "training_state": None,
        "training_context": None,
        "user_id": str(user_id) if user_id else None,
        "shared_with_org": False,
    }


def _ownership_filter(
    user_id: Optional[_uuid.UUID],
    org_id: Optional[_uuid.UUID],
    model_cls: type,
):
    """Build a WHERE clause: own rows + org-shared rows + legacy unowned rows."""
    if user_id is None:
        return True  # no auth — return everything (backward compat)
    org_user_ids = select(User.id).where(User.organization_id == org_id).scalar_subquery()
    return or_(
        model_cls.user_id == user_id,
        and_(model_cls.shared_with_org.is_(True), model_cls.user_id.in_(org_user_ids)),
        model_cls.user_id.is_(None),
    )


def list_experiments(
    user_id: Optional[_uuid.UUID] = None,
    org_id: Optional[_uuid.UUID] = None,
) -> list[dict[str, object]]:
    with get_db_session() as session:
        stmt = (
            select(Experiment)
            .where(_ownership_filter(user_id, org_id, Experiment))
            .order_by(Experiment.updated_at.desc())
        )
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "goal": r.goal,
                "status": r.status,
                "chat_thread_id": r.chat_thread_id,
                "linked_datasets": r.linked_datasets,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                "last_message": (
                    r.chat_history[-1].get("content", "")[:80]
                    if r.chat_history and isinstance(r.chat_history, list) and r.chat_history
                    else None
                ),
                "user_id": str(r.user_id) if r.user_id else None,
                "shared_with_org": r.shared_with_org,
                "is_owner": r.user_id == user_id if user_id else True,
            }
            for r in rows
        ]


def get_experiment(experiment_id: str) -> Optional[dict[str, object]]:
    with get_db_session() as session:
        exp = session.get(Experiment, experiment_id)
        if exp is None:
            return None
        return {
            "id": exp.id,
            "name": exp.name,
            "goal": exp.goal,
            "status": exp.status,
            "chat_thread_id": exp.chat_thread_id,
            "chat_history": exp.chat_history or [],
            "training_state": exp.training_state,
            "training_context": exp.training_context,
            "linked_datasets": exp.linked_datasets,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
            "user_id": str(exp.user_id) if exp.user_id else None,
            "shared_with_org": exp.shared_with_org,
        }


def update_experiment(experiment_id: str, updates: dict[str, object]) -> bool:
    with get_db_session() as session:
        exp = session.get(Experiment, experiment_id)
        if exp is None:
            return False
        for key, value in updates.items():
            if hasattr(exp, key) and key not in ("id", "created_at"):
                setattr(exp, key, value)
        return True


def delete_experiment(experiment_id: str) -> bool:
    with get_db_session() as session:
        exp = session.get(Experiment, experiment_id)
        if exp is None:
            return False
        session.delete(exp)
        return True


def save_experiment_chat_history(
    experiment_id: str, chat_history: list[dict],
) -> None:
    with get_db_session() as session:
        exp = session.get(Experiment, experiment_id)
        if exp:
            exp.chat_history = chat_history
