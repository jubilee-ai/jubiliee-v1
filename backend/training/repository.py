import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select

from backend.shared.database import get_db_session
from backend.shared.models import (
    AgentCheckpoint, ChatThread, Experiment, Model, TrainingSummary, TrainingJob,
)
from backend.shared.state import (
    chat_threads_with_context,
    last_training_context,
    simple_agent_store,
    training_contexts,
    training_jobs,
    training_states,
)

# LangGraph compiled graphs / savers are not JSON-serializable; never persist repr strings.
_STORE_NON_PERSISTED_KEYS = frozenset({"agent", "checkpointer"})


def _extract_model_name_from_chat_history(chat_history: object) -> Optional[str]:
    if not isinstance(chat_history, list):
        return None
    for msg in reversed(chat_history):
        if not isinstance(msg, dict):
            continue
        step_id = msg.get("stepId") or msg.get("step_id")
        content = msg.get("content")
        if step_id != "training" or not isinstance(content, str):
            continue
        match = re.search(r"Trained \*\*(.+?)\*\*", content)
        if match:
            model_name = match.group(1).strip()
            if model_name:
                return model_name
    return None


def _build_training_state_from_report(
    report: dict[str, object],
    report_path: Path,
    existing_state: dict[str, object],
    fallback_goal: Optional[str],
) -> dict[str, object]:
    model = report.get("model") if isinstance(report.get("model"), dict) else {}
    data = report.get("data") if isinstance(report.get("data"), dict) else {}
    label_definition = (
        report.get("label_definition")
        if isinstance(report.get("label_definition"), dict)
        else {}
    )
    training_results = (
        report.get("training_results")
        if isinstance(report.get("training_results"), dict)
        else {}
    )
    validation_metrics = (
        training_results.get("validation_metrics")
        if isinstance(training_results.get("validation_metrics"), dict)
        else {}
    )
    test_metrics = (
        training_results.get("test_metrics")
        if isinstance(training_results.get("test_metrics"), dict)
        else {}
    )
    audit_trace = report.get("audit_trace")
    training_metrics = {
        "success": training_results.get("success"),
        "model_name": model.get("name"),
        "model_type": model.get("type"),
        "val_accuracy": validation_metrics.get("accuracy"),
        "val_roc_auc": validation_metrics.get("roc_auc"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_roc_auc": test_metrics.get("roc_auc"),
        "iterations": training_results.get("iterations", []),
        "num_iterations": training_results.get("num_iterations"),
        "best_iteration": training_results.get("best_iteration"),
        "summary": training_results.get("summary"),
    }
    next_state = {
        **existing_state,
        "goal": report.get("goal") or fallback_goal or existing_state.get("goal") or "",
        "selected_model": model.get("type") or existing_state.get("selected_model"),
        "model_explanation": model.get("explanation") or existing_state.get("model_explanation"),
        "collected_dataset_ref": data.get("collected_dataset") or existing_state.get("collected_dataset_ref"),
        "cleaned_dataset_ref": data.get("cleaned_dataset") or existing_state.get("cleaned_dataset_ref"),
        "transformed_train_ref": data.get("train_dataset") or existing_state.get("transformed_train_ref"),
        "transformed_val_ref": data.get("val_dataset") or existing_state.get("transformed_val_ref"),
        "transformed_test_ref": data.get("test_dataset") or existing_state.get("transformed_test_ref"),
        "label_definition": label_definition or existing_state.get("label_definition"),
        "training_metrics": training_metrics,
        "report_path": str(report_path),
        "audit_trace": audit_trace if isinstance(audit_trace, list) else existing_state.get("audit_trace", []),
        "current_step": "generate_report",
        "error": None,
    }
    return next_state


def _maybe_backfill_experiment(exp: Experiment) -> None:
    state = dict(exp.training_state or {})
    if state.get("training_metrics") or state.get("report_path"):
        return
    model_name = _extract_model_name_from_chat_history(exp.chat_history or [])
    if not model_name:
        return
    report_path = Path(__file__).parent.parent.parent / "trained_models" / f"{model_name}_report.json"
    if not report_path.exists():
        return
    try:
        report = json.loads(report_path.read_text())
    except Exception:
        return
    next_state = _build_training_state_from_report(
        report=report,
        report_path=report_path,
        existing_state=state,
        fallback_goal=exp.goal,
    )
    exp.training_state = next_state
    if not exp.goal and next_state.get("goal"):
        exp.goal = str(next_state["goal"])
    if next_state.get("training_metrics", {}).get("success") is True:
        exp.status = "completed"


def _store_agent_is_runnable(agent: object) -> bool:
    stream = getattr(agent, "stream", None)
    return callable(stream)


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


def ensure_training_job(
    job_id: str,
    request_payload: Optional[dict[str, object]] = None,
) -> None:
    """Ensure a ``training_jobs`` row exists for this id (graph/simple SSE use thread_id as id).

    Idempotent: no-op if the row already exists. Required so ``run_dataset_links``
    FK to ``training_jobs`` succeeds for LangGraph checkpoint thread ids.
    """
    request_payload = request_payload or {}
    with get_db_session() as session:
        if session.get(TrainingJob, job_id) is not None:
            return
        job = TrainingJob(
            id=job_id,
            status="pending",
            progress=0,
            goal=request_payload.get("goal"),
            linked_datasets=request_payload.get("linked_datasets"),
            model_preference=request_payload.get("model_preference"),
        )
        session.add(job)

    if job_id not in training_jobs:
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
    """Return job status, preferring persisted DB state so any API replica sees worker updates."""
    mem = training_jobs.get(job_id)
    with get_db_session() as session:
        job = session.get(TrainingJob, job_id)
        if job is None and mem is None:
            return None
        db_dict = _job_to_dict(job) if job is not None else None
    if db_dict is not None:
        if mem:
            merged = {**mem, **db_dict}
            return merged
        return db_dict
    return mem


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

    db_value = {
        k: v for k, v in value.items() if k not in _STORE_NON_PERSISTED_KEYS
    }
    serializable = _make_serializable(db_value)
    with get_db_session() as session:
        existing = session.get(AgentCheckpoint, thread_id)
        if existing:
            existing.state = serializable
        else:
            session.add(AgentCheckpoint(thread_id=thread_id, state=serializable))


def get_simple_agent_store(thread_id: str) -> Optional[dict[str, object]]:
    if thread_id in simple_agent_store:
        store = simple_agent_store[thread_id]
    else:
        with get_db_session() as session:
            row = session.get(AgentCheckpoint, thread_id)
            if row is None:
                return None
            store = row.state

    if not isinstance(store, dict):
        return None
    agent = store.get("agent")
    if agent is None or not _store_agent_is_runnable(agent):
        return None
    return store


def thread_exists(thread_id: str) -> bool:
    return get_simple_agent_store(thread_id) is not None


def save_interrupt_ids(thread_id: str, interrupt_ids: list[str]) -> None:
    if thread_id in simple_agent_store:
        simple_agent_store[thread_id]["interrupt_ids"] = interrupt_ids

    with get_db_session() as session:
        row = session.get(AgentCheckpoint, thread_id)
        if row:
            row.interrupt_ids = interrupt_ids


# --- Training summaries (for chat) ---

def extract_model_name_from_training_state(
    final_values: Optional[dict[str, object]],
) -> Optional[str]:
    """Match `_build_context_dict` model_name resolution: training_metrics.model_name, else model_weights_path."""
    if not final_values:
        return None
    raw_metrics = final_values.get("training_metrics")
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    mn = metrics.get("model_name")
    if isinstance(mn, str):
        s = mn.strip()
        if s:
            return s
    mwp = final_values.get("model_weights_path")
    if not isinstance(mwp, str):
        return None
    s = mwp.strip()
    if not s:
        return None
    # State often stores the logical model name here; if it looks like a filesystem path, use basename/stem.
    if "/" in s or "\\" in s or s.startswith((".", "~")):
        base = Path(s).name
        if not base:
            return None
        stem = Path(base).stem
        return stem if stem else base
    return s


def link_model_to_experiment(model_name: Optional[str], experiment_id: Optional[str]) -> None:
    """Persist experiment link on the model row (JSONB `properties`) — no extra DDL required."""
    if not model_name or not experiment_id:
        return
    with get_db_session() as session:
        row = session.query(Model).filter(Model.name == model_name).first()
        if row is None:
            return
        exp = session.get(Experiment, experiment_id)
        exp_name = exp.name if exp else None
        props = dict(row.properties or {})
        props["experiment_id"] = experiment_id
        if exp_name:
            props["experiment_name"] = exp_name
        row.properties = props


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
        "report_storage_key": shared_state.get("report_storage_key"),
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
            "summary", "num_iterations", "report_storage_key",
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
            if session.get(TrainingJob, job_id) is None:
                return
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
) -> dict[str, object]:
    with get_db_session() as session:
        exp = Experiment(
            id=experiment_id,
            name=name,
            chat_thread_id=chat_thread_id,
            linked_datasets=linked_datasets or [],
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
    }


def list_experiments() -> list[dict[str, object]]:
    with get_db_session() as session:
        rows = session.execute(
            select(Experiment).order_by(Experiment.updated_at.desc())
        ).scalars().all()
        for row in rows:
            _maybe_backfill_experiment(row)
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
                "lab_mode": (
                    (r.training_state or {}).get("lab_mode")
                    if isinstance(r.training_state, dict)
                    else None
                ),
                "task_status": (
                    (r.training_state or {}).get("task_status")
                    if isinstance(r.training_state, dict)
                    else None
                ),
            }
            for r in rows
        ]


def get_experiment(experiment_id: str) -> Optional[dict[str, object]]:
    with get_db_session() as session:
        exp = session.get(Experiment, experiment_id)
        if exp is None:
            return None
        _maybe_backfill_experiment(exp)
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


def _compact_experiment_training_patch(patch: dict[str, object]) -> dict[str, object]:
    """Avoid huge JSONB rows: trim long lists and oversized string values."""
    out: dict[str, object] = {}
    max_list = 24
    max_str = 120_000
    for k, v in patch.items():
        if k == "task_step_events" and isinstance(v, list) and len(v) > max_list:
            out[k] = v[-max_list:]
            out["task_step_events_truncated"] = True
        elif k == "experiment_grid_summary" and isinstance(v, list) and len(v) > max_list:
            out[k] = v[:max_list]
            out["experiment_grid_summary_truncated"] = True
        elif k == "audit_trace" and isinstance(v, list) and len(v) > max_list:
            out[k] = v[-max_list:]
            out["audit_trace_truncated"] = True
        elif isinstance(v, str) and len(v) > max_str:
            out[k] = v[:max_str] + "…(truncated for DB)"
        else:
            out[k] = v
    return out


def merge_experiment_training_state(experiment_id: str, patch: dict[str, object]) -> bool:
    """Deep-shallow merge JSON `training_state` on an experiment (patch wins for top-level keys)."""
    patch = _compact_experiment_training_patch(dict(patch))
    with get_db_session() as session:
        exp = session.get(Experiment, experiment_id)
        if exp is None:
            return False
        cur = dict(exp.training_state or {})
        for k, v in patch.items():
            cur[k] = v
        exp.training_state = cur
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
