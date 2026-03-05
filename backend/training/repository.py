from datetime import datetime
from typing import Optional

from backend.shared.state import (
    chat_threads_with_context,
    last_training_context,
    simple_agent_store,
    training_jobs,
)


def start_training(job_id: str, request_payload: dict[str, object]) -> None:
    training_jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "current_step": None,
        "state": None,
        "error": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        **request_payload,
    }


def get_training_status(job_id: str) -> Optional[dict[str, object]]:
    return training_jobs.get(job_id)


def update_training_status(job_id: str, updates: dict[str, object]) -> None:
    if job_id in training_jobs:
        training_jobs[job_id].update(updates)


def cancel_training(job_id: str) -> bool:
    if job_id not in training_jobs:
        return False
    del training_jobs[job_id]
    return True


def put_simple_agent_store(thread_id: str, value: dict[str, object]) -> None:
    simple_agent_store[thread_id] = value


def get_simple_agent_store(thread_id: str) -> Optional[dict[str, object]]:
    return simple_agent_store.get(thread_id)


def thread_exists(thread_id: str) -> bool:
    return thread_id in simple_agent_store


def save_interrupt_ids(thread_id: str, interrupt_ids: list[str]) -> None:
    if thread_id in simple_agent_store:
        simple_agent_store[thread_id]["interrupt_ids"] = interrupt_ids


def save_training_context(shared_state: dict[str, object]) -> None:
    metrics = shared_state.get("training_metrics", {})
    label_def = shared_state.get("label_definition") or {}
    last_training_context.clear()
    last_training_context.update(
        {
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
    )
    chat_threads_with_context.clear()
