from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, TypedDict


class SSEEnvelope(TypedDict, total=False):
    type: str
    experiment_id: str | None
    ts: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sse(payload: dict[str, Any], experiment_id: str | None = None) -> str:
    payload.setdefault("experiment_id", experiment_id)
    payload.setdefault("ts", _now_iso())
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Typed event builders
# ---------------------------------------------------------------------------

def stream_start(
    experiment_id: str | None = None,
    *,
    training_graph: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {"type": "stream.start", "experiment_id": experiment_id}
    if training_graph:
        out["training_graph"] = True
    return out


def stream_end(
    experiment_id: str | None = None,
    *,
    pipeline_completed: bool = False,
) -> dict[str, Any]:
    return {
        "type": "stream.end",
        "experiment_id": experiment_id,
        "pipeline_completed": pipeline_completed,
    }


def token(content: str, experiment_id: str | None = None, phase: str | None = None) -> dict[str, Any]:
    evt: dict[str, Any] = {"type": "token", "content": content, "experiment_id": experiment_id}
    if phase:
        evt["phase"] = phase
    return evt


def tool_start(
    tool: str,
    args: dict[str, Any],
    headline: str,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "tool.start",
        "tool": tool,
        "args": args,
        "headline": headline,
        "experiment_id": experiment_id,
    }


def tool_end(
    tool: str,
    result: Any,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "tool.end",
        "tool": tool,
        "result": result,
        "experiment_id": experiment_id,
    }


def step_complete(
    node: str,
    progress: float,
    summary: str,
    details: Any,
    state: Any,
    headline: str,
    stream_step_key: str | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "step.complete",
        "node": node,
        "progress": progress,
        "summary": summary,
        "details": details,
        "state": state,
        "headline": headline,
        "stream_step_key": stream_step_key,
        "experiment_id": experiment_id,
    }


def step_skipped(
    node: str,
    headline: str,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "step.skipped",
        "node": node,
        "headline": headline,
        "experiment_id": experiment_id,
    }


def step_progress(
    phase: str,
    message: str,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "step.progress",
        "phase": phase,
        "message": message,
        "experiment_id": experiment_id,
    }


def review_required(
    node: str,
    summary: str,
    message: str,
    state_snapshot: Any,
    review_prompt: str,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "review.required",
        "node": node,
        "summary": summary,
        "message": message,
        "state_snapshot": state_snapshot,
        "review_prompt": review_prompt,
        "experiment_id": experiment_id,
    }


def review_auto_approved(
    node: str,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "review.auto_approved",
        "node": node,
        "experiment_id": experiment_id,
    }


def predict_start(
    model: str,
    dataset: str,
    headline: str,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "predict.start",
        "model": model,
        "dataset": dataset,
        "headline": headline,
        "experiment_id": experiment_id,
    }


def predict_complete(
    model: str,
    rows_predicted: int,
    headline: str,
    result_ref: str | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "predict.complete",
        "model": model,
        "rows_predicted": rows_predicted,
        "headline": headline,
        "result_ref": result_ref,
        "experiment_id": experiment_id,
    }


def dataset_resolved(
    ref: str,
    dataset_info: Any = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "dataset.resolved",
        "ref": ref,
        "dataset_info": dataset_info,
        "experiment_id": experiment_id,
    }


def dataset_error(
    ref: str,
    error: str,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "dataset.error",
        "ref": ref,
        "error": error,
        "experiment_id": experiment_id,
    }


def error_event(
    error_msg: str,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    return {"type": "error", "error": error_msg, "experiment_id": experiment_id}


# ---------------------------------------------------------------------------
# Validation & public API
# ---------------------------------------------------------------------------

ALL_EVENT_TYPES: list[str] = [
    "stream.start",
    "stream.end",
    "token",
    "tool.start",
    "tool.end",
    "step.complete",
    "step.skipped",
    "step.progress",
    "review.required",
    "review.auto_approved",
    "predict.start",
    "predict.complete",
    "dataset.resolved",
    "dataset.error",
    "error",
]


def format_sse(payload: dict[str, Any], experiment_id: str | None = None) -> str:
    """Public API — stamp *experiment_id* + *ts* and return an SSE ``data:`` line."""
    return _sse(payload, experiment_id=experiment_id)
