"""
Compact training state for DB/Redis and optionally persist full snapshots to object storage.

Large lists (audit traces, iteration logs, experiment grids) stay in durable storage;
runtime payloads carry references via `full_state_storage_key` / `report_storage_key`.
"""

from __future__ import annotations

import copy
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from backend.shared.settings import get_settings


def compact_training_state_for_db(state: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly copy with heavy fields trimmed (does not mutate input)."""
    out: dict[str, Any] = copy.deepcopy(state)

    max_audit = 48
    max_grid = 32
    max_iterations = 12
    max_model_comparison = 24
    max_str = 120_000

    at = out.get("audit_trace")
    if isinstance(at, list) and len(at) > max_audit:
        out["audit_trace"] = at[-max_audit:]
        out["audit_trace_truncated"] = True

    eg = out.get("experiment_grid_summary")
    if isinstance(eg, list) and len(eg) > max_grid:
        out["experiment_grid_summary"] = eg[:max_grid]
        out["experiment_grid_summary_truncated"] = True

    tse = out.get("task_step_events")
    if isinstance(tse, list) and len(tse) > max_grid:
        out["task_step_events"] = tse[-max_grid:]
        out["task_step_events_truncated"] = True

    mc = out.get("model_comparison")
    if isinstance(mc, list) and len(mc) > max_model_comparison:
        out["model_comparison"] = mc[:max_model_comparison]
        out["model_comparison_truncated"] = True

    tm = out.get("training_metrics")
    if isinstance(tm, dict):
        tm = dict(tm)
        it = tm.get("iterations")
        if isinstance(it, list) and len(it) > max_iterations:
            tm["iterations"] = it[-max_iterations:]
            tm["iterations_truncated"] = True
        out["training_metrics"] = tm

    er = out.get("experiment_result")
    if isinstance(er, dict) and isinstance(er.get("signal_features"), list):
        sf = er["signal_features"]
        if len(sf) > 64:
            er = dict(er)
            er["signal_features"] = sf[:64]
            er["signal_features_truncated"] = True
            out["experiment_result"] = er

    for k, v in list(out.items()):
        if isinstance(v, str) and len(v) > max_str:
            out[k] = v[:max_str] + "…(truncated for DB)"

    return out


def _json_size_bytes(obj: Any) -> int:
    return len(json.dumps(obj, default=str).encode("utf-8"))


def persist_full_training_state_blob(storage_key: str, full_state: dict[str, Any]) -> str:
    """Upload full state JSON to the artifact store; returns the storage key."""
    from backend.shared.artifact_store import get_artifact_store

    store = get_artifact_store()
    return store.upload_json(storage_key, full_state)


def prepare_job_state_for_persistence(
    job_id: str,
    full_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Produce the dict to store on ``TrainingJob.state`` / Celery result: compact summary,
    with an object-storage pointer when the full snapshot is large and R2 is enabled.
    """
    settings = get_settings()
    compact = compact_training_state_for_db(full_state)
    if not settings.r2_enabled:
        return compact

    min_b = settings.TRAINING_STATE_BLOB_MIN_BYTES
    if _json_size_bytes(full_state) < min_b:
        return compact

    key = f"training-state/jobs/{job_id}.json"
    try:
        persist_full_training_state_blob(key, full_state)
        compact["full_state_storage_key"] = key
    except Exception:
        # Still persist compact summary if upload fails
        pass
    return compact


def prepare_experiment_training_state_for_persistence(
    experiment_id: str,
    full_state: dict[str, Any],
) -> dict[str, Any]:
    """Merge-ready training_state patch: compact fields plus optional full blob key."""
    settings = get_settings()
    compact = compact_training_state_for_db(full_state)
    if not settings.r2_enabled:
        return compact

    min_b = settings.TRAINING_STATE_BLOB_MIN_BYTES
    if _json_size_bytes(full_state) < min_b:
        return compact

    safe = re.sub(r"[^\w\-.]", "_", experiment_id)[:200]
    key = f"training-state/experiments/{safe}.json"
    try:
        persist_full_training_state_blob(key, full_state)
        compact["full_state_storage_key"] = key
    except Exception:
        pass
    return compact


def download_json_artifact(key: str) -> dict[str, Any]:
    """Load a JSON object from the artifact store by key."""
    from backend.shared.artifact_store import get_artifact_store

    store = get_artifact_store()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        store.download(key, tmp_path)
        return json.loads(tmp_path.read_text(encoding="utf-8"))
    finally:
        tmp_path.unlink(missing_ok=True)
