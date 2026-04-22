"""Evidently-based drift / performance tools for the monitor subagent."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


def _dt_path() -> Path:
    return Path(__file__).resolve().parents[3] / "tools" / "data-tools"


def _ensure_dt() -> None:
    p = str(_dt_path())
    if p not in sys.path:
        sys.path.insert(0, p)


@tool("evidently_data_drift")
def evidently_data_drift_tool(
    reference_dataset_ref: str,
    current_dataset_ref: str,
    columns_json: str = "[]",
) -> str:
    """Run Evidently DatasetDriftMetric / DataDriftPreset on two registered frames."""
    _ensure_dt()
    try:
        from utils import get_registered_dataset

        ref = get_registered_dataset(reference_dataset_ref)
        cur = get_registered_dataset(current_dataset_ref)
        if ref is None or cur is None:
            return json.dumps({"ok": False, "error": "dataset ref not found"})
        cols = json.loads(columns_json or "[]")
        if cols:
            ref = ref[[c for c in cols if c in ref.columns]]
            cur = cur[[c for c in cols if c in cur.columns]]
        snap: dict | list | str
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            rep = Report(metrics=[DataDriftPreset()])
            my_run = rep.run(reference_data=ref, current_data=cur)
            snap = my_run.json()
        except Exception:
            snap = {
                "fallback": "evidently_unavailable_or_failed",
                "reference_shape": list(ref.shape),
                "current_shape": list(cur.shape),
                "column_mean_delta": {
                    c: float(cur[c].mean() - ref[c].mean())
                    for c in ref.columns
                    if c in cur.columns and ref[c].dtype.kind in "fiu" and cur[c].dtype.kind in "fiu"
                },
            }
        return json.dumps({"ok": True, "report_json": snap}, default=str)[:50_000]
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool("evidently_model_performance")
def evidently_model_performance_tool(
    reference_dataset_ref: str,
    current_dataset_ref: str,
    prediction_column: str,
    target_column: str,
) -> str:
    """Compare regression/classification performance columns between two snapshots (simple column stats)."""
    _ensure_dt()
    try:
        from utils import get_registered_dataset

        ref = get_registered_dataset(reference_dataset_ref)
        cur = get_registered_dataset(current_dataset_ref)
        if ref is None or cur is None:
            return json.dumps({"ok": False, "error": "dataset ref not found"})
        if prediction_column not in cur.columns or target_column not in cur.columns:
            return json.dumps({"ok": False, "error": "missing prediction or target column"})
        err = (cur[prediction_column] - cur[target_column]).abs().mean()
        return json.dumps(
            {
                "ok": True,
                "mean_absolute_error_current": float(err),
                "n_current": len(cur),
                "n_reference": len(ref),
            }
        )
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


_RETRAIN_REQUESTS: list[dict] = []


@tool("open_retrain_request")
def open_retrain_request_tool(model_name: str, reason: str, evidence_uri: str = "") -> str:
    """Record a retrain request for the orchestrator / human (in-process queue)."""
    rid = str(uuid.uuid4())
    _RETRAIN_REQUESTS.append(
        {
            "id": rid,
            "model_name": model_name,
            "reason": reason,
            "evidence_uri": evidence_uri,
        }
    )
    return json.dumps({"ok": True, "retrain_request_id": rid})


@tool("load_recent_inferences")
def load_recent_inferences_tool(model_name: str, since_iso: str = "") -> str:
    """Placeholder: return empty list until inference logging is wired to Postgres."""
    return json.dumps(
        {
            "ok": True,
            "model_name": model_name,
            "since": since_iso,
            "rows": [],
            "note": "Wire predictions table to populate this tool.",
        }
    )


def build_drift_tools():
    return [
        evidently_data_drift_tool,
        evidently_model_performance_tool,
        open_retrain_request_tool,
        load_recent_inferences_tool,
    ]


def drain_retrain_requests() -> list[dict]:
    """Pop all pending retrain requests (for scheduler / API)."""
    global _RETRAIN_REQUESTS
    out = list(_RETRAIN_REQUESTS)
    _RETRAIN_REQUESTS = []
    return out
