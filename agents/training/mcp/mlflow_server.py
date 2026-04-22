"""MLflow helper tools for training / evaluator subagents (optional tracking URI)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool


def _uri() -> str:
    try:
        from backend.shared.settings import get_settings

        return (getattr(get_settings(), "MLFLOW_TRACKING_URI", None) or "").strip()
    except Exception:
        return ""


@tool
def mlflow_start_run(run_name: str, nested: bool = True) -> str:
    """Start an MLflow run (nested=True when already inside a parent run)."""
    uri = _uri()
    if not uri:
        return json.dumps({"ok": False, "error": "MLFLOW_TRACKING_URI not set"})
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        run = mlflow.start_run(run_name=run_name, nested=nested)
        rid = run.info.run_id
        return json.dumps({"ok": True, "run_id": rid})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def mlflow_log_params_json(params_json: str) -> str:
    """Log parameters from a JSON object string."""
    uri = _uri()
    if not uri:
        return json.dumps({"ok": False, "error": "MLFLOW_TRACKING_URI not set"})
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        p = json.loads(params_json or "{}")
        if not isinstance(p, dict):
            return json.dumps({"ok": False, "error": "params_json must be a JSON object"})
        mlflow.log_params({str(k): str(v)[:500] for k, v in p.items()})
        return json.dumps({"ok": True})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def mlflow_log_metrics_json(metrics_json: str) -> str:
    """Log metrics from a JSON object (numeric values)."""
    uri = _uri()
    if not uri:
        return json.dumps({"ok": False, "error": "MLFLOW_TRACKING_URI not set"})
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        m = json.loads(metrics_json or "{}")
        if not isinstance(m, dict):
            return json.dumps({"ok": False, "error": "metrics_json must be a JSON object"})
        flat = {str(k): float(v) for k, v in m.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(flat)
        return json.dumps({"ok": True, "logged": list(flat.keys())})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def mlflow_log_text_artifact(text: str, artifact_file: str) -> str:
    """Write text as an MLflow artifact (path under the active run)."""
    uri = _uri()
    if not uri:
        return json.dumps({"ok": False, "error": "MLFLOW_TRACKING_URI not set"})
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.log_text(text, artifact_file=artifact_file)
        return json.dumps({"ok": True, "artifact_file": artifact_file})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def mlflow_set_tags_json(tags_json: str) -> str:
    """Set tags on the active run from a JSON object."""
    uri = _uri()
    if not uri:
        return json.dumps({"ok": False, "error": "MLFLOW_TRACKING_URI not set"})
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        t = json.loads(tags_json or "{}")
        if not isinstance(t, dict):
            return json.dumps({"ok": False, "error": "tags_json must be a JSON object"})
        mlflow.set_tags({str(k): str(v)[:500] for k, v in t.items()})
        return json.dumps({"ok": True})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def mlflow_end_run() -> str:
    """End the active MLflow run."""
    uri = _uri()
    if not uri:
        return json.dumps({"ok": False, "error": "MLFLOW_TRACKING_URI not set"})
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.end_run()
        return json.dumps({"ok": True})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool
def mlflow_set_model_alias(name: str, alias: str, version: str) -> str:
    """Set a Model Registry alias (e.g. champion, challenger) on a registered model."""
    uri = _uri()
    if not uri:
        return json.dumps({"ok": False, "error": "MLFLOW_TRACKING_URI not set"})
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=uri)
        client.set_registered_model_alias(name, alias, int(version))
        return json.dumps({"ok": True, "model_uri": f"models:/{name}@{alias}"})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool("mlflow_evaluate_sklearn_model")
def mlflow_evaluate_sklearn_model(
    model_name: str,
    dataset_ref: str,
    target_column: str,
    model_type: str = "classifier",
) -> str:
    """Run Jubilee ``evaluate_model`` and optionally log the report to MLflow as text."""
    uri = _uri()
    _mt = Path(__file__).resolve().parents[3] / "tools" / "models-tools" / "training"
    import sys

    if str(_mt) not in sys.path:
        sys.path.insert(0, str(_mt))
    try:
        from model_storage import evaluate_model_tool

        report = evaluate_model_tool.invoke(
            {
                "model_name": model_name,
                "dataset_ref": dataset_ref,
                "target_column": target_column,
            }
        )
        if uri:
            try:
                import mlflow

                mlflow.set_tracking_uri(uri.strip())
                with mlflow.start_run(run_name=f"evaluate_{model_name}", nested=True):
                    mlflow.log_text(str(report), artifact_file="evaluation_report.md")
            except Exception:
                pass
        return json.dumps({"ok": True, "report_markdown": str(report)[:50_000]})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


@tool("mlflow_models_serve_command")
def mlflow_models_serve_command(model_uri: str, port: int = 5001, host: str = "127.0.0.1") -> str:
    """Return a shell command to serve a registered model URI."""
    return json.dumps(
        {
            "ok": True,
            "command": f"mlflow models serve -m {model_uri!r} --host {host} -p {port} --env-manager local",
        }
    )


@tool("smoke_test_json_endpoint")
def smoke_test_json_endpoint_tool(endpoint: str, payload_json: str = "{}") -> str:
    """POST JSON to an endpoint; returns status and short body."""
    import urllib.error
    import urllib.request

    try:
        data = payload_json.encode("utf-8")
        req = urllib.request.Request(
            endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read()[:2000].decode("utf-8", errors="replace")
            return json.dumps({"ok": True, "status": resp.status, "body_preview": body})
    except urllib.error.HTTPError as e:
        return json.dumps({"ok": False, "status": e.code, "error": str(e)})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


def build_mlflow_tools():
    return [
        mlflow_start_run,
        mlflow_log_params_json,
        mlflow_log_metrics_json,
        mlflow_log_text_artifact,
        mlflow_set_tags_json,
        mlflow_end_run,
        mlflow_set_model_alias,
        mlflow_evaluate_sklearn_model,
        mlflow_models_serve_command,
        smoke_test_json_endpoint_tool,
    ]
