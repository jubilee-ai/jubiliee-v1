"""
Jubilee MCP Server — Exposes ML tools via the Model Context Protocol.

Tools:
  list_models       - Available model families + trained model versions
  list_datasets     - Dataset catalog with optional search
  predict           - Run inference with a trained model
  train_model       - Train a new ML model (long-running)
  assign_task       - High-level NL interface: ask a prediction question, get an answer
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Path bootstrap — ensure project root and tool dirs are importable.
# Done eagerly at import time so tools resolve regardless of cwd.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

for _p in [
    _PROJECT_ROOT,
    _PROJECT_ROOT / "tools" / "data-tools",
    _PROJECT_ROOT / "tools" / "models-tools" / "training",
]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Eagerly load .env so DATABASE_URL / OPENAI_API_KEY are available
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env", override=True)

from backend.shared.settings import bootstrap_paths  # noqa: E402

bootstrap_paths()


@asynccontextmanager
async def _lifespan(server: FastMCP):
    """Verify DB connectivity on startup (best-effort)."""
    try:
        from backend.shared.database import get_db_session

        with get_db_session() as session:
            session.execute(__import__("sqlalchemy").text("SELECT 1"))
        print("[jubilee-mcp] Database connected", flush=True)
    except Exception as exc:
        print(f"[jubilee-mcp] Database unavailable, using fallbacks: {exc}", flush=True)
    yield {}


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Jubilee ML Platform",
    instructions=(
        "Jubilee is an ML training, prediction, and dataset management platform. "
        "Use list_models / list_datasets to discover what's available, "
        "predict to run inference, train_model to train a new model, "
        "and assign_task for natural-language prediction questions."
    ),
    host="0.0.0.0",
    port=int(os.environ.get("PORT", "8080")),
    lifespan=_lifespan,
)


# ============================================================================
# Tool 1 — list_models
# ============================================================================

@mcp.tool()
def list_models(include_trained: bool = True) -> dict:
    """List available model families and optionally trained model versions.

    Returns:
      - model_families: model types you can train (logistic_regression, xgboost, etc.)
      - trained_models: already-trained models with metrics, features, and version info
    """
    from backend.catalog import repository

    result: dict[str, Any] = {
        "model_families": repository.get_models(),
    }

    if include_trained:
        try:
            import model_storage

            result["trained_models"] = model_storage.list_models()
        except Exception:
            try:
                raw = repository.get_trained_models()
                result["trained_models"] = list(raw.values()) if isinstance(raw, dict) else raw
            except Exception:
                result["trained_models"] = []

    return result


# ============================================================================
# Tool 2 — list_datasets
# ============================================================================

@mcp.tool()
def list_datasets(query: str | None = None) -> dict:
    """List available datasets, optionally filtered by a search query.

    Returns:
      - catalog: datasets from the database / catalog.json (persistent)
      - registered: in-memory datasets from recent transformations
      - sql_tables: tables in the SQL warehouse
    """
    result: dict[str, Any] = {"catalog": [], "registered": [], "sql_tables": []}

    try:
        from backend.catalog import repository

        result["catalog"] = repository.get_datasets()
    except Exception:
        pass

    try:
        from utils import get_all_available_datasets

        all_ds = get_all_available_datasets()
        result["registered"] = all_ds.get("registered", [])
        result["sql_tables"] = all_ds.get("sql_tables", [])
    except Exception:
        pass

    if query:
        q = query.lower()
        result["catalog"] = [
            ds for ds in result["catalog"]
            if q in json.dumps(ds, default=str).lower()
        ]
        result["registered"] = [
            ds for ds in result["registered"]
            if q in ds.get("ref", "").lower()
        ]
        result["sql_tables"] = [
            ds for ds in result["sql_tables"]
            if q in ds.get("ref", "").lower()
            or q in " ".join(ds.get("column_names", [])).lower()
        ]

    return result


# ============================================================================
# Tool 3 — predict
# ============================================================================

@mcp.tool()
def predict(model_name: str, input_data: list[dict]) -> dict:
    """Run inference on one or more records using a trained model.

    Args:
        model_name: Name of a trained model (use list_models to see available).
        input_data: List of dicts — each dict is one sample with feature values.
                    Use the raw column names (e.g. age, income, employed).
                    The server handles any preprocessing the model requires.

    Returns predictions with class probabilities for classifiers,
    or predicted values for regressors.
    """
    import numpy as np
    import pandas as pd

    from model_storage import (
        get_model_info,
        list_models as storage_list_models,
        load_model,
    )

    info = get_model_info(model_name)
    if info is None:
        available = [m["model_name"] for m in storage_list_models()]
        return {
            "error": f"Model '{model_name}' not found.",
            "available_models": available,
        }

    try:
        model = load_model(model_name)
    except Exception as exc:
        return {"error": f"Failed to load model: {exc}"}

    df = pd.DataFrame(input_data)

    # The model may be an sklearn Pipeline with a preprocessor that expects
    # raw feature names. If it's a bare estimator that expects preprocessed
    # column names (num__age, cat__employed_yes, ...), we try to align.
    expected_features = info.get("feature_names", [])

    try:
        predictions_raw = model.predict(df)
    except Exception as first_err:
        if not expected_features:
            return {"error": f"Prediction failed: {first_err}"}

        try:
            df_aligned = _align_features(df, expected_features)
            predictions_raw = model.predict(df_aligned)
            df = df_aligned
        except Exception:
            return {
                "error": f"Prediction failed: {first_err}",
                "hint": f"Model expects features: {expected_features}",
                "received": list(df.columns),
            }

    is_classifier = hasattr(model, "predict_proba") and hasattr(model, "classes_")
    predictions = []

    if is_classifier:
        try:
            probabilities = model.predict_proba(df)
        except Exception:
            probabilities = None

        classes = [str(c) for c in model.classes_] if hasattr(model, "classes_") else []

        for i, pred in enumerate(predictions_raw):
            entry: dict[str, Any] = {"prediction": _jsonable(pred)}
            if probabilities is not None and classes:
                entry["probabilities"] = {
                    c: round(float(p), 4) for c, p in zip(classes, probabilities[i])
                }
            predictions.append(entry)
    else:
        for pred in predictions_raw:
            predictions.append({"prediction": _jsonable(pred)})

    return {
        "model_name": model_name,
        "model_type": info.get("model_type", ""),
        "target_column": info.get("target_column", ""),
        "num_records": len(input_data),
        "predictions": predictions,
    }


def _align_features(df: "pd.DataFrame", expected: list[str]) -> "pd.DataFrame":
    """Try to one-hot-encode / rename columns to match what the model expects.

    Handles the common pattern where the model was trained with a
    ColumnTransformer that produced columns like num__age, cat__employed_yes.
    """
    import pandas as pd

    num_cols = [c.replace("num__", "") for c in expected if c.startswith("num__")]
    cat_cols_raw: dict[str, set[str]] = {}
    for c in expected:
        if c.startswith("cat__"):
            parts = c[5:].rsplit("_", 1)
            if len(parts) == 2:
                cat_cols_raw.setdefault(parts[0], set()).add(parts[1])

    aligned = pd.DataFrame()

    for col in num_cols:
        if col in df.columns:
            aligned[f"num__{col}"] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col, values in cat_cols_raw.items():
        if col in df.columns:
            for val in values:
                aligned[f"cat__{col}_{val}"] = (df[col].astype(str) == val).astype(float)

    return aligned[expected] if set(expected).issubset(aligned.columns) else aligned


def _jsonable(val: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python types."""
    import numpy as np

    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


# ============================================================================
# Tool 4 — train_model
# ============================================================================

@mcp.tool()
def train_model(
    goal: str,
    dataset_ref: str | None = None,
    model_preference: str | None = None,
    use_external_sources: bool = False,
) -> dict:
    """Train a new ML model end-to-end. This is a LONG-RUNNING operation
    (typically 2-10 minutes).

    Runs the full pipeline: model selection -> data collection -> cleaning ->
    label/split definition -> feature engineering -> training -> evaluation.

    Args:
        goal: What to predict, in natural language.
              Example: 'Predict loan default using the Loan Default dataset'.
        dataset_ref: Dataset reference name to train on (from list_datasets).
                     If omitted the agent discovers suitable data.
        model_preference: Model family hint — 'supervised', 'unsupervised',
                          or 'neural_networks'. Omit to auto-select.
        use_external_sources: If True, search Kaggle/HuggingFace for data
                              when local data is insufficient.

    Returns training results with metrics, model name, and report path.
    """
    try:
        from agent import _run_training_to_completion
    except Exception as exc:
        return {"error": f"Cannot import training agent: {exc}"}

    linked = [dataset_ref] if dataset_ref else None

    try:
        state = _run_training_to_completion(
            goal=goal,
            linked_datasets=linked,
            model_preference=model_preference,
            use_external_sources=use_external_sources,
        )
    except Exception as exc:
        return {
            "error": f"Training failed: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    metrics = state.get("training_metrics") or {}
    return {
        "status": "completed" if metrics else "incomplete",
        "selected_model": state.get("selected_model"),
        "model_explanation": state.get("model_explanation"),
        "model_name": metrics.get("model_name"),
        "model_type": metrics.get("model_type"),
        "metrics": {
            k: _jsonable(v) for k, v in metrics.items()
            if k not in ("model_name", "model_type", "summary", "recommendations")
        },
        "summary": metrics.get("summary"),
        "recommendations": metrics.get("recommendations"),
        "report_path": state.get("report_path"),
        "model_weights_path": state.get("model_weights_path"),
    }


# ============================================================================
# Tool 5 — assign_task
# ============================================================================

@mcp.tool()
def assign_task(task: str, data_source_ref: str | None = None) -> dict:
    """Give Jubilee a natural-language prediction question and get back an answer.

    Jubilee autonomously decides the best approach:
      - Use an existing pretrained model for instant answers
      - Analyze data in your datasets
      - Train a new model if needed

    Args:
        task: Your question or prediction request in plain English.
              Examples:
                'What is the default risk for a 35-year-old with 50k income?'
                'What are the top predictors of insurance charges?'
                'Predict financial distress for companies with ratio x1=0.5'
        data_source_ref: Optional dataset reference to scope the analysis to.

    Returns the answer, plus the thread_id for follow-up questions.
    """
    try:
        from agent import agent as orchestrator
    except Exception as exc:
        return {"error": f"Cannot import orchestrator agent: {exc}"}

    prompt = task
    if data_source_ref:
        prompt += f"\n\nUse this dataset: {data_source_ref}"

    thread_id = f"mcp-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = orchestrator.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
        )
    except Exception as exc:
        return {
            "error": f"Task failed: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    messages = result.get("messages", [])
    answer = messages[-1].content if messages else "No response generated."

    return {
        "answer": answer,
        "thread_id": thread_id,
        "data_source_ref": data_source_ref,
    }
