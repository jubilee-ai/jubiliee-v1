"""
Training Agent — Step 7 of the ML Training Pipeline.

Skill-based architecture: the agent reads a skill's SKILL.md (injected into
context) to choose an estimator, calls train_with_skill to train, evaluates,
and iterates.  No hardcoded model lists — the skill drives everything.
"""

import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..utils.graph_stream_hooks import emit_graph_stream
from ..utils.prompts import TRAINING_SYSTEM_PROMPT

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

_MODEL_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "models-tools" / "training"
if str(_MODEL_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_TOOLS_DIR))

from model_storage import (classification_roc_auc, delete_model,
                           evaluate_model_tool, get_model_info,
                           get_model_info_tool, list_models,
                           list_trained_models_tool, load_model)
from utils import get_registered_dataset

SKILLS_DIR = Path(__file__).parent.parent / "skills"


# =============================================================================
# SKILL HELPERS
# =============================================================================


def _load_skill_prompt(skill_name: str) -> str:
    skill_md = SKILLS_DIR / skill_name / "SKILL.md"
    if not skill_md.exists():
        raise ValueError(f"SKILL.md not found for skill '{skill_name}'")
    return skill_md.read_text()


def _run_skill(skill_name: str, params: dict) -> str:
    train_py = SKILLS_DIR / skill_name / "train.py"
    if not train_py.exists():
        raise ValueError(f"train.py not found for skill '{skill_name}'")

    spec = importlib.util.spec_from_file_location(
        f"skill_{skill_name}_train", str(train_py)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "run"):
        raise ValueError(f"Skill '{skill_name}' train.py must define a run(params) function")

    return module.run(params)


# =============================================================================
# TOOLS
# =============================================================================


class TrainWithSkillInput(BaseModel):
    model_config = {"extra": "allow"}

    skill_name: str = Field(
        description="Name of the skill to use (e.g. 'supervised')"
    )
    params: Optional[dict] = Field(
        default=None,
        description="Training parameters as a JSON object. Must include "
        "'estimator', 'model_name', and 'train_dataset_ref'. "
        "'target_column' is required for supervised skills only.",
    )


@tool("train_with_skill", args_schema=TrainWithSkillInput)
def train_with_skill_tool(skill_name: str, params: Optional[dict] = None, **kwargs) -> str:
    """Train a model using a skill.

    The skill automatically discovers the estimator's hyperparameters,
    runs cross-validated tuning, and returns training results along with
    a summary of all tunable parameters for the next iteration.
    """
    if params is None:
        params = {}
    if kwargs:
        params = {**kwargs, **params}
    try:
        return _run_skill(skill_name, params)
    except Exception as e:
        return f"Training failed: {e}"


class FeatureRedoRequest(BaseModel):
    recommendation: str = Field(
        description="Specific recommendation for what to change in feature engineering."
    )
    reason: str = Field(
        description="Why you believe the current features are limiting model performance."
    )
    suspected_issues: list[str] = Field(default_factory=list)


_feature_redo_request: Optional[FeatureRedoRequest] = None


def _get_and_clear_feature_redo_request() -> Optional[FeatureRedoRequest]:
    global _feature_redo_request
    request = _feature_redo_request
    _feature_redo_request = None
    return request


@tool
def request_feature_engineering_redo_tool(
    recommendation: str,
    reason: str,
    suspected_issues: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Request to redo feature engineering with specific recommendations.

    Only call when multiple estimators all perform poorly despite tuning
    and you have specific recommendations for feature changes.
    """
    global _feature_redo_request
    _feature_redo_request = FeatureRedoRequest(
        recommendation=recommendation,
        reason=reason,
        suspected_issues=suspected_issues or [],
    )
    return {
        "status": "registered",
        "message": "Feature engineering redo requested. Training will stop after this call.",
        "recommendation": recommendation,
        "reason": reason,
        "suspected_issues": suspected_issues or [],
    }


class BatchTrainConfig(BaseModel):
    estimator: str = Field(description="Estimator class name, e.g. 'HistGradientBoostingClassifier'")
    model_name: str = Field(description="Unique name for this model, e.g. 'hgb_v1'")
    hyperparams: dict = Field(default_factory=dict, description="Hyperparameter overrides for this estimator")


class BatchTrainInput(BaseModel):
    skill_name: str = Field(description="Skill to use for all configs (e.g. 'supervised')")
    configs: list[BatchTrainConfig] = Field(
        description="List of 2-3 training configs to run in parallel",
        min_length=1,
        max_length=4,
    )
    train_dataset_ref: str = Field(description="Registered training dataset ref")
    val_dataset_ref: Optional[str] = Field(default=None, description="Registered validation dataset ref")
    target_column: str = Field(default="", description="Target column name (empty for unsupervised)")


@tool("batch_train_with_skill", args_schema=BatchTrainInput)
def batch_train_with_skill_tool(
    skill_name: str,
    configs: list[BatchTrainConfig],
    train_dataset_ref: str,
    val_dataset_ref: Optional[str] = None,
    target_column: str = "",
) -> str:
    """Train 2-3 models in parallel using a skill and compare results.

    Each config specifies an estimator and hyperparameters. All configs run
    concurrently via threads, then results are compared side-by-side.
    Use this instead of calling train_with_skill multiple times sequentially.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    configs = configs[:4]

    def _run_one(cfg: BatchTrainConfig) -> dict:
        params = {
            "estimator": cfg.estimator,
            "model_name": cfg.model_name,
            "train_dataset_ref": train_dataset_ref,
            "target_column": target_column,
            **cfg.hyperparams,
        }
        if val_dataset_ref:
            params["val_dataset_ref"] = val_dataset_ref
        try:
            raw = _run_skill(skill_name, params)
            return {"model_name": cfg.model_name, "estimator": cfg.estimator, "success": True, "raw_output": raw}
        except Exception as e:
            return {"model_name": cfg.model_name, "estimator": cfg.estimator, "success": False, "error": str(e)}

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(configs), 3)) as pool:
        futures = {pool.submit(_run_one, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            results.append(future.result())

    lines = [f"## Batch Training Results ({len(results)} models)\n"]
    for i, r in enumerate(results, 1):
        if r["success"]:
            output_preview = r["raw_output"][:2000] if isinstance(r["raw_output"], str) else str(r["raw_output"])[:2000]
            lines.append(f"### {i}. {r['model_name']} ({r['estimator']}) — SUCCESS\n{output_preview}\n")
        else:
            lines.append(f"### {i}. {r['model_name']} ({r['estimator']}) — FAILED\n{r['error']}\n")

    lines.append(
        "\nUse `evaluate_model` or `get_model_info` on individual models above to compare metrics, "
        "then pick the best for further tuning or final evaluation."
    )
    return "\n".join(lines)


TRAINING_TOOLS = [
    train_with_skill_tool,
    batch_train_with_skill_tool,
    evaluate_model_tool,
    list_trained_models_tool,
    get_model_info_tool,
    request_feature_engineering_redo_tool,
]


# =============================================================================
# STRUCTURED OUTPUT SCHEMA
# =============================================================================


class TrainingIteration(BaseModel):
    model_name: str = Field(description="Name of the model for this iteration")
    tool_used: str = Field(description="Skill and estimator used")
    hyperparams: dict = Field(default_factory=dict)
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_roc_auc: Optional[float] = None
    train_r2: Optional[float] = None
    val_r2: Optional[float] = None
    val_rmse: Optional[float] = None
    val_mae: Optional[float] = None
    test_r2: Optional[float] = None
    test_rmse: Optional[float] = None
    test_mae: Optional[float] = None
    silhouette_score: Optional[float] = None
    davies_bouldin: Optional[float] = None
    inertia: Optional[float] = None
    reconstruction_loss: Optional[float] = None
    success: bool = Field(description="Whether this iteration succeeded")
    error: Optional[str] = None


class TrainingResult(BaseModel):
    success: bool
    best_model_name: str
    model_type: str = Field(description="Estimator class name")
    val_accuracy: Optional[float] = None
    val_roc_auc: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_roc_auc: Optional[float] = None
    train_r2: Optional[float] = None
    val_r2: Optional[float] = None
    val_rmse: Optional[float] = None
    val_mae: Optional[float] = None
    test_r2: Optional[float] = None
    test_rmse: Optional[float] = None
    test_mae: Optional[float] = None
    silhouette_score: Optional[float] = None
    davies_bouldin: Optional[float] = None
    inertia: Optional[float] = None
    reconstruction_loss: Optional[float] = None
    iterations: list[TrainingIteration] = Field(default_factory=list)
    num_iterations: int
    summary: str = Field(
        description="Concise narrative: experiment arc, why best_model_name won, key metrics stated once (no duplicate numbers).",
    )
    recommendations: Optional[str] = Field(
        default=None,
        description="Brief actionable bullets for best_model_name (deploy, thresholds, monitoring); other models only for short comparison.",
    )
    feature_redo_requested: bool = False


# =============================================================================
# HELPERS
# =============================================================================


def _infer_task_type(goal: str, estimator_hint: Optional[str] = None, selected_model: Optional[str] = None) -> str:
    """Infer 'classification', 'regression', or 'unsupervised' from the goal/model."""
    if selected_model == "unsupervised":
        return "unsupervised"
    text = (goal + " " + (estimator_hint or "")).lower()
    if any(
        kw in text
        for kw in (
            "unsupervised",
            "cluster",
            "clustering",
            "segmentation",
            "anomaly",
            "outlier",
            "dimensionality reduction",
            "pca",
        )
    ):
        return "unsupervised"
    if "regress" in text or any(
        kw in text for kw in ("forecast", "predict value", "continuous", "amount", "price", "cost")
    ):
        return "regression"
    return "classification"


def _run_quick_baseline(
    train_df,
    val_df,
    target_column: str,
    task_type: str,
) -> dict[str, float | None]:
    """Run a quick tree-based baseline to establish a performance floor.

    Returns dict with baseline metrics (accuracy, roc_auc or r2, rmse).
    This runs in <10s even on large datasets and gives the NN agent a
    concrete target to beat.
    """
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    result: dict[str, float | None] = {"model": "HistGradientBoosting"}
    try:
        feature_cols = [c for c in train_df.columns if c != target_column]
        X_train = train_df[feature_cols]
        y_train = train_df[target_column]
        X_val = val_df[feature_cols]
        y_val = val_df[target_column]

        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            from sklearn.preprocessing import OrdinalEncoder
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_train = X_train.copy()
            X_val = X_val.copy()
            X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
            X_val[cat_cols] = oe.transform(X_val[cat_cols])

        le = None
        if y_train.dtype == object or y_train.dtype.name == "category":
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_val = le.transform(y_val)

        if task_type == "regression":
            model = HistGradientBoostingRegressor(max_iter=200, max_depth=6, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            result["r2"] = round(float(r2_score(y_val, y_pred)), 4)
        else:
            model = HistGradientBoostingClassifier(max_iter=200, max_depth=6, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            result["accuracy"] = round(float(accuracy_score(y_val, y_pred)), 4)
            try:
                if y_proba.shape[1] == 2:
                    result["roc_auc"] = round(float(roc_auc_score(y_val, y_proba[:, 1])), 4)
                else:
                    result["roc_auc"] = round(float(
                        roc_auc_score(y_val, y_proba, multi_class="ovr", average="weighted")
                    ), 4)
            except (ValueError, TypeError):
                pass

        print(f"[training_agent] Quick baseline ({result['model']}): {result}")
    except Exception as e:
        print(f"[training_agent] Quick baseline failed (non-fatal): {e}")
    return result


def _cleanup_intermediate_models(
    best_model_name: str,
    base_model_name: str,
    iteration_model_names: Optional[list[str]] = None,
) -> list[str]:
    deleted = []
    version_pattern = rf"^{re.escape(base_model_name)}_v\d+$"
    iteration_set = set(iteration_model_names or [])

    for model_info in list_models():
        name = model_info["model_name"]
        if name == best_model_name:
            continue
        if re.match(version_pattern, name) or name in iteration_set:
            if delete_model(name):
                deleted.append(name)
                print(f"[training_agent] Cleaned up intermediate model: {name}")
    return deleted


def _extract_training_result(result: dict) -> TrainingResult:
    structured = result.get("structured_response")
    if isinstance(structured, TrainingResult):
        return structured

    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", None)
        if content is None:
            continue
        if isinstance(content, TrainingResult):
            return content
        if isinstance(content, dict):
            try:
                return TrainingResult(**content)
            except Exception:
                continue
        if isinstance(content, str):
            try:
                return TrainingResult(**json.loads(content))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    raise ValueError("Failed to extract TrainingResult from agent response")


def _should_continue_iterating(
    iterations: list[TrainingIteration],
    max_iterations: int,
    task_type: str,
) -> bool:
    """Decide whether the agent should be re-invoked for more experiments.

    Returns True when there are iterations remaining AND at least one of:
    - Fewer than 3 successful experiments have been run
    - The most recent iteration improved the best metric
    """
    n_used = len(iterations)
    if n_used >= max_iterations:
        return False

    successful = [it for it in iterations if it.success]
    if len(successful) < 3:
        return True

    if len(successful) < 2:
        return True

    def _metric(it: TrainingIteration) -> float:
        if task_type == "unsupervised":
            return it.silhouette_score if it.silhouette_score is not None else -float("inf")
        if task_type == "regression":
            return it.val_r2 or it.train_r2 or -float("inf")
        return it.val_roc_auc or it.val_accuracy or -float("inf")

    best_before_last = max(_metric(it) for it in successful[:-1])
    last_metric = _metric(successful[-1])
    if last_metric > best_before_last:
        return True

    return False


def _format_best_metric(best_iteration: dict, task_type: str) -> str:
    if task_type == "unsupervised":
        sil = best_iteration.get("silhouette_score")
        db = best_iteration.get("davies_bouldin")
        parts = []
        if sil is not None:
            parts.append(f"Silhouette={sil:.4f}")
        if db is not None:
            parts.append(f"Davies-Bouldin={db:.4f}")
        return ", ".join(parts) if parts else "N/A"
    if task_type == "regression":
        r2 = best_iteration.get("val_r2")
        return f"R²={r2:.4f}" if r2 is not None else "N/A"
    roc = best_iteration.get("val_roc_auc")
    acc = best_iteration.get("val_accuracy")
    parts = []
    if roc is not None:
        parts.append(f"ROC-AUC={roc:.4f}")
    if acc is not None:
        parts.append(f"Accuracy={acc:.4f}")
    return ", ".join(parts) if parts else "N/A"


def _primary_metric(it: TrainingIteration, task_type: str) -> float:
    if task_type == "unsupervised":
        return it.silhouette_score if it.silhouette_score is not None else -float("inf")
    if task_type == "regression":
        return it.val_r2 or it.train_r2 or -float("inf")
    return it.val_roc_auc or it.val_accuracy or -float("inf")


def _diagnose_trend(
    iterations: list[TrainingIteration],
    task_type: str,
) -> dict[str, Any]:
    """Analyze the experiment history and diagnose the current situation."""
    successful = [it for it in iterations if it.success]
    if not successful:
        return {"trend": "no_data", "diagnosis": "No successful experiments yet."}

    metrics = [_primary_metric(it, task_type) for it in successful]
    best_metric = max(metrics)
    best_idx = metrics.index(best_metric)
    last_metric = metrics[-1]

    if len(metrics) >= 2:
        recent_deltas = [metrics[i] - metrics[i - 1] for i in range(1, len(metrics))]
        improving = recent_deltas[-1] > 0.001
        plateauing = all(abs(d) < 0.005 for d in recent_deltas[-2:]) if len(recent_deltas) >= 2 else False
    else:
        improving = False
        plateauing = False

    last_was_best = (best_idx == len(metrics) - 1)
    gap_to_best = best_metric - last_metric if not last_was_best else 0.0

    if plateauing:
        trend = "plateau"
        diagnosis = (
            f"Metrics have plateaued — last {min(3, len(recent_deltas))} changes "
            f"moved the metric by < 0.5%. Consider a more radical change: "
            f"different architecture family, different optimizer, or different LR schedule."
        )
    elif improving:
        trend = "improving"
        diagnosis = (
            f"Last change improved the metric. Continue in this direction — "
            f"make a similar-magnitude change to the same aspect, or try refining "
            f"the improvement further."
        )
    elif last_was_best:
        trend = "at_best"
        diagnosis = "Current config is the best so far. Try a single targeted change."
    else:
        trend = "regressed"
        diagnosis = (
            f"Last experiment regressed ({last_metric:.4f} vs best {best_metric:.4f}). "
            f"Revert to the best config and try a different change."
        )

    return {
        "trend": trend,
        "diagnosis": diagnosis,
        "best_metric": best_metric,
        "last_metric": last_metric,
        "best_model": successful[best_idx].model_name,
        "n_successful": len(successful),
        "improving": improving,
        "plateauing": plateauing,
        "gap_to_best": gap_to_best,
    }


def _build_continuation_message(
    iterations: list[TrainingIteration],
    max_iterations: int,
    task_type: str,
    baseline_metrics: dict | None = None,
) -> str:
    """Build a dynamic continuation prompt based on experiment history."""
    remaining = max_iterations - len(iterations)
    diag = _diagnose_trend(iterations, task_type)

    lines = [
        f"You have **{remaining} iterations** remaining.",
        f"",
        f"## Experiment History",
    ]
    for it in iterations:
        m = _primary_metric(it, task_type)
        status = "OK" if it.success else "FAIL"
        lines.append(f"- [{status}] `{it.model_name}`: {m:.4f}")
    lines.append("")

    if baseline_metrics:
        baseline_val = baseline_metrics.get("roc_auc") or baseline_metrics.get("r2")
        if baseline_val is not None and diag.get("best_metric") is not None:
            gap = baseline_val - diag["best_metric"]
            if gap > 0:
                lines.append(
                    f"## Baseline Gap\n"
                    f"Tree-based baseline: {baseline_val:.4f}. "
                    f"Your best: {diag['best_metric']:.4f}. "
                    f"**Gap: {gap:.4f}** — focus on closing this.\n"
                )
            else:
                lines.append(
                    f"## Baseline Beaten\n"
                    f"Your best ({diag['best_metric']:.4f}) exceeds the tree baseline "
                    f"({baseline_val:.4f}). Keep pushing for further gains.\n"
                )

    lines.append(f"## Diagnosis\n{diag['diagnosis']}\n")

    lines.append(
        "## Next Steps\n"
        "Consult the **Hyperparameter Strategy Matrix** and **Architecture Decision Guide** "
        "in the SKILL.md to choose your next experiment based on the diagnosis above. "
        "Change exactly ONE thing. Run the experiment, evaluate, and report results.\n"
        "In the structured `iterations` array, append only **new** attempts from this round "
        "(prior attempts are already stored). Your **`summary`** and **`recommendations`** "
        "must still cover the **entire run** and the model you set as **`best_model_name`** "
        "(the artifact headline metrics will follow), not only the iterations added here."
    )

    return "\n".join(lines)


def _find_best_iteration(iterations: list[dict], task_type: str) -> Optional[dict]:
    """Pick the best successful iteration by validation metrics.

    For classification, ROC-AUC is the primary metric (more reliable than
    accuracy, especially on imbalanced data). Accuracy is the tiebreaker.
    For unsupervised, silhouette_score is primary (higher is better).
    """
    best, best_score = None, (-float("inf"), -float("inf"))
    for it in iterations:
        if not it.get("success"):
            continue
        if task_type == "unsupervised":
            score = (
                it.get("silhouette_score") or -float("inf"),
                -(it.get("davies_bouldin") or float("inf")),
            )
        elif task_type == "regression":
            primary = it.get("val_r2") or it.get("train_r2") or -float("inf")
            score = (primary, 0.0)
        else:
            score = (
                it.get("val_roc_auc") or -float("inf"),
                it.get("val_accuracy") or -float("inf"),
            )
        if score > best_score:
            best_score = score
            best = it
    return best


def _training_result_updates_from_best_iteration(
    best: Optional[dict],
    fallback_best_name: str,
) -> dict:
    """Fields to merge into TrainingResult so logs/API match cleanup and test eval.

    The LLM may set ``best_model_name`` using narrative criteria; we always
    reconcile to :func:`_find_best_iteration` before logging or returning.
    """
    updates: dict = {"best_model_name": best["model_name"] if best else fallback_best_name}
    if not best:
        return updates
    for fld in (
        "val_accuracy",
        "val_roc_auc",
        "test_accuracy",
        "test_roc_auc",
        "train_r2",
        "val_r2",
        "val_rmse",
        "val_mae",
        "test_r2",
        "test_rmse",
        "test_mae",
        "silhouette_score",
        "davies_bouldin",
        "inertia",
        "reconstruction_loss",
    ):
        v = best.get(fld)
        if v is not None:
            updates[fld] = v
    tool = best.get("tool") or best.get("tool_used")
    if tool:
        updates["model_type"] = str(tool).rsplit(".", 1)[-1]
    return updates


def _maybe_clarify_summary_vs_saved_model(training_result: TrainingResult) -> TrainingResult:
    """If the LLM's prose focused on a later experiment that is not the saved best model, prepend context.

    Metrics and ``model_name`` are reconciled to the best validation iteration after the agent
    returns; ``summary`` text often still describes only the last continuation round.
    """
    saved = (training_result.best_model_name or "").strip()
    if not saved:
        return training_result
    succ_named = [it for it in training_result.iterations if it.success and (it.model_name or "").strip()]
    if not succ_named:
        return training_result
    last_name = succ_named[-1].model_name
    if last_name == saved:
        return training_result
    body = (training_result.summary or "").strip()
    if not body:
        return training_result
    prefix = (
        f"The headline test metrics and saved artifact refer to **{saved}**, chosen by validation scores. "
        f"The text below discusses a later experiment (**{last_name}**) that was not selected as the best model.\n\n"
    )
    return training_result.model_copy(update={"summary": prefix + body})


def _iteration_to_dict(it: TrainingIteration) -> dict:
    d = it.model_dump()
    d["tool"] = d.pop("tool_used")
    d["metrics"] = {
        "train_accuracy": it.train_accuracy,
        "val_accuracy": it.val_accuracy,
        "roc_auc": it.val_roc_auc,
    }
    if not d.get("hyperparams") and it.model_name:
        info = get_model_info(it.model_name)
        if info and info.get("hyperparameters"):
            d["hyperparams"] = info["hyperparameters"]
    return d


def _evaluate_model_on_test(
    model_name: str,
    test_ref: str,
    target_column: str,
    task_type: str,
) -> dict[str, float | None]:
    import numpy as np
    from sklearn.metrics import accuracy_score

    result: dict[str, float | None] = {}
    if task_type == "unsupervised":
        return result
    try:
        model = load_model(model_name)
        test_df = get_registered_dataset(test_ref)
        if model is None or test_df is None:
            return result

        y_true = test_df[target_column]
        X = test_df[[c for c in test_df.columns if c != target_column]]

        if task_type == "regression":
            from sklearn.metrics import (mean_absolute_error,
                                         mean_squared_error, r2_score)
            y_pred = model.predict(X)
            result["test_r2"] = float(r2_score(y_true, y_pred))
            result["test_rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            result["test_mae"] = float(mean_absolute_error(y_true, y_pred))
        else:
            y_pred = model.predict(X)
            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)
            if y_true_arr.dtype != y_pred_arr.dtype:
                y_pred_arr = y_pred_arr.astype(y_true_arr.dtype)
            result["test_accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
            roc = classification_roc_auc(model, X, y_true_arr)
            if roc is not None:
                result["test_roc_auc"] = float(roc)
        print(f"[training_agent] Programmatic test evaluation: {result}")
    except Exception as exc:
        print(f"[training_agent] Programmatic test evaluation failed: {exc}")
    return result


def _aggregate_transformed_importances_to_raw(
    transformed_names: list[str],
    importances: np.ndarray,
    raw_feature_cols: list[str],
) -> dict[str, float]:
    """Map preprocessor output names (e.g. num__Age, cat__Education_MBA) back to DataFrame columns."""
    agg: dict[str, float] = defaultdict(float)
    raw_sorted = sorted(raw_feature_cols, key=len, reverse=True)

    for fname, imp in zip(transformed_names, importances):
        key = fname.split("__", 1)[-1] if "__" in fname else fname
        matched = None
        for col in raw_sorted:
            if key == col or key.startswith(col + "_"):
                matched = col
                break
        if matched is None:
            matched = key
        agg[matched] += float(imp)

    rounded = {k: round(v, 4) for k, v in sorted(agg.items(), key=lambda x: x[1], reverse=True)}
    return rounded


def _extract_feature_importances(model_name: str, feature_columns: list[str]) -> dict[str, float]:
    """Extract feature importances from a trained sklearn model.

    Supervised `train.py` fits a Pipeline(preprocessor, model) where the tree/linear
    step sees one-hot-encoded columns. Importances length matches
    `preprocessor.get_feature_names_out()`, not raw `feature_columns`, so we
    aggregate OHE splits back onto original column names when possible.
    """
    try:
        model = load_model(model_name)
        if model is None:
            return {}

        outer = model
        if hasattr(outer, "best_estimator_"):
            outer = outer.best_estimator_

        preproc = None
        final_est = None
        if hasattr(outer, "named_steps"):
            ns = outer.named_steps
            preproc = ns.get("preprocessor")
            final_est = ns.get("model")
        if final_est is None and getattr(outer, "steps", None):
            final_est = outer.steps[-1][1]
            if preproc is None and len(outer.steps) >= 2:
                first_name, first_step = outer.steps[0]
                if first_name == "preprocessor":
                    preproc = first_step

        importances_arr: np.ndarray | None = None
        if final_est is not None:
            if hasattr(final_est, "feature_importances_"):
                importances_arr = np.asarray(final_est.feature_importances_, dtype=float)
            elif hasattr(final_est, "coef_"):
                coef = final_est.coef_
                folded = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
                importances_arr = np.asarray(folded, dtype=float)

        if importances_arr is None:
            est = outer
            if hasattr(est, "feature_importances_"):
                importances_arr = np.asarray(est.feature_importances_, dtype=float)
            elif hasattr(est, "coef_"):
                coef = est.coef_
                folded = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
                importances_arr = np.asarray(folded, dtype=float)

        if importances_arr is None or importances_arr.size == 0:
            return {}

        transformed_names: list[str] | None = None
        if preproc is not None:
            try:
                transformed_names = [str(x) for x in preproc.get_feature_names_out()]
            except (AttributeError, ValueError, NotImplementedError, TypeError):
                transformed_names = None

        if transformed_names is not None and len(transformed_names) == len(importances_arr):
            if feature_columns:
                return _aggregate_transformed_importances_to_raw(
                    transformed_names, importances_arr, feature_columns,
                )
            detail = {n: round(float(v), 4) for n, v in zip(transformed_names, importances_arr)}
            return dict(sorted(detail.items(), key=lambda x: x[1], reverse=True))

        info = get_model_info(model_name)
        reg_names = (info or {}).get("feature_names") or []
        if isinstance(reg_names, list) and len(reg_names) == len(importances_arr):
            if feature_columns:
                return _aggregate_transformed_importances_to_raw(
                    [str(x) for x in reg_names], importances_arr, feature_columns,
                )
            detail = {str(n): round(float(v), 4) for n, v in zip(reg_names, importances_arr)}
            return dict(sorted(detail.items(), key=lambda x: x[1], reverse=True))

        if len(importances_arr) == len(feature_columns):
            result = {
                col: round(float(v), 4)
                for col, v in zip(feature_columns, importances_arr)
            }
            return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

        return {}
    except Exception:
        return {}


def _log_training_results(training_result: TrainingResult, task_type: str):
    print(f"\n[training_agent] Training complete!")
    print(f"  Success: {training_result.success}")
    print(f"  Iterations: {training_result.num_iterations}")
    print()
    print("  ITERATION LOG:")
    print("  " + "-" * 60)
    for i, it in enumerate(training_result.iterations, 1):
        status = "OK" if it.success else "FAIL"
        hp_str = ", ".join(f"{k}={v}" for k, v in it.hyperparams.items() if v is not None)
        print(f"  [{status}] Iter {i}: {it.model_name}")
        print(f"     Estimator: {it.tool_used}")
        print(f"     Hyperparams: {hp_str[:60]}...")
        if task_type == "unsupervised":
            if it.silhouette_score is not None:
                print(f"     Silhouette: {it.silhouette_score}, Davies-Bouldin: {it.davies_bouldin}")
        elif task_type == "regression":
            if it.val_r2 is not None:
                print(f"     Val R²: {it.val_r2}, Val RMSE: {it.val_rmse}")
        else:
            if it.val_accuracy is not None or it.val_roc_auc is not None:
                print(f"     Val Accuracy: {it.val_accuracy}, Val ROC-AUC: {it.val_roc_auc}")
        if it.error:
            print(f"     Error: {it.error}")
    print("  " + "-" * 60)
    print(f"  Best Model: {training_result.best_model_name}")
    if task_type == "unsupervised":
        print(f"  Silhouette: {training_result.silhouette_score}")
        print(f"  Davies-Bouldin: {training_result.davies_bouldin}")
    elif task_type == "regression":
        print(f"  Val R²: {training_result.val_r2}")
        print(f"  Test R²: {training_result.test_r2}")
    else:
        print(f"  Val Accuracy: {training_result.val_accuracy}")
        print(f"  Val ROC-AUC: {training_result.val_roc_auc}")
        print(f"  Test Accuracy: {training_result.test_accuracy}")
        print(f"  Test ROC-AUC: {training_result.test_roc_auc}")
    print(f"\n  Summary: {training_result.summary}")
    if training_result.recommendations:
        print(f"  Recommendations: {training_result.recommendations}")


# =============================================================================
# MAIN
# =============================================================================


def run_training_agent(
    train_ref: str,
    val_ref: Optional[str],
    test_ref: Optional[str],
    target_column: str,
    selected_model: str,
    goal: str,
    model_name: Optional[str] = None,
    max_iterations: int = 8,
    llm_model: str = "openai:gpt-5.1",
    estimator_hint: Optional[str] = None,
    experiment_result: Optional[dict[str, Any]] = None,
    feature_rankings: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Run the training agent.

    The agent reads the skill's SKILL.md (injected in context) to select the
    right estimator, trains via train_with_skill, evaluates, and iterates.
    """
    train_df = get_registered_dataset(train_ref)
    val_df = get_registered_dataset(val_ref) if val_ref else None
    test_df = get_registered_dataset(test_ref) if test_ref else None

    if train_df is None:
        raise ValueError(f"Training dataset not found: {train_ref}")
    for ref, df, label in [(val_ref, val_df, "Validation"), (test_ref, test_df, "Test")]:
        if ref and df is None:
            raise ValueError(f"{label} dataset not found: {ref}")

    task_type = _infer_task_type(goal, estimator_hint, selected_model=selected_model)

    available_skills = [d.name for d in SKILLS_DIR.iterdir() if (d / "train.py").exists()]
    skill_name = selected_model if selected_model in available_skills else "supervised"

    if not model_name:
        model_name = f"{estimator_hint or skill_name}_{int(time.time())}"

    feature_columns = [c for c in train_df.columns if c != target_column] if target_column else list(train_df.columns)

    print(f"[training_agent] Starting training...")
    print(f"  Skill: {skill_name} | Target: {target_column or '(none)'} | Task: {task_type}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df) if val_df is not None else 0} | Test: {len(test_df) if test_df is not None else 0} | Features: {len(feature_columns)}")
    if estimator_hint:
        print(f"  Estimator hint: {estimator_hint}")

    emit_graph_stream({"type": "progress", "message": f"Starting training with {skill_name} skill...", "phase": "training"})

    if task_type == "unsupervised":
        class_counts = {}
        imbalance_note = "Unsupervised task — no target variable."
    else:
        class_counts = train_df[target_column].value_counts().to_dict()
        total = sum(class_counts.values())
        minority_ratio = min(class_counts.values()) / total if total > 0 else 0
        imbalance_note = (
            f"IMBALANCED — minority class is {minority_ratio:.1%}. "
            "Consider class_weight='balanced'."
            if minority_ratio < 0.3 else "Balanced classes"
        )

    try:
        skill_docs = _load_skill_prompt(skill_name)
    except ValueError:
        skill_docs = f"(No SKILL.md found for '{skill_name}')"

    estimator_section = ""
    if estimator_hint:
        estimator_section = (
            f"\n## Preferred Estimator\n"
            f"Start with **{estimator_hint}**. Focus on tuning it first. "
            f"Only try alternatives if it clearly under-performs.\n"
        )

    features_preview = feature_columns[:10]
    ellipsis = "..." if len(feature_columns) > 10 else ""

    baseline_section = ""
    baseline_metrics = None
    if skill_name == "neural_networks" and task_type != "unsupervised":
        baseline_metrics = _run_quick_baseline(train_df, val_df, target_column, task_type)
        if baseline_metrics.get("roc_auc") or baseline_metrics.get("r2"):
            baseline_section = (
                f"\n## Performance Baseline (HistGradientBoosting — auto-computed)\n"
                f"A quick tree-based model achieved these metrics on the same data:\n"
            )
            for k, v in baseline_metrics.items():
                if k != "model" and v is not None:
                    baseline_section += f"- {k}: {v}\n"
            baseline_section += (
                "\n**Your neural network must beat these numbers.** "
                "If it can't match the baseline after several iterations, "
                "focus on matching it first before trying to exceed it.\n"
            )

    experiment_section = ""
    if experiment_result and experiment_result.get("total_scouts", 0) > 0:
        exp = experiment_result
        experiment_section = f"\n## Feature Experiment Results\n"
        experiment_section += (
            f"The pipeline tested **{exp.get('total_variants', 0)}** feature-set variants "
            f"x 2 model families = **{exp.get('total_scouts', 0)}** scout experiments in parallel.\n"
            f"Best variant: **{exp.get('best_variant_name', 'full')}** "
            f"(metric={exp.get('best_metric', 0):.4f}).\n\n"
        )
        signal = exp.get("signal_features", [])
        dropped = exp.get("dropped_features", [])
        if signal:
            experiment_section += f"**Signal features** (consistently high importance): {signal[:15]}\n"
        if dropped:
            experiment_section += f"**Low-signal features** (consistently near-zero): {dropped[:15]}\n"
        experiment_section += (
            "\nThe winning feature set is already loaded. Focus your iterations on **model "
            "selection and hyperparameter tuning** — the feature selection has been validated "
            "by the experiment grid.\n"
        )

    if feature_rankings:
        top_ranked = list(feature_rankings.items())[:10]
        if top_ranked:
            experiment_section += "\n**Feature importance rankings** (cross-variant weighted average):\n"
            for fname, imp in top_ranked:
                experiment_section += f"- {fname}: {imp:.4f}\n"

    if task_type == "unsupervised":
        target_line = "- Target column: N/A (unsupervised)"
        class_section = ""
        next_step = (
            "Begin training now. Maximize unsupervised objective quality by exploring "
            "estimators and hyperparameters. Use the dataset refs above. "
            "Do not call evaluate_model because no target labels are required."
        )
    else:
        target_line = f"- Target column: `{target_column}`"
        class_section = f"\n**Class distribution:** {class_counts}\n{imbalance_note}\n"
        next_step = ""

    context = f"""## Goal
{goal}

## Skill: `{skill_name}`

Follow the skill documentation below — it covers model selection and training.

<skill_documentation>
{skill_docs}
</skill_documentation>
{estimator_section}{baseline_section}{experiment_section}
## Data
- Task type: {task_type}
{target_line}
- Training: {len(train_df)} rows (ref: `{train_ref}`)
- Validation: {len(val_df) if val_df is not None else 0} rows (ref: `{val_ref or "N/A"}`)
- Test: {len(test_df) if test_df is not None else 0} rows (ref: `{test_ref or "N/A"}`)
- Features ({len(feature_columns)}): {features_preview}{ellipsis}
{class_section}
## Constraints
- Max iterations: {max_iterations}

## Sample Data (first 3 rows)
{train_df.head(3).to_dict(orient="records")}

{next_step}
"""

    llm = init_chat_model(llm_model)
    agent = create_agent(
        model=llm,
        tools=TRAINING_TOOLS,
        system_prompt=TRAINING_SYSTEM_PROMPT,
        response_format=ToolStrategy(schema=TrainingResult),
    )

    if task_type == "unsupervised":
        start_instruction = (
            "Begin training now. Maximize unsupervised objective quality by exploring "
            "estimators and hyperparameters. Use the dataset refs above. "
            "Do not call evaluate_model because no target labels are required."
        )
    elif skill_name == "neural_networks":
        start_instruction = (
            "Begin training now. You are using the neural_networks skill — "
            "write PyTorch training code following the SKILL.md templates and data pipeline exactly. "
            "Pass your code via train_with_skill(skill_name='neural_networks', params={...}). "
            "The params dict MUST include 'code', 'train_dataset_ref', 'target_column', and 'model_name'. "
            "If a run fails with an EXECUTION ERROR, read the traceback, fix the code, and try again. "
            "Do NOT give up after one failure — you have multiple iterations. "
            "You MUST use at least 3 iterations: baseline, then at least 2 experiments "
            "(architecture, hyperparameter, or regularization changes). "
            "Maximize validation performance. Run final test evaluation on your best model before finishing."
        )
    else:
        start_instruction = (
            "Begin training now. Maximize validation performance by exploring "
            "different estimators and hyperparameters. Use the dataset refs above. "
            "Run final test evaluation on your best model before finishing."
        )

    messages = [
        {"role": "user", "content": context},
        {"role": "user", "content": start_instruction},
    ]

    all_iterations: list[TrainingIteration] = []
    continuation_round = 0
    max_continuation_rounds = 2
    _baseline = baseline_metrics

    try:
        result = agent.invoke({"messages": messages})
        final_messages = result.get("messages", [])
        training_result = _extract_training_result(result)
        all_iterations.extend(training_result.iterations)

        while (
            continuation_round < max_continuation_rounds
            and training_result.success
            and not training_result.feature_redo_requested
            and _should_continue_iterating(all_iterations, max_iterations, task_type)
        ):
            continuation_round += 1
            emit_graph_stream({"type": "progress", "message": f"Training iteration {continuation_round + 1} — exploring hyperparameters...", "phase": "training"})
            continuation_msg = _build_continuation_message(
                all_iterations, max_iterations, task_type, _baseline,
            )

            diag = _diagnose_trend(all_iterations, task_type)
            best_str = _format_best_metric(
                _find_best_iteration(
                    [_iteration_to_dict(it) for it in all_iterations], task_type
                ) or {},
                task_type,
            )
            print(f"\n[training_agent] Continuing training (round {continuation_round}, "
                  f"{max_iterations - len(all_iterations)} remaining, "
                  f"best: {best_str}, trend: {diag['trend']})")

            cont_messages = final_messages + [
                {"role": "user", "content": continuation_msg},
            ]

            result = agent.invoke({"messages": cont_messages})
            final_messages = result.get("messages", [])
            training_result = _extract_training_result(result)
            existing_names = {it.model_name for it in all_iterations}
            for it in training_result.iterations:
                if it.model_name not in existing_names:
                    all_iterations.append(it)
                    existing_names.add(it.model_name)

        training_result.iterations = all_iterations
        training_result.num_iterations = len(all_iterations)

        feature_redo_request = _get_and_clear_feature_redo_request()
        feature_redo_requested = feature_redo_request is not None or training_result.feature_redo_requested

        iterations_dict = [_iteration_to_dict(it) for it in training_result.iterations]
        best_iteration = _find_best_iteration(iterations_dict, task_type)
        actual_best_name = best_iteration["model_name"] if best_iteration else training_result.best_model_name

        training_result = training_result.model_copy(
            update=_training_result_updates_from_best_iteration(
                best_iteration, training_result.best_model_name
            )
        )
        training_result = _maybe_clarify_summary_vs_saved_model(training_result)
        _log_training_results(training_result, task_type)

        # Extract feature importances BEFORE cleanup so we can try all models
        feat_imp = {}
        if training_result.success and actual_best_name:
            feat_imp = _extract_feature_importances(actual_best_name, feature_columns)
            if not feat_imp:
                for it in training_result.iterations:
                    if it.success and it.model_name and it.model_name != actual_best_name:
                        feat_imp = _extract_feature_importances(it.model_name, feature_columns)
                        if feat_imp:
                            break

        if training_result.success and actual_best_name:
            iteration_model_names = [it.model_name for it in training_result.iterations if it.model_name]
            deleted = _cleanup_intermediate_models(
                best_model_name=actual_best_name,
                base_model_name=model_name,
                iteration_model_names=iteration_model_names,
            )
            if deleted:
                print(f"\n[training_agent] Cleaned up {len(deleted)} intermediate models")

        output = training_result.model_dump(exclude={"best_model_name", "feature_redo_requested", "iterations"})

        if best_iteration:
            if task_type == "unsupervised":
                metric_keys = ("silhouette_score", "davies_bouldin", "inertia", "reconstruction_loss")
            elif task_type == "regression":
                metric_keys = ("val_r2", "val_rmse", "val_mae", "train_r2")
            else:
                metric_keys = ("val_accuracy", "val_roc_auc")
            for key in metric_keys:
                if best_iteration.get(key) is not None:
                    output[key] = best_iteration[key]

        if training_result.success and actual_best_name and task_type != "unsupervised":
            test_metrics = _evaluate_model_on_test(
                model_name=actual_best_name,
                test_ref=test_ref,
                target_column=target_column,
                task_type=task_type,
            )
            for key, val in test_metrics.items():
                if val is not None:
                    output[key] = val

            emit_graph_stream({"type": "progress", "message": f"Training complete — evaluating final model...", "phase": "training"})

        # Feature importances already extracted above (before cleanup)

        redo_rec = feature_redo_request.recommendation if feature_redo_request else None
        if feat_imp and redo_rec:
            top_5 = list(feat_imp.items())[:5]
            bottom_5 = [kv for kv in list(feat_imp.items())[-5:] if kv[1] < 0.01]
            imp_summary = (
                f"\n\nFeature importances from best model ({actual_best_name}):\n"
                f"  Top features: {dict(top_5)}\n"
            )
            if bottom_5:
                imp_summary += f"  Near-zero features (consider dropping): {dict(bottom_5)}\n"
            redo_rec += imp_summary

        output.update({
            "model_name": actual_best_name,
            "task_type": task_type,
            "target_column": target_column,
            "train_size": len(train_df),
            "val_size": len(val_df) if val_df is not None else 0,
            "test_size": len(test_df) if test_df is not None else 0,
            "iterations": iterations_dict,
            "best_iteration": best_iteration,
            "messages": final_messages,
            "feature_redo_requested": feature_redo_requested,
            "feature_redo_recommendation": redo_rec,
            "feature_redo_reason": feature_redo_request.reason if feature_redo_request else None,
            "feature_redo_suspected_issues": feature_redo_request.suspected_issues if feature_redo_request else None,
            "feature_importances": feat_imp,
        })
        return output

    except Exception as e:
        print(f"[training_agent] Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name,
            "model_type": selected_model,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_training_agent",
    "TrainingResult",
    "TrainingIteration",
    "TRAINING_TOOLS",
    "FeatureRedoRequest",
    "request_feature_engineering_redo_tool",
    "batch_train_with_skill_tool",
    "BatchTrainConfig",
    "BatchTrainInput",
]
