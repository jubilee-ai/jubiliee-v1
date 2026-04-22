"""
Training Agent — Step 7 of the ML Training Pipeline.

Skill-based architecture: the agent reads a skill's SKILL.md (injected into
context) to choose an estimator, calls train_with_skill to train, evaluates,
and iterates.  No hardcoded model lists — the skill drives everything.
"""

import contextvars
import importlib
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
from pydantic import BaseModel, ConfigDict, Field

from ..core.task_inference import (
    infer_supervised_task_type_from_target_column,
    infer_task_type,
)
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
                           evaluate_model_tool, generate_model_path,
                           get_model_info, get_model_info_tool, list_models,
                           list_trained_models_tool, load_model, register_model)
from utils import get_registered_dataset

SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Injected into train_with_skill / batch_train params for supervised sklearn alignment.
_training_expected_task_type: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "training_expected_task_type", default=None
)

# Per run_training_agent run: number of train_with_skill / batch configs executed (None = no cap).
_MAX_TRAINED_MODELS_PER_RUN = 10
_training_models_trained: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "training_models_trained", default=None
)

# Current iteration list for get_experiment_diagnosis / get_best_iteration_by_metric tools.
_training_iterations_ctx: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "training_iterations_ctx", default=None
)


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

    module = importlib.import_module(f"agents.training.skills.{skill_name}.train")

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
    et = _training_expected_task_type.get()
    if et and "expected_task_type" not in params:
        params = {**params, "expected_task_type": et}
    used = _training_models_trained.get()
    if used is not None and used >= _MAX_TRAINED_MODELS_PER_RUN:
        return (
            f"TRAINING BLOCKED: this run allows at most {_MAX_TRAINED_MODELS_PER_RUN} trained models "
            f"({used} already completed). Use evaluate_model / get_model_info on existing models "
            "or finish your structured result without further training."
        )
    try:
        out = _run_skill(skill_name, params)
    except Exception as e:
        return f"Training failed: {e}"
    cur = _training_models_trained.get()
    if cur is not None:
        _training_models_trained.set(cur + 1)
    return out


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


def _tool_request_more_iterations_in_messages(messages: Any) -> bool:
    """True if the agent called ``request_more_iterations`` this turn."""
    if not messages:
        return False
    for msg in messages:
        name = getattr(msg, "name", None)
        if name == "request_more_iterations":
            return True
        if getattr(msg, "type", None) == "tool" and getattr(msg, "name", None) == "request_more_iterations":
            return True
        for tc in getattr(msg, "tool_calls", None) or []:
            n = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            if n == "request_more_iterations":
                return True
    return False


@tool("request_more_iterations")
def request_more_iterations_tool(reason: str) -> dict[str, Any]:
    """Request another full training agent turn (same session) when you need more budget.

    The orchestrator re-invokes you with an extra user message if under max continuation rounds.
    """
    return {
        "status": "registered",
        "reason": reason,
        "message": "Extension recorded; finish this turn with your TrainingResult, then you may get another turn.",
    }


@tool("get_experiment_diagnosis")
def get_experiment_diagnosis_tool(task_type: str) -> str:
    """Summarize experiment trend (improving / plateau / regressed) from iterations so far."""
    raw = _training_iterations_ctx.get()
    iters = list(raw) if raw else []
    if not iters:
        return json.dumps({"trend": "no_data", "diagnosis": "No iterations yet.", "n_iterations": 0})
    diag = _diagnose_trend(iters, task_type)
    diag["n_iterations"] = len(iters)
    return json.dumps(diag, default=str)


@tool("get_best_iteration_by_metric")
def get_best_iteration_by_metric_tool(task_type: str) -> str:
    """Return the best successful iteration by validation metric (ROC-AUC / R² / silhouette)."""
    raw = _training_iterations_ctx.get()
    iters = list(raw) if raw else []
    dicts = [_iteration_to_dict(it) for it in iters]
    best = _find_best_iteration(dicts, task_type)
    return json.dumps({"best": best}, default=str)


@tool("cleanup_intermediate_models")
def cleanup_intermediate_models_tool(
    best_model_name: str,
    base_model_name: str,
    iteration_model_names_json: str = "[]",
) -> str:
    """Delete intermediate trial models; keep ``best_model_name``."""
    try:
        names = json.loads(iteration_model_names_json or "[]")
        if not isinstance(names, list):
            names = []
        deleted = _cleanup_intermediate_models(
            best_model_name=best_model_name,
            base_model_name=base_model_name,
            iteration_model_names=names,
        )
        return json.dumps({"deleted": deleted, "n_deleted": len(deleted)})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("evaluate_champion_on_test")
def evaluate_champion_on_test_tool(
    model_name: str,
    test_ref: str,
    target_column: str,
    task_type: str,
) -> str:
    """Compute holdout test metrics for a registered model (call before submitting TrainingResult)."""
    m = _evaluate_model_on_test(model_name, test_ref, target_column, task_type)
    return json.dumps({"metrics": m}, default=str)


@tool("run_tree_baseline")
def run_tree_baseline_tool(
    train_ref: str,
    val_ref: str,
    target_column: str,
    task_type: str,
) -> str:
    """Quick HistGradientBoosting baseline on train/val (optional for neural_networks or comparisons)."""
    train_df = get_registered_dataset(train_ref)
    val_df = get_registered_dataset(val_ref)
    if train_df is None or val_df is None:
        return json.dumps({"error": "train or val dataset not found"})
    b = _run_quick_baseline(train_df, val_df, target_column, task_type)
    return json.dumps(b, default=str)


@tool("stack_registered_models")
def stack_registered_models_tool(
    model_names_json: str,
    new_model_name: str,
    train_dataset_ref: str,
    val_dataset_ref: str,
    target_column: str,
    task_type: str,
    meta_learner: str = "logistic",
) -> str:
    """Build a sklearn Voting soft ensemble (or hard if no proba) from existing registered models."""
    import joblib
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    try:
        names = json.loads(model_names_json or "[]")
        if not isinstance(names, list) or len(names) < 2:
            return json.dumps({"ok": False, "error": "model_names_json must be a JSON list of at least 2 names"})
        train_df = get_registered_dataset(train_dataset_ref)
        val_df = get_registered_dataset(val_dataset_ref)
        if train_df is None or val_df is None:
            return json.dumps({"ok": False, "error": "datasets not found"})
        ests = []
        for n in names:
            m = load_model(n)
            if m is None:
                return json.dumps({"ok": False, "error": f"model not found: {n}"})
            ests.append((str(n).replace(" ", "_")[:40], m))
        feature_cols = [c for c in train_df.columns if c != target_column]
        X_tr = train_df[feature_cols]
        y_tr = train_df[target_column]
        if task_type == "regression":
            vr = VotingRegressor(estimators=ests)
            vr.fit(X_tr, y_tr)
            fitted = vr
        else:
            try:
                vc = VotingClassifier(estimators=ests, voting="soft")
                vc.fit(X_tr, y_tr)
                fitted = vc
            except Exception:
                vc = VotingClassifier(estimators=ests, voting="hard")
                vc.fit(X_tr, y_tr)
                fitted = vc
        save_path = generate_model_path(new_model_name)
        joblib.dump(fitted, save_path)

        y_val = val_df[target_column]
        X_val = val_df[feature_cols]
        if task_type == "regression":
            from sklearn.metrics import mean_squared_error, r2_score

            pred = fitted.predict(X_val)
            metrics = {
                "val_r2": float(r2_score(y_val, pred)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
            }
        else:
            from sklearn.metrics import accuracy_score, roc_auc_score

            pred = fitted.predict(X_val)
            metrics = {"val_accuracy": float(accuracy_score(y_val, pred))}
            try:
                proba = fitted.predict_proba(X_val)
                if proba.shape[1] == 2:
                    metrics["val_roc_auc"] = float(roc_auc_score(y_val, proba[:, 1]))
            except Exception:
                pass
        reg = register_model(
            model_name=new_model_name,
            model_path=save_path,
            model_type="sklearn_stacked_voting",
            description=f"Voting ensemble of {names}",
            metrics=metrics,
            feature_names=feature_cols,
            target_column=target_column,
            hyperparameters={"base_models": names, "meta": meta_learner},
            training_samples=len(train_df),
            classes=[str(x) for x in sorted(y_tr.unique())] if task_type != "regression" else [],
        )
        used = _training_models_trained.get()
        if used is not None:
            _training_models_trained.set(used + 1)
        return json.dumps({"ok": True, "registered": reg}, default=str)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


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
    used = _training_models_trained.get()
    if used is not None:
        remaining = _MAX_TRAINED_MODELS_PER_RUN - used
        if remaining <= 0:
            return (
                f"TRAINING BLOCKED: this run allows at most {_MAX_TRAINED_MODELS_PER_RUN} trained models "
                f"({used} already completed). Use evaluate_model / get_model_info on existing models "
                "or finish without further batch training."
            )
        if len(configs) > remaining:
            configs = configs[:remaining]

    expected_tt = _training_expected_task_type.get()

    def _run_one(cfg: BatchTrainConfig) -> dict:
        params = {
            "estimator": cfg.estimator,
            "model_name": cfg.model_name,
            "train_dataset_ref": train_dataset_ref,
            "target_column": target_column,
            "hyperparameters": cfg.hyperparams,
        }
        if expected_tt:
            params["expected_task_type"] = expected_tt
        if val_dataset_ref:
            params["val_dataset_ref"] = val_dataset_ref
        try:
            raw = _run_skill(skill_name, params)
            return {"model_name": cfg.model_name, "estimator": cfg.estimator, "success": True, "raw_output": raw}
        except Exception as e:
            return {"model_name": cfg.model_name, "estimator": cfg.estimator, "success": False, "error": str(e)}

    def _max_parallel_batch() -> int:
        try:
            from backend.shared.settings import get_settings

            return max(1, int(get_settings().MAX_PARALLEL_BATCH_TRAIN))
        except Exception:
            return 1

    results: list[dict] = []
    mw = min(len(configs), _max_parallel_batch())
    with ThreadPoolExecutor(max_workers=mw) as pool:
        futures = {pool.submit(_run_one, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            results.append(future.result())

    cur = _training_models_trained.get()
    if cur is not None:
        _training_models_trained.set(cur + len(results))

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


from agents.training.mcp.h2o_server import build_h2o_tools
from agents.training.mcp.mlflow_server import build_mlflow_tools

TRAINING_TOOLS = [
    train_with_skill_tool,
    batch_train_with_skill_tool,
    evaluate_model_tool,
    list_trained_models_tool,
    get_model_info_tool,
    request_feature_engineering_redo_tool,
    request_more_iterations_tool,
    get_experiment_diagnosis_tool,
    get_best_iteration_by_metric_tool,
    cleanup_intermediate_models_tool,
    evaluate_champion_on_test_tool,
    run_tree_baseline_tool,
    stack_registered_models_tool,
] + list(build_h2o_tools()) + list(build_mlflow_tools())


# =============================================================================
# STRUCTURED OUTPUT SCHEMA
# =============================================================================


class TrainingIteration(BaseModel):
    model_name: str = Field(description="Name of the model for this iteration")
    tool_used: str = Field(
        description="Estimator class name only (e.g. HistGradientBoostingClassifier), not skill or import paths.",
    )
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
    model_config = ConfigDict(extra="ignore")

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
        description=(
            "User-facing narrative: open with plain-language 'what this model is for' and why it's useful; "
            "then concise experiment arc, why best_model_name won, key metrics once each. "
            "Avoid hyperparameter dumps, formulas, and implementation detail."
        ),
    )
    feature_redo_requested: bool = False


# =============================================================================
# HELPERS
# =============================================================================


def _infer_task_type(
    goal: str,
    estimator_hint: Optional[str] = None,
    selected_model: Optional[str] = None,
) -> str:
    """Thin wrapper around :func:`infer_task_type` for readability at call sites."""
    return infer_task_type(
        goal,
        selected_model=selected_model,
        estimator_hint=estimator_hint,
    )


def _final_estimator_from_fitted(model: Any) -> Any:
    """Last step of a Pipeline / RandomizedSearchCV, for sklearn is_classifier checks."""
    outer = model
    if outer is None:
        return None
    if hasattr(outer, "best_estimator_"):
        outer = outer.best_estimator_
    if hasattr(outer, "named_steps"):
        ns = outer.named_steps
        if "model" in ns:
            return ns["model"]
        if ns:
            return list(ns.values())[-1]
    steps = getattr(outer, "steps", None)
    if steps:
        return steps[-1][1]
    return outer


def _validate_estimator_matches_task(
    model_name: str, task_type: str, skill_name: str
) -> tuple[bool, str]:
    """Ensure the saved sklearn model matches classification vs regression intent."""
    if task_type == "unsupervised" or skill_name in ("neural_networks",):
        return True, ""
    if task_type not in ("classification", "regression"):
        return True, ""
    try:
        from sklearn.base import is_classifier, is_regressor

        model = load_model(model_name)
        if model is None:
            return True, ""
        est = _final_estimator_from_fitted(model)
        if est is None:
            return True, ""
        if task_type == "classification":
            if not is_classifier(est):
                return (
                    False,
                    f"Task type is classification but the trained estimator "
                    f"({type(est).__name__}) is not a classifier. Adjust the goal, "
                    f"training plan, or estimator choice so they align.",
                )
        elif task_type == "regression":
            if not is_regressor(est):
                return (
                    False,
                    f"Task type is regression but the trained estimator "
                    f"({type(est).__name__}) is not a regressor. Adjust the goal, "
                    f"training plan, or estimator choice so they align.",
                )
    except Exception as exc:
        return False, f"Estimator vs task_type validation failed: {exc}"
    return True, ""


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


def _format_training_plan_section(training_plan: dict | None) -> str:
    """Inject human-approved plan from training_approval into the training agent context."""
    if not training_plan:
        return ""
    skip = {"data_summary"}
    parts: list[str] = []
    for key in (
        "model_type",
        "task_type",
        "hyperparameters",
        "class_weight",
        "strategy_notes",
        "expected_metrics",
    ):
        if key in skip:
            continue
        val = training_plan.get(key)
        if val is None or val == "" or val == {}:
            continue
        if key == "hyperparameters" and isinstance(val, dict):
            parts.append(f"- **{key}**: {json.dumps(val, default=str)}")
        else:
            parts.append(f"- **{key}**: {val}")
    if not parts:
        return ""
    return (
        "\n## Human-approved training plan\n\n"
        "The pipeline (or user) approved this configuration — **start here**, then refine using "
        "validation metrics. You may override hyperparameters if diagnostics demand it.\n\n"
        + "\n".join(parts)
        + "\n\n"
    )


def _truncate_jsonish(obj: Any, max_len: int = 420) -> str:
    s = json.dumps(obj, default=str, sort_keys=True) if isinstance(obj, (dict, list)) else str(obj)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _format_best_iteration_continuation_block(
    iterations: list[TrainingIteration],
    task_type: str,
) -> str:
    """Concrete anchor so continuation rounds refine the validation leader, not random models."""
    dicts = [_iteration_to_dict(it) for it in iterations]
    best = _find_best_iteration(dicts, task_type)
    if not best:
        return ""

    name = best.get("model_name", "")
    raw_tool = best.get("tool") or best.get("tool_used") or ""
    tool_short = str(raw_tool).rsplit(".", 1)[-1].split("/")[-1] if raw_tool else ""
    hp = best.get("hyperparams") or {}
    hp_s = _truncate_jsonish(hp) if hp else "(defaults or see model registry)"

    if task_type == "unsupervised":
        metrics_line = (
            f"- silhouette={best.get('silhouette_score')}, "
            f"davies_bouldin={best.get('davies_bouldin')}"
        )
        metric_name = "unsupervised objective"
    elif task_type == "regression":
        metrics_line = (
            f"- val R²={best.get('val_r2')}, val RMSE={best.get('val_rmse')}"
        )
        metric_name = "validation R² (primary)"
    else:
        metrics_line = (
            f"- val ROC-AUC={best.get('val_roc_auc')}, val accuracy={best.get('val_accuracy')}"
        )
        metric_name = "validation ROC-AUC (primary), then accuracy"

    return (
        f"## Current validation best — refine THIS\n\n"
        f"Leader by **{metric_name}**:\n"
        f"- **model_name**: `{name}`\n"
        f"- **estimator**: `{tool_short}`\n"
        f"- **hyperparams**: {hp_s}\n"
        f"{metrics_line}\n\n"
        f"**Next experiment:** Improve this configuration (same estimator family) unless the last "
        f"runs were within noise (~1%) of each other *and* train/val diagnostics clearly favor "
        f"switching families. If you switch, state why in your reasoning.\n\n"
    )


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
    lines.append(_format_best_iteration_continuation_block(iterations, task_type))

    lines.append(
        "## Next Steps\n"
        "Consult the **Hyperparameter Strategy Matrix** and **Architecture Decision Guide** "
        "in the SKILL.md. Prefer **refining the validation leader** above; change exactly ONE "
        "focused thing per attempt. Run the experiment, evaluate, and report results.\n"
        "In the structured `iterations` array, append only **new** attempts from this round "
        "(prior attempts are already stored). Your **`summary`** must still cover the **entire run** "
        "and the model you set as **`best_model_name`** "
        "(the artifact headline metrics will follow), not only the iterations added here."
    )

    return "\n".join(lines)


def _coerce_iteration_history(raw_iterations: Any) -> list[TrainingIteration]:
    """Best-effort parse of persisted iteration history from prior runs."""
    if not isinstance(raw_iterations, list):
        return []

    parsed: list[TrainingIteration] = []
    for raw in raw_iterations:
        if not isinstance(raw, dict):
            continue
        try:
            parsed.append(TrainingIteration.model_validate(raw))
        except Exception:
            continue
    return parsed


def _build_prior_history_section(
    prior_iterations: list[TrainingIteration],
    task_type: str,
) -> str:
    """Summarize prior training runs so restarts avoid repeating dead ends."""
    if not prior_iterations:
        return ""

    lines = [
        "## Previous Training History",
        "A prior training run already explored these configurations. Reuse this context so you do not restart blindly.",
        "",
    ]
    for it in prior_iterations[-6:]:
        metric = _primary_metric(it, task_type)
        metric_str = f"{metric:.4f}" if metric is not None else "N/A"
        status = "OK" if it.success else "FAIL"
        hp = _truncate_jsonish(it.hyperparams) if it.hyperparams else "(defaults or unavailable)"
        lines.extend([
            f"- [{status}] `{it.model_name}` via `{it.tool_used}`",
            f"  metric={metric_str}; hyperparams={hp}",
        ])

    best = _find_best_iteration(
        [_iteration_to_dict(it) for it in prior_iterations],
        task_type,
    )
    if best:
        best_metric = _format_best_metric(best, task_type)
        best_hp = _truncate_jsonish(best.get("hyperparams") or {})
        lines.extend([
            "",
            f"Best prior validation result: `{best.get('model_name')}` with {best_metric}.",
            f"Best prior hyperparams: {best_hp}",
        ])

    lines.extend([
        "",
        "Avoid repeating the exact same losing configurations unless you are validating a specific hypothesis or data changed.",
        "",
    ])
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
        f"The saved model is **{saved}** (picked by validation). The narrative below still mentions a later run "
        f"(**{last_name}**) that was not kept.\n\n"
    )
    return training_result.model_copy(update={"summary": prefix + body})


def _iteration_to_dict(it: TrainingIteration) -> dict:
    d = it.model_dump()
    d.pop("tool_used", None)
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
            import pandas as pd

            y_pred = model.predict(X)
            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)
            if y_true_arr.dtype != y_pred_arr.dtype:
                y_pred_arr = y_pred_arr.astype(y_true_arr.dtype)
            result["test_accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
            y_series = pd.Series(y_true)
            n_unique = int(y_series.nunique(dropna=True))
            n_rows = len(y_series)
            looks_continuous = (
                pd.api.types.is_numeric_dtype(y_series)
                and n_unique > min(50, max(10, n_rows // 20))
            )
            if looks_continuous:
                print(
                    f"[training_agent] Skipping ROC-AUC on test: target has {n_unique} unique "
                    f"numeric values (continuous-like for a classification task)."
                )
            else:
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
    max_iterations: int = 4,
    llm_model: str = "openai:gpt-5.4",
    estimator_hint: Optional[str] = None,
    experiment_result: Optional[dict[str, Any]] = None,
    feature_rankings: Optional[dict[str, float]] = None,
    training_plan: Optional[dict[str, Any]] = None,
    prior_training_metrics: Optional[dict[str, Any]] = None,
    max_continuation_rounds: int = 3,
    explicit_task_type: Optional[str] = None,
) -> dict[str, Any]:
    """Run the training agent.

    The agent reads the skill's SKILL.md (injected in context) to select the
    right estimator, trains via train_with_skill, evaluates, and iterates.

    ``training_plan`` (from training_approval) is injected into context so the
    sub-agent aligns with the approved hyperparameters and strategy.
    """
    train_df = get_registered_dataset(train_ref)
    val_df = get_registered_dataset(val_ref) if val_ref else None
    test_df = get_registered_dataset(test_ref) if test_ref else None

    if train_df is None:
        raise ValueError(f"Training dataset not found: {train_ref}")
    for ref, df, label in [(val_ref, val_df, "Validation"), (test_ref, test_df, "Test")]:
        if ref and df is None:
            raise ValueError(f"{label} dataset not found: {ref}")

    plan_tt: Optional[str] = None
    if training_plan and isinstance(training_plan, dict):
        raw_tt = training_plan.get("task_type")
        if isinstance(raw_tt, str) and raw_tt:
            plan_tt = raw_tt
    data_tt: Optional[str] = None
    if target_column and target_column in train_df.columns:
        data_tt = infer_supervised_task_type_from_target_column(train_df, target_column)
    base_tt = (
        explicit_task_type
        or plan_tt
        or _infer_task_type(goal, estimator_hint, selected_model=selected_model)
    )
    task_type = base_tt
    if data_tt and base_tt != data_tt:
        if {data_tt, base_tt} == {"classification", "regression"}:
            task_type = data_tt

    available_skills = [d.name for d in SKILLS_DIR.iterdir() if (d / "train.py").exists()]
    skill_name = selected_model if selected_model in available_skills else "supervised"

    if not model_name:
        model_name = f"{estimator_hint or skill_name}_{int(time.time())}"

    if target_column:
        feature_columns = [c for c in train_df.columns if c != target_column]
    else:
        feature_columns = list(train_df.columns)
    if task_type == "unsupervised":
        no_dt = [c for c in feature_columns if not str(train_df[c].dtype).startswith("datetime")]
        if no_dt:
            feature_columns = no_dt

    print(f"[training_agent] Starting training...")
    print(f"  Skill: {skill_name} | Target: {target_column or '(none)'} | Task: {task_type}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df) if val_df is not None else 0} | Test: {len(test_df) if test_df is not None else 0} | Features: {len(feature_columns)}")
    if estimator_hint:
        print(f"  Estimator hint: {estimator_hint}")

    emit_graph_stream({"type": "progress", "message": f"Model training in progress...", "phase": "training"})

    if task_type == "unsupervised":
        class_counts = {}
        imbalance_note = "Unsupervised task — no target variable."
    elif task_type == "regression":
        ys = train_df[target_column].dropna()
        class_counts = {
            "target_mean": float(ys.mean()) if len(ys) else None,
            "target_std": float(ys.std()) if len(ys) else None,
            "target_min": float(ys.min()) if len(ys) else None,
            "target_max": float(ys.max()) if len(ys) else None,
        }
        imbalance_note = "Regression target — distribution summary above (not class balance)."
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
        baseline_section = (
            "\n## Tree baseline (optional)\n"
            "Call `run_tree_baseline` with the train ref, val ref, target, and task_type if you want "
            "a quick HistGradientBoosting reference — **you decide** whether to run it.\n"
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

    plan_section = _format_training_plan_section(training_plan)
    prior_iterations = _coerce_iteration_history(
        (prior_training_metrics or {}).get("iterations")
        if isinstance(prior_training_metrics, dict)
        else None
    )
    prior_history_section = _build_prior_history_section(prior_iterations, task_type)

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
        if task_type == "regression":
            class_section = f"\n**Target summary:** {class_counts}\n{imbalance_note}\n"
        else:
            class_section = f"\n**Class distribution:** {class_counts}\n{imbalance_note}\n"
        next_step = ""

    context = f"""## Goal
{goal}

## Skill: `{skill_name}`

Follow the skill documentation below — it covers model selection and training.

<skill_documentation>
{skill_docs}
</skill_documentation>
{estimator_section}{baseline_section}{experiment_section}{plan_section}{prior_history_section}## Data
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
            "When sample size allows, pass eval_holdout_fraction around 0.15 in train_with_skill "
            "params so metrics include less optimistic val_* scores. "
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
            "Begin training now. Maximize **validation** performance (ROC-AUC primary, then "
            "accuracy for classification; R² for regression).\n\n"
            "Keep experiments cheap and deliberate. When comparing fresh estimators, use small search budgets. "
            "When refining a promising config, prefer passing exact hyperparameters and `auto_tune=false` so the tool "
            "does a direct fit instead of a broad CV search unless you truly need a focused re-search.\n\n"
            "**Iteration protocol:**\n"
            "1) **First** experiment: use `batch_train_with_skill` to compare 2–3 estimators "
            "from different families in parallel (unless the Human-approved training plan "
            "already prescribes a single starting point — then align with it first).\n"
            "2) Identify the **validation leader** from that batch.\n"
            "3) **All further iterations** must **refine that leader** (same estimator family) "
            "by tuning hyperparameters — one focused change at a time. Do **not** jump to "
            "unrelated estimators unless leaders tie within ~1% and diagnostics clearly "
            "suggest a different failure mode; if you switch, explain why.\n"
            "4) After each run, compare metrics to your **best validation score so far** and "
            "state what you will change next to beat it.\n\n"
            "Use the dataset refs above. Run final test evaluation on your best model before finishing."
        )

    messages = [
        {"role": "user", "content": context},
        {"role": "user", "content": start_instruction},
    ]

    all_iterations: list[TrainingIteration] = []
    continuation_round = 0
    cont_cap = max(1, min(6, max_continuation_rounds))
    if training_plan and isinstance(training_plan.get("max_continuation_rounds"), int):
        cont_cap = max(1, min(6, int(training_plan["max_continuation_rounds"])))

    token = _training_expected_task_type.set(task_type)
    budget_token = _training_models_trained.set(0)
    try:
        extensions_used = 0
        result: dict[str, Any] = {}
        final_messages: list[Any] = []
        training_result: TrainingResult | None = None

        while True:
            it_tok = _training_iterations_ctx.set(list(all_iterations))
            try:
                result = agent.invoke({"messages": messages})
            finally:
                _training_iterations_ctx.reset(it_tok)

            final_messages = result.get("messages", [])
            training_result = _extract_training_result(result)
            existing_names = {it.model_name for it in all_iterations}
            for it in training_result.iterations:
                if it.model_name not in existing_names:
                    all_iterations.append(it)
                    existing_names.add(it.model_name)

            want_more = _tool_request_more_iterations_in_messages(final_messages)
            if (
                want_more
                and extensions_used < cont_cap
                and training_result.success
                and not training_result.feature_redo_requested
            ):
                extensions_used += 1
                continuation_round = extensions_used
                emit_graph_stream(
                    {
                        "type": "progress",
                        "message": f"Training extension {extensions_used} — additional turn granted...",
                        "phase": "training",
                    }
                )
                messages = final_messages + [
                    {
                        "role": "user",
                        "content": (
                            "You requested more training budget. Continue experimenting. "
                            "Optional: call `get_experiment_diagnosis` or `get_best_iteration_by_metric` first, "
                            "then train/evaluate. Call `request_more_iterations` again if you still need another turn."
                        ),
                    },
                ]
                continue
            break

        assert training_result is not None
        training_result.iterations = all_iterations
        training_result.num_iterations = len(all_iterations)

        feature_redo_request = _get_and_clear_feature_redo_request()
        feature_redo_requested = feature_redo_request is not None or training_result.feature_redo_requested

        iterations_dict = [_iteration_to_dict(it) for it in training_result.iterations]
        best_iteration = _find_best_iteration(iterations_dict, task_type)
        from backend.shared.settings import get_settings as _gs_tr

        _s = _gs_tr()
        actual_best_name = (training_result.best_model_name or "").strip()
        if not actual_best_name and best_iteration:
            actual_best_name = str(best_iteration.get("model_name") or "")

        if _s.TRAINING_METRIC_LEADER_OVERRIDE and best_iteration:
            training_result = training_result.model_copy(
                update=_training_result_updates_from_best_iteration(
                    best_iteration, training_result.best_model_name
                )
            )
            training_result = _maybe_clarify_summary_vs_saved_model(training_result)
            actual_best_name = (training_result.best_model_name or "").strip()

        if training_result.success and actual_best_name:
            est_ok, est_msg = _validate_estimator_matches_task(
                actual_best_name, task_type, skill_name
            )
            if not est_ok:
                return {
                    "success": False,
                    "error": est_msg,
                    "model_name": actual_best_name,
                    "model_type": selected_model,
                    "task_type": task_type,
                    "target_column": target_column,
                    "train_size": len(train_df),
                    "val_size": len(val_df) if val_df is not None else 0,
                    "test_size": len(test_df) if test_df is not None else 0,
                    "iterations": iterations_dict,
                    "num_iterations": len(all_iterations),
                    "messages": final_messages,
                    "summary": est_msg,
                }

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

        if training_result.success and actual_best_name and _s.TRAINING_AUTO_CLEANUP_INTERMEDIATES:
            iteration_model_names = [it.model_name for it in training_result.iterations if it.model_name]
            deleted = _cleanup_intermediate_models(
                best_model_name=actual_best_name,
                base_model_name=model_name,
                iteration_model_names=iteration_model_names,
            )
            if deleted:
                print(f"\n[training_agent] Cleaned up {len(deleted)} intermediate models")

        output = training_result.model_dump(exclude={"best_model_name", "feature_redo_requested", "iterations"})

        metric_row = (
            best_iteration
            if _s.TRAINING_METRIC_LEADER_OVERRIDE and best_iteration
            else next(
                (d for d in iterations_dict if d.get("model_name") == actual_best_name),
                best_iteration,
            )
        )
        if metric_row:
            if task_type == "unsupervised":
                metric_keys = ("silhouette_score", "davies_bouldin", "inertia", "reconstruction_loss")
            elif task_type == "regression":
                metric_keys = ("val_r2", "val_rmse", "val_mae", "train_r2")
            else:
                metric_keys = ("val_accuracy", "val_roc_auc")
            for key in metric_keys:
                if metric_row.get(key) is not None:
                    output[key] = metric_row[key]

        if training_result.success and actual_best_name and task_type != "unsupervised" and _s.TRAINING_AUTO_TEST_EVAL:
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
    finally:
        _training_models_trained.reset(budget_token)
        _training_expected_task_type.reset(token)


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
