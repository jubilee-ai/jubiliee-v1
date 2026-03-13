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
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..utils.prompts import TRAINING_SYSTEM_PROMPT

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

_MODEL_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "models-tools" / "training"
if str(_MODEL_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_TOOLS_DIR))

from model_storage import (delete_model, evaluate_model_tool, get_model_info,
                           get_model_info_tool, list_models, load_model,
                           list_trained_models_tool)
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
        "'estimator', 'model_name', 'train_dataset_ref', and 'target_column'.",
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


TRAINING_TOOLS = [
    train_with_skill_tool,
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
    iterations: list[TrainingIteration] = Field(default_factory=list)
    num_iterations: int
    summary: str
    recommendations: Optional[str] = None
    feature_redo_requested: bool = False


# =============================================================================
# HELPERS
# =============================================================================


def _infer_task_type(goal: str, estimator_hint: Optional[str] = None) -> str:
    """Infer 'classification' or 'regression' from the goal text and estimator name."""
    text = (goal + " " + (estimator_hint or "")).lower()
    if "regress" in text or any(
        kw in text for kw in ("forecast", "predict value", "continuous", "amount", "price", "cost")
    ):
        return "regression"
    return "classification"


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


def _find_best_iteration(iterations: list[dict], task_type: str) -> Optional[dict]:
    """Pick the best successful iteration by validation metrics."""
    best, best_score = None, (-float("inf"), -float("inf"))
    for it in iterations:
        if not it.get("success"):
            continue
        if task_type == "regression":
            primary = it.get("val_r2") or it.get("train_r2") or -float("inf")
            score = (primary, 0.0)
        else:
            score = (
                it.get("val_accuracy") or -float("inf"),
                it.get("val_roc_auc") or -float("inf"),
            )
        if score > best_score:
            best_score = score
            best = it
    return best


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
    from sklearn.metrics import accuracy_score, roc_auc_score

    result: dict[str, float | None] = {}
    try:
        model = load_model(model_name)
        test_df = get_registered_dataset(test_ref)
        if model is None or test_df is None:
            return result

        y_true = test_df[target_column]
        X = test_df[[c for c in test_df.columns if c != target_column]]

        if task_type == "regression":
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            y_pred = model.predict(X)
            result["test_r2"] = float(r2_score(y_true, y_pred))
            result["test_rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            result["test_mae"] = float(mean_absolute_error(y_true, y_pred))
        else:
            y_pred = model.predict(X)
            result["test_accuracy"] = float(accuracy_score(y_true, y_pred))
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)
                if y_proba.shape[1] == 2:
                    result["test_roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    result["test_roc_auc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                    )
        print(f"[training_agent] Programmatic test evaluation: {result}")
    except Exception as exc:
        print(f"[training_agent] Programmatic test evaluation failed: {exc}")
    return result


def _extract_feature_importances(model_name: str, feature_columns: list[str]) -> dict[str, float]:
    """Extract feature importances from a trained sklearn model.

    Works for tree-based (`.feature_importances_`) and linear (`.coef_`) models.
    Returns a dict mapping feature name to importance, sorted descending.
    """
    try:
        model = load_model(model_name)
        if model is None:
            return {}
        estimator = model
        if hasattr(model, "best_estimator_"):
            estimator = model.best_estimator_

        importances = None
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            if coef.ndim > 1:
                importances = np.mean(np.abs(coef), axis=0)
            else:
                importances = np.abs(coef)

        if importances is None or len(importances) != len(feature_columns):
            return {}

        result = {col: round(float(v), 4) for col, v in zip(feature_columns, importances)}
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
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
        if task_type == "regression":
            if it.val_r2 is not None:
                print(f"     Val R²: {it.val_r2}, Val RMSE: {it.val_rmse}")
        else:
            if it.val_accuracy is not None or it.val_roc_auc is not None:
                print(f"     Val Accuracy: {it.val_accuracy}, Val ROC-AUC: {it.val_roc_auc}")
        if it.error:
            print(f"     Error: {it.error}")
    print("  " + "-" * 60)
    print(f"  Best Model: {training_result.best_model_name}")
    if task_type == "regression":
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
    val_ref: str,
    test_ref: str,
    target_column: str,
    selected_model: str,
    goal: str,
    model_name: Optional[str] = None,
    max_iterations: int = 6,
    llm_model: str = "openai:gpt-5.1",
    estimator_hint: Optional[str] = None,
) -> dict[str, Any]:
    """Run the training agent.

    The agent reads the skill's SKILL.md (injected in context) to select the
    right estimator, trains via train_with_skill, evaluates, and iterates.
    """
    train_df = get_registered_dataset(train_ref)
    val_df = get_registered_dataset(val_ref)
    test_df = get_registered_dataset(test_ref)

    for ref, df, label in [
        (train_ref, train_df, "Training"),
        (val_ref, val_df, "Validation"),
        (test_ref, test_df, "Test"),
    ]:
        if df is None:
            raise ValueError(f"{label} dataset not found: {ref}")

    task_type = _infer_task_type(goal, estimator_hint)

    available_skills = [d.name for d in SKILLS_DIR.iterdir() if (d / "train.py").exists()]
    skill_name = selected_model if selected_model in available_skills else "supervised"

    if not model_name:
        model_name = f"{estimator_hint or skill_name}_{int(time.time())}"

    feature_columns = [c for c in train_df.columns if c != target_column]

    print(f"[training_agent] Starting training...")
    print(f"  Skill: {skill_name} | Target: {target_column} | Task: {task_type}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} | Features: {len(feature_columns)}")
    if estimator_hint:
        print(f"  Estimator hint: {estimator_hint}")

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

    context = f"""## Goal
{goal}

## Skill: `{skill_name}`

Follow the skill documentation below — it covers model selection and training.

<skill_documentation>
{skill_docs}
</skill_documentation>
{estimator_section}
## Data
- Task type: {task_type}
- Target column: `{target_column}`
- Training: {len(train_df)} rows (ref: `{train_ref}`)
- Validation: {len(val_df)} rows (ref: `{val_ref}`)
- Test: {len(test_df)} rows (ref: `{test_ref}`)
- Features ({len(feature_columns)}): {features_preview}{ellipsis}

**Class distribution:** {class_counts}
{imbalance_note}

## Constraints
- Max iterations: {max_iterations}

## Sample Data (first 3 rows)
{train_df.head(3).to_dict(orient="records")}
"""

    llm = init_chat_model(llm_model)
    agent = create_agent(
        model=llm,
        tools=TRAINING_TOOLS,
        system_prompt=TRAINING_SYSTEM_PROMPT,
        response_format=ToolStrategy(schema=TrainingResult),
    )

    messages = [
        {"role": "user", "content": context},
        {"role": "user", "content": (
            "Begin training now. Maximize validation performance by exploring "
            "different estimators and hyperparameters. Use the dataset refs above. "
            "Run final test evaluation on your best model before finishing."
        )},
    ]

    try:
        result = agent.invoke({"messages": messages})
        final_messages = result.get("messages", [])
        training_result = _extract_training_result(result)

        feature_redo_request = _get_and_clear_feature_redo_request()
        feature_redo_requested = feature_redo_request is not None or training_result.feature_redo_requested

        _log_training_results(training_result, task_type)

        iterations_dict = [_iteration_to_dict(it) for it in training_result.iterations]
        best_iteration = _find_best_iteration(iterations_dict, task_type)

        actual_best_name = best_iteration["model_name"] if best_iteration else training_result.best_model_name
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
            metric_keys = (
                ("val_r2", "val_rmse", "val_mae", "train_r2")
                if task_type == "regression"
                else ("val_accuracy", "val_roc_auc")
            )
            for key in metric_keys:
                if best_iteration.get(key) is not None:
                    output[key] = best_iteration[key]

        if training_result.success and actual_best_name:
            test_metrics = _evaluate_model_on_test(
                model_name=actual_best_name,
                test_ref=test_ref,
                target_column=target_column,
                task_type=task_type,
            )
            for key, val in test_metrics.items():
                if val is not None:
                    output[key] = val

        # Extract feature importances from the best model
        feat_imp = {}
        if training_result.success and actual_best_name:
            feat_imp = _extract_feature_importances(actual_best_name, feature_columns)

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
            "val_size": len(val_df),
            "test_size": len(test_df),
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
]
