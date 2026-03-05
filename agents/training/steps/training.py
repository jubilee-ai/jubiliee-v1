"""
Training Agent - Step 7 of the ML Training Pipeline.

Uses a standard LangChain agent with skill docs injected into the context:
- Selected skill's SKILL.md is loaded once and embedded in the prompt
- Alternative skills can be loaded on-demand via get_skill_prompt tool
- Training executed via train_with_skill tool
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from ..utils.prompts import TRAINING_SYSTEM_PROMPT

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "models-tools" / "training"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from model_storage import (delete_model, evaluate_model_tool,
                           get_model_info_tool, list_models,
                           list_trained_models_tool, load_model,
                           predict_with_model_tool)
from utils import get_registered_dataset

from ..skill_registry import (_load_skill_prompt, get_available_models_for_task,
                              get_skill_prompt_tool, train_with_skill_tool)

# =============================================================================
# FEATURE ENGINEERING REDO TOOL
# =============================================================================


class FeatureRedoRequest(BaseModel):
    """Request to redo feature engineering with specific recommendations."""
    recommendation: str = Field(
        description="Specific recommendation for what to change in feature engineering. "
        "Be specific about which features to add, remove, or modify and why."
    )
    reason: str = Field(
        description="Why you believe the current features are limiting model performance. "
        "Include evidence from training results (e.g., specific metrics, patterns observed)."
    )
    suspected_issues: list[str] = Field(
        default_factory=list,
        description="List of suspected feature issues: "
        "'missing_interactions', 'high_cardinality', 'data_leakage', 'irrelevant_features', "
        "'missing_transformations', 'scale_issues', 'temporal_issues'"
    )


_feature_redo_request: Optional[FeatureRedoRequest] = None


def _get_and_clear_feature_redo_request() -> Optional[FeatureRedoRequest]:
    """Get the feature redo request and clear it."""
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
    """
    Request to redo feature engineering with specific recommendations.
    
    IMPORTANT: Only call this tool when you have strong evidence that the current features
    are the bottleneck limiting model performance. You must have tried multiple models
    and hyperparameter configurations first.
    
    Valid reasons to call this tool:
    - All model types show similar poor performance despite tuning
    - Feature diagnostics show high correlation or potential leakage
    - Performance is far below expected baseline for the task
    - You've exhausted reasonable hyperparameter tuning
    
    Args:
        recommendation: Specific recommendation for what to change in feature engineering.
        reason: Why you believe features are limiting performance.
        suspected_issues: List of suspected issues like 'missing_interactions', 
                         'high_cardinality', 'irrelevant_features', etc.
    
    Returns:
        Confirmation that the request was registered.
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


# =============================================================================
# TRAINING TOOLS & CONSTANTS
# =============================================================================

TRAINING_TOOLS = [
    train_with_skill_tool,
    get_skill_prompt_tool,
    list_trained_models_tool,
    predict_with_model_tool,
    get_model_info_tool,
    evaluate_model_tool,
    request_feature_engineering_redo_tool,
]


# =============================================================================
# TRAINING RESULT SCHEMA
# =============================================================================

class TrainingIteration(BaseModel):
    """A single training iteration attempt."""
    model_name: str = Field(description="Name of the model for this iteration")
    tool_used: str = Field(description="Training tool used (e.g., sklearn_logistic_regression)")
    hyperparams: dict = Field(default_factory=dict, description="Hyperparameters used")
    # Classification metrics
    train_accuracy: Optional[float] = Field(default=None, description="Training accuracy")
    val_accuracy: Optional[float] = Field(default=None, description="Validation accuracy")
    val_roc_auc: Optional[float] = Field(default=None, description="Validation ROC-AUC")
    # Regression metrics
    train_r2: Optional[float] = Field(default=None, description="Training R²")
    val_r2: Optional[float] = Field(default=None, description="Validation R²")
    val_rmse: Optional[float] = Field(default=None, description="Validation RMSE")
    val_mae: Optional[float] = Field(default=None, description="Validation MAE")
    test_r2: Optional[float] = Field(default=None, description="Test R²")
    test_rmse: Optional[float] = Field(default=None, description="Test RMSE")
    test_mae: Optional[float] = Field(default=None, description="Test MAE")
    success: bool = Field(description="Whether this iteration succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class TrainingResult(BaseModel):
    """Structured output for training completion."""
    success: bool = Field(description="Whether training completed successfully")
    best_model_name: str = Field(description="Name of the best trained model")
    model_type: str = Field(description="Type of model trained (e.g., logistic_regression, random_forest)")
    # Classification metrics
    val_accuracy: Optional[float] = Field(default=None, description="Best model validation accuracy")
    val_roc_auc: Optional[float] = Field(default=None, description="Best model validation ROC-AUC")
    test_accuracy: Optional[float] = Field(default=None, description="Best model test accuracy")
    test_roc_auc: Optional[float] = Field(default=None, description="Best model test ROC-AUC")
    # Regression metrics
    train_r2: Optional[float] = Field(default=None, description="Training R²")
    val_r2: Optional[float] = Field(default=None, description="Validation R²")
    val_rmse: Optional[float] = Field(default=None, description="Validation RMSE")
    val_mae: Optional[float] = Field(default=None, description="Validation MAE")
    test_r2: Optional[float] = Field(default=None, description="Test R²")
    test_rmse: Optional[float] = Field(default=None, description="Test RMSE")
    test_mae: Optional[float] = Field(default=None, description="Test MAE")
    # Iteration tracking
    iterations: list[TrainingIteration] = Field(default_factory=list, description="All training iterations")
    num_iterations: int = Field(description="Total number of training iterations attempted")
    # Summary
    summary: str = Field(description="Summary of training process and results")
    recommendations: Optional[str] = Field(default=None, description="Recommendations for improvement")
    feature_redo_requested: bool = Field(
        default=False, 
        description="Whether a feature engineering redo was requested via request_feature_engineering_redo tool"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

_REGRESSION_ESTIMATORS = {
    "svr", "linearsvr", "nusvr", "kneighborsregressor", "decisiontreeregressor",
    "gradientboostingregressor", "histgradientboostingregressor",
    "adaboostregressor", "baggingregressor",
    "extratreesregressor", "mlpregressor", "linearregression", "ridge",
    "lasso", "elasticnet", "huberregressor", "sgdregressor",
    "passiveaggressiveregressor",
}

_REGRESSION_GOAL_KEYWORDS = (
    "regress", "predict value", "forecast", "amount", "price", "cost",
    "revenue", "salary", "estimate number", "continuous",
)


def _get_task_type(selected_model: str, goal: str) -> Literal["classification", "regression"]:
    """Infer task type from model selection and goal."""
    goal_lower = goal.lower()
    model_lower = selected_model.lower()
    if any(w in goal_lower for w in _REGRESSION_GOAL_KEYWORDS):
        return "regression"
    if any(w in model_lower for w in ("glm", "regression")) and "logistic" not in model_lower:
        return "regression"
    # Detect regression estimators mentioned in goal for sklearn_generic
    if model_lower == "sklearn_generic":
        for est in _REGRESSION_ESTIMATORS:
            if est in goal_lower:
                return "regression"
    return "classification"


def _cleanup_intermediate_models(
    best_model_name: str,
    base_model_name: str,
    iteration_model_names: Optional[list[str]] = None,
) -> list[str]:
    """Clean up intermediate model versions, keeping only the best model."""
    deleted = []
    version_pattern = rf"^{re.escape(base_model_name)}_v\d+$"
    iteration_set = set(iteration_model_names) if iteration_model_names else set()

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
    """Extract TrainingResult from agent response, trying multiple strategies."""
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
    """Find the best successful iteration using task-appropriate metrics.

    Classification: primary = val_accuracy, tiebreak = val_roc_auc.
    Regression:     primary = val_r2, fallback = train_r2.
    """
    best, best_score = None, (-float("inf"), -float("inf"))
    for it in iterations:
        if not it.get("success"):
            continue
        if task_type == "regression":
            val_r2 = it.get("val_r2")
            train_r2 = it.get("train_r2")
            primary = val_r2 if val_r2 is not None else (train_r2 if train_r2 is not None else -float("inf"))
            score = (primary, 0.0)
        else:
            val_acc = it.get("val_accuracy")
            val_roc = it.get("val_roc_auc")
            primary = val_acc if val_acc is not None else -float("inf")
            secondary = val_roc if val_roc is not None else -float("inf")
            score = (primary, secondary)
        if score > best_score:
            best_score = score
            best = it
    return best


def _iteration_to_dict(it: TrainingIteration) -> dict:
    """Convert a TrainingIteration to a dict with backward-compatible 'metrics' sub-key."""
    d = it.model_dump()
    d["tool"] = d.pop("tool_used")
    d["metrics"] = {
        "train_accuracy": it.train_accuracy,
        "val_accuracy": it.val_accuracy,
        "roc_auc": it.val_roc_auc,
    }
    return d


def _maybe_discretize_target(y_true, model_name: str):
    """Apply target discretization if the model was trained with an auto-discretized target.

    Checks the model registry for a stored discretization threshold and
    binarizes y_true using that threshold so evaluation metrics are meaningful.
    """
    import numpy as np
    from model_storage import get_model_info

    info = get_model_info(model_name)
    if info is None:
        return y_true

    hp = info.get("hyperparameters") or {}
    if hp.get("target_discretized") and hp.get("target_discretization_threshold") is not None:
        threshold = float(hp["target_discretization_threshold"])
        y_true = (y_true > threshold).astype(int)
    return y_true


def _evaluate_model_on_test(
    model_name: str,
    test_ref: str,
    target_column: str,
    task_type: str,
) -> dict[str, float | None]:
    """Programmatically evaluate a model on the test set.

    Returns a dict with test_accuracy, test_roc_auc (classification)
    or test_r2, test_rmse, test_mae (regression). All values are None
    on failure so callers can safely fall through.
    """
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

        if task_type == "classification":
            y_true = _maybe_discretize_target(y_true, model_name)

        if task_type == "regression":
            from sklearn.metrics import (mean_absolute_error,
                                         mean_squared_error, r2_score)
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


def _log_training_results(training_result: TrainingResult, task_type: str):
    """Print structured training results to stdout."""
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
        print(f"     Tool: {it.tool_used}")
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
# MAIN TRAINING FUNCTION
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
    """
    Run the training agent to train and evaluate a model.
    
    Args:
        train_ref: Reference to training dataset
        val_ref: Reference to validation dataset
        test_ref: Reference to test dataset
        target_column: Column to predict
        selected_model: Skill name (e.g. "sklearn_generic")
        goal: The ML goal/objective
        model_name: Optional name for the model (auto-generated if not provided)
        max_iterations: Maximum training iterations
        llm_model: LLM to use for the agent
        estimator_hint: Specific sklearn estimator class name (e.g. "GradientBoostingClassifier")
    
    Returns:
        Dict with training results
    """
    train_df = get_registered_dataset(train_ref)
    val_df = get_registered_dataset(val_ref)
    test_df = get_registered_dataset(test_ref)

    if train_df is None:
        raise ValueError(f"Training dataset not found: {train_ref}")
    if val_df is None:
        raise ValueError(f"Validation dataset not found: {val_ref}")
    if test_df is None:
        raise ValueError(f"Test dataset not found: {test_ref}")

    task_type = _get_task_type(selected_model, goal)
    available_skills = get_available_models_for_task(task_type)

    if not model_name:
        model_name = f"{selected_model}_{int(time.time())}"

    feature_columns = [c for c in train_df.columns if c != target_column]

    print(f"[training_agent] Starting training...")
    print(f"  Skill: {selected_model}")
    if estimator_hint:
        print(f"  Estimator: {estimator_hint}")
    print(f"  Target: {target_column}")
    print(f"  Task type: {task_type}")
    print(f"  Train size: {len(train_df)}")
    print(f"  Val size: {len(val_df)}")
    print(f"  Test size: {len(test_df)}")
    print(f"  Features: {len(feature_columns)}")

    class_counts = train_df[target_column].value_counts().to_dict()
    total = sum(class_counts.values())
    minority_ratio = min(class_counts.values()) / total if total > 0 else 0
    is_imbalanced = minority_ratio < 0.3

    other_skills = [s for s in available_skills if s["skill"] != selected_model]
    other_skills_str = (
        "\n".join(f"- `{s['skill']}` — {s['label']}" for s in other_skills)
        if other_skills else "(none for this task type)"
    )
    imbalance_note = (
        f"\u26a0\ufe0f IMBALANCED DATA \u2014 minority class is {minority_ratio:.1%}. "
        "Use class_weight='balanced' (LR/RF/SVC/DT/GradientBoosting/ExtraTrees) "
        "or scale_pos_weight (XGB)."
        if is_imbalanced else "\u2713 Balanced classes"
    )

    sample_rows = train_df.head(3).to_dict(orient="records")
    features_preview = feature_columns[:10]
    ellipsis = "..." if len(feature_columns) > 10 else ""

    # Load the selected skill's documentation directly into context
    try:
        skill_docs = _load_skill_prompt(selected_model)
    except ValueError:
        skill_docs = f"(No SKILL.md found for '{selected_model}')"

    estimator_section = ""
    if estimator_hint:
        estimator_section = (
            f"\n## IMPORTANT: Preferred Estimator\n"
            f"The user specifically requested **{estimator_hint}**. "
            f"You MUST start with `\"{estimator_hint}\"` as the `estimator` parameter "
            f"in your first `train_with_skill` call. Focus on tuning this estimator. "
            f"Only try alternatives if it clearly under-performs.\n"
        )

    context = f"""## Goal
{goal}

## Selected Skill: `{selected_model}`

The full documentation for this skill is below. Use it to set hyperparameters.

<skill_documentation>
{skill_docs}
</skill_documentation>
{estimator_section}
## Alternative Skills (use `get_skill_prompt` to load docs before trying)
{other_skills_str}

Start with **{selected_model}**{f' using `{estimator_hint}`' if estimator_hint else ''}. If you want to try a different model, call `get_skill_prompt(skill_name)` first to get its parameters, then `train_with_skill`.

## Data
- Task type: {task_type}
- Target column: `{target_column}`
- Training: {len(train_df)} rows (ref: `{train_ref}`)
- Validation: {len(val_df)} rows (ref: `{val_ref}`)
- Test: {len(test_df)} rows (ref: `{test_ref}`)
- Features ({len(feature_columns)}): {features_preview}{ellipsis}

**Class distribution:** {class_counts}
{imbalance_note}

## Instructions
- Max iterations: {max_iterations}
- Name models descriptively: `lr_v1`, `rf_v1`, `xgb_v1`, `rf_v2`, etc.
- For train_with_skill params: `"train_dataset_ref": "{train_ref}"`, `"target_column": "{target_column}"`
- For evaluate_model: `dataset_ref="{val_ref}"` (validation) or `dataset_ref="{test_ref}"` (final test)
- **Focus on the {'preferred estimator (' + estimator_hint + ')' if estimator_hint else 'selected model'} first** — tune its hyperparameters for 1-2 iterations before considering alternatives.
- Only switch to a different model if the selected model clearly under-performs despite tuning.
- **AVOID slow models on large datasets** (>{len(train_df)} rows): GradientBoosting, SVC, KNeighbors, and MLP are very slow at this scale. Prefer LogisticRegression, RandomForest, ExtraTrees, LinearSVC, Ridge, or SGD for fast iteration.

## Sample Data (first 3 rows)
{sample_rows}
"""

    llm = init_chat_model(llm_model)
    agent = create_agent(
        model=llm,
        tools=TRAINING_TOOLS,
        system_prompt=TRAINING_SYSTEM_PROMPT,
        response_format=ToolStrategy(schema=TrainingResult),
        checkpointer=MemorySaver(),
    )

    messages = [
        {"role": "user", "content": context},
        {"role": "user", "content": (
            "Begin training now. Maximize validation performance by exploring "
            "different models and hyperparameters. Use the dataset refs above. "
            "Run final test evaluation on your best model before finishing."
        )},
    ]

    try:
        result = agent.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": f"training_{int(time.time())}"}},
        )
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

        # Override top-level val metrics with best iteration's values so
        # the report always reflects the actual best model.
        if best_iteration:
            if task_type == "regression":
                for key in ("val_r2", "val_rmse", "val_mae", "train_r2"):
                    if best_iteration.get(key) is not None:
                        output[key] = best_iteration[key]
            else:
                for key in ("val_accuracy", "val_roc_auc"):
                    if best_iteration.get(key) is not None:
                        output[key] = best_iteration[key]

        # Programmatic test evaluation — always evaluate the best model
        # on the test set ourselves instead of trusting the LLM's output.
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
            "feature_redo_recommendation": feature_redo_request.recommendation if feature_redo_request else None,
            "feature_redo_reason": feature_redo_request.reason if feature_redo_request else None,
            "feature_redo_suspected_issues": feature_redo_request.suspected_issues if feature_redo_request else None,
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
