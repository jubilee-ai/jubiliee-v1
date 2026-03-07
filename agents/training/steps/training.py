"""
Training Agent - Step 7 of the ML Training Pipeline.

Uses a skill-based architecture (progressive disclosure) instead of hardcoded
model tools. The agent:
  1. Reads the skill's SKILL.md to choose the right estimator
  2. Calls get_model_params to discover hyperparameters (TODO)
  3. Calls train_with_skill to train via the skill's train.py
  4. Evaluates and iterates
"""

import importlib.util
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
from pydantic import BaseModel, Field

from ..utils.prompts import TRAINING_SYSTEM_PROMPT

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

_MODEL_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "models-tools" / "training"
if str(_MODEL_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_TOOLS_DIR))

from model_storage import (delete_model, evaluate_model_tool,
                           get_model_info_tool, list_models, load_model,
                           list_trained_models_tool, predict_with_model_tool)
from utils import get_registered_dataset

from ..skills.scripts.extract_params import extract_estimator_params_tool

SKILLS_DIR = Path(__file__).parent.parent / "skills"


# =============================================================================
# SKILL LOADING
# =============================================================================


def _load_skill_prompt(skill_name: str) -> str:
    """Load a skill's SKILL.md content."""
    skill_md = SKILLS_DIR / skill_name / "SKILL.md"
    if not skill_md.exists():
        raise ValueError(f"SKILL.md not found for skill '{skill_name}'")
    return skill_md.read_text()


def _run_skill(skill_name: str, params: dict) -> str:
    """Dynamically import a skill's train.py and call its run(params) function."""
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
# SKILL TOOLS (LangChain)
# =============================================================================


class GetSkillPromptInput(BaseModel):
    skill_name: str = Field(
        description="Name of the skill to load (e.g. 'supervised')"
    )


@tool("get_skill_prompt", args_schema=GetSkillPromptInput)
def get_skill_prompt_tool(skill_name: str) -> str:
    """Load the full documentation for a training skill.

    Returns the skill's model selection guide, workflow instructions,
    and available estimators. Read this to choose the right model.
    """
    try:
        return _load_skill_prompt(skill_name)
    except ValueError as e:
        return f"Error: {e}"


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

    Pass skill_name and a params dict with estimator, model_name,
    train_dataset_ref, target_column, and any optional hyperparameters.
    """
    if params is None:
        params = {}
    if kwargs:
        params = {**kwargs, **params}
    try:
        return _run_skill(skill_name, params)
    except Exception as e:
        return f"Training failed: {e}"


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
# TRAINING TOOLS
# =============================================================================

TRAINING_TOOLS = [
    extract_estimator_params_tool,
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
    tool_used: str = Field(description="Skill and estimator used (e.g., supervised/HistGradientBoostingClassifier)")
    hyperparams: dict = Field(default_factory=dict, description="Hyperparameters used")
    train_accuracy: Optional[float] = Field(default=None, description="Training accuracy")
    val_accuracy: Optional[float] = Field(default=None, description="Validation accuracy")
    val_roc_auc: Optional[float] = Field(default=None, description="Validation ROC-AUC")
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
    model_type: str = Field(description="Estimator used (e.g., HistGradientBoostingClassifier)")
    val_accuracy: Optional[float] = Field(default=None, description="Best model validation accuracy")
    val_roc_auc: Optional[float] = Field(default=None, description="Best model validation ROC-AUC")
    test_accuracy: Optional[float] = Field(default=None, description="Best model test accuracy")
    test_roc_auc: Optional[float] = Field(default=None, description="Best model test ROC-AUC")
    train_r2: Optional[float] = Field(default=None, description="Training R²")
    val_r2: Optional[float] = Field(default=None, description="Validation R²")
    val_rmse: Optional[float] = Field(default=None, description="Validation RMSE")
    val_mae: Optional[float] = Field(default=None, description="Validation MAE")
    test_r2: Optional[float] = Field(default=None, description="Test R²")
    test_rmse: Optional[float] = Field(default=None, description="Test RMSE")
    test_mae: Optional[float] = Field(default=None, description="Test MAE")
    iterations: list[TrainingIteration] = Field(default_factory=list, description="All training iterations")
    num_iterations: int = Field(description="Total number of training iterations attempted")
    summary: str = Field(description="Summary of training process and results")
    recommendations: Optional[str] = Field(default=None, description="Recommendations for improvement")
    feature_redo_requested: bool = Field(
        default=False,
        description="Whether a feature engineering redo was requested via request_feature_engineering_redo tool"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

_REGRESSION_KEYWORDS = (
    "regress", "predict value", "forecast", "amount", "price", "cost",
    "revenue", "salary", "estimate number", "continuous",
)

_REGRESSION_ESTIMATORS = {
    "svr", "linearsvr", "nusvr", "kneighborsregressor", "decisiontreeregressor",
    "gradientboostingregressor", "histgradientboostingregressor",
    "adaboostregressor", "baggingregressor", "extratreesregressor",
    "mlpregressor", "linearregression", "ridge", "lasso", "elasticnet",
    "huberregressor", "sgdregressor", "passiveaggressiveregressor",
    "randomforestregressor", "gaussianprocessregressor",
    "isotonicregression", "quantileregressor",
}


def _get_task_type(goal: str, estimator_hint: Optional[str] = None) -> Literal["classification", "regression"]:
    """Infer task type from goal text and optional estimator hint."""
    goal_lower = goal.lower()
    if any(kw in goal_lower for kw in _REGRESSION_KEYWORDS):
        return "regression"
    if estimator_hint and estimator_hint.lower() in _REGRESSION_ESTIMATORS:
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
    """Apply target discretization if the model was trained with an auto-discretized target."""
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
    """Programmatically evaluate a model on the test set."""
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
# MAIN TRAINING FUNCTION
# =============================================================================

SUPERVISED_SKILL = "supervised"


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
    """Run the training agent using the skill-based architecture.

    The agent reads the supervised skill's SKILL.md to select the right
    estimator, then trains via train_with_skill. No hardcoded model list —
    the skill documentation drives model selection.

    Args:
        train_ref: Reference to training dataset
        val_ref: Reference to validation dataset
        test_ref: Reference to test dataset
        target_column: Column to predict
        selected_model: Skill name (default: "supervised")
        goal: The ML goal/objective
        model_name: Optional name for the model (auto-generated if not provided)
        max_iterations: Maximum training iterations
        llm_model: LLM to use for the agent
        estimator_hint: Specific sklearn estimator class name to start with
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

    task_type = _get_task_type(goal, estimator_hint)
    skill_name = SUPERVISED_SKILL

    if not model_name:
        model_name = f"{estimator_hint or skill_name}_{int(time.time())}"

    feature_columns = [c for c in train_df.columns if c != target_column]

    print(f"[training_agent] Starting training...")
    print(f"  Skill: {skill_name}")
    if estimator_hint:
        print(f"  Estimator hint: {estimator_hint}")
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

    imbalance_note = (
        f"\u26a0\ufe0f IMBALANCED DATA \u2014 minority class is {minority_ratio:.1%}. "
        "Consider estimators that support class_weight='balanced' or oversample."
        if is_imbalanced else "\u2713 Balanced classes"
    )

    sample_rows = train_df.head(3).to_dict(orient="records")
    features_preview = feature_columns[:10]
    ellipsis = "..." if len(feature_columns) > 10 else ""

    try:
        skill_docs = _load_skill_prompt(skill_name)
    except ValueError:
        skill_docs = f"(No SKILL.md found for '{skill_name}')"

    estimator_section = ""
    if estimator_hint:
        estimator_section = (
            f"\n## Preferred Estimator\n"
            f"The user specifically requested **{estimator_hint}**. "
            f"Start with this estimator. Focus on tuning it first. "
            f"Only try alternatives if it clearly under-performs.\n"
        )

    context = f"""## Goal
{goal}

## Skill: `{skill_name}`

Follow the skill documentation below. It covers model selection, parameter
discovery, and training. Use the Workflow section as your step-by-step guide.

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
- **AVOID slow estimators on large datasets** (>{len(train_df)} rows): GradientBoosting, SVC, KNeighbors, and MLP are very slow at scale. Prefer HistGradientBoosting, RandomForest, LinearSVC, Ridge, or SGD.

## Sample Data (first 3 rows)
{sample_rows}
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
            if task_type == "regression":
                for key in ("val_r2", "val_rmse", "val_mae", "train_r2"):
                    if best_iteration.get(key) is not None:
                        output[key] = best_iteration[key]
            else:
                for key in ("val_accuracy", "val_roc_auc"):
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
