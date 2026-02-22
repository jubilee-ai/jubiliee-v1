"""
Simple ML Training Agent using the Deep Agents SDK.

Replaces the rigid LangGraph StateGraph with a single deep agent
whose LLM decides step ordering, guided by a system prompt.
All 9 pipeline steps are exposed as tools with shared state via closure.
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

from .core.state import TrainingAgentState, create_initial_state

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))
from utils import get_registered_dataset, register_dataset

from .steps.cleaning_simple import run_cleaning_simple
from .steps.data_collection import data_collection as _data_collection_impl
from .steps.feature_engineering_executor import execute_feature_spec_split
from .steps.feature_engineering_simple import run_feature_engineering_simple
from .steps.label_and_split import (apply_split, compute_split_indices,
                                    run_label_split_definition)
from .steps.select_model import select_model as _select_model_impl
from .steps.training import run_training_agent as _run_training

SYSTEM_PROMPT = """\
You are an ML model training agent. Your job is to train the best possible model \
by calling the provided step tools. You have full flexibility to revisit any step.

## Available Steps (recommended first-pass order)
- select_model — Choose the model type for the task
- data_collection — Load/collect the dataset
- cleaning — Clean and standardize the data
- label_split_definition — Define the target column and create train/val/test splits
- feature_selection_specification — Analyze data and specify features to engineer
- feature_engineering_executor — Execute the feature spec to produce transformed datasets
- training_approval — Propose hyperparameters and training configuration
- training — Train the model and evaluate metrics
- generate_report — Save the final report

## First Pass
On the very first run, follow the order above sequentially.

## Iterating and Going Back
You can and SHOULD go back to earlier steps when it would improve results:
- Poor training metrics? Go back to feature_selection_specification to redesign \
features, or select_model to try a different algorithm, or both.
- User asks to change the model? Call select_model, then re-run training_approval \
and training (features can often be reused).
- User asks to change features? Call feature_selection_specification, then \
feature_engineering_executor, training_approval, and training.
- User asks to change data cleaning? Call cleaning, then re-run all downstream \
steps from label_split_definition onward.
- Feature engineering validation fails? Retry feature_selection_specification \
with adjusted specs.
- When you re-run a step, you MUST also re-run all steps that depend on its output.

## Rules
- After each step, briefly report its outcome before moving to the next.
- After generate_report, summarize the final results for the user.
- If a step errors, you may retry it once before giving up.
- Use your judgment about which steps to revisit — optimize for the best model.
"""


def _infer_task_type(goal: str, selected_model: str) -> str:
    goal_lower = goal.lower()
    model_lower = selected_model.lower()
    if any(w in model_lower for w in ["regress", "continuous", "numeric"]):
        return "regression"
    if any(w in goal_lower for w in ["regress", "predict value", "forecast", "amount", "price", "cost"]):
        return "regression"
    if any(w in model_lower for w in ["glm", "regression"]) and "logistic" not in model_lower:
        return "regression"
    return "classification"


def create_simple_training_agent(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None,
    model: str = "openai:gpt-4o-mini",
):
    """Create a simple deep-agent-based training pipeline.

    Returns a compiled deep agent that can be invoked with:
        agent.invoke({"messages": [{"role": "user", "content": goal}]})
    """
    state: dict = create_initial_state(goal, linked_datasets, user_model_preference)

    # -- tool wrappers (each closes over `state`) ---------------------------

    # Keys produced by each step, used to invalidate downstream state on re-runs
    _STEP_OUTPUTS = {
        "select_model": ["selected_model", "model_explanation"],
        "data_collection": ["collected_dataset_ref"],
        "cleaning": ["cleaned_dataset_ref", "cleaning_summary", "cleaning_transformations"],
        "label_split_definition": [
            "label_definition", "split_indices",
            "train_dataset_ref", "val_dataset_ref", "test_dataset_ref",
        ],
        "feature_selection_specification": [
            "feature_spec", "analysis_trace",
            "feature_redo_requested", "feature_redo_recommendation",
            "feature_redo_reason", "feature_redo_iteration",
        ],
        "feature_engineering_executor": [
            "transformed_train_ref", "transformed_val_ref",
            "transformed_test_ref", "transformed_dataset_ref",
            "feature_validation_passed",
        ],
        "training_approval": ["training_plan", "training_plan_approved"],
        "training": [
            "model_weights_path", "training_metrics", "training_iteration",
            "feature_redo_requested", "feature_redo_recommendation",
            "feature_redo_reason",
        ],
        "generate_report": ["report_path"],
    }
    _STEP_ORDER = [
        "select_model", "data_collection", "cleaning", "label_split_definition",
        "feature_selection_specification", "feature_engineering_executor",
        "training_approval", "training", "generate_report",
    ]

    def _invalidate_downstream(step_name: str):
        """Clear state produced by all steps after `step_name`."""
        idx = _STEP_ORDER.index(step_name)
        for later_step in _STEP_ORDER[idx + 1:]:
            for key in _STEP_OUTPUTS.get(later_step, []):
                state.pop(key, None)

    def tool_select_model() -> str:
        """Select the best ML model type for the training goal. Can be re-called to switch models."""
        nonlocal state
        _invalidate_downstream("select_model")
        result = _select_model_impl(state)
        state.update(result)
        return (
            f"Selected model: {state.get('selected_model', 'unknown')}\n"
            f"Reason: {state.get('model_explanation', 'N/A')}"
        )

    def tool_data_collection() -> str:
        """Collect or load the dataset. Can be re-called to reload or change data sources."""
        nonlocal state
        _invalidate_downstream("data_collection")
        result = _data_collection_impl(state)
        state.update(result)
        audit = next(
            (t for t in state.get("audit_trace", []) if t.get("step") == "data_collection"),
            {},
        )
        cols = len(audit.get("columns", [])) if audit.get("columns") else "?"
        return (
            f"Dataset: {state.get('collected_dataset_ref', 'unknown')}\n"
            f"Rows: {audit.get('rows', '?')}, Columns: {cols}"
        )

    def tool_cleaning() -> str:
        """Clean and standardize the collected dataset. Can be re-called to apply different cleaning."""
        nonlocal state
        _invalidate_downstream("cleaning")
        dataset_ref = state["collected_dataset_ref"]
        try:
            df = get_registered_dataset(dataset_ref)
            num_columns = len(df.columns) if df is not None else 20
            max_iters = min(80, max(30, 30 + num_columns))
        except Exception:
            max_iters = 50

        result = run_cleaning_simple(
            dataset_ref=dataset_ref,
            goal=state.get("goal", ""),
            max_iterations=max_iters,
        )
        state["cleaned_dataset_ref"] = result["cleaned_ref"]
        state["cleaning_summary"] = result.get("cleaning_summary")
        state["cleaning_transformations"] = result.get("transformations", [])
        state["current_step"] = "label_split_definition"
        return (
            f"Cleaned dataset: {result['cleaned_ref']}\n"
            f"Transformations applied: {len(result.get('transformations', []))}\n"
            f"{result.get('cleaning_summary', '')}"
        )

    def tool_label_split_definition() -> str:
        """Define the target column, split strategy, and create train/val/test splits. Can be re-called."""
        nonlocal state
        _invalidate_downstream("label_split_definition")
        existing = state.get("label_definition") or {}
        label_def = run_label_split_definition(
            dataset_ref=state["cleaned_dataset_ref"],
            goal=state.get("goal", ""),
            selected_model=state.get("selected_model"),
            model_explanation=state.get("model_explanation"),
            target_column=existing.get("target_column"),
            prediction_horizon=existing.get("prediction_horizon"),
            grain=existing.get("grain"),
            as_of_cutoff=existing.get("as_of_cutoff"),
            split_strategy=existing.get("split_strategy"),
            forbidden_columns=existing.get("forbidden_columns"),
        )

        df = get_registered_dataset(state["cleaned_dataset_ref"])
        split_indices = compute_split_indices(
            df=df, label_definition=label_def,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        )
        train_df, val_df, test_df = apply_split(df, split_indices)

        base_ref = state["cleaned_dataset_ref"]
        train_ref = f"{base_ref}_train"
        val_ref = f"{base_ref}_val"
        test_ref = f"{base_ref}_test"
        register_dataset(train_ref, train_df)
        register_dataset(val_ref, val_df)
        register_dataset(test_ref, test_df)

        state.update({
            "label_definition": label_def,
            "split_indices": split_indices,
            "train_dataset_ref": train_ref,
            "val_dataset_ref": val_ref,
            "test_dataset_ref": test_ref,
            "current_step": "feature_selection_specification",
        })
        return (
            f"Target column: {label_def.get('target_column', '?')}\n"
            f"Split strategy: {label_def.get('split_strategy', '?')}\n"
            f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows"
        )

    def tool_feature_selection_specification() -> str:
        """Analyze data and specify which features to engineer. Can be re-called to redesign features."""
        nonlocal state
        _invalidate_downstream("feature_selection_specification")
        label_def = state.get("label_definition") or {}
        train_ref = state.get("train_dataset_ref")
        target_column = label_def.get("target_column", "")
        if not train_ref or not target_column:
            return "Cannot run yet — label_split_definition must run first to define target and splits."

        recommendation = state.get("feature_redo_recommendation") if state.get("feature_redo_requested") else None

        result = run_feature_engineering_simple(
            train_ref=train_ref,
            goal=state.get("goal", ""),
            target_column=target_column,
            grain=label_def.get("grain", ""),
            recomendation=recommendation,
            val_ref=state.get("val_dataset_ref"),
            test_ref=state.get("test_dataset_ref"),
            task_type=_infer_task_type(state.get("goal", ""), state.get("selected_model", "")),
            forbidden_columns=label_def.get("forbidden_columns", []),
            as_of_cutoff=label_def.get("as_of_cutoff"),
            prediction_horizon=label_def.get("prediction_horizon"),
        )

        feature_spec = result.get("feature_spec")
        validation = result.get("validation", {})
        if validation and not validation.get("valid", True) and feature_spec and "features" in feature_spec:
            features_valid = validation.get("features_valid", {})
            feature_spec["features"] = [
                f for f in feature_spec["features"]
                if features_valid.get(f.get("name"), {}).get("valid", True)
            ]

        redo_iter = state.get("feature_redo_iteration", 0)
        state.update({
            "feature_spec": feature_spec,
            "analysis_trace": [{"step": "feature_selection_specification",
                                "key_stats": result.get("key_stats", {}),
                                "is_redo": state.get("feature_redo_requested", False)}],
            "feature_redo_requested": False,
            "feature_redo_recommendation": None,
            "feature_redo_reason": None,
            "feature_redo_iteration": redo_iter + 1 if state.get("feature_redo_requested") else redo_iter,
        })
        features = (feature_spec or {}).get("features", [])
        names = [f.get("name", "?") for f in features[:10]]
        return f"Features specified: {len(features)}\nNames: {', '.join(names)}"

    def tool_feature_engineering_executor() -> str:
        """Execute the feature specification to produce transformed train/val/test datasets. Can be re-called."""
        nonlocal state
        _invalidate_downstream("feature_engineering_executor")
        label_def = state.get("label_definition") or {}
        train_ref = state.get("train_dataset_ref")
        feature_spec = state.get("feature_spec")
        target_column = label_def.get("target_column", "")
        if not train_ref or not feature_spec or not target_column:
            return "Cannot run yet — feature_selection_specification must run first to define features."

        result = execute_feature_spec_split(
            train_ref=train_ref,
            val_ref=state.get("val_dataset_ref"),
            test_ref=state.get("test_dataset_ref"),
            feature_spec=feature_spec,
            target_column=target_column,
            grain=label_def.get("grain", ""),
            as_of_cutoff=label_def.get("as_of_cutoff"),
        )

        errors = result.get("errors", [])
        features_created = result.get("features_created", [])
        passed = len(features_created) > 0 and len(errors) < len(features_created)

        state.update({
            "transformed_train_ref": result.get("train_ref"),
            "transformed_val_ref": result.get("val_ref"),
            "transformed_test_ref": result.get("test_ref"),
            "transformed_dataset_ref": result.get("train_ref"),
            "feature_validation_passed": passed,
            "audit_trace": state.get("audit_trace", []) + [{
                "step": "feature_engineering_executor",
                "features_created": features_created,
                "errors": errors,
                "shapes": result.get("shapes"),
            }],
        })
        shapes = result.get("shapes", {})
        status = "PASSED" if passed else "FAILED"
        summary = (
            f"Validation: {status}\n"
            f"Features created: {len(features_created)}\n"
            f"Errors: {len(errors)}\n"
            f"Train shape: {shapes.get('train', '?')} | Val: {shapes.get('val', '?')} | Test: {shapes.get('test', '?')}"
        )
        if not passed:
            summary += "\n\nFeature engineering had issues. Consider going back to feature_selection_specification."
        return summary

    def tool_training_approval() -> str:
        """Propose a training configuration (hyperparameters, strategy). Re-call after changing model or features."""
        nonlocal state
        _invalidate_downstream("training_approval")
        train_ref = state.get("transformed_train_ref")
        val_ref = state.get("transformed_val_ref")
        label_def = state.get("label_definition") or {}
        target_column = label_def.get("target_column", "")
        selected_model = state.get("selected_model", "logistic_regression")
        task_type = _infer_task_type(state.get("goal", ""), selected_model)

        train_df = get_registered_dataset(train_ref)
        val_df = get_registered_dataset(val_ref) if val_ref else None
        if train_df is None:
            return "Cannot run yet — feature_engineering_executor must run first to produce training data."

        n_rows = len(train_df)
        n_features = len([c for c in train_df.columns if c != target_column])
        class_counts = train_df[target_column].value_counts().to_dict()
        total = sum(class_counts.values())
        minority_ratio = min(class_counts.values()) / total if total > 0 else 0
        is_imbalanced = minority_ratio < 0.3

        feature_names = [f.get("name") for f in (state.get("feature_spec") or {}).get("features", [])][:20]

        prompt = f"""You are an ML expert. Propose a training configuration.

Goal: {state.get('goal', '')}
Task Type: {task_type}
Model: {selected_model}
Target: {target_column}
Training rows: {n_rows}, Features: {n_features}
Feature names (first 20): {feature_names}
Val rows: {len(val_df) if val_df is not None else 'N/A'}
Class distribution: {json.dumps(class_counts)}
{"Imbalanced data - minority class is " + f"{minority_ratio:.1%}" if is_imbalanced else "Balanced classes"}

Respond with JSON:
{{
    "model_type": "{selected_model}",
    "task_type": "{task_type}",
    "hyperparameters": {{}},
    "class_weight": "balanced" or null,
    "max_iterations": 3,
    "strategy_notes": "...",
    "expected_metrics": "..."
}}"""

        response = init_chat_model("openai:gpt-5.1").invoke([{"role": "user", "content": prompt}])
        try:
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response.content)
            training_plan = json.loads(json_match.group(1)) if json_match else json.loads(response.content)
        except (json.JSONDecodeError, AttributeError):
            training_plan = {
                "model_type": selected_model, "task_type": task_type,
                "hyperparameters": {},
                "class_weight": "balanced" if is_imbalanced else None,
                "max_iterations": 3,
                "strategy_notes": "Default configuration",
                "expected_metrics": "Standard metrics",
            }

        training_plan.setdefault("model_type", selected_model)
        training_plan.setdefault("task_type", task_type)
        training_plan.setdefault("max_iterations", 3)
        training_plan["data_summary"] = {
            "train_rows": n_rows,
            "val_rows": len(val_df) if val_df is not None else None,
            "n_features": n_features,
            "class_distribution": class_counts,
            "is_imbalanced": is_imbalanced,
        }

        state["training_plan"] = training_plan
        state["training_plan_approved"] = True
        state["current_step"] = "training"

        hp = training_plan.get("hyperparameters", {})
        hp_str = ", ".join(f"{k}={v}" for k, v in hp.items()) if hp else "(defaults)"
        return (
            f"Training plan approved.\n"
            f"Model: {training_plan.get('model_type')}\n"
            f"Hyperparameters: {hp_str}\n"
            f"Strategy: {training_plan.get('strategy_notes', 'N/A')}"
        )

    def tool_training() -> str:
        """Train the model and evaluate on validation/test sets. Can be re-called with different config or features."""
        nonlocal state
        _invalidate_downstream("training")
        label_def = state.get("label_definition") or {}
        target_column = label_def.get("target_column", "")
        selected_model = state.get("selected_model", "logistic_regression")
        train_ref = state.get("transformed_train_ref")
        if not train_ref or not target_column:
            return "Cannot run yet — feature_engineering_executor and label_split_definition must complete first."

        model_name = f"{selected_model}_{int(time.time())}"
        result = _run_training(
            train_ref=train_ref,
            val_ref=state.get("transformed_val_ref"),
            test_ref=state.get("transformed_test_ref"),
            target_column=target_column,
            selected_model=selected_model,
            goal=state.get("goal", ""),
            model_name=model_name,
            max_iterations=3,
        )

        feature_redo_requested = result.get("feature_redo_requested", False)
        state.update({
            "model_weights_path": result.get("model_name"),
            "training_metrics": {
                "success": result.get("success"),
                "model_name": result.get("model_name"),
                "model_type": result.get("model_type"),
                "val_accuracy": result.get("val_accuracy"),
                "val_roc_auc": result.get("val_roc_auc"),
                "test_accuracy": result.get("test_accuracy"),
                "test_roc_auc": result.get("test_roc_auc"),
                "train_r2": result.get("train_r2"),
                "val_r2": result.get("val_r2"),
                "val_rmse": result.get("val_rmse"),
                "test_r2": result.get("test_r2"),
                "test_rmse": result.get("test_rmse"),
                "num_iterations": result.get("num_iterations", 0),
                "summary": result.get("summary"),
                "recommendations": result.get("recommendations"),
                "feature_redo_requested": feature_redo_requested,
            },
            "training_iteration": state.get("training_iteration", 0) + 1,
            "feature_redo_requested": feature_redo_requested,
            "feature_redo_recommendation": result.get("feature_redo_recommendation"),
            "feature_redo_reason": result.get("feature_redo_reason"),
            "audit_trace": state.get("audit_trace", []) + [{
                "step": "training",
                "model_name": result.get("model_name"),
                "success": result.get("success"),
                "num_iterations": result.get("num_iterations", 0),
                "feature_redo_requested": feature_redo_requested,
            }],
        })

        lines = [f"Training {'succeeded' if result.get('success') else 'FAILED'}"]
        lines.append(f"Model: {result.get('model_name', '?')}")
        for key, label in [("val_accuracy", "Val Accuracy"), ("val_roc_auc", "Val ROC-AUC"),
                           ("test_accuracy", "Test Accuracy"), ("val_r2", "Val R²"), ("test_r2", "Test R²")]:
            v = result.get(key)
            if v is not None:
                lines.append(f"{label}: {v:.4f}")
        if result.get("summary"):
            lines.append(f"Summary: {result['summary']}")
        if feature_redo_requested:
            lines.append(f"\nFeature redo recommended: {result.get('feature_redo_reason')}")
            lines.append("Consider going back to feature_selection_specification.")
        return "\n".join(lines)

    def tool_generate_report() -> str:
        """Generate and save the final training report. Call when satisfied with training results."""
        nonlocal state
        metrics = state.get("training_metrics", {})
        label_def = state.get("label_definition") or {}

        report = {
            "generated_at": datetime.now().isoformat(),
            "goal": state.get("goal"),
            "model": {
                "type": state.get("selected_model"),
                "name": metrics.get("model_name"),
                "explanation": state.get("model_explanation"),
            },
            "data": {
                "collected_dataset": state.get("collected_dataset_ref"),
                "cleaned_dataset": state.get("cleaned_dataset_ref"),
                "train_dataset": state.get("transformed_train_ref"),
                "val_dataset": state.get("transformed_val_ref"),
                "test_dataset": state.get("transformed_test_ref"),
            },
            "label_definition": {
                "target_column": label_def.get("target_column"),
                "split_strategy": label_def.get("split_strategy"),
                "grain": label_def.get("grain"),
            },
            "training_results": {
                "success": metrics.get("success"),
                "num_iterations": metrics.get("num_iterations", 0),
                "validation_metrics": {
                    "accuracy": metrics.get("val_accuracy"),
                    "roc_auc": metrics.get("val_roc_auc"),
                },
                "test_metrics": {
                    "accuracy": metrics.get("test_accuracy"),
                    "roc_auc": metrics.get("test_roc_auc"),
                },
                "summary": metrics.get("summary"),
                "recommendations": metrics.get("recommendations"),
            },
            "audit_trace": state.get("audit_trace", []),
        }

        report_dir = Path(__file__).parent.parent.parent / "trained_models"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"{metrics.get('model_name', 'unknown')}_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        state["report_path"] = str(report_path)
        state["audit_trace"] = state.get("audit_trace", []) + [
            {"step": "generate_report", "path": str(report_path)}
        ]
        return f"Report saved to {report_path}"

    # -- build the deep agent -----------------------------------------------

    agent = create_deep_agent(
        model=model,
        tools=[
            tool_select_model,
            tool_data_collection,
            tool_cleaning,
            tool_label_split_definition,
            tool_feature_selection_specification,
            tool_feature_engineering_executor,
            tool_training_approval,
            tool_training,
            tool_generate_report,
        ],
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def invoke_simple_training_agent(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None,
    model: str = "openai:gpt-4o-mini",
):
    """Convenience function: create and invoke the simple training agent in one call."""
    agent = create_simple_training_agent(goal, linked_datasets, user_model_preference, model)
    result = agent.invoke({"messages": [{"role": "user", "content": goal}]})
    return result
