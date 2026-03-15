"""
Simple ML Training Agent using LangChain's create_agent.

Uses a single tool-calling agent whose LLM decides step ordering,
guided by a system prompt.
All 9 pipeline steps are exposed as tools with shared state via closure.
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from pydantic import BaseModel, Field

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
from .steps.orchestrator import _infer_target_column
from .steps.select_model import MODEL_FAMILIES
from .steps.select_model import select_model as _select_model_impl
from .steps.training import run_training_agent as _run_training


# =============================================================================
# STRUCTURED OUTPUT SCHEMAS
# =============================================================================


class TrainingPlan(BaseModel):
    """Training plan proposed by the LLM before model training begins."""

    model_type: str = Field(description="Model family: 'supervised', 'unsupervised', or 'neural_networks'")
    task_type: Literal["classification", "regression", "unsupervised"] = Field(description="Task type: classification, regression, or unsupervised")
    hyperparameters: dict = Field(
        default_factory=dict,
        description="Starting hyperparameters. For neural_networks: architecture, optimizer, lr, weight_decay, epochs, patience. For supervised: estimator-specific params.",
    )
    class_weight: Optional[str] = Field(
        default=None,
        description="Class weighting strategy for imbalanced data. E.g. 'balanced', 'use CrossEntropyLoss weight param', or null.",
    )
    max_iterations: int = Field(
        default=5,
        description="Number of experiment iterations the training agent should run",
    )
    strategy_notes: str = Field(description="High-level training strategy and experiment plan")
    expected_metrics: str = Field(description="Expected range of validation metrics for this task")


SYSTEM_PROMPT = """\
You are an ML pipeline agent. Execute the pipeline steps to train the best model.

## CRITICAL RULES
1. Call EXACTLY ONE tool at a time. NEVER call multiple tools simultaneously.
2. Wait for each tool to return before calling the next.
3. Do NOT skip steps. Do NOT repeat a step that already succeeded.
4. After each tool returns, briefly note the result, then call the next tool.

## First Pass — Execute in THIS EXACT ORDER
1. data_collection — Retrieve the dataset(s) and load it
2. select_model — Choose the model family (supervised, unsupervised, or neural_networks) based on the goal
3. cleaning — Clean and standardize the data
4. label_split_definition — Define target column and train/val/test splits
5. feature_selection_specification — Analyze data and specify features
6. feature_engineering_executor — Execute feature transformations
7. training_approval — Propose training configuration
8. training — Train the model and evaluate metrics
9. generate_report — Save the final report

## After Training — Optimization (Optional)
After training completes, evaluate the metrics. If they are unsatisfactory:
- Weak features → go back to feature_selection_specification
- Wrong model family → go back to select_model
- Poor hyperparameters → go back to training_approval
- Unsuitable dataset (near-zero R² or accuracy near random) → go back to data_collection
After changing any step, re-run all downstream steps in order.
Do not loop more than 4 total training iterations.

## After generate_report
Summarize the final results.
"""


def _infer_task_type(goal: str, selected_model: str) -> str:
    if selected_model == "unsupervised":
        return "unsupervised"
    goal_lower = goal.lower()
    model_lower = selected_model.lower()
    if "logistic" not in model_lower:
        if any(w in model_lower for w in ["regress", "continuous", "numeric"]):
            return "regression"
    if any(w in goal_lower for w in ["regress", "predict value", "forecast", "amount", "price", "cost", "salary", "revenue", "income"]):
        return "regression"
    if any(w in model_lower for w in ["glm", "regression"]) and "logistic" not in model_lower:
        return "regression"
    return "classification"


def create_simple_training_agent(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None,
    model: str = "openai:gpt-5.1",
    hitl: bool = True,
    checkpointer=None,
    use_external_sources: bool = False,
):
    """Create a simple deep-agent-based training pipeline.

    Args:
        hitl: If True, tools call interrupt() after completing their work so
              the user can review results before the agent proceeds.
              If False, all tools run without interruption.
        checkpointer: LangGraph checkpointer for state persistence (required for
                      HITL). A MemorySaver is created automatically if hitl=True
                      and no checkpointer is provided.
        use_external_sources: If True, data collection will fall back to the
                              Dataset Curator (Kaggle + HuggingFace) when local
                              retrieval fails.

    Returns a compiled deep agent that can be invoked with:
        agent.invoke({"messages": [{"role": "user", "content": goal}]}, config=...)
    """
    state: dict = create_initial_state(goal, linked_datasets, user_model_preference, use_external_sources)

    def _hitl_gate(node_name: str, summary: str) -> dict:
        """Interrupt for human review after a step completes. Returns the decision."""
        if not hitl:
            return {"approved": True}
        decision = interrupt({
            "node": node_name,
            "summary": summary,
            "message": "Approve to continue, or provide feedback to redo.",
        })
        if isinstance(decision, dict):
            return decision
        return {"approved": True}

    KNOWN_FAMILIES = set(MODEL_FAMILIES.keys())

    # -- tool wrappers (each closes over `state`) ---------------------------

    # Track which steps have successfully completed to prevent redundant calls
    _completed_steps: set[str] = set()

    # Keys produced by each step, used to invalidate downstream state on re-runs
    _STEP_OUTPUTS = {
        "select_model": ["selected_model", "model_explanation"],
        "data_collection": ["collected_dataset_ref", "data_source"],
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
        "data_collection", "select_model", "cleaning", "label_split_definition",
        "feature_selection_specification", "feature_engineering_executor",
        "training_approval", "training", "generate_report",
    ]

    def _invalidate_downstream(step_name: str):
        """Clear state produced by all steps after `step_name`."""
        idx = _STEP_ORDER.index(step_name)
        for later_step in _STEP_ORDER[idx + 1:]:
            for key in _STEP_OUTPUTS.get(later_step, []):
                state.pop(key, None)
            _completed_steps.discard(later_step)

    def tool_select_model() -> str:
        """Select the best ML model family for the training goal. Can be re-called to switch families."""
        nonlocal state
        if "select_model" in _completed_steps and not state.get("_redo_feedback_select_model"):
            return f"SKIP: Model family already selected: {state.get('selected_model')}. Proceed to the next step."
        if state.get("selected_model") is not None:
            _invalidate_downstream("select_model")

        redo_fb = state.pop("_redo_feedback_select_model", None)
        if redo_fb:
            fb_lower = redo_fb.lower()
            for m in KNOWN_FAMILIES:
                if m in fb_lower or m.replace("_", " ") in fb_lower:
                    state["user_model_preference"] = m
                    break
            else:
                state["_select_model_redo_hint"] = redo_fb

        result = _select_model_impl(state)
        state.update(result)
        state.pop("_select_model_redo_hint", None)

        summary = (
            f"Selected model family **{state.get('selected_model', 'unknown')}** for this task.\n"
            f"Reason: {state.get('model_explanation', 'N/A')}"
        )
        decision = _hitl_gate("select_model", summary)
        if not decision.get("approved", True):
            fb = decision.get("feedback", "Please reconsider the model family choice.")
            state["_redo_feedback_select_model"] = fb
            state.pop("user_model_preference", None)
            return f"REJECTED by user: {fb}. Please redo model family selection."

        _completed_steps.add("select_model")
        return (
            f"Selected model family: {state.get('selected_model', 'unknown')}\n"
            f"Reason: {state.get('model_explanation', 'N/A')}"
        )

    def tool_data_collection() -> str:
        """Collect or load the dataset. Can be re-called to reload or change data sources."""
        nonlocal state
        if "data_collection" in _completed_steps and not state.get("_redo_feedback_data_collection"):
            ref = state.get("collected_dataset_ref")
            if ref:
                return f"SKIP: Dataset already loaded: {ref}. Proceed to the next step."
        _invalidate_downstream("data_collection")

        redo_fb = state.pop("_redo_feedback_data_collection", None)
        if redo_fb:
            state["_data_collection_redo_hint"] = redo_fb

        result = _data_collection_impl(state)
        state.update(result)
        state.pop("_data_collection_redo_hint", None)

        if not state.get("collected_dataset_ref"):
            error = state.get("error", "Unknown error")
            return (
                f"FAILED: Could not find a suitable dataset. Error: {error}\n"
                "Please try calling data_collection again or adjust the goal."
            )

        audit = next(
            (t for t in state.get("audit_trace", []) if t.get("step") == "data_collection"),
            {},
        )
        cols = len(audit.get("columns", [])) if audit.get("columns") else "?"

        summary = (
            f"Loaded **{state.get('collected_dataset_ref', 'unknown')}** "
            f"({audit.get('rows', '?')} rows, {cols} columns)"
        )
        decision = _hitl_gate("data_collection", summary)
        if not decision.get("approved", True):
            fb = decision.get("feedback", "Please redo data collection.")
            state["_redo_feedback_data_collection"] = fb
            return f"REJECTED by user: {fb}"

        _completed_steps.add("data_collection")
        return (
            f"Dataset: {state.get('collected_dataset_ref', 'unknown')}\n"
            f"Rows: {audit.get('rows', '?')}, Columns: {cols}"
        )

    def tool_cleaning() -> str:
        """Clean and standardize the collected dataset. Can be re-called to apply different cleaning."""
        nonlocal state
        if "cleaning" in _completed_steps and not state.get("_redo_feedback_cleaning"):
            return f"SKIP: Data already cleaned: {state.get('cleaned_dataset_ref')}. Proceed to the next step."

        dataset_ref = state.get("collected_dataset_ref")
        if not dataset_ref:
            return "SKIP: Cannot run — data_collection must run first."
        _invalidate_downstream("cleaning")
        redo_fb = state.pop("_redo_feedback_cleaning", None)
        try:
            df = get_registered_dataset(dataset_ref)
            num_columns = len(df.columns) if df is not None else 20
            max_iters = min(80, max(30, 30 + num_columns))
        except Exception:
            max_iters = 50

        cleaning_goal = state.get("goal", "")
        if redo_fb:
            cleaning_goal += f"\n\nIMPORTANT user feedback on previous cleaning: {redo_fb}"

        sel_model = state.get("selected_model", "")
        target_col = _infer_target_column(cleaning_goal, dataset_ref)
        task_type = _infer_task_type(cleaning_goal, sel_model) if target_col else None

        result = run_cleaning_simple(
            dataset_ref=dataset_ref,
            goal=cleaning_goal,
            max_iterations=max_iters,
            target_col=target_col,
            task_type=task_type,
            selected_model=sel_model,
        )
        state["cleaned_dataset_ref"] = result["cleaned_ref"]
        state["cleaning_summary"] = result.get("cleaning_summary")
        state["cleaning_transformations"] = result.get("transformations", [])
        state["current_step"] = "label_split_definition"

        n_transforms = len(result.get("transformations", []))
        summary = (
            f"Applied {n_transforms} transformations to clean the data.\n"
            f"{result.get('cleaning_summary', '')}"
        )
        decision = _hitl_gate("cleaning", summary)
        if not decision.get("approved", True):
            fb = decision.get("feedback", "Please redo cleaning.")
            state["_redo_feedback_cleaning"] = fb
            return f"REJECTED by user: {fb}"

        _completed_steps.add("cleaning")
        return (
            f"Cleaned dataset: {result['cleaned_ref']}\n"
            f"Transformations applied: {n_transforms}\n"
            f"{result.get('cleaning_summary', '')}"
        )

    def tool_label_split_definition() -> str:
        """Define the target column, split strategy, and create train/val/test splits. Can be re-called."""
        nonlocal state
        if "label_split_definition" in _completed_steps and not state.get("_redo_feedback_label_split"):
            ld = state.get("label_definition") or {}
            return f"SKIP: Label/split already defined. Target: {ld.get('target_column')}. Proceed to the next step."
        dataset_ref = state.get("cleaned_dataset_ref")
        if not dataset_ref:
            return "SKIP: Cannot run — cleaning must run first."
        _invalidate_downstream("label_split_definition")

        if state.get("selected_model") == "unsupervised":
            df = get_registered_dataset(dataset_ref)
            if df is None:
                return f"SKIP: Dataset not found: {dataset_ref}"
            train_ref = f"{dataset_ref}_train"
            register_dataset(train_ref, df)
            state.update({
                "label_definition": {
                    "target_column": "",
                    "prediction_horizon": None,
                    "grain": "",
                    "as_of_cutoff": None,
                    "split_strategy": "random",
                    "forbidden_columns": [],
                },
                "split_indices": None,
                "train_dataset_ref": train_ref,
                "val_dataset_ref": None,
                "test_dataset_ref": None,
                "current_step": "feature_selection_specification",
            })
            _completed_steps.add("label_split_definition")
            print(f"🏷️ label_split: unsupervised bypass — train={train_ref}", flush=True)
            return f"Unsupervised flow: label/split skipped. Train: {train_ref}"

        redo_fb = state.pop("_redo_feedback_label_split", None)

        existing = state.get("label_definition") or {}
        if redo_fb:
            existing = {}

        label_goal = state.get("goal", "")
        if redo_fb:
            label_goal += f"\n\nIMPORTANT user feedback on previous label/split: {redo_fb}"

        label_def = run_label_split_definition(
            dataset_ref=dataset_ref,
            goal=label_goal,
            selected_model=state.get("selected_model"),
            model_explanation=state.get("model_explanation"),
            target_column=existing.get("target_column"),
            prediction_horizon=existing.get("prediction_horizon"),
            grain=existing.get("grain"),
            as_of_cutoff=existing.get("as_of_cutoff"),
            split_strategy=existing.get("split_strategy"),
            forbidden_columns=existing.get("forbidden_columns"),
        )

        df = get_registered_dataset(dataset_ref)
        split_indices = compute_split_indices(
            df=df, label_definition=label_def,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        )
        target_transform = label_def.get("target_transform")
        train_df, val_df, test_df = apply_split(
            df, split_indices,
            target_column=label_def.get("target_column"),
            target_transform=target_transform,
        )

        base_ref = dataset_ref
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

        transform_note = f" (target transformed: {target_transform})" if target_transform else ""
        summary = (
            f"Target: **{label_def.get('target_column', '?')}**{transform_note}, "
            f"{label_def.get('split_strategy', '?')} split — "
            f"Train: {len(train_df)} / Val: {len(val_df)} / Test: {len(test_df)}"
        )
        decision = _hitl_gate("label_split_definition", summary)
        if not decision.get("approved", True):
            fb = decision.get("feedback", "Please redo label/split definition.")
            state["_redo_feedback_label_split"] = fb
            return f"REJECTED by user: {fb}"

        _completed_steps.add("label_split_definition")
        return (
            f"Target column: {label_def.get('target_column', '?')}\n"
            f"Split strategy: {label_def.get('split_strategy', '?')}\n"
            f"Train: {len(train_df)} rows | Val: {len(val_df)} rows | Test: {len(test_df)} rows"
        )

    def tool_feature_selection_specification() -> str:
        """Analyze data and specify which features to engineer. Can be re-called to redesign features."""
        nonlocal state
        if "feature_selection_specification" in _completed_steps and not state.get("feature_redo_requested") and not state.get("_redo_feedback_feature_selection"):
            fs = (state.get("feature_spec") or {}).get("features", [])
            return f"SKIP: Features already specified ({len(fs)} features). Proceed to the next step."
        label_def = state.get("label_definition") or {}
        if state.get("selected_model") == "unsupervised":
            state.update({
                "feature_spec": {"features": []},
                "analysis_trace": [],
                "feature_redo_requested": False,
                "feature_redo_recommendation": None,
                "feature_redo_reason": None,
            })
            _completed_steps.add("feature_selection_specification")
            return "Unsupervised flow: feature specification skipped (direct training on cleaned columns)."
        train_ref = state.get("train_dataset_ref")
        target_column = label_def.get("target_column", "")
        if not train_ref or not target_column:
            return "SKIP: Cannot run — label_split_definition must run first."
        _invalidate_downstream("feature_selection_specification")

        redo_fb = state.pop("_redo_feedback_feature_selection", None)
        if redo_fb:
            state["feature_redo_requested"] = True
            state["feature_redo_recommendation"] = redo_fb

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
            selected_model=state.get("selected_model"),
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

        summary = f"Specified **{len(features)}** features: {', '.join(names)}"
        if len(features) > 10:
            summary += f" … and {len(features) - 10} more"
        decision = _hitl_gate("feature_selection_specification", summary)
        if not decision.get("approved", True):
            fb = decision.get("feedback", "Please revise feature selection.")
            state["_redo_feedback_feature_selection"] = fb
            return f"REJECTED by user: {fb}"

        _completed_steps.add("feature_selection_specification")
        return f"Features specified: {len(features)}\nNames: {', '.join(names)}"

    def tool_feature_engineering_executor() -> str:
        """Execute the feature specification to produce transformed train/val/test datasets. Can be re-called."""
        nonlocal state
        if "feature_engineering_executor" in _completed_steps and state.get("feature_validation_passed"):
            return f"SKIP: Features already engineered. Proceed to the next step."
        label_def = state.get("label_definition") or {}
        if state.get("selected_model") == "unsupervised":
            train_ref = state.get("train_dataset_ref")
            if not train_ref:
                return "SKIP: Cannot run — label_split_definition must run first."
            state.update({
                "transformed_train_ref": train_ref,
                "transformed_val_ref": None,
                "transformed_test_ref": None,
                "transformed_dataset_ref": train_ref,
                "feature_validation_passed": True,
            })
            _completed_steps.add("feature_engineering_executor")
            return "Unsupervised flow: feature engineering passthrough complete."
        train_ref = state.get("train_dataset_ref")
        feature_spec = state.get("feature_spec")
        target_column = label_def.get("target_column", "")
        if not train_ref or not feature_spec or not target_column:
            return "SKIP: Cannot run — feature_selection_specification must run first."
        _invalidate_downstream("feature_engineering_executor")

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
        if passed:
            _completed_steps.add("feature_engineering_executor")
        status = "PASSED" if passed else "FAILED"

        def _fmt_shape(s):
            if isinstance(s, (list, tuple)) and len(s) >= 2:
                return f"{s[0]} × {s[1]}"
            return str(s) if s else "?"

        summary = (
            f"Validation: {status}\n"
            f"Features created: {len(features_created)}\n"
            f"Errors: {len(errors)}\n"
            f"Train shape: {_fmt_shape(shapes.get('train'))} | "
            f"Val: {_fmt_shape(shapes.get('val'))} | "
            f"Test: {_fmt_shape(shapes.get('test'))}"
        )
        if not passed:
            summary += "\n\nFeature engineering had issues. Consider going back to feature_selection_specification."
        return summary

    def tool_training_approval() -> str:
        """Propose a training configuration (hyperparameters, strategy). Re-call after changing model or features."""
        nonlocal state
        if "training_approval" in _completed_steps and not state.get("_redo_feedback_training_approval"):
            tp = state.get("training_plan") or {}
            return f"SKIP: Training plan already approved ({tp.get('model_type')}). Proceed to the next step."
        _invalidate_downstream("training_approval")
        train_ref = state.get("transformed_train_ref")
        val_ref = state.get("transformed_val_ref")
        label_def = state.get("label_definition") or {}
        target_column = label_def.get("target_column", "")
        selected_model = state.get("selected_model", "supervised")
        unsupervised = selected_model == "unsupervised"
        task_type = "unsupervised" if unsupervised else _infer_task_type(state.get("goal", ""), selected_model)

        train_df = get_registered_dataset(train_ref)
        val_df = get_registered_dataset(val_ref) if val_ref else None
        if train_df is None:
            return "SKIP: Cannot run — feature_engineering_executor must run first."

        n_rows = len(train_df)
        if target_column:
            n_features = len([c for c in train_df.columns if c != target_column])
            class_counts = train_df[target_column].value_counts().to_dict()
            total = sum(class_counts.values())
            minority_ratio = min(class_counts.values()) / total if total > 0 else 0
            is_imbalanced = minority_ratio < 0.3
        else:
            n_features = len(train_df.columns)
            class_counts = {}
            minority_ratio = 0
            is_imbalanced = False

        feature_names = [f.get("name") for f in (state.get("feature_spec") or {}).get("features", [])][:20]

        redo_fb = state.pop("_redo_feedback_training_approval", None)
        redo_section = ""
        if redo_fb:
            redo_section = f"\n\nIMPORTANT - The user rejected the previous training plan with this feedback:\n\"{redo_fb}\"\nPlease adjust accordingly.\n"

        size_bucket = "small" if n_rows < 1000 else ("medium" if n_rows < 10000 else ("large" if n_rows < 100000 else "very large"))

        if selected_model == "neural_networks":
            arch_rec = {
                "small": "1-2 layers, 32-64 units, dropout 0.3-0.5",
                "medium": "2-3 layers, 64-128 units, dropout 0.2-0.3",
                "large": "2-4 layers, 128-256 units, BatchNorm + dropout 0.1-0.3",
                "very large": "3-5 layers, 256-512 units, BatchNorm, lower dropout",
            }[size_bucket]
            model_section = (
                f"Model: neural_networks (PyTorch code execution)\n"
                f"Architecture recommendation for {size_bucket} dataset: {arch_rec}\n\n"
                f"This uses the code-execution workflow — the training agent writes PyTorch code.\n"
                f"Propose a plan following the autoresearch experiment protocol:\n"
                f"1. Baseline: default 2-layer network with AdamW, early stopping\n"
                f"2. Architecture search: try different depths/widths (short runs, 30 epochs each)\n"
                f"3. Refinement: tune the winning architecture (dropout, BatchNorm, LR schedule)\n"
                f"4. Final training: full run with best config, early stopping"
            )
        else:
            model_section = f"Model: {selected_model}"

        if task_type == "unsupervised":
            imbalance_note = "Unsupervised task — no target column or class distribution."
        elif is_imbalanced:
            imbalance_note = f"Imbalanced data — minority class is {minority_ratio:.1%}. Consider class weights."
        else:
            imbalance_note = "Balanced classes"

        prompt = (
            f"You are an ML expert. Propose a training plan.\n\n"
            f"Goal: {state.get('goal', '')}\n"
            f"Task type: {task_type}\n"
            f"{model_section}\n"
            f"Target: {target_column or '(none — unsupervised)'}\n"
            f"Training rows: {n_rows} ({size_bucket}), Features: {n_features}\n"
            f"Feature names (first 20): {feature_names}\n"
            f"Val rows: {len(val_df) if val_df is not None else 'N/A'}\n"
            f"Class distribution: {json.dumps(class_counts) if class_counts else 'N/A (unsupervised)'}\n"
            f"{imbalance_note}"
            f"{redo_section}"
        )

        structured_llm = init_chat_model("openai:gpt-5.1").with_structured_output(
            TrainingPlan, method="function_calling"
        )
        training_plan = structured_llm.invoke(prompt).model_dump()
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

        summary = (
            f"Training config: **{training_plan.get('model_type')}**\n"
            f"Hyperparameters: {hp_str}\n"
            f"Strategy: {training_plan.get('strategy_notes', 'N/A')}"
        )
        decision = _hitl_gate("training_approval", summary)
        if not decision.get("approved", True):
            fb = decision.get("feedback", "Please adjust the training configuration.")
            state["_redo_feedback_training_approval"] = fb
            return f"REJECTED by user: {fb}"

        _completed_steps.add("training_approval")
        return (
            f"Training plan proposed.\n"
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
        selected_model = state.get("selected_model", "supervised")
        task_type = _infer_task_type(state.get("goal", ""), selected_model)
        train_ref = state.get("transformed_train_ref")
        if not train_ref:
            return "SKIP: Cannot run — feature_engineering_executor must complete first."
        if not target_column and task_type != "unsupervised":
            return "SKIP: Cannot run — label_split_definition must define a target column first."

        model_name = f"{selected_model}_{int(time.time())}"
        training_plan = state.get("training_plan") or {}
        plan_max_iters = training_plan.get("max_iterations", 5 if selected_model == "neural_networks" else 3)
        result = _run_training(
            train_ref=train_ref,
            val_ref=state.get("transformed_val_ref"),
            test_ref=state.get("transformed_test_ref"),
            target_column=target_column,
            selected_model=selected_model,
            goal=state.get("goal", ""),
            model_name=model_name,
            max_iterations=plan_max_iters,
        )

        feature_redo_requested = result.get("feature_redo_requested", False)
        best_model_type = result.get("model_type", selected_model)
        task_type = _infer_task_type(state.get("goal", ""), selected_model)

        # --- Programmatic quality gates ---
        iteration_num = state.get("training_iteration", 0) + 1
        if result.get("success") and not feature_redo_requested:
            if task_type == "regression":
                val_r2 = result.get("val_r2")
                if val_r2 is not None and val_r2 < 0.05 and iteration_num <= 2:
                    feature_redo_requested = True
                    extra = result.get("feature_redo_recommendation") or ""
                    result["feature_redo_recommendation"] = (
                        extra + "\n[AUTO] Val R² < 0.05 — features may lack predictive signal. "
                        "Try adding interactions, polynomial terms, or different encodings."
                    )
                    result["feature_redo_reason"] = f"Auto-triggered: val_r2={val_r2:.4f}"
            else:
                val_roc = result.get("val_roc_auc")
                val_acc = result.get("val_accuracy")
                if val_roc is not None and val_roc < 0.55 and iteration_num <= 2:
                    feature_redo_requested = True
                    extra = result.get("feature_redo_recommendation") or ""
                    result["feature_redo_recommendation"] = (
                        extra + "\n[AUTO] Val ROC-AUC < 0.55 — near-random performance. "
                        "Features may be uninformative or target is noisy."
                    )
                    result["feature_redo_reason"] = f"Auto-triggered: val_roc_auc={val_roc:.4f}"
                elif val_acc is not None and val_acc < 0.55 and val_roc is None and iteration_num <= 2:
                    feature_redo_requested = True
                    extra = result.get("feature_redo_recommendation") or ""
                    result["feature_redo_recommendation"] = (
                        extra + "\n[AUTO] Val Accuracy < 0.55 — near-random."
                    )
                    result["feature_redo_reason"] = f"Auto-triggered: val_accuracy={val_acc:.4f}"

        state.update({
            "model_weights_path": result.get("model_name"),
            "selected_model": best_model_type,
            "training_metrics": {
                "success": result.get("success"),
                "model_name": result.get("model_name"),
                "model_type": best_model_type,
                "val_accuracy": result.get("val_accuracy"),
                "val_roc_auc": result.get("val_roc_auc"),
                "test_accuracy": result.get("test_accuracy"),
                "test_roc_auc": result.get("test_roc_auc"),
                "train_r2": result.get("train_r2"),
                "val_r2": result.get("val_r2"),
                "val_rmse": result.get("val_rmse"),
                "val_mae": result.get("val_mae"),
                "test_r2": result.get("test_r2"),
                "test_rmse": result.get("test_rmse"),
                "test_mae": result.get("test_mae"),
                "silhouette_score": result.get("silhouette_score"),
                "davies_bouldin": result.get("davies_bouldin"),
                "inertia": result.get("inertia"),
                "reconstruction_loss": result.get("reconstruction_loss"),
                "iterations": result.get("iterations", []),
                "num_iterations": result.get("num_iterations", 0),
                "best_iteration": result.get("best_iteration"),
                "summary": result.get("summary"),
                "recommendations": result.get("recommendations"),
                "feature_redo_requested": feature_redo_requested,
                "feature_importances": result.get("feature_importances", {}),
            },
            "training_iteration": iteration_num,
            "feature_redo_requested": feature_redo_requested,
            "feature_redo_recommendation": result.get("feature_redo_recommendation"),
            "feature_redo_reason": result.get("feature_redo_reason"),
            "audit_trace": state.get("audit_trace", []) + [{
                "step": "training",
                "model_name": result.get("model_name"),
                "model_type": best_model_type,
                "success": result.get("success"),
                "num_iterations": result.get("num_iterations", 0),
                "feature_redo_requested": feature_redo_requested,
            }],
        })

        lines = [f"Training {'succeeded' if result.get('success') else 'FAILED'}"]
        lines.append(f"Model: {result.get('model_name', '?')}")
        metric_parts = []
        for key, label in [
            ("val_accuracy", "Val Accuracy"), ("val_roc_auc", "Val ROC-AUC"),
            ("test_accuracy", "Test Accuracy"), ("val_r2", "Val R²"), ("test_r2", "Test R²"),
            ("silhouette_score", "Silhouette"), ("davies_bouldin", "Davies-Bouldin"),
            ("inertia", "Inertia"),
        ]:
            v = result.get(key)
            if v is not None:
                lines.append(f"{label}: {v:.4f}")
                metric_parts.append(f"{label}: {v:.4f}")
        if result.get("summary"):
            lines.append(f"Summary: {result['summary']}")
        if feature_redo_requested:
            lines.append(f"\nFeature redo recommended: {result.get('feature_redo_reason')}")
            lines.append("Consider going back to feature_selection_specification.")

        status = "succeeded" if result.get("success") else "FAILED"
        summary = f"Training **{status}** — {', '.join(metric_parts)}" if metric_parts else f"Training {status}"
        decision = _hitl_gate("training", summary)
        if not decision.get("approved", True):
            fb = decision.get("feedback", "Please adjust training approach.")
            fb_lower = fb.lower()
            routed = False
            for m in KNOWN_FAMILIES:
                if m in fb_lower or m.replace("_", " ") in fb_lower:
                    state["_redo_feedback_select_model"] = fb
                    routed = True
                    break
            if not routed:
                state["_redo_feedback_training_approval"] = fb
            return f"REJECTED by user: {fb}"

        _completed_steps.add("training")
        return "\n".join(lines)

    def tool_generate_report() -> str:
        """Generate and save the final training report. Call when satisfied with training results."""
        nonlocal state
        if "generate_report" in _completed_steps:
            return f"SKIP: Report already generated: {state.get('report_path')}."
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
                "iterations": metrics.get("iterations", []),
                "best_iteration": metrics.get("best_iteration"),
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
        _completed_steps.add("generate_report")
        return f"Report saved to {report_path}"

    # -- build the deep agent -----------------------------------------------

    all_tools = [
        tool_data_collection,
        tool_select_model,
        tool_cleaning,
        tool_label_split_definition,
        tool_feature_selection_specification,
        tool_feature_engineering_executor,
        tool_training_approval,
        tool_training,
        tool_generate_report,
    ]

    if checkpointer is None and hitl:
        checkpointer = MemorySaver()

    llm = init_chat_model(model) if isinstance(model, str) else model

    kwargs: dict = {
        "model": llm,
        "tools": all_tools,
        "system_prompt": SYSTEM_PROMPT,
    }
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer

    agent = create_agent(**kwargs)
    return agent, state


def invoke_simple_training_agent(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None,
    model: str = "openai:gpt-5.1",
    use_external_sources: bool = False,
):
    """Convenience function: create and invoke the simple training agent (no HITL)."""
    agent, _state = create_simple_training_agent(
        goal, linked_datasets, user_model_preference, model, hitl=False,
        use_external_sources=use_external_sources,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": goal}]})
    return result
