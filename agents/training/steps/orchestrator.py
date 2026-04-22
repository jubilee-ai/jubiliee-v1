"""
Node function implementations for the ML Training Agent.
Each node represents a step in the training pipeline.
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from langchain.chat_models import init_chat_model
from langgraph.types import interrupt

from ..core.hitl import (_invalidate_split_downstream, make_serializable,
                         parse_decision, run_with_hitl)
from ..core.state import TrainingAgentState

# Import utilities for dataset registration
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))
from utils import get_registered_dataset, register_dataset

# Import node implementations from separate modules
from .cleaning_simple import run_cleaning_simple
from .data_collection import data_collection as _data_collection_impl
from .feature_engineering_executor import execute_feature_spec_split
from .feature_engineering_simple import run_feature_engineering_simple
from .feature_experiment_runner import run_experiment_grid
from .label_and_split import (apply_split, compute_split_indices,
                              normalize_label_definition_for_df,
                              run_label_split_definition)
from .select_model import select_model as _select_model_impl
from .training import run_training_agent as _run_training

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _add_feedback_to_goal(goal: str, step: str, feedback: Optional[str]) -> str:
    """Append user feedback to goal if provided."""
    return f"{goal}\n\nUser feedback on {step}: {feedback}" if feedback else goal


def _get_label_def(state: TrainingAgentState) -> dict:
    """Get label definition from state with empty dict fallback."""
    return state.get("label_definition") or {}


def _is_unsupervised(state: TrainingAgentState) -> bool:
    return state.get("selected_model") == "unsupervised"


def _infer_task_type(goal: str, selected_model: str) -> str:
    """Infer task type from goal and model selection — thin wrapper around the shared helper."""
    from ..core.task_inference import infer_task_type

    return infer_task_type(goal, selected_model=selected_model)


def _compact_experiment_feedback(state: TrainingAgentState) -> Optional[str]:
    """Compact scout results into a short recommendation string for feature redesign."""
    exp = state.get("experiment_result") or {}
    rankings = state.get("feature_rankings") or {}
    if not exp and not rankings:
        return None

    lines: list[str] = ["Scout experiment feedback:"]
    best_variant = exp.get("best_variant_name")
    best_metric = exp.get("best_metric")
    if best_variant:
        metric_text = f" ({best_metric:.4f})" if isinstance(best_metric, (int, float)) else ""
        lines.append(f"- Best variant: {best_variant}{metric_text}")
    signal = list(exp.get("signal_features") or [])[:8]
    if signal:
        lines.append(f"- Strong signals: {', '.join(signal)}")
    dropped = list(exp.get("dropped_features") or [])[:8]
    if dropped:
        lines.append(f"- Weak features to reconsider: {', '.join(dropped)}")
    top_ranked = list(rankings.items())[:8]
    if top_ranked:
        ranked_text = ", ".join(f"{name} ({score:.3f})" for name, score in top_ranked)
        lines.append(f"- Top ranked: {ranked_text}")
    return "\n".join(lines)


def _infer_target_column(goal: str, dataset_ref: str) -> Optional[str]:
    """Best-effort inference of the target column from the goal text and dataset columns.

    Uses three strategies in order:
    1. Exact match — a column name appears as a standalone word in the goal.
    2. Keyword mapping — goal contains a domain keyword that maps to common column
       name patterns (e.g. "churn" in goal → column named "churn" / "is_churn").
    3. Generic fallback — columns named "target", "label", "y", etc.

    Returns None if no plausible target can be identified.
    """
    df = get_registered_dataset(dataset_ref)
    if df is None:
        return None

    goal = goal or ""
    goal_lower = goal.lower()
    goal_words = set(goal_lower.split())
    columns_lower_map = {c.lower(): c for c in df.columns}

    # Strategy 1: column name appears verbatim in the goal
    for col_lower, col in columns_lower_map.items():
        if len(col_lower) >= 2 and col_lower in goal_words:
            return col

    # Strategy 2: domain keyword → column name pattern mapping
    _keyword_patterns: dict[str, list[str]] = {
        "churn": ["churn", "churned", "is_churn", "has_churned"],
        "default": ["default", "defaulted", "is_default", "loan_default"],
        "fraud": ["fraud", "is_fraud", "fraudulent"],
        "surviv": ["survived", "survival", "event"],
        "cancel": ["cancelled", "canceled", "cancellation"],
        "attrit": ["attrition", "attrition_flag", "attrited"],
        "spam": ["spam", "is_spam"],
        "click": ["clicked", "click"],
        "conver": ["converted", "conversion"],
        "diagnos": ["diagnosis", "diagnosed"],
        "price": ["price", "sale_price", "selling_price"],
        "salary": ["salary", "wage", "compensation"],
        "revenue": ["revenue", "total_revenue"],
        "satisf": ["satisfaction", "rating"],
    }
    for keyword, patterns in _keyword_patterns.items():
        if keyword in goal_lower:
            for pattern in patterns:
                if pattern in columns_lower_map:
                    return columns_lower_map[pattern]

    # Strategy 3: generic target column names
    for generic in ["target", "label", "y", "class", "outcome"]:
        if generic in columns_lower_map:
            return columns_lower_map[generic]

    return None


# =============================================================================
# NODE FUNCTIONS
# =============================================================================


def select_model(state: TrainingAgentState) -> TrainingAgentState:
    """Step 1: Select Model Family with HITL approval."""

    def do_work(s: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        if feedback:
            return _select_model_impl({**s, "goal": _add_feedback_to_goal(s.get("goal", ""), "model family selection", feedback)})
        return _select_model_impl(s)

    def get_summary(r: TrainingAgentState) -> str:
        return "Setup is ready. Approve to continue, or share feedback to adjust."

    return run_with_hitl("select_model", state, do_work, get_summary)


def data_collection(state: TrainingAgentState) -> TrainingAgentState:
    """Step 2: Data Collection with HITL approval."""

    def do_work(s: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        if feedback:
            return _data_collection_impl({**s, "goal": _add_feedback_to_goal(s.get("goal", ""), "data collection", feedback)})
        return _data_collection_impl(s)

    def get_summary(r: TrainingAgentState) -> str:
        from ..utils.streaming import build_node_update

        u = build_node_update("data_collection", dict(r))
        headline = (u.get("headline") or "").strip()
        if headline:
            return headline
        ref = r.get("collected_dataset_ref")
        if ref:
            short = str(ref).rsplit("/", 1)[-1] or str(ref)
            return f"Dataset **{short}** is ready. Approve to continue."
        return "Data collection finished. Approve to continue."

    return run_with_hitl("data_collection", state, do_work, get_summary)


def cleaning_node(state: TrainingAgentState) -> TrainingAgentState:
    """Step 3: Cleaning & Standardization with HITL approval."""

    def do_work(s: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        dataset_ref = s.get("collected_dataset_ref")
        if not dataset_ref:
            return {
                **s,
                "error": "No collected_dataset_ref in state — data_collection must complete first",
                "current_step": "data_collection",
            }

        try:
            df = get_registered_dataset(dataset_ref)
            num_columns = len(df.columns) if df is not None else 20
            max_iters = min(80, max(30, 30 + num_columns))
        except Exception:
            max_iters = 50

        # Infer target column and task type so cleaning is target-aware
        goal = s.get("goal", "")
        selected_model = s.get("selected_model", "")
        target_col = _infer_target_column(goal, dataset_ref)
        task_type = _infer_task_type(goal, selected_model) if target_col else None

        if target_col:
            print(f"[cleaning] Inferred target column: '{target_col}' (task: {task_type}) — will be protected during cleaning")
        else:
            print("[cleaning] Could not infer target column — cleaning will proceed without target awareness")

        result = run_cleaning_simple(
            dataset_ref=dataset_ref,
            goal=_add_feedback_to_goal(goal, "cleaning", feedback),
            max_iterations=max_iters,
            target_col=target_col,
            task_type=task_type,
            selected_model=selected_model,
        )

        return {
            **s,
            "cleaned_dataset_ref": result["cleaned_ref"],
            "cleaning_summary": result.get("cleaning_summary"),
            "cleaning_transformations": result.get("transformations", []),
            "current_step": "label_split_definition",
        }

    def get_summary(r: TrainingAgentState) -> str:
        return (
            f"Cleaned dataset: {r.get('cleaned_dataset_ref', 'unknown')}\n"
            f"Transformations applied: {len(r.get('cleaning_transformations', []))}\n\n"
            f"{r.get('cleaning_summary', '')}"
        )

    return run_with_hitl("cleaning", state, do_work, get_summary)


def label_split_definition(state: TrainingAgentState) -> TrainingAgentState:
    """Step 3.5: Label + Split Definition + Data Splitting with HITL approval."""

    def do_work(s: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        if _is_unsupervised(s):
            dataset_ref = s.get("cleaned_dataset_ref")
            if not dataset_ref:
                return {
                    **s,
                    "error": "Cannot run unsupervised split passthrough — cleaning must run first.",
                    "current_step": "cleaning",
                }
            df = get_registered_dataset(dataset_ref)
            if df is None:
                return {
                    **s,
                    "error": f"Cannot load cleaned dataset {dataset_ref} for unsupervised training.",
                    "current_step": "cleaning",
                }
            return {
                **s,
                "label_definition": {
                    "target_column": "",
                    "prediction_horizon": None,
                    "grain": "",
                    "as_of_cutoff": None,
                    "split_strategy": "none",
                    "forbidden_columns": [],
                },
                "split_indices": None,
                "train_dataset_ref": dataset_ref,
                "val_dataset_ref": None,
                "test_dataset_ref": None,
                "current_step": "feature_selection_specification",
            }

        dataset_ref = s.get("cleaned_dataset_ref")
        if not dataset_ref:
            return {
                **s,
                "error": "Cannot run label/split definition — cleaning must run first to produce a cleaned dataset.",
                "current_step": "cleaning",
            }

        existing = _get_label_def(s)
        goal = _add_feedback_to_goal(s.get("goal", ""), "label/split definition", feedback)

        # Get label definition from LLM
        label_def = run_label_split_definition(
            dataset_ref=dataset_ref,
            goal=goal,
            selected_model=s.get("selected_model"),
            model_explanation=s.get("model_explanation"),
            target_column=existing.get("target_column"),
            prediction_horizon=existing.get("prediction_horizon"),
            grain=existing.get("grain"),
            as_of_cutoff=existing.get("as_of_cutoff"),
            split_strategy=existing.get("split_strategy"),
            forbidden_columns=existing.get("forbidden_columns"),
        )

        # Compute and apply split
        df = get_registered_dataset(dataset_ref)
        if df is None:
            return {
                **s,
                "error": f"Could not load dataset {dataset_ref} for label/split.",
                "current_step": "cleaning",
            }

        label_def = normalize_label_definition_for_df(df, label_def)

        task_type = s.get("task_type")
        tgt_col = label_def.get("target_column")
        if task_type == "classification" and tgt_col and tgt_col in df.columns:
            col = df[tgt_col]
            if pd.api.types.is_numeric_dtype(col):
                nunique = int(col.nunique(dropna=True))
                n = len(col)
                if n > 0 and nunique > min(50, max(10, n // 5)):
                    print(
                        f"[label_split_definition] Target `{tgt_col}` has {nunique} distinct "
                        f"numeric values on {n} rows — auto-correcting task_type to regression."
                    )
                    task_type = "regression"
                    s = {**s, "task_type": "regression"}

        print(f"[label_split_definition] Computing {label_def.get('split_strategy', 'random')} split...")
        
        split_indices = compute_split_indices(df=df, label_definition=label_def, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        train_df, val_df, test_df = apply_split(df, split_indices)
        print(f"[label_split_definition] Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # Register split datasets
        train_ref, val_ref, test_ref = f"{dataset_ref}_train", f"{dataset_ref}_val", f"{dataset_ref}_test"
        register_dataset(train_ref, train_df)
        register_dataset(val_ref, val_df)
        register_dataset(test_ref, test_df)
        print(f"[label_split_definition] Registered: {train_ref}, {val_ref}, {test_ref}")

        return {
            **s,
            "label_definition": label_def,
            "split_indices": split_indices,
            "train_dataset_ref": train_ref,
            "val_dataset_ref": val_ref,
            "test_dataset_ref": test_ref,
            "current_step": "feature_selection_specification",
        }

    def get_summary(r: TrainingAgentState) -> str:
        if _is_unsupervised(r):
            return (
                "Unsupervised flow: no target column or train/val/test split.\n"
                f"Full cleaned dataset for training: {r.get('train_dataset_ref')}\n"
                "Validation: N/A\n"
                "Test: N/A"
            )
        ld = _get_label_def(r)
        return (
            f"Target column: {ld.get('target_column', 'unknown')}\n"
            f"Split strategy: {ld.get('split_strategy', 'unknown')}\n"
            f"Grain: {ld.get('grain', 'unknown')}\n"
            f"Forbidden columns: {ld.get('forbidden_columns', [])}\n"
            f"Train: {r.get('train_dataset_ref')}\n"
            f"Val: {r.get('val_dataset_ref')}\n"
            f"Test: {r.get('test_dataset_ref')}"
        )

    return run_with_hitl("label_split_definition", state, do_work, get_summary)


def feature_selection_specification(state: TrainingAgentState) -> TrainingAgentState:
    """Step 4: Feature Selection & Specification with HITL approval."""

    def do_work(s: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        if _is_unsupervised(s):
            train_ref = s.get("train_dataset_ref")
            if not train_ref:
                raise ValueError("No train_dataset_ref in state - step 3.5 must complete first")
            return {
                **s,
                "feature_spec": {"features": []},
                "analysis_trace": [{
                    "step": "feature_selection_specification",
                    "analysis_results": {},
                    "key_stats": {},
                    "validation": {"valid": True},
                    "is_redo": False,
                    "redo_recommendation": None,
                }],
                "feature_redo_requested": False,
                "feature_redo_recommendation": None,
                "feature_redo_reason": None,
            }

        train_ref, val_ref, test_ref = s.get("train_dataset_ref"), s.get("val_dataset_ref"), s.get("test_dataset_ref")
        label_def = _get_label_def(s)
        goal = _add_feedback_to_goal(s.get("goal", ""), "feature selection", feedback)
        
        target_column = label_def.get("target_column", "")
        if not train_ref:
            raise ValueError("No train_dataset_ref in state - step 3.5 must complete first")
        if not target_column:
            raise ValueError("No target_column in label_definition - step 3.5 must complete first")

        # Handle feature redo from training
        feature_redo_requested = s.get("feature_redo_requested", False)
        feature_redo_recommendation = s.get("feature_redo_recommendation")
        feature_redo_iteration = s.get("feature_redo_iteration", 0)

        recommendation = feature_redo_recommendation if feature_redo_requested else None
        if feedback:
            recommendation = f"{recommendation}\n{feedback}" if recommendation else feedback
        experiment_feedback = _compact_experiment_feedback(s)
        if experiment_feedback:
            recommendation = f"{recommendation}\n{experiment_feedback}" if recommendation else experiment_feedback

        if feature_redo_requested:
            print(f"[feature_selection_specification] REDO iteration {feature_redo_iteration + 1}")
            print(f"[feature_selection_specification] Recommendation from training: {feature_redo_recommendation}")

        print(f"[feature_selection_specification] Running analysis on TRAINING data only ({train_ref})...")

        result = run_feature_engineering_simple(
            train_ref=train_ref, goal=goal, target_column=target_column,
            grain=label_def.get("grain", ""), recomendation=recommendation,
            val_ref=val_ref, test_ref=test_ref,
            task_type=_infer_task_type(goal, s.get("selected_model", "")),
            forbidden_columns=label_def.get("forbidden_columns", []),
            as_of_cutoff=label_def.get("as_of_cutoff"),
            prediction_horizon=label_def.get("prediction_horizon"),
            selected_model=s.get("selected_model"),
        )

        feature_spec, validation = result.get("feature_spec"), result.get("validation", {})
        key_stats = result.get("key_stats", {})

        # Log key statistics
        if key_stats:
            overview = key_stats.get("dataset_overview", {})
            print(f"[feature_selection_specification] Key statistics extracted:")
            print(f"  - Dataset: {overview.get('rows', '?')} rows, {overview.get('columns', '?')} columns")
            if key_stats.get("feature_correlations"):
                top = key_stats["feature_correlations"][0]
                print(f"  - Top correlation: {top.get('feature')} (r={top.get('correlation')})")
            if key_stats.get("leakage_warnings"):
                print(f"  - ⚠️ Leakage warnings: {len(key_stats['leakage_warnings'])} features")
            if key_stats.get("high_correlation_pairs"):
                print(f"  - High correlation pairs: {len(key_stats['high_correlation_pairs'])}")

        # Auto-remove invalid features
        if validation and not validation.get("valid", True):
            print(f"[WARNING] Feature spec validation issues: {validation.get('errors', [])}")
            features_valid = validation.get("features_valid", {})
            if feature_spec and "features" in feature_spec:
                original_count = len(feature_spec["features"])
                feature_spec["features"] = [
                    f for f in feature_spec["features"]
                    if features_valid.get(f.get("name"), {}).get("valid", True)
                ]
                removed = original_count - len(feature_spec["features"])
                if removed > 0:
                    print(f"[INFO] Auto-removed {removed} invalid feature(s)")

        return {
            **s,
            "feature_spec": feature_spec,
            "analysis_trace": [{
                "step": "feature_selection_specification",
                "analysis_results": result.get("analysis_results", {}),
                "key_stats": key_stats,
                "validation": validation,
                "is_redo": feature_redo_requested,
                "redo_recommendation": feature_redo_recommendation,
            }],
            "feature_redo_requested": False,
            "feature_redo_recommendation": None,
            "feature_redo_reason": None,
            "feature_redo_iteration": feature_redo_iteration + 1 if feature_redo_requested else feature_redo_iteration,
        }

    def get_summary(r: TrainingAgentState) -> str:
        if _is_unsupervised(r):
            return "Unsupervised flow: skipped feature specification and will train on cleaned features directly."
        features = (r.get("feature_spec") or {}).get("features", [])
        names = [f.get("name", "?") for f in features[:10]]
        more = f" (+{len(features) - 10} more)" if len(features) > 10 else ""
        return f"Features selected: {len(features)}\nFeature names: {', '.join(names)}{more}"

    return run_with_hitl("feature_selection_specification", state, do_work, get_summary)


def feature_engineering_executor(state: TrainingAgentState) -> TrainingAgentState:
    """Step 5: Feature Engineering Executor with HITL approval."""

    def do_work(s: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        if _is_unsupervised(s):
            train_ref = s.get("train_dataset_ref")
            if not train_ref:
                raise ValueError("No train_dataset_ref in state - step 3.5 must complete first")
            return {
                **s,
                "transformed_train_ref": train_ref,
                "transformed_val_ref": None,
                "transformed_test_ref": None,
                "transformed_dataset_ref": train_ref,
                "feature_validation_passed": True,
                "feature_pipeline_mode": "passthrough",
                "feature_redo_recommendation": None,
                "audit_trace": s.get("audit_trace", []) + [{
                    "step": "feature_engineering_executor",
                    "features_created": [],
                    "errors": [],
                    "shapes": {"train": None, "val": None, "test": None},
                    "temporal_constraints_applied": 0,
                    "mode": "unsupervised_passthrough",
                }],
            }

        train_ref, val_ref, test_ref = s.get("train_dataset_ref"), s.get("val_dataset_ref"), s.get("test_dataset_ref")
        feature_spec, label_def = s.get("feature_spec"), _get_label_def(s)
        target_column = label_def.get("target_column", "")

        if not train_ref:
            raise ValueError("No train_dataset_ref in state - step 3.5 must complete first")
        if not feature_spec:
            raise ValueError("No feature_spec in state - step 4 must complete first")
        if not target_column:
            raise ValueError("No target_column in label_definition")

        if feedback:
            print(f"[feature_engineering_executor] Feedback received, will be passed to feature selection: {feedback}")

        print("[feature_engineering_executor] Executing feature spec (fit on train, transform all)...")

        result = execute_feature_spec_split(
            train_ref=train_ref, val_ref=val_ref, test_ref=test_ref,
            feature_spec=feature_spec, target_column=target_column,
            grain=label_def.get("grain", ""), as_of_cutoff=label_def.get("as_of_cutoff"),
        )

        errors, features_created = result.get("errors", []), result.get("features_created", [])
        if errors:
            print(f"[WARNING] Feature engineering had {len(errors)} errors:")
            for err in errors:
                print(f"  - {err}")

        print(f"[feature_engineering_executor] Created {len(features_created)} features")
        print(f"[feature_engineering_executor] Shapes: {result.get('shapes')}")

        return {
            **s,
            "transformed_train_ref": result.get("train_ref"),
            "transformed_val_ref": result.get("val_ref"),
            "transformed_test_ref": result.get("test_ref"),
            "transformed_dataset_ref": result.get("train_ref"),
            "feature_validation_passed": len(features_created) > 0 and len(errors) < len(features_created),
            "feature_pipeline_mode": "engineered",
            "feature_redo_recommendation": feedback if feedback else s.get("feature_redo_recommendation"),
            "audit_trace": s.get("audit_trace", []) + [{
                "step": "feature_engineering_executor",
                "features_created": features_created,
                "errors": errors,
                "shapes": result.get("shapes"),
                "temporal_constraints_applied": result.get("temporal_constraints_applied", 0),
            }],
        }

    def _fmt_shape(s):
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            return f"{s[0]} × {s[1]}"
        return str(s) if s else "?"

    def get_summary(r: TrainingAgentState) -> str:
        if _is_unsupervised(r):
            return "Unsupervised flow: feature engineering executor passthrough complete."
        audit = next((t for t in r.get("audit_trace", []) if t.get("step") == "feature_engineering_executor"), {})
        shapes = audit.get("shapes", {})
        return (
            f"Features created: {len(audit.get('features_created', []))}\n"
            f"Train shape: {_fmt_shape(shapes.get('train'))}\n"
            f"Val shape: {_fmt_shape(shapes.get('val'))}\n"
            f"Test shape: {_fmt_shape(shapes.get('test'))}\n"
            f"Errors: {len(audit.get('errors', []))}"
        )

    return run_with_hitl("feature_engineering_executor", state, do_work, get_summary)


def feature_experiment_runner(state: TrainingAgentState) -> TrainingAgentState:
    """Step 5.5: Parallel feature-set experimentation with scout models."""

    def do_work(s: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        if _is_unsupervised(s):
            print("[feature_experiment_runner] Unsupervised — skipping experiment grid")
            return {
                **s,
                "experiment_result": None,
                "feature_rankings": None,
                "experiment_grid_summary": None,
            }

        train_ref = s.get("transformed_train_ref") or s.get("train_dataset_ref")
        val_ref = s.get("transformed_val_ref") or s.get("val_dataset_ref")
        test_ref = s.get("transformed_test_ref") or s.get("test_dataset_ref")
        label_def = _get_label_def(s)
        target_column = label_def.get("target_column", "")
        feature_spec = s.get("feature_spec")
        goal = s.get("goal", "")
        selected_model = s.get("selected_model", "supervised")

        if not train_ref or not val_ref:
            missing = []
            if not train_ref:
                missing.append("train")
            if not val_ref:
                missing.append("validation")
            raise ValueError(
                "feature_experiment_runner requires registered train and validation dataset refs "
                f"(missing: {', '.join(missing)}). "
                "Re-run label_split_definition after cleaning, or ensure the prior steps "
                "completed successfully (check state.error)."
            )

        if not feature_spec or not feature_spec.get("features"):
            print("[feature_experiment_runner] Empty feature spec — skipping")
            return {**s, "experiment_result": None, "feature_rankings": None, "experiment_grid_summary": None}

        train_df = get_registered_dataset(train_ref)
        if train_df is None or len(train_df) < 500 or len(feature_spec.get("features", [])) < 5:
            n_rows = len(train_df) if train_df is not None else 0
            n_feats = len(feature_spec.get("features", []))
            print(f"[feature_experiment_runner] Dataset too small or too few features "
                  f"({n_rows} rows, {n_feats} features) — skipping experiment grid")
            return {**s, "experiment_result": None, "feature_rankings": None, "experiment_grid_summary": None}

        task_type = s.get("task_type") or _infer_task_type(goal, selected_model)

        # Use the raw (pre-feature-engineering) refs so each variant
        # can apply its own feature spec from scratch
        raw_train_ref = s.get("train_dataset_ref")
        raw_val_ref = s.get("val_dataset_ref")
        raw_test_ref = s.get("test_dataset_ref")

        print(f"[feature_experiment_runner] Running parallel experiment grid...")
        exp_result = run_experiment_grid(
            feature_spec=feature_spec,
            train_ref=raw_train_ref or train_ref,
            val_ref=raw_val_ref or val_ref,
            test_ref=raw_test_ref or test_ref,
            target_column=target_column,
            task_type=task_type,
            grain=label_def.get("grain", ""),
            as_of_cutoff=label_def.get("as_of_cutoff"),
        )

        updated = {
            **s,
            "experiment_result": {
                "best_variant_name": exp_result.best_variant_name,
                "best_metric": exp_result.best_metric,
                "total_variants": exp_result.total_variants,
                "total_scouts": exp_result.total_scouts,
                "wall_time_seconds": exp_result.wall_time_seconds,
                "signal_features": exp_result.signal_features,
                "dropped_features": exp_result.dropped_features,
            },
            "feature_rankings": exp_result.feature_rankings,
            # Keep only a bounded summary in state to limit memory and JSONB size
            "experiment_grid_summary": (exp_result.experiment_grid or [])[:12],
        }

        if exp_result.best_variant_name != "full":
            updated["transformed_train_ref"] = exp_result.transformed_train_ref
            updated["transformed_val_ref"] = exp_result.transformed_val_ref
            updated["transformed_test_ref"] = exp_result.transformed_test_ref
            updated["feature_spec"] = exp_result.best_feature_spec
            print(f"[feature_experiment_runner] Switched to variant "
                  f"'{exp_result.best_variant_name}' (better than full baseline)")
        else:
            print(f"[feature_experiment_runner] Full baseline was best — keeping original features")

        updated["audit_trace"] = s.get("audit_trace", []) + [{
            "step": "feature_experiment_runner",
            "best_variant": exp_result.best_variant_name,
            "best_metric": exp_result.best_metric,
            "total_variants": exp_result.total_variants,
            "total_scouts": exp_result.total_scouts,
            "wall_time": exp_result.wall_time_seconds,
            "signal_features": exp_result.signal_features[:10],
            "dropped_features": exp_result.dropped_features[:10],
        }]

        return updated

    def get_summary(r: TrainingAgentState) -> str:
        exp = r.get("experiment_result")
        if not exp:
            return "Feature experiment runner: skipped (unsupervised, too few features, or small dataset)"
        grid = r.get("experiment_grid_summary") or []
        lines = [
            f"Tested {exp.get('total_variants', 0)} feature-set variants "
            f"x 2 model families = {exp.get('total_scouts', 0)} scouts "
            f"in {exp.get('wall_time_seconds', 0):.1f}s",
            f"Best variant: {exp.get('best_variant_name')} "
            f"(metric={exp.get('best_metric', 0):.4f})",
        ]
        signal = exp.get("signal_features", [])
        dropped = exp.get("dropped_features", [])
        if signal:
            lines.append(f"Signal features ({len(signal)}): {', '.join(signal[:8])}")
        if dropped:
            lines.append(f"Low-signal features ({len(dropped)}): {', '.join(dropped[:8])}")
        if grid:
            lines.append("\nScout grid results:")
            for entry in grid[:6]:
                m = entry.get("metrics", {})
                metric_str = ", ".join(f"{k}={v:.4f}" for k, v in m.items()) if m else "N/A"
                lines.append(f"  {entry.get('variant')} + {entry.get('model_family')}: {metric_str}")
            if len(grid) > 6:
                lines.append(f"  ... and {len(grid) - 6} more")
        return "\n".join(lines)

    return run_with_hitl("feature_experiment_runner", state, do_work, get_summary)


def training_approval(state: TrainingAgentState) -> TrainingAgentState:
    """Step 6.5: Training Approval - Propose training configuration for user approval."""
    feedback = None

    while True:
        train_ref = state.get("transformed_train_ref") or state.get("train_dataset_ref") or state.get("cleaned_dataset_ref") or state.get("collected_dataset_ref")
        val_ref = state.get("transformed_val_ref") or state.get("val_dataset_ref")
        label_def, feature_spec = _get_label_def(state), state.get("feature_spec") or {}
        target_column = label_def.get("target_column", "")
        selected_model = state.get("selected_model", "supervised")
        goal = state.get("goal", "")
        unsupervised = selected_model == "unsupervised"

        train_df = get_registered_dataset(train_ref) if train_ref else None
        val_df = get_registered_dataset(val_ref) if val_ref else None
        if train_df is None:
            return {
                **state,
                "error": f"Training dataset not available (ref={train_ref}). Earlier pipeline steps may have failed.",
                "current_step": "training_approval",
            }

        if not unsupervised and target_column and target_column not in train_df.columns:
            return {
                **state,
                "error": (
                    f"Target column '{target_column}' is missing from training dataset "
                    f"(ref={train_ref}). Label definition may reference a non-existent "
                    "column, or feature engineering may have dropped it."
                ),
                "current_step": "training_approval",
            }

        n_rows, n_features = len(train_df), len([c for c in train_df.columns if c != target_column]) if target_column else len(train_df.columns)
        resolved_task_type = state.get("task_type")
        if not isinstance(resolved_task_type, str) or not resolved_task_type:
            resolved_task_type = "unsupervised" if unsupervised else _infer_task_type(
                goal, selected_model
            )
        task_type = resolved_task_type
        class_counts = (
            train_df[target_column].value_counts().to_dict()
            if task_type == "classification" and target_column
            else {}
        )
        total = sum(class_counts.values()) if class_counts else 0
        minority_ratio = (min(class_counts.values()) / total) if total > 0 else 0
        is_imbalanced = (minority_ratio < 0.3) if class_counts else False

        features_list = [f.get("name") for f in feature_spec.get("features", [])][:20]
        imbalance_msg = (
            f"⚠️ IMBALANCED DATA - minority class is {minority_ratio:.1%}"
            if is_imbalanced else ("✓ Balanced classes" if class_counts else "N/A for unsupervised")
        )
        feedback_section = f"## User Feedback to Incorporate:\n{feedback}" if feedback else ""

        prompt = f"""You are an ML expert. Propose a training configuration for the following task.

## Task
Goal: {goal}
Task Type: {task_type}
Selected Model: {selected_model}
Target Column: {target_column if target_column else "N/A (unsupervised)"}

## Data Characteristics
- Training rows: {n_rows}
- Features: {n_features}
- Feature names (first 20): {features_list}
- Validation rows: {len(val_df) if val_df is not None else 'N/A'}

## Class Distribution (Training)
{json.dumps(class_counts, indent=2) if class_counts else "N/A (unsupervised)"}
{imbalance_msg}

{feedback_section}

## Your Task
Propose a training configuration. Respond with a JSON object containing:

```json
{{
    "model_type": "{selected_model}",
    "task_type": "{task_type}",
    "hyperparameters": {{}},
    "class_weight": "balanced" or null,
    "max_iterations": 4,
    "strategy_notes": "2–4 sentences for a stakeholder: what this training setup is meant to achieve and why it fits the problem — plain language, no formulas, no hyperparameter dumps or library-specific kwargs (those belong in hyperparameters only)",
    "expected_metrics": "What metrics to optimize and expected performance range"
}}
```

Be specific with **hyperparameter** values in the `hyperparameters` object (that is the right place for technical detail).

For **`strategy_notes`** only: explain in **high-level terms** what this configuration is for (e.g. handling rare events, controlling overfitting on tabular data) and why it matches the goal — as if briefing someone who will use the model but does not tune it. Do **not** paste numeric settings, `scale_pos_weight` arithmetic, or long lists of kwargs into `strategy_notes`.

Also consider when choosing values:
- Dataset size ({n_rows} rows) - larger datasets can support more complex models
- Number of features ({n_features}) - may need regularization if many features
- Class imbalance - use class_weight="balanced" if imbalanced (skip for unsupervised)
- Model type - choose appropriate hyperparameters for {selected_model}
"""

        response = init_chat_model("openai:gpt-5.1").invoke([{"role": "user", "content": prompt}])

        # Parse JSON from response
        try:
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response.content)
            training_plan = json.loads(json_match.group(1)) if json_match else json.loads(response.content)
        except json.JSONDecodeError:
            training_plan = {
                "model_type": selected_model, "task_type": task_type, "hyperparameters": {},
                "class_weight": "balanced" if is_imbalanced else None, "max_iterations": 4,
                "strategy_notes": "Default configuration - LLM response could not be parsed",
                "expected_metrics": "Standard metrics for the task type",
            }
        if unsupervised:
            training_plan["class_weight"] = None

        # Ensure required fields and add data summary
        training_plan.setdefault("model_type", selected_model)
        training_plan.setdefault("task_type", task_type)
        training_plan.setdefault("max_iterations", 4)
        training_plan["data_summary"] = {
            "train_rows": n_rows, "val_rows": len(val_df) if val_df is not None else None,
            "n_features": n_features, "class_distribution": class_counts, "is_imbalanced": is_imbalanced,
        }

        print(f"[training_approval] Proposed training plan:")
        print(f"  Model: {training_plan.get('model_type')}")
        print(f"  Hyperparameters: {training_plan.get('hyperparameters')}")
        print(f"  Strategy: {training_plan.get('strategy_notes')}")

        hyperparams = training_plan.get("hyperparameters", {})
        hyperparams_str = "\n".join([f"  - {k}: {v}" for k, v in hyperparams.items()]) if hyperparams else "  (default values)"

        summary = f"""## Proposed Training Configuration

**Model:** {training_plan.get('model_type')}
**Task Type:** {training_plan.get('task_type')}
**Max Iterations:** {training_plan.get('max_iterations')}

**Hyperparameters:**
{hyperparams_str}

**Class Weight:** {training_plan.get('class_weight', 'None')}

**Strategy:**
{training_plan.get('strategy_notes', 'No notes provided')}

**Expected Metrics:**
{training_plan.get('expected_metrics', 'Standard metrics for the task')}

**Data Summary:**
- Training: {n_rows} rows, {n_features} features
- Validation: {len(val_df) if val_df is not None else 'N/A'} rows
- Class balance: {'Imbalanced' if is_imbalanced else ('Balanced' if class_counts else 'N/A for unsupervised')}
"""

        training_plan = make_serializable(training_plan)
        if state.get("hitl_auto_approve"):
            return {**state, "training_plan": training_plan, "training_plan_approved": True, "current_step": "training"}

        decision = interrupt({
            "node": "training_approval", "summary": summary,
            "message": "Review the proposed training configuration. Approve to start training, or provide feedback to adjust the plan.",
            "state_snapshot": {"training_plan": training_plan, "selected_model": selected_model, "target_column": target_column},
        })

        approved, new_feedback = parse_decision(decision)
        if approved:
            return {**state, "training_plan": training_plan, "training_plan_approved": True, "current_step": "training"}
        
        feedback = new_feedback or "Please adjust the training configuration."
        print(f"[training_approval] Plan rejected. Feedback: {feedback}")


def training(state: TrainingAgentState) -> TrainingAgentState:
    """Step 7: Training with HITL approval."""

    def do_work(s: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        train_ref = s.get("transformed_train_ref") or s.get("train_dataset_ref") or s.get("cleaned_dataset_ref") or s.get("collected_dataset_ref")
        val_ref = s.get("transformed_val_ref") or s.get("val_dataset_ref")
        test_ref = s.get("transformed_test_ref") or s.get("test_dataset_ref")
        label_def = _get_label_def(s)
        target_column = label_def.get("target_column", "")
        selected_model = s.get("selected_model", "supervised")
        goal = _add_feedback_to_goal(s.get("goal", ""), "training", feedback)
        unsupervised = selected_model == "unsupervised"

        if not train_ref:
            raise ValueError("No training dataset in state — data collection or feature engineering must complete first")
        task_type = s.get("task_type") or _infer_task_type(goal, selected_model)
        if not target_column and task_type != "unsupervised":
            raise ValueError("No target_column in label_definition")

        model_name = f"{selected_model}_{int(time.time())}"
        training_plan = s.get("training_plan") or {}
        plan_max_iters = training_plan.get("max_iterations", 4)
        print(f"[training] Starting training with {selected_model}...")
        print(f"  Train: {train_ref}\n  Val: {val_ref}\n  Test: {test_ref}\n  Target: {target_column}")

        plan_for_agent = training_plan if training_plan else None
        explicit_tt = s.get("task_type")
        if not explicit_tt and isinstance(training_plan, dict):
            explicit_tt = training_plan.get("task_type")
        result = _run_training(
            train_ref=train_ref, val_ref=val_ref, test_ref=test_ref,
            target_column=target_column, selected_model=selected_model,
            goal=goal, model_name=model_name, max_iterations=plan_max_iters,
            experiment_result=s.get("experiment_result"),
            feature_rankings=s.get("feature_rankings"),
            training_plan=plan_for_agent,
            prior_training_metrics=s.get("training_metrics"),
            explicit_task_type=explicit_tt if isinstance(explicit_tt, str) else None,
        )

        if result.get("success"):
            print(f"[training] Model trained successfully: {result.get('model_name')}")
        else:
            print(f"[training] Training failed: {result.get('error')}")

        feature_redo_requested = result.get("feature_redo_requested", False)
        if feature_redo_requested:
            print(f"[training] Feature engineering redo requested!")
            print(f"  Reason: {result.get('feature_redo_reason')}")
            print(f"  Recommendation: {result.get('feature_redo_recommendation')}")

        best_model_type = result.get("model_type", selected_model)

        return {
            **s,
            "model_weights_path": result.get("model_name"),
            "selected_model": best_model_type,
            "error": result.get("error"),
            "training_metrics": {
                "success": result.get("success"), "model_name": result.get("model_name"), "model_type": best_model_type,
                "val_accuracy": result.get("val_accuracy"), "val_roc_auc": result.get("val_roc_auc"),
                "test_accuracy": result.get("test_accuracy"), "test_roc_auc": result.get("test_roc_auc"),
                "train_r2": result.get("train_r2"), "val_r2": result.get("val_r2"), "val_rmse": result.get("val_rmse"), "val_mae": result.get("val_mae"),
                "test_r2": result.get("test_r2"), "test_rmse": result.get("test_rmse"), "test_mae": result.get("test_mae"),
                "silhouette_score": result.get("silhouette_score"), "davies_bouldin": result.get("davies_bouldin"),
                "inertia": result.get("inertia"), "reconstruction_loss": result.get("reconstruction_loss"),
                "iterations": result.get("iterations", []), "num_iterations": result.get("num_iterations", 0),
                "best_iteration": result.get("best_iteration"), "summary": result.get("summary"),
                "feature_redo_requested": feature_redo_requested,
            },
            "training_iteration": s.get("training_iteration", 0) + 1,
            "feature_redo_requested": feature_redo_requested,
            "feature_redo_recommendation": result.get("feature_redo_recommendation"),
            "feature_redo_reason": result.get("feature_redo_reason"),
            "audit_trace": s.get("audit_trace", []) + [{
                "step": "training", "model_name": result.get("model_name"), "model_type": best_model_type, "success": result.get("success"),
                "num_iterations": result.get("num_iterations", 0), "val_accuracy": result.get("val_accuracy"),
                "val_roc_auc": result.get("val_roc_auc"), "test_accuracy": result.get("test_accuracy"),
                "test_roc_auc": result.get("test_roc_auc"), "error": result.get("error"),
                "feature_redo_requested": feature_redo_requested, "feature_redo_recommendation": result.get("feature_redo_recommendation"),
            }],
        }

    def get_summary(r: TrainingAgentState) -> str:
        metrics = r.get("training_metrics") or {}
        lines = [f"Training {'succeeded' if metrics.get('success') else 'failed'}", f"Model: {metrics.get('model_name', 'unknown')}"]
        for key, label in [
            ("val_accuracy", "Val Accuracy"),
            ("val_roc_auc", "Val ROC-AUC"),
            ("test_accuracy", "Test Accuracy"),
            ("val_r2", "Val R²"),
            ("test_r2", "Test R²"),
            ("silhouette_score", "Silhouette"),
            ("davies_bouldin", "Davies-Bouldin"),
            ("inertia", "Inertia"),
            ("reconstruction_loss", "Reconstruction loss"),
        ]:
            if metrics.get(key) is not None:
                lines.append(f"{label}: {metrics.get(key):.4f}")
        if metrics.get("summary"):
            lines.append(f"\nSummary: {metrics.get('summary')}")
        return "\n".join(lines)

    return run_with_hitl("training", state, do_work, get_summary)


def generate_report(state: TrainingAgentState) -> TrainingAgentState:
    """Step 8: Generate Report with HITL approval."""

    def do_work(s: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        raw_metrics = s.get("training_metrics")
        training_metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        label_def = _get_label_def(s)

        if not training_metrics.get("success"):
            return {
                **s,
                "error": s.get("error")
                or "Cannot generate report because training did not succeed.",
                "current_step": "generate_report",
            }
        if not training_metrics.get("model_name"):
            return {
                **s,
                "error": "Cannot generate report because no trained model name is available.",
                "current_step": "generate_report",
            }

        report = {
            "generated_at": datetime.now().isoformat(),
            "goal": s.get("goal"),
            "model": {"type": s.get("selected_model"), "name": training_metrics.get("model_name"), "explanation": s.get("model_explanation")},
            "data": {
                "collected_dataset": s.get("collected_dataset_ref"), "cleaned_dataset": s.get("cleaned_dataset_ref"),
                "train_dataset": s.get("transformed_train_ref"), "val_dataset": s.get("transformed_val_ref"), "test_dataset": s.get("transformed_test_ref"),
            },
            "label_definition": {"target_column": label_def.get("target_column"), "split_strategy": label_def.get("split_strategy"), "grain": label_def.get("grain")},
            "training_results": {
                "success": training_metrics.get("success"), "num_iterations": training_metrics.get("num_iterations", 0),
                "validation_metrics": {"accuracy": training_metrics.get("val_accuracy"), "roc_auc": training_metrics.get("val_roc_auc")},
                "test_metrics": {"accuracy": training_metrics.get("test_accuracy"), "roc_auc": training_metrics.get("test_roc_auc")},
                "iterations": training_metrics.get("iterations", []), "best_iteration": training_metrics.get("best_iteration"),
                "summary": training_metrics.get("summary"),
            },
            "audit_trace": s.get("audit_trace", []),
            "user_feedback": feedback,
        }

        report_dir = Path(__file__).parent.parent.parent.parent / "trained_models"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"{training_metrics.get('model_name', 'unknown')}_report.json"
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[generate_report] Report saved to: {report_path}")

        report_storage_key: Optional[str] = None
        try:
            from backend.shared.artifact_store import get_artifact_store

            store = get_artifact_store()
            safe_name = re.sub(r"[^\w\-.]", "_", str(training_metrics.get("model_name", "unknown")))
            storage_key = f"reports/{safe_name}_report.json"
            report_storage_key = store.upload(report_path, storage_key)
            print(f"[generate_report] Report uploaded to object storage: {report_storage_key}")
        except Exception as ex:
            print(f"[generate_report] Object storage upload skipped: {ex}")

        return {
            **s,
            "report_path": str(report_path),
            "report_storage_key": report_storage_key,
            "audit_trace": s.get("audit_trace", []) + [{"step": "generate_report", "path": str(report_path), "storage_key": report_storage_key}],
        }

    def get_summary(r: TrainingAgentState) -> str:
        return f"Report generated successfully.\nPath: {r.get('report_path')}\nModel: {r.get('model_weights_path')}"

    return run_with_hitl("generate_report", state, do_work, get_summary)
