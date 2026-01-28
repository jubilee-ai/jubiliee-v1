"""
ML Model Training Agent - LangGraph Structure
Based on the architecture defined in README.md
"""

# Import utilities for dataset registration
import sys
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .cleaning_simple import run_cleaning_simple
from .data_collection import data_collection
from .feature_engineering_executor import (execute_feature_spec,
                                           execute_feature_spec_split)
from .feature_engineering_simple import run_feature_engineering_simple
from .label_and_split import (apply_split, compute_split_indices,
                              run_label_split_definition)
# Import node implementations
from .select_model import select_model

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))
from utils import get_registered_dataset, register_dataset

# =============================================================================
# STATE DEFINITIONS
# =============================================================================


class LabelDefinition(TypedDict):
    """Label and split configuration from step 3.5"""
    target_column: str
    prediction_horizon: Optional[str]
    grain: str
    as_of_cutoff: Optional[str]
    split_strategy: Literal["random", "time_based", "entity_based"]
    forbidden_columns: list[str]


class FeatureSpec(TypedDict):
    """Feature specification output from step 4"""
    features: list[dict[str, Any]]  # Each feature has: name, formula, source_tables, window, grain, as_of_constraint, encoding


class TrainingAgentState(TypedDict):
    """Main state object passed through the graph"""
    # Inputs
    goal: str
    linked_datasets: Optional[list[str]]
    user_model_preference: Optional[str]
    
    # Step 1: Model Selection
    selected_model: Optional[str]
    model_explanation: Optional[str]
    model_regen_count: int
    
    # Step 2: Data Collection
    collected_dataset_ref: Optional[str]
    
    # Step 3: Cleaning & Standardization
    cleaned_dataset_ref: Optional[str]
    cleaning_transformations: list[dict[str, Any]]
    cleaning_summary: Optional[str]  # Summary message from cleaning agent
    
    # Step 3.5: Label & Split Definition
    label_definition: Optional[LabelDefinition]
    split_indices: Optional[dict[str, Any]]  # Output of compute_split_indices
    train_dataset_ref: Optional[str]  # Registered train dataset
    val_dataset_ref: Optional[str]    # Registered val dataset
    test_dataset_ref: Optional[str]   # Registered test dataset
    
    # Step 4: Feature Selection & Specification
    feature_spec: Optional[FeatureSpec]
    analysis_trace: list[dict[str, Any]]
    
    # Step 5: Feature Engineering
    transformed_dataset_ref: Optional[str]  # Legacy - for backward compat
    transformed_train_ref: Optional[str]    # NEW: transformed train dataset
    transformed_val_ref: Optional[str]      # NEW: transformed val dataset
    transformed_test_ref: Optional[str]     # NEW: transformed test dataset
    feature_validation_passed: bool
    
    # Step 6: Human Confirmation
    human_confirmed: bool
    
    # Step 7: Training
    training_params: Optional[dict[str, Any]]
    model_weights_path: Optional[str]
    training_metrics: Optional[dict[str, Any]]
    training_iteration: int
    
    # Feature Engineering Redo (loop back from training)
    feature_redo_requested: bool
    feature_redo_recommendation: Optional[str]
    feature_redo_reason: Optional[str]
    feature_redo_iteration: int  # Track how many times we've looped back
    
    # Step 8: Report
    report_path: Optional[str]
    
    # Audit & Trace
    audit_trace: list[dict[str, Any]]
    explanations: list[str]
    
    # Control flow
    current_step: str
    error: Optional[str]


# =============================================================================
# NODE FUNCTIONS (NOT IMPLEMENTED - STRUCTURE ONLY)
# =============================================================================

# select_model is imported from .select_model
# data_collection is imported from .data_collection

# TODO: Need a way for users to link their datasets from the start


def cleaning_node(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 3: Cleaning & Standardization using simple cleaning agent.
    
    Calls run_cleaning_simple which handles its own iteration loop internally.
    Returns updated state with cleaned dataset reference.
    """
    # Dynamically set max iterations based on dataset complexity
    # More columns = more potential cleaning operations needed
    try:
        from utils import get_registered_dataset
        df = get_registered_dataset(state["collected_dataset_ref"])
        num_columns = len(df.columns) if df is not None else 20
        # Base of 30 iterations + 1 per column, capped at 80
        max_iters = min(80, max(30, 30 + num_columns))
    except Exception:
        max_iters = 50
    
    result = run_cleaning_simple(
        dataset_ref=state["collected_dataset_ref"],
        goal=state["goal"],
        max_iterations=max_iters,
    )
    
    # Extract transformations from the cleaning result
    transformations = result.get("transformations", [])
    
    return {
        **state,
        "cleaned_dataset_ref": result["cleaned_ref"],
        "cleaning_summary": result.get("cleaning_summary"),
        "cleaning_transformations": transformations,
        "current_step": "label_split_definition",
    }


def label_split_definition(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 3.5: Label + Split Definition + Data Splitting
    
    Uses LLM to infer the 6 key parameters for supervised learning:
    1. Target column
    2. Prediction horizon
    3. Grain (what does one row represent?)
    4. As-of cutoff
    5. Split strategy (random / time-based / entity-based)
    6. Forbidden columns (not available at prediction time)
    
    Then SPLITS the data into train/val/test using the defined strategy.
    
    If label_definition is already set in state, uses those values.
    Otherwise, LLM infers based on goal, model, and schema context.
    
    Locked definitions + split datasets passed to downstream steps (4, 5, 7)
    """
    # Extract any pre-provided label definition values
    existing = state.get("label_definition") or {}
    
    # Step 1: Get label definition from LLM
    label_def = run_label_split_definition(
        dataset_ref=state["cleaned_dataset_ref"],
        goal=state["goal"],
        selected_model=state.get("selected_model"),
        model_explanation=state.get("model_explanation"),
        target_column=existing.get("target_column"),
        prediction_horizon=existing.get("prediction_horizon"),
        grain=existing.get("grain"),
        as_of_cutoff=existing.get("as_of_cutoff"),
        split_strategy=existing.get("split_strategy"),
        forbidden_columns=existing.get("forbidden_columns"),
    )
    
    # Step 2: Get the cleaned dataset
    df = get_registered_dataset(state["cleaned_dataset_ref"])
    
    # Step 3: Compute split indices based on strategy
    print(f"[label_split_definition] Computing {label_def.get('split_strategy', 'random')} split...")
    split_indices = compute_split_indices(
        df=df,
        label_definition=label_def,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    # Step 4: Apply split to get train/val/test DataFrames
    train_df, val_df, test_df = apply_split(df, split_indices)
    
    print(f"[label_split_definition] Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Step 5: Register the split datasets
    base_ref = state["cleaned_dataset_ref"]
    train_ref = f"{base_ref}_train"
    val_ref = f"{base_ref}_val"
    test_ref = f"{base_ref}_test"
    
    register_dataset(train_ref, train_df)
    register_dataset(val_ref, val_df)
    register_dataset(test_ref, test_df)
    
    print(f"[label_split_definition] Registered: {train_ref}, {val_ref}, {test_ref}")
    
    return {
        **state,
        "label_definition": label_def,
        "split_indices": split_indices,
        "train_dataset_ref": train_ref,
        "val_dataset_ref": val_ref,
        "test_dataset_ref": test_ref,
        "current_step": "feature_selection_specification",
    }


def feature_selection_specification(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 4: Feature Selection & Specification
    
    Uses the simple approach: run all analysis tools upfront, then one LLM call.
    
    IMPORTANT: Analysis runs on TRAINING DATA ONLY to prevent data leakage.
    
    Inputs from state:
    - train_dataset_ref: The training dataset from step 3.5
    - val_dataset_ref: The validation dataset from step 3.5
    - test_dataset_ref: The test dataset from step 3.5
    - goal: The ML goal
    - label_definition: Contains target_column, grain, as_of_cutoff, forbidden_columns
    
    Output: feature_spec (structured contract for step 5)
    """
    # Get inputs from state
    train_ref = state.get("train_dataset_ref")
    val_ref = state.get("val_dataset_ref")
    test_ref = state.get("test_dataset_ref")
    goal = state.get("goal", "")
    label_def = state.get("label_definition") or {}
    
    target_column = label_def.get("target_column", "")
    grain = label_def.get("grain", "")
    as_of_cutoff = label_def.get("as_of_cutoff")
    forbidden_columns = label_def.get("forbidden_columns", [])
    prediction_horizon = label_def.get("prediction_horizon")
    
    # Infer task type from goal or model selection
    # Default to classification, could be enhanced to detect from goal text
    selected_model = state.get("selected_model", "").lower()
    if any(word in selected_model for word in ["regress", "continuous", "numeric"]):
        task_type = "regression"
    elif any(word in goal.lower() for word in ["regress", "predict value", "forecast amount"]):
        task_type = "regression"
    else:
        task_type = "classification"
    
    # Validate we have required inputs
    if not train_ref:
        raise ValueError("No train_dataset_ref in state - step 3.5 must complete first")
    if not target_column:
        raise ValueError("No target_column in label_definition - step 3.5 must complete first")
    
    # Check if this is a feature engineering redo from training
    feature_redo_requested = state.get("feature_redo_requested", False)
    feature_redo_recommendation = state.get("feature_redo_recommendation")
    feature_redo_iteration = state.get("feature_redo_iteration", 0)
    
    if feature_redo_requested:
        print(f"[feature_selection_specification] REDO iteration {feature_redo_iteration + 1}")
        print(f"[feature_selection_specification] Recommendation from training: {feature_redo_recommendation}")
    
    print(f"[feature_selection_specification] Running analysis on TRAINING data only ({train_ref})...")
    
    # Run feature engineering - analysis on train only!
    result = run_feature_engineering_simple(
        train_ref=train_ref,
        goal=goal,
        target_column=target_column,
        grain=grain,
        recomendation=feature_redo_recommendation if feature_redo_requested else None,
        val_ref=val_ref,
        test_ref=test_ref,
        task_type=task_type,
        forbidden_columns=forbidden_columns,
        as_of_cutoff=as_of_cutoff,
        prediction_horizon=prediction_horizon,
    )
    
    # Extract feature_spec and key statistics
    feature_spec = result.get("feature_spec")
    validation = result.get("validation", {})
    analysis_results = result.get("analysis_results", {})
    key_stats = result.get("key_stats", {})
    
    # Log key statistics for visibility
    if key_stats:
        print(f"[feature_selection_specification] Key statistics extracted:")
        print(f"  - Dataset: {key_stats.get('dataset_overview', {}).get('rows', '?')} rows, {key_stats.get('dataset_overview', {}).get('columns', '?')} columns")
        if key_stats.get("feature_correlations"):
            top_corr = key_stats["feature_correlations"][0]
            print(f"  - Top correlation: {top_corr.get('feature')} (r={top_corr.get('correlation')})")
        if key_stats.get("leakage_warnings"):
            print(f"  - ⚠️ Leakage warnings: {len(key_stats['leakage_warnings'])} features")
        if key_stats.get("high_correlation_pairs"):
            print(f"  - High correlation pairs: {len(key_stats['high_correlation_pairs'])}")
    
    # Check validation and auto-remove invalid features (e.g., leakage)
    if validation and not validation.get("valid", True):
        errors = validation.get("errors", [])
        print(f"[WARNING] Feature spec validation issues: {errors}")
        
        # Auto-remove invalid features
        features_valid = validation.get("features_valid", {})
        if feature_spec and "features" in feature_spec:
            original_count = len(feature_spec["features"])
            feature_spec["features"] = [
                f for f in feature_spec["features"]
                if features_valid.get(f.get("name"), {}).get("valid", True)
            ]
            removed_count = original_count - len(feature_spec["features"])
            if removed_count > 0:
                print(f"[INFO] Auto-removed {removed_count} invalid feature(s)")
    
    # Update state - clear redo flags and increment redo iteration if this was a redo
    new_redo_iteration = feature_redo_iteration + 1 if feature_redo_requested else feature_redo_iteration
    
    return {
        **state,
        "feature_spec": feature_spec,
        "analysis_trace": [
            {
                "step": "feature_selection_specification",
                "analysis_results": analysis_results,
                "key_stats": key_stats,  # Structured stats for frontend display
                "validation": validation,
                "is_redo": feature_redo_requested,
                "redo_recommendation": feature_redo_recommendation,
            }
        ],
        # Clear redo flags after processing
        "feature_redo_requested": False,
        "feature_redo_recommendation": None,
        "feature_redo_reason": None,
        "feature_redo_iteration": new_redo_iteration,
    }


def feature_engineering_executor(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 5: Feature Engineering Executor (DETERMINISTIC - no LLM reasoning loop)
    
    IMPORTANT: Fits transformations on TRAINING data, applies to all splits.
    This prevents data leakage from val/test into feature computation.
    
    Inputs from state:
    - train_dataset_ref, val_dataset_ref, test_dataset_ref: Split datasets from step 3.5
    - feature_spec: The feature specification from step 4
    - label_definition: Contains target_column, grain, as_of_cutoff
    
    Executes each feature in the spec using the appropriate transformation.
    No iterative LLM decision-making; execution follows the spec exactly.
    
    Output: transformed_train_ref, transformed_val_ref, transformed_test_ref
    """
    # Get inputs from state
    train_ref = state.get("train_dataset_ref")
    val_ref = state.get("val_dataset_ref")
    test_ref = state.get("test_dataset_ref")
    feature_spec = state.get("feature_spec")
    label_def = state.get("label_definition") or {}
    
    target_column = label_def.get("target_column", "")
    grain = label_def.get("grain", "")
    as_of_cutoff = label_def.get("as_of_cutoff")
    
    # Validate inputs
    if not train_ref:
        raise ValueError("No train_dataset_ref in state - step 3.5 must complete first")
    if not feature_spec:
        raise ValueError("No feature_spec in state - step 4 must complete first")
    if not target_column:
        raise ValueError("No target_column in label_definition")
    
    print(f"[feature_engineering_executor] Executing feature spec (fit on train, transform all)...")
    
    # Execute the feature spec with split-aware logic
    result = execute_feature_spec_split(
        train_ref=train_ref,
        val_ref=val_ref,
        test_ref=test_ref,
        feature_spec=feature_spec,
        target_column=target_column,
        grain=grain,
        as_of_cutoff=as_of_cutoff,
    )
    
    # Check for errors
    errors = result.get("errors", [])
    if errors:
        print(f"[WARNING] Feature engineering had {len(errors)} errors:")
        for err in errors:
            print(f"  - {err}")
    
    # Determine if validation passed (no critical errors)
    # For now, consider passed if at least some features were created
    features_created = result.get("features_created", [])
    validation_passed = len(features_created) > 0 and len(errors) < len(features_created)
    
    print(f"[feature_engineering_executor] Created {len(features_created)} features")
    print(f"[feature_engineering_executor] Shapes: {result.get('shapes')}")
    
    # Update state
    return {
        **state,
        # NEW: Split-aware outputs
        "transformed_train_ref": result.get("train_ref"),
        "transformed_val_ref": result.get("val_ref"),
        "transformed_test_ref": result.get("test_ref"),
        # Legacy: for backward compatibility
        "transformed_dataset_ref": result.get("train_ref"),
        "feature_validation_passed": validation_passed,
        "audit_trace": state.get("audit_trace", []) + [
            {
                "step": "feature_engineering_executor",
                "features_created": features_created,
                "errors": errors,
                "shapes": result.get("shapes"),
                "temporal_constraints_applied": result.get("temporal_constraints_applied", 0),
            }
        ],
    }


def human_confirmation(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 6: Show trace of everything and get human confirmation
    - Present full audit trace
    - Get human to confirm we can proceed to training
    
    For automated runs, this auto-confirms. In production, would await user input.
    """
    print("\n" + "="*70)
    print("HUMAN CONFIRMATION CHECKPOINT")
    print("="*70)
    print(f"Goal: {state.get('goal')}")
    print(f"Selected Model: {state.get('selected_model')}")
    print(f"Target Column: {state.get('label_definition', {}).get('target_column', 'N/A')}")
    print(f"Train Dataset: {state.get('transformed_train_ref')}")
    print(f"Val Dataset: {state.get('transformed_val_ref')}")
    print(f"Test Dataset: {state.get('transformed_test_ref')}")
    print("\nAudit Trace:")
    for item in state.get("audit_trace", []):
        print(f"  - {item.get('step', 'unknown')}: {item}")
    print("="*70)
    print("Auto-confirming for automated run...")
    print("="*70 + "\n")
    
    return {
        **state,
        "human_confirmed": True,
        "audit_trace": state.get("audit_trace", []) + [
            {"step": "human_confirmation", "confirmed": True, "mode": "auto"}
        ],
    }


def training(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 7: Iterative Training Subagent
    Tools: glm + logistic_regression + random_forest + survival_analysis + 
           xgboost_model + model_storage
    
    Inputs from step 5: train_ref, val_ref, test_ref (already transformed with features)
    Inputs from step 3.5: label_definition (target_column, etc.)
    
    Uses the training agent to:
    i. Train on the training set
    ii. Evaluate on validation set
    iii. Evaluate on test set
    iv. Output model weights and metrics
    """
    from .training import run_training_agent as _run_training

    # Get inputs from state
    train_ref = state.get("transformed_train_ref")
    val_ref = state.get("transformed_val_ref")
    test_ref = state.get("transformed_test_ref")
    label_def = state.get("label_definition") or {}
    
    target_column = label_def.get("target_column", "")
    selected_model = state.get("selected_model", "logistic_regression")
    goal = state.get("goal", "")
    
    # Validate inputs
    if not train_ref:
        raise ValueError("No transformed_train_ref in state - step 5 must complete first")
    if not target_column:
        raise ValueError("No target_column in label_definition")
    
    # Generate model name
    import time
    model_name = f"{selected_model}_{int(time.time())}"
    
    print(f"[training] Starting training with {selected_model}...")
    print(f"  Train: {train_ref}")
    print(f"  Val: {val_ref}")
    print(f"  Test: {test_ref}")
    print(f"  Target: {target_column}")
    
    # Run training using the iterative training agent
    result = _run_training(
        train_ref=train_ref,
        val_ref=val_ref,
        test_ref=test_ref,
        target_column=target_column,
        selected_model=selected_model,
        goal=goal,
        model_name=model_name,
        max_iterations=3,
    )
    
    if result.get("success"):
        print(f"[training] Model trained successfully: {result.get('model_name')}")
    else:
        print(f"[training] Training failed: {result.get('error')}")
    
    # Check for feature engineering redo request
    feature_redo_requested = result.get("feature_redo_requested", False)
    feature_redo_recommendation = result.get("feature_redo_recommendation")
    feature_redo_reason = result.get("feature_redo_reason")
    
    if feature_redo_requested:
        print(f"[training] Feature engineering redo requested!")
        print(f"  Reason: {feature_redo_reason}")
        print(f"  Recommendation: {feature_redo_recommendation}")
    
    # Update state with extracted metrics and iteration logs
    return {
        **state,
        "model_weights_path": result.get("model_name"),  # Model is saved in registry
        "training_metrics": {
            "success": result.get("success"),
            "model_name": result.get("model_name"),
            "model_type": result.get("model_type"),
            # Classification metrics
            "val_accuracy": result.get("val_accuracy"),
            "val_roc_auc": result.get("val_roc_auc"),
            "test_accuracy": result.get("test_accuracy"),
            "test_roc_auc": result.get("test_roc_auc"),
            # Regression metrics
            "train_r2": result.get("train_r2"),
            "val_r2": result.get("val_r2"),
            "val_rmse": result.get("val_rmse"),
            "val_mae": result.get("val_mae"),
            "test_r2": result.get("test_r2"),
            "test_rmse": result.get("test_rmse"),
            "test_mae": result.get("test_mae"),
            # Iteration logs
            "iterations": result.get("iterations", []),
            "num_iterations": result.get("num_iterations", 0),
            "best_iteration": result.get("best_iteration"),
            # Summary and recommendations (important for regression results)
            "summary": result.get("summary"),
            "recommendations": result.get("recommendations"),
            # Feature engineering redo
            "feature_redo_requested": result.get("feature_redo_requested", False),
        },
        "training_iteration": state.get("training_iteration", 0) + 1,
        # Feature engineering redo fields
        "feature_redo_requested": feature_redo_requested,
        "feature_redo_recommendation": feature_redo_recommendation,
        "feature_redo_reason": feature_redo_reason,
        "audit_trace": state.get("audit_trace", []) + [
            {
                "step": "training",
                "model_name": result.get("model_name"),
                "success": result.get("success"),
                "num_iterations": result.get("num_iterations", 0),
                "val_accuracy": result.get("val_accuracy"),
                "val_roc_auc": result.get("val_roc_auc"),
                "test_accuracy": result.get("test_accuracy"),
                "test_roc_auc": result.get("test_roc_auc"),
                "error": result.get("error"),
                "feature_redo_requested": feature_redo_requested,
                "feature_redo_recommendation": feature_redo_recommendation,
            }
        ],
    }


def generate_report(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 8: Generate Report
    - Generate a report on everything to accompany the weights file
    """
    import json
    from datetime import datetime
    from pathlib import Path
    
    training_metrics = state.get("training_metrics", {})
    label_def = state.get("label_definition", {})
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "goal": state.get("goal"),
        "model": {
            "type": state.get("selected_model"),
            "name": training_metrics.get("model_name"),
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
            "success": training_metrics.get("success"),
            "num_iterations": training_metrics.get("num_iterations", 0),
            "validation_metrics": {
                "accuracy": training_metrics.get("val_accuracy"),
                "roc_auc": training_metrics.get("val_roc_auc"),
            },
            "test_metrics": {
                "accuracy": training_metrics.get("test_accuracy"),
                "roc_auc": training_metrics.get("test_roc_auc"),
            },
            "iterations": training_metrics.get("iterations", []),
            "best_iteration": training_metrics.get("best_iteration"),
            "summary": training_metrics.get("summary"),
            "recommendations": training_metrics.get("recommendations"),
        },
        "audit_trace": state.get("audit_trace", []),
    }
    
    # Save report to file
    report_dir = Path(__file__).parent.parent.parent / "trained_models"
    report_dir.mkdir(exist_ok=True)
    
    model_name = training_metrics.get("model_name", "unknown")
    report_path = report_dir / f"{model_name}_report.json"
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n[generate_report] Report saved to: {report_path}")
    
    return {
        **state,
        "report_path": str(report_path),
        "audit_trace": state.get("audit_trace", []) + [
            {"step": "generate_report", "path": str(report_path)}
        ],
    }


# =============================================================================
# CONDITIONAL EDGE FUNCTIONS
# =============================================================================

def should_regen_model(state: TrainingAgentState) -> Literal["regen", "continue"]:
    """Check if user wants to regenerate model selection (max 3 times)"""
    # For automated runs, always continue with selected model
    regen_count = state.get("model_regen_count", 0)
    if regen_count >= 3:
        print("[should_regen_model] Max regen count reached, continuing...")
        return "continue"
    return "continue"


def should_skip_label_definition(state: TrainingAgentState) -> Literal["skip", "define"]:
    """Check if label/split definition is relevant or should be skipped"""
    # Always define labels for supervised learning
    return "define"


def feature_validation_result(state: TrainingAgentState) -> Literal["passed", "failed"]:
    """Check if feature engineering validation passed or needs spec revision"""
    if state.get("transformed_train_ref"):
        return "passed"
    return "failed"


def human_confirmed_proceed(state: TrainingAgentState) -> Literal["proceed", "abort"]:
    """Check if human confirmed to proceed to training"""
    if state.get("human_confirmed"):
        return "proceed"
    return "proceed"


def training_decision(state: TrainingAgentState) -> Literal["iterate", "complete", "redo_features"]:
    """Check if training should iterate, complete, or redo feature engineering."""
    
    # Check if feature engineering redo was requested
    if state.get("feature_redo_requested"):
        redo_iteration = state.get("feature_redo_iteration", 0)
        # Limit feature redo iterations to prevent infinite loops
        if redo_iteration >= 2:
            print("[training_decision] Max feature redo iterations (2) reached, completing...")
            return "complete"
        print("[training_decision] Feature engineering redo requested, routing back...")
        return "redo_features"
    
    training_metrics = state.get("training_metrics", {})
    if training_metrics.get("success"):
        return "complete"
    iteration = state.get("training_iteration", 0)
    if iteration >= 3:
        return "complete"
    return "complete"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_training_agent_graph() -> StateGraph:
    """
    Build the ML Training Agent graph with all nodes and edges.
    
    Flow:
    1. select_model → (regen loop or continue)
    2. data_collection
    3. cleaning (uses simple cleaning agent with internal iteration)
    3.5. label_split_definition → (skip if not relevant)
    4. feature_selection_specification
    5. feature_engineering_executor → (back to 4 if validation fails)
    6. human_confirmation → (abort or proceed)
    7. training → (iterative with human checkpoints)
    8. generate_report → END
    """
    
    # Initialize the graph with state schema
    graph = StateGraph(TrainingAgentState)
    
    # -------------------------------------------------------------------------
    # ADD NODES
    # -------------------------------------------------------------------------
    graph.add_node("select_model", select_model)
    graph.add_node("data_collection", data_collection)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("label_split_definition", label_split_definition)
    graph.add_node("feature_selection_specification", feature_selection_specification)
    graph.add_node("feature_engineering_executor", feature_engineering_executor)
    graph.add_node("human_confirmation", human_confirmation)
    graph.add_node("training", training)
    graph.add_node("generate_report", generate_report)
    
    # -------------------------------------------------------------------------
    # ADD EDGES
    # -------------------------------------------------------------------------
    
    # Entry point
    graph.set_entry_point("select_model")
    
    # Step 1 → Step 2 (with potential regen loop)
    graph.add_conditional_edges(
        "select_model",
        should_regen_model,
        {
            "regen": "select_model",  # Loop back for regeneration
            "continue": "data_collection"
        }
    )
    
    # Step 2 → Step 3
    graph.add_edge("data_collection", "cleaning")
    
    # Step 3 → Step 3.5 (cleaning handles its own iteration internally)
    graph.add_edge("cleaning", "label_split_definition")
    
    # Step 3.5 → Step 4 (with skip option)
    graph.add_conditional_edges(
        "label_split_definition",
        should_skip_label_definition,
        {
            "skip": "feature_selection_specification",
            "define": "feature_selection_specification"  # Both go to step 4, but with different state
        }
    )
    
    # Step 4 → Step 5
    graph.add_edge("feature_selection_specification", "feature_engineering_executor")
    
    # Step 5 → Step 6 or back to Step 4 (validation check)
    graph.add_conditional_edges(
        "feature_engineering_executor",
        feature_validation_result,
        {
            "passed": "human_confirmation",
            "failed": "feature_selection_specification"  # Back to step 4 for spec revision
        }
    )
    
    # Step 6 → Step 7 or abort (human confirmation)
    graph.add_conditional_edges(
        "human_confirmation",
        human_confirmed_proceed,
        {
            "proceed": "training",
            "abort": END
        }
    )
    
    # Step 7 → Step 8, iterate training, or redo feature engineering
    graph.add_conditional_edges(
        "training",
        training_decision,
        {
            "iterate": "training",  # Back to training for another iteration
            "complete": "generate_report",
            "redo_features": "feature_selection_specification",  # Loop back to feature engineering with recommendation
        }
    )
    
    # Step 8 → END
    graph.add_edge("generate_report", END)
    
    return graph


# =============================================================================
# GRAPH COMPILATION & INVOCATION
# =============================================================================

# TODO: Params like dataset_ref, model_type, goal/prompt...
def create_training_agent():
    """Create and compile the training agent graph"""
    graph = build_training_agent_graph()
    return graph.compile()


def _create_initial_state(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None
) -> TrainingAgentState:
    """Create the initial state for the training agent."""
    return {
        # Inputs
        "goal": goal,
        "linked_datasets": linked_datasets,
        "user_model_preference": user_model_preference,
        
        # Initialize all other fields
        "selected_model": None,
        "model_explanation": None,
        "model_regen_count": 0,
        "collected_dataset_ref": None,
        "cleaned_dataset_ref": None,
        "cleaning_transformations": [],
        "cleaning_summary": None,
        "label_definition": None,
        "feature_spec": None,
        "analysis_trace": [],
        "transformed_dataset_ref": None,
        "feature_validation_passed": False,
        "human_confirmed": False,
        "training_params": None,
        "model_weights_path": None,
        "training_metrics": None,
        "training_iteration": 0,
        "feature_redo_requested": False,
        "feature_redo_recommendation": None,
        "feature_redo_reason": None,
        "feature_redo_iteration": 0,
        "report_path": None,
        "audit_trace": [],
        "explanations": [],
        "current_step": "select_model",
        "error": None,
    }


def invoke_training_agent(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None
) -> TrainingAgentState:
    """
    Invoke the training agent with the given inputs (non-streaming).
    
    Args:
        goal: The training goal/objective from user
        linked_datasets: Optional list of dataset references to use
        user_model_preference: Optional model type preference from user
        
    Returns:
        Final state containing:
        - audit_trace: Full lineage trace
        - explanations: Explanations of what was done
        - model_weights_path: Path to the trained weights file
        - report_path: Path to the generated report
    """
    agent = create_training_agent()
    initial_state = _create_initial_state(goal, linked_datasets, user_model_preference)
    return agent.invoke(initial_state)


from typing import Generator, Any

def stream_training_agent(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None
) -> Generator[dict[str, Any], None, TrainingAgentState]:
    """
    Stream the training agent with the given inputs.
    
    Yields state updates after each node completes, allowing real-time
    progress tracking in the UI.
    
    Args:
        goal: The training goal/objective from user
        linked_datasets: Optional list of dataset references to use
        user_model_preference: Optional model type preference from user
        
    Yields:
        Dict with:
        - node: Name of the node that just completed
        - state: Current state after this node
        
    Returns:
        Final state (accessible via generator's return value)
    """
    agent = create_training_agent()
    initial_state = _create_initial_state(goal, linked_datasets, user_model_preference)
    
    final_state = None
    
    # Stream with "values" mode to get full state after each node
    for event in agent.stream(initial_state, stream_mode="values"):
        # event is the full state after each node
        current_step = event.get("current_step", "unknown")
        
        yield {
            "type": "state_update",
            "node": current_step,
            "state": event,
        }
        
        final_state = event
    
    return final_state


def stream_training_agent_with_updates(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None
) -> Generator[dict[str, Any], None, None]:
    """
    Stream the training agent and yield structured updates.
    
    This is a simpler version that yields update dicts suitable for SSE.
    
    Args:
        goal: The training goal/objective from user
        linked_datasets: Optional list of dataset references to use
        user_model_preference: Optional model type preference from user
        
    Yields:
        Update dicts with type, node, progress, and relevant state fields
    """
    agent = create_training_agent()
    initial_state = _create_initial_state(goal, linked_datasets, user_model_preference)
    
    # Define step order for progress calculation
    step_order = [
        "select_model",
        "data_collection", 
        "cleaning_and_standardization",
        "label_split_definition",
        "feature_selection_specification",
        "feature_engineering_executor",
        "human_confirmation",
        "training",
        "generate_report",
    ]
    
    yield {
        "type": "started",
        "node": "init",
        "progress": 0,
        "message": "Training agent started",
    }
    
    # Stream with "updates" mode to see which node produced what
    for event in agent.stream(initial_state, stream_mode="updates"):
        # event is a dict like {node_name: node_output}
        for node_name, node_output in event.items():
            # Calculate progress
            try:
                step_idx = step_order.index(node_name)
                progress = int(((step_idx + 1) / len(step_order)) * 100)
            except ValueError:
                progress = 50
            
            # Extract key info from the node output
            update = {
                "type": "node_complete",
                "node": node_name,
                "progress": progress,
                "state": node_output,  # Full state after this node
            }
            
            # Add step-specific detailed info
            if node_name == "select_model":
                update["summary"] = {
                    "selected_model": node_output.get("selected_model"),
                    "explanation": node_output.get("model_explanation"),
                }
                update["details"] = {
                    "title": "Model Selection Complete",
                    "description": f"Selected **{node_output.get('selected_model', 'unknown')}** as the optimal model for this task.",
                    "reasoning": node_output.get("model_explanation", "No explanation provided."),
                }
                
            elif node_name == "data_collection":
                # Get audit trace entry for data collection details
                audit = next((t for t in node_output.get("audit_trace", []) if t.get("step") == "data_collection"), {})
                update["summary"] = {
                    "dataset": node_output.get("collected_dataset_ref"),
                    "rows": audit.get("rows"),
                    "columns": audit.get("columns"),
                    "source": audit.get("source", "collected"),
                }
                update["details"] = {
                    "title": "Data Collection Complete",
                    "description": f"Loaded dataset: **{node_output.get('collected_dataset_ref', 'unknown')}**",
                    "stats": {
                        "rows": audit.get("rows", "unknown"),
                        "columns": len(audit.get("columns", [])) if audit.get("columns") else "unknown",
                        "column_names": audit.get("columns", []),
                    },
                }
                
            elif node_name == "cleaning_and_standardization":
                transformations = node_output.get("cleaning_transformations", [])
                cleaning_summary = node_output.get("cleaning_summary") or ""
                
                # Parse cleaning summary for key info
                rows_cleaned = "unknown"
                cols_cleaned = "unknown" 
                reason = ""
                if cleaning_summary:
                    for line in cleaning_summary.split("\n"):
                        if "Rows:" in line:
                            rows_cleaned = line.split("Rows:")[1].strip().split()[0] if "Rows:" in line else "unknown"
                        if "Columns:" in line:
                            cols_cleaned = line.split("Columns:")[1].strip().split()[0] if "Columns:" in line else "unknown"
                        if "Reason:" in line:
                            reason = line.split("Reason:")[1].strip() if "Reason:" in line else ""
                
                update["summary"] = {
                    "cleaned_dataset": node_output.get("cleaned_dataset_ref"),
                    "num_transformations": len(transformations),
                    "transformations": transformations[:10],  # First 10 for chat preview
                    "cleaning_message": cleaning_summary,  # Full message from agent
                    "rows": rows_cleaned,
                    "columns": cols_cleaned,
                    "reason": reason,
                }
                update["details"] = {
                    "title": "Data Cleaning Complete",
                    "description": reason or f"Applied {len(transformations)} transformations to clean the data.",
                    "output_dataset": node_output.get("cleaned_dataset_ref"),
                    "all_transformations": transformations,  # Full list for report
                    "cleaning_summary": cleaning_summary,
                    "transformations_applied": [
                        t if isinstance(t, dict) else {"op": str(t)} for t in transformations
                    ],
                }
                
            elif node_name == "label_split_definition":
                label_def = node_output.get("label_definition", {}) or {}
                update["summary"] = {
                    "target_column": label_def.get("target_column"),
                    "split_strategy": label_def.get("split_strategy"),
                    "grain": label_def.get("grain"),
                    "train_ref": node_output.get("train_dataset_ref"),
                    "val_ref": node_output.get("val_dataset_ref"),
                    "test_ref": node_output.get("test_dataset_ref"),
                }
                update["details"] = {
                    "title": "Label & Split Definition Complete",
                    "description": f"Target column: **{label_def.get('target_column', 'unknown')}** with {label_def.get('split_strategy', 'random')} split.",
                    "label_definition": {
                        "target": label_def.get("target_column"),
                        "strategy": label_def.get("split_strategy"),
                        "grain": label_def.get("grain"),
                        "forbidden_columns": label_def.get("forbidden_columns", []),
                    },
                    "datasets": {
                        "train": node_output.get("train_dataset_ref"),
                        "validation": node_output.get("val_dataset_ref"),
                        "test": node_output.get("test_dataset_ref"),
                    },
                }
                
            elif node_name == "feature_selection_specification":
                feature_spec = node_output.get("feature_spec", {}) or {}
                features = feature_spec.get("features", [])
                analysis_trace = node_output.get("analysis_trace", [])
                
                # Extract key_stats from analysis_trace
                key_stats = {}
                if analysis_trace:
                    key_stats = analysis_trace[0].get("key_stats", {}) if len(analysis_trace) > 0 else {}
                
                update["summary"] = {
                    "num_features": len(features),
                    "feature_names": [f.get("name") for f in features[:10]],  # First 10
                    # Include all key statistics in summary for chat display
                    "dataset_overview": key_stats.get("dataset_overview", {}),
                    "target_analysis": key_stats.get("target_analysis", {}),
                    "numeric_summaries": key_stats.get("numeric_summaries", []),  # Full numeric stats
                    "feature_correlations": key_stats.get("feature_correlations", []),  # All correlations
                    "correlation_matrix": key_stats.get("correlation_matrix", {}),
                    "high_correlation_pairs": key_stats.get("high_correlation_pairs", []),
                    "leakage_warnings": key_stats.get("leakage_warnings", []),
                    "feature_health": key_stats.get("feature_health", []),
                    "distribution_stats": key_stats.get("distribution_stats", []),  # With histograms
                    "group_summaries": key_stats.get("group_summaries", []),  # Categorical analysis
                    "concentration_analysis": key_stats.get("concentration_analysis", []),  # With Lorenz curves
                    "categorical_summaries": key_stats.get("categorical_summaries", []),
                    "schema": key_stats.get("schema", []),
                    "summary_text": key_stats.get("summary_text", ""),
                }
                update["details"] = {
                    "title": "Feature Selection Complete",
                    "description": f"Specified {len(features)} features for the model.",
                    "features": [
                        {
                            "name": f.get("name"),
                            "encoding": f.get("encoding"),
                            "formula": str(f.get("formula")) if f.get("formula") else None,
                        }
                        for f in features
                    ],
                    # Full analysis data for report display
                    "key_stats": key_stats,
                    "analysis_trace": analysis_trace,
                }
                
            elif node_name == "feature_engineering_executor":
                # Get audit trace entry for feature engineering details
                audit = next((t for t in node_output.get("audit_trace", []) if t.get("step") == "feature_engineering_executor"), {})
                shapes = audit.get("shapes", {})
                update["summary"] = {
                    "train_ref": node_output.get("transformed_train_ref"),
                    "val_ref": node_output.get("transformed_val_ref"),
                    "test_ref": node_output.get("transformed_test_ref"),
                    "validation_passed": node_output.get("feature_validation_passed"),
                    "features_created": audit.get("features_created", []),
                    "shapes": shapes,
                }
                update["details"] = {
                    "title": "Feature Engineering Complete",
                    "description": f"Created {len(audit.get('features_created', []))} features.",
                    "features_created": audit.get("features_created", []),
                    "dataset_shapes": {
                        "train": f"{shapes.get('train', ['?', '?'])[0]} rows × {shapes.get('train', ['?', '?'])[1]} cols" if shapes.get('train') else "unknown",
                        "validation": f"{shapes.get('val', ['?', '?'])[0]} rows × {shapes.get('val', ['?', '?'])[1]} cols" if shapes.get('val') else "unknown",
                        "test": f"{shapes.get('test', ['?', '?'])[0]} rows × {shapes.get('test', ['?', '?'])[1]} cols" if shapes.get('test') else "unknown",
                    },
                    "errors": audit.get("errors", []),
                }
                
            elif node_name == "human_confirmation":
                # Get audit trace entry for confirmation details
                audit = next((t for t in node_output.get("audit_trace", []) if t.get("step") == "human_confirmation"), {})
                update["summary"] = {
                    "confirmed": node_output.get("human_confirmed"),
                    "mode": audit.get("mode", "auto"),
                }
                update["details"] = {
                    "title": "Confirmation Checkpoint",
                    "description": "Ready to proceed with training." if node_output.get("human_confirmed") else "Awaiting confirmation.",
                    "confirmed": node_output.get("human_confirmed"),
                    "selected_model": node_output.get("selected_model"),
                    "target_column": node_output.get("label_definition", {}).get("target_column") if node_output.get("label_definition") else None,
                }
                
            elif node_name == "training":
                metrics = node_output.get("training_metrics", {}) or {}
                iterations = metrics.get("iterations", [])
                update["summary"] = {
                    "success": metrics.get("success"),
                    "model_name": metrics.get("model_name"),
                    "model_type": metrics.get("model_type"),
                    "num_iterations": metrics.get("num_iterations"),
                    # Classification metrics
                    "val_accuracy": metrics.get("val_accuracy"),
                    "val_roc_auc": metrics.get("val_roc_auc"),
                    "test_accuracy": metrics.get("test_accuracy"),
                    "test_roc_auc": metrics.get("test_roc_auc"),
                    # Regression metrics
                    "val_r2": metrics.get("val_r2"),
                    "val_rmse": metrics.get("val_rmse"),
                    "test_r2": metrics.get("test_r2"),
                    "test_rmse": metrics.get("test_rmse"),
                    "test_mae": metrics.get("test_mae"),
                }
                update["details"] = {
                    "title": "Training Complete",
                    "description": f"Trained **{metrics.get('model_name', 'model')}** successfully." if metrics.get("success") else "Training completed with issues.",
                    "model": {
                        "name": metrics.get("model_name"),
                        "type": metrics.get("model_type"),
                        "path": node_output.get("model_weights_path"),
                    },
                    "metrics": {
                        # Include both classification and regression metrics
                        "validation": {
                            "accuracy": metrics.get("val_accuracy"),
                            "roc_auc": metrics.get("val_roc_auc"),
                            "r2": metrics.get("val_r2"),
                            "rmse": metrics.get("val_rmse"),
                            "mae": metrics.get("val_mae"),
                        },
                        "test": {
                            "accuracy": metrics.get("test_accuracy"),
                            "roc_auc": metrics.get("test_roc_auc"),
                            "r2": metrics.get("test_r2"),
                            "rmse": metrics.get("test_rmse"),
                            "mae": metrics.get("test_mae"),
                        },
                    },
                    "iterations": [
                        {
                            "model_name": it.get("model_name"),
                            "tool": it.get("tool"),
                            "success": it.get("success"),
                            "val_r2": it.get("val_r2"),
                            "test_r2": it.get("test_r2"),
                        }
                        for it in iterations
                    ],
                    "summary": metrics.get("summary"),
                    "recommendations": metrics.get("recommendations"),
                }
                
            elif node_name == "generate_report":
                update["summary"] = {
                    "report_path": node_output.get("report_path"),
                    "model_path": node_output.get("model_weights_path"),
                }
                update["details"] = {
                    "title": "Report Generated",
                    "description": f"Final report saved to: **{node_output.get('report_path', 'unknown')}**",
                    "report_path": node_output.get("report_path"),
                    "model_weights_path": node_output.get("model_weights_path"),
                    "audit_trace_length": len(node_output.get("audit_trace", [])),
                }
            
            yield update
    
    yield {
        "type": "completed",
        "node": "end",
        "progress": 100,
        "message": "Training completed",
    }
