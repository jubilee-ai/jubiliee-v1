"""
ML Model Training Agent - LangGraph Structure
Based on the architecture defined in README.md
"""

# Import utilities for dataset registration
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from .cleaning_simple import run_cleaning_simple
from .data_collection import data_collection as _data_collection_impl
from .feature_engineering_executor import (execute_feature_spec,
                                           execute_feature_spec_split)
from .feature_engineering_simple import run_feature_engineering_simple
from .label_and_split import (apply_split, compute_split_indices,
                              run_label_split_definition)
# Import node implementations
from .select_model import select_model as _select_model_impl

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))
from utils import get_registered_dataset, register_dataset

# =============================================================================
# HUMAN-IN-THE-LOOP HELPER
# =============================================================================

def _make_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to Python native types for serialization."""
    import numpy as np
    
    if obj is None:
        return None
    elif isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_serializable(item) for item in obj)
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # Try to convert to string as last resort
        try:
            return str(obj)
        except Exception:
            return None


def run_with_hitl(
    node_name: str,
    state: "TrainingAgentState",
    work_fn: Callable[["TrainingAgentState", Optional[str]], "TrainingAgentState"],
    get_summary_fn: Optional[Callable[["TrainingAgentState"], str]] = None,
) -> "TrainingAgentState":
    """
    Run node work with human-in-the-loop approval.
    
    After the work completes, pauses for human review. User can:
    - Approve: continue to next node
    - Reject with feedback: re-run the node with the feedback
    
    Args:
        node_name: Name of the node (for display)
        state: Current agent state
        work_fn: Function that takes (state, feedback) and returns updated state
        get_summary_fn: Optional function to create human-readable summary from result
        
    Returns:
        Updated state after human approval
    """
    feedback = None
    
    while True:
        # Run the actual work (passing feedback if this is a redo)
        result = work_fn(state, feedback)
        
        # Sanitize the entire result to ensure all values are serializable
        # This is critical for LangGraph's checkpointer (msgpack)
        result = {k: _make_serializable(v) for k, v in result.items()}
        
        # Create summary for human review
        if get_summary_fn:
            summary = get_summary_fn(result)
        else:
            summary = f"Node '{node_name}' completed successfully."
        
        # Build state snapshot with only key fields
        state_snapshot_keys = [
            "selected_model", "model_explanation", "collected_dataset_ref",
            "cleaned_dataset_ref", "cleaning_summary", "label_definition",
            "feature_spec", "transformed_train_ref", "training_plan",
            "training_metrics", "report_path", "error"
        ]
        state_snapshot = {
            k: result.get(k) 
            for k in state_snapshot_keys 
            if k in result
        }
        
        # Interrupt for human review
        decision = interrupt({
            "node": node_name,
            "summary": summary,
            "message": "Approve to continue, or provide feedback to redo.",
            "state_snapshot": state_snapshot
        })
        
        # Handle the decision
        # Accept: {"approved": True} or just True or "yes"
        # Reject: {"approved": False, "feedback": "..."} or {"feedback": "..."}
        if isinstance(decision, bool):
            approved = decision
            feedback = None
        elif isinstance(decision, str):
            # String response - "yes"/"y" means approved, anything else is feedback
            if decision.lower() in ["yes", "y", "ok", "approve", "approved", "continue"]:
                approved = True
                feedback = None
            else:
                approved = False
                feedback = decision
        elif isinstance(decision, dict):
            approved = decision.get("approved", True)
            feedback = decision.get("feedback")
        else:
            # Default to approved if unclear
            approved = True
            feedback = None
        
        if approved:
            return result
        else:
            # User rejected - loop will redo with feedback
            if not feedback:
                feedback = "Please redo this step."
            print(f"[HITL] Node '{node_name}' rejected. Feedback: {feedback}")


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
    
    # Step 6.5: Training Approval (pre-training plan)
    training_plan: Optional[dict[str, Any]]  # Approved training configuration
    training_plan_approved: bool
    
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
# NODE FUNCTIONS WITH HITL
# =============================================================================

def select_model(state: TrainingAgentState) -> TrainingAgentState:
    """Step 1: Select Model with HITL approval."""
    
    def do_work(state: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        # If there's feedback, add it to the goal for the LLM to consider
        if feedback:
            modified_state = {
                **state,
                "goal": f"{state['goal']}\n\nUser feedback on model selection: {feedback}"
            }
            return _select_model_impl(modified_state)
        return _select_model_impl(state)
    
    def get_summary(result: TrainingAgentState) -> str:
        model = result.get("selected_model", "unknown")
        explanation = result.get("model_explanation", "No explanation provided")
        return f"Selected model: {model}\n\nReason: {explanation}"
    
    return run_with_hitl("select_model", state, do_work, get_summary)


def data_collection(state: TrainingAgentState) -> TrainingAgentState:
    """Step 2: Data Collection with HITL approval."""
    
    def do_work(state: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        if feedback:
            modified_state = {
                **state,
                "goal": f"{state['goal']}\n\nUser feedback on data collection: {feedback}"
            }
            return _data_collection_impl(modified_state)
        return _data_collection_impl(state)
    
    def get_summary(result: TrainingAgentState) -> str:
        ref = result.get("collected_dataset_ref", "unknown")
        # Try to get row/column info from audit trace
        audit = next((t for t in result.get("audit_trace", []) if t.get("step") == "data_collection"), {})
        rows = audit.get("rows", "?")
        cols = len(audit.get("columns", [])) if audit.get("columns") else "?"
        return f"Collected dataset: {ref}\nRows: {rows}, Columns: {cols}"
    
    return run_with_hitl("data_collection", state, do_work, get_summary)


def cleaning_node(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 3: Cleaning & Standardization using simple cleaning agent with HITL approval.
    
    Calls run_cleaning_simple which handles its own iteration loop internally.
    Returns updated state with cleaned dataset reference.
    """
    
    def do_work(state: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        # Dynamically set max iterations based on dataset complexity
        try:
            df = get_registered_dataset(state["collected_dataset_ref"])
            num_columns = len(df.columns) if df is not None else 20
            max_iters = min(80, max(30, 30 + num_columns))
        except Exception:
            max_iters = 50
        
        goal = state["goal"]
        if feedback:
            goal = f"{goal}\n\nUser feedback on cleaning: {feedback}"
        
        result = run_cleaning_simple(
            dataset_ref=state["collected_dataset_ref"],
            goal=goal,
            max_iterations=max_iters,
        )
        
        transformations = result.get("transformations", [])
        
        return {
            **state,
            "cleaned_dataset_ref": result["cleaned_ref"],
            "cleaning_summary": result.get("cleaning_summary"),
            "cleaning_transformations": transformations,
            "current_step": "label_split_definition",
        }
    
    def get_summary(result: TrainingAgentState) -> str:
        ref = result.get("cleaned_dataset_ref", "unknown")
        summary = result.get("cleaning_summary", "")
        num_transforms = len(result.get("cleaning_transformations", []))
        return f"Cleaned dataset: {ref}\nTransformations applied: {num_transforms}\n\n{summary}"
    
    return run_with_hitl("cleaning", state, do_work, get_summary)


def label_split_definition(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 3.5: Label + Split Definition + Data Splitting with HITL approval.
    
    Uses LLM to infer the 6 key parameters for supervised learning:
    1. Target column
    2. Prediction horizon
    3. Grain (what does one row represent?)
    4. As-of cutoff
    5. Split strategy (random / time-based / entity-based)
    6. Forbidden columns (not available at prediction time)
    
    Then SPLITS the data into train/val/test using the defined strategy.
    """
    
    def do_work(state: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        existing = state.get("label_definition") or {}
        
        goal = state["goal"]
        if feedback:
            goal = f"{goal}\n\nUser feedback on label/split definition: {feedback}"
        
        # Step 1: Get label definition from LLM
        label_def = run_label_split_definition(
            dataset_ref=state["cleaned_dataset_ref"],
            goal=goal,
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
    
    def get_summary(result: TrainingAgentState) -> str:
        label_def = result.get("label_definition", {})
        return (
            f"Target column: {label_def.get('target_column', 'unknown')}\n"
            f"Split strategy: {label_def.get('split_strategy', 'unknown')}\n"
            f"Grain: {label_def.get('grain', 'unknown')}\n"
            f"Forbidden columns: {label_def.get('forbidden_columns', [])}\n"
            f"Train: {result.get('train_dataset_ref')}\n"
            f"Val: {result.get('val_dataset_ref')}\n"
            f"Test: {result.get('test_dataset_ref')}"
        )
    
    return run_with_hitl("label_split_definition", state, do_work, get_summary)


def feature_selection_specification(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 4: Feature Selection & Specification with HITL approval.
    
    Uses the simple approach: run all analysis tools upfront, then one LLM call.
    IMPORTANT: Analysis runs on TRAINING DATA ONLY to prevent data leakage.
    """
    
    def do_work(state: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        # Get inputs from state
        train_ref = state.get("train_dataset_ref")
        val_ref = state.get("val_dataset_ref")
        test_ref = state.get("test_dataset_ref")
        goal = state.get("goal", "")
        label_def = state.get("label_definition") or {}
        
        # Add feedback to goal if provided
        if feedback:
            goal = f"{goal}\n\nUser feedback on feature selection: {feedback}"
        
        target_column = label_def.get("target_column", "")
        grain = label_def.get("grain", "")
        as_of_cutoff = label_def.get("as_of_cutoff")
        forbidden_columns = label_def.get("forbidden_columns", [])
        prediction_horizon = label_def.get("prediction_horizon")
        
        # Infer task type from goal or model selection
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
        
        # Combine feedback with redo recommendation
        recommendation = None
        if feature_redo_requested and feature_redo_recommendation:
            recommendation = feature_redo_recommendation
        if feedback:
            recommendation = f"{recommendation}\n{feedback}" if recommendation else feedback
        
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
            recomendation=recommendation,
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
                    "key_stats": key_stats,
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
    
    def get_summary(result: TrainingAgentState) -> str:
        feature_spec = result.get("feature_spec", {}) or {}
        features = feature_spec.get("features", [])
        feature_names = [f.get("name", "?") for f in features[:10]]
        more = f" (+{len(features) - 10} more)" if len(features) > 10 else ""
        return (
            f"Features selected: {len(features)}\n"
            f"Feature names: {', '.join(feature_names)}{more}"
        )
    
    return run_with_hitl("feature_selection_specification", state, do_work, get_summary)


def feature_engineering_executor(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 5: Feature Engineering Executor with HITL approval.
    
    IMPORTANT: Fits transformations on TRAINING data, applies to all splits.
    This prevents data leakage from val/test into feature computation.
    
    Note: This is deterministic execution, so feedback triggers going back to
    feature_selection_specification rather than re-running this node.
    """
    
    def do_work(state: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
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
        
        # If feedback is provided, store it for the next iteration of feature selection
        # (this executor is deterministic, so we can't change features here)
        if feedback:
            print(f"[feature_engineering_executor] Feedback received, will be passed to feature selection: {feedback}")
        
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
        
        # Determine if validation passed
        features_created = result.get("features_created", [])
        validation_passed = len(features_created) > 0 and len(errors) < len(features_created)
        
        print(f"[feature_engineering_executor] Created {len(features_created)} features")
        print(f"[feature_engineering_executor] Shapes: {result.get('shapes')}")
        
        # Update state
        return {
            **state,
            "transformed_train_ref": result.get("train_ref"),
            "transformed_val_ref": result.get("val_ref"),
            "transformed_test_ref": result.get("test_ref"),
            "transformed_dataset_ref": result.get("train_ref"),
            "feature_validation_passed": validation_passed,
            # Store feedback for feature selection if user rejects
            "feature_redo_recommendation": feedback if feedback else state.get("feature_redo_recommendation"),
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
    
    def get_summary(result: TrainingAgentState) -> str:
        audit = next((t for t in result.get("audit_trace", []) if t.get("step") == "feature_engineering_executor"), {})
        features = audit.get("features_created", [])
        shapes = audit.get("shapes", {})
        errors = audit.get("errors", [])
        return (
            f"Features created: {len(features)}\n"
            f"Train shape: {shapes.get('train', '?')}\n"
            f"Val shape: {shapes.get('val', '?')}\n"
            f"Test shape: {shapes.get('test', '?')}\n"
            f"Errors: {len(errors)}"
        )
    
    return run_with_hitl("feature_engineering_executor", state, do_work, get_summary)


def training_approval(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 6.5: Training Approval - Propose training configuration for user approval.
    
    This node runs BEFORE training. It:
    1. Analyzes the prepared data (shapes, class distribution, features)
    2. Uses LLM to propose hyperparameters and training strategy
    3. Pauses for user approval BEFORE any training happens
    4. User can approve or provide feedback to adjust the plan
    """
    import json

    from langchain.chat_models import init_chat_model
    
    feedback = None
    
    while True:
        # Get inputs from state
        train_ref = state.get("transformed_train_ref")
        val_ref = state.get("transformed_val_ref")
        test_ref = state.get("transformed_test_ref")
        label_def = state.get("label_definition") or {}
        feature_spec = state.get("feature_spec") or {}
        
        target_column = label_def.get("target_column", "")
        selected_model = state.get("selected_model", "logistic_regression")
        goal = state.get("goal", "")
        
        # Load training data to analyze
        train_df = get_registered_dataset(train_ref)
        val_df = get_registered_dataset(val_ref) if val_ref else None
        
        if train_df is None:
            raise ValueError(f"Training dataset not found: {train_ref}")
        
        # Analyze data characteristics
        n_rows = len(train_df)
        n_features = len([c for c in train_df.columns if c != target_column])
        
        # Check class distribution for classification
        class_counts = train_df[target_column].value_counts().to_dict()
        total = sum(class_counts.values())
        minority_ratio = min(class_counts.values()) / total if total > 0 else 0
        is_imbalanced = minority_ratio < 0.3
        
        # Determine task type
        goal_lower = goal.lower()
        model_lower = selected_model.lower()
        if any(word in goal_lower for word in ["regress", "predict value", "forecast", "amount", "price", "cost"]):
            task_type = "regression"
        elif any(word in model_lower for word in ["glm", "regression"]) and "logistic" not in model_lower:
            task_type = "regression"
        else:
            task_type = "classification"
        
        # Build context for LLM
        features_list = [f.get("name") for f in feature_spec.get("features", [])][:20]
        
        prompt = f"""You are an ML expert. Propose a training configuration for the following task.

## Task
Goal: {goal}
Task Type: {task_type}
Selected Model: {selected_model}
Target Column: {target_column}

## Data Characteristics
- Training rows: {n_rows}
- Features: {n_features}
- Feature names (first 20): {features_list}
- Validation rows: {len(val_df) if val_df is not None else 'N/A'}

## Class Distribution (Training)
{json.dumps(class_counts, indent=2)}
{"⚠️ IMBALANCED DATA - minority class is " + f"{minority_ratio:.1%}" if is_imbalanced else "✓ Balanced classes"}

{f"## User Feedback to Incorporate:{chr(10)}{feedback}" if feedback else ""}

## Your Task
Propose a training configuration. Respond with a JSON object containing:

```json
{{
    "model_type": "{selected_model}",
    "task_type": "{task_type}",
    "hyperparameters": {{
        // Model-specific hyperparameters
        // For logistic_regression: C, penalty, solver, max_iter, class_weight
        // For random_forest: n_estimators, max_depth, min_samples_split, class_weight
        // For xgboost: n_estimators, max_depth, learning_rate, subsample, colsample_bytree
        // For glm: family, alpha, l1_ratio
    }},
    "class_weight": "balanced" or null,
    "max_iterations": 3,
    "strategy_notes": "Brief explanation of why these hyperparameters were chosen",
    "expected_metrics": "What metrics to optimize and expected performance range"
}}
```

Be specific with hyperparameter values. Consider:
- Dataset size ({n_rows} rows) - larger datasets can support more complex models
- Number of features ({n_features}) - may need regularization if many features
- Class imbalance - use class_weight="balanced" if imbalanced
- Model type - choose appropriate hyperparameters for {selected_model}
"""

        # Call LLM to propose training plan
        llm = init_chat_model("openai:gpt-4o-mini")
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        # Parse the response to extract JSON
        response_text = response.content
        training_plan = None
        
        # Try to extract JSON from response
        try:
            # Look for JSON block in markdown code fence
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                training_plan = json.loads(json_match.group(1))
            else:
                # Try parsing entire response as JSON
                training_plan = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback to default plan
            training_plan = {
                "model_type": selected_model,
                "task_type": task_type,
                "hyperparameters": {},
                "class_weight": "balanced" if is_imbalanced else None,
                "max_iterations": 3,
                "strategy_notes": "Default configuration - LLM response could not be parsed",
                "expected_metrics": "Standard metrics for the task type",
            }
        
        # Ensure required fields exist
        training_plan["model_type"] = training_plan.get("model_type", selected_model)
        training_plan["task_type"] = training_plan.get("task_type", task_type)
        training_plan["max_iterations"] = training_plan.get("max_iterations", 3)
        
        # Add data summary to the plan
        training_plan["data_summary"] = {
            "train_rows": n_rows,
            "val_rows": len(val_df) if val_df is not None else None,
            "n_features": n_features,
            "class_distribution": class_counts,
            "is_imbalanced": is_imbalanced,
        }
        
        print(f"[training_approval] Proposed training plan:")
        print(f"  Model: {training_plan.get('model_type')}")
        print(f"  Hyperparameters: {training_plan.get('hyperparameters')}")
        print(f"  Strategy: {training_plan.get('strategy_notes')}")
        
        # Build summary for user review
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
- Class balance: {'Imbalanced' if is_imbalanced else 'Balanced'}
"""
        
        # Sanitize for serialization
        training_plan = _make_serializable(training_plan)
        
        # Interrupt for human review BEFORE training
        decision = interrupt({
            "node": "training_approval",
            "summary": summary,
            "message": "Review the proposed training configuration. Approve to start training, or provide feedback to adjust the plan.",
            "state_snapshot": {
                "training_plan": training_plan,
                "selected_model": selected_model,
                "target_column": target_column,
            }
        })
        
        # Handle the decision
        if isinstance(decision, bool):
            approved = decision
            feedback = None
        elif isinstance(decision, str):
            if decision.lower() in ["yes", "y", "ok", "approve", "approved", "continue"]:
                approved = True
                feedback = None
            else:
                approved = False
                feedback = decision
        elif isinstance(decision, dict):
            approved = decision.get("approved", True)
            feedback = decision.get("feedback")
        else:
            approved = True
            feedback = None
        
        if approved:
            # User approved - store plan and continue to training
            return {
                **state,
                "training_plan": training_plan,
                "training_plan_approved": True,
                "current_step": "training",
            }
        else:
            # User rejected - loop with feedback
            if not feedback:
                feedback = "Please adjust the training configuration."
            print(f"[training_approval] Plan rejected. Feedback: {feedback}")
            # Loop continues with feedback incorporated into next LLM call


def training(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 7: Training with HITL approval.
    
    Trains the model on the prepared data and evaluates on validation/test sets.
    """
    from .training import run_training_agent as _run_training

    def do_work(state: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
        # Get inputs from state
        train_ref = state.get("transformed_train_ref")
        val_ref = state.get("transformed_val_ref")
        test_ref = state.get("transformed_test_ref")
        label_def = state.get("label_definition") or {}
        
        target_column = label_def.get("target_column", "")
        selected_model = state.get("selected_model", "logistic_regression")
        goal = state.get("goal", "")
        
        # Add feedback to goal if provided
        if feedback:
            goal = f"{goal}\n\nUser feedback on training: {feedback}"
        
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
        
        return {
            **state,
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
                "val_mae": result.get("val_mae"),
                "test_r2": result.get("test_r2"),
                "test_rmse": result.get("test_rmse"),
                "test_mae": result.get("test_mae"),
                "iterations": result.get("iterations", []),
                "num_iterations": result.get("num_iterations", 0),
                "best_iteration": result.get("best_iteration"),
                "summary": result.get("summary"),
                "recommendations": result.get("recommendations"),
                "feature_redo_requested": result.get("feature_redo_requested", False),
            },
            "training_iteration": state.get("training_iteration", 0) + 1,
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
    
    def get_summary(result: TrainingAgentState) -> str:
        metrics = result.get("training_metrics", {}) or {}
        success = metrics.get("success", False)
        model_name = metrics.get("model_name", "unknown")
        
        # Build metrics summary based on what's available
        lines = [f"Training {'succeeded' if success else 'failed'}"]
        lines.append(f"Model: {model_name}")
        
        if metrics.get("val_accuracy") is not None:
            lines.append(f"Val Accuracy: {metrics.get('val_accuracy'):.4f}")
        if metrics.get("val_roc_auc") is not None:
            lines.append(f"Val ROC-AUC: {metrics.get('val_roc_auc'):.4f}")
        if metrics.get("test_accuracy") is not None:
            lines.append(f"Test Accuracy: {metrics.get('test_accuracy'):.4f}")
        if metrics.get("val_r2") is not None:
            lines.append(f"Val R²: {metrics.get('val_r2'):.4f}")
        if metrics.get("test_r2") is not None:
            lines.append(f"Test R²: {metrics.get('test_r2'):.4f}")
        if metrics.get("summary"):
            lines.append(f"\nSummary: {metrics.get('summary')}")
        
        return "\n".join(lines)
    
    return run_with_hitl("training", state, do_work, get_summary)


def generate_report(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 8: Generate Report with HITL approval.
    
    Generates a report on everything to accompany the weights file.
    """
    import json
    from datetime import datetime
    
    def do_work(state: TrainingAgentState, feedback: Optional[str]) -> TrainingAgentState:
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
            "user_feedback": feedback,
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
    
    def get_summary(result: TrainingAgentState) -> str:
        return (
            f"Report generated successfully.\n"
            f"Path: {result.get('report_path')}\n"
            f"Model: {result.get('model_weights_path')}"
        )
    
    return run_with_hitl("generate_report", state, do_work, get_summary)


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
    6. training_approval → (user approves hyperparameters BEFORE training)
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
    graph.add_node("training_approval", training_approval)  # NEW: Approval before training
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
    
    # Step 5 → Step 6 (training_approval) or back to Step 4 (validation check)
    graph.add_conditional_edges(
        "feature_engineering_executor",
        feature_validation_result,
        {
            "passed": "training_approval",  # Go to approval step first
            "failed": "feature_selection_specification"  # Back to step 4 for spec revision
        }
    )
    
    # Step 6.5 → Step 7 (training_approval → training)
    graph.add_edge("training_approval", "training")
    
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

# Global checkpointer instance for persistence across invocations
_CHECKPOINTER = MemorySaver()


def create_training_agent(checkpointer=None):
    """
    Create and compile the training agent graph with HITL support.
    
    Args:
        checkpointer: Optional checkpointer for persistence. Uses global MemorySaver if not provided.
                     In production, use a persistent checkpointer like SqliteSaver or PostgresSaver.
    """
    graph = build_training_agent_graph()
    return graph.compile(checkpointer=checkpointer or _CHECKPOINTER)


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
        "training_plan": None,
        "training_plan_approved": False,
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
    user_model_preference: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Invoke the training agent with the given inputs (non-streaming).
    
    This function now supports human-in-the-loop. When a node completes, the agent
    will pause and return with an '__interrupt__' field. Use resume_training_agent()
    to continue after reviewing.
    
    Args:
        goal: The training goal/objective from user
        linked_datasets: Optional list of dataset references to use
        user_model_preference: Optional model type preference from user
        thread_id: Optional thread ID for persistence. Generated if not provided.
        
    Returns:
        Result dict containing:
        - If interrupted: '__interrupt__' field with node info and summary
        - If complete: Final state with audit_trace, model_weights_path, report_path
    """
    import uuid
    
    agent = create_training_agent()
    initial_state = _create_initial_state(goal, linked_datasets, user_model_preference)
    
    # Generate thread_id if not provided
    if not thread_id:
        thread_id = f"training-{uuid.uuid4().hex[:8]}"
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = agent.invoke(initial_state, config=config)
    
    # Add thread_id to result for resumption
    if isinstance(result, dict):
        result["_thread_id"] = thread_id
    
    return result


def resume_training_agent(
    decision: Any,
    thread_id: str,
) -> dict[str, Any]:
    """
    Resume the training agent after an interrupt with the user's decision.
    
    Args:
        decision: The user's decision. Can be:
            - True or "yes": Approve and continue
            - {"approved": True}: Approve and continue
            - "feedback text": Reject with feedback (will redo the node)
            - {"approved": False, "feedback": "..."}: Reject with feedback
        thread_id: The thread ID from the previous invocation
        
    Returns:
        Result dict (same as invoke_training_agent)
    """
    agent = create_training_agent()
    config = {"configurable": {"thread_id": thread_id}}
    
    result = agent.invoke(Command(resume=decision), config=config)
    
    # Add thread_id to result for further resumption
    if isinstance(result, dict):
        result["_thread_id"] = thread_id
    
    return result


from typing import Generator


def stream_training_agent(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Generator[dict[str, Any], None, TrainingAgentState]:
    """
    Stream the training agent with the given inputs.
    
    Yields state updates after each node completes, allowing real-time
    progress tracking in the UI. With HITL, will yield interrupt events
    that require user response.
    
    Args:
        goal: The training goal/objective from user
        linked_datasets: Optional list of dataset references to use
        user_model_preference: Optional model type preference from user
        thread_id: Optional thread ID for persistence
        
    Yields:
        Dict with:
        - type: "state_update" or "interrupt"
        - node: Name of the node that just completed
        - state: Current state after this node (for state_update)
        - interrupt: Interrupt info (for interrupt events)
        - thread_id: Thread ID for resumption
        
    Returns:
        Final state (accessible via generator's return value)
    """
    import uuid
    
    agent = create_training_agent()
    initial_state = _create_initial_state(goal, linked_datasets, user_model_preference)
    
    if not thread_id:
        thread_id = f"training-{uuid.uuid4().hex[:8]}"
    
    config = {"configurable": {"thread_id": thread_id}}
    
    final_state = None
    
    # Stream with "values" mode to get full state after each node
    for event in agent.stream(initial_state, config=config, stream_mode="values"):
        # Check for interrupt
        if "__interrupt__" in event:
            yield {
                "type": "interrupt",
                "interrupt": event["__interrupt__"],
                "thread_id": thread_id,
            }
            return event
        
        # Regular state update
        current_step = event.get("current_step", "unknown")
        
        yield {
            "type": "state_update",
            "node": current_step,
            "state": event,
            "thread_id": thread_id,
        }
        
        final_state = event
    
    return final_state


def stream_resume_training_agent(
    decision: Any,
    thread_id: str,
) -> Generator[dict[str, Any], None, TrainingAgentState]:
    """
    Resume streaming the training agent after an interrupt.
    
    Args:
        decision: The user's decision (same format as resume_training_agent)
        thread_id: The thread ID from the previous invocation
        
    Yields:
        Same as stream_training_agent
    """
    agent = create_training_agent()
    config = {"configurable": {"thread_id": thread_id}}
    
    final_state = None
    
    for event in agent.stream(Command(resume=decision), config=config, stream_mode="values"):
        if "__interrupt__" in event:
            yield {
                "type": "interrupt",
                "interrupt": event["__interrupt__"],
                "thread_id": thread_id,
            }
            return event
        
        current_step = event.get("current_step", "unknown")
        
        yield {
            "type": "state_update",
            "node": current_step,
            "state": event,
            "thread_id": thread_id,
        }
        
        final_state = event
    
    return final_state


def stream_training_agent_with_updates(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Stream the training agent and yield structured updates.
    
    This version yields update dicts suitable for SSE, including HITL interrupts.
    
    Args:
        goal: The training goal/objective from user
        linked_datasets: Optional list of dataset references to use
        user_model_preference: Optional model type preference from user
        thread_id: Optional thread ID for persistence
        
    Yields:
        Update dicts with type, node, progress, and relevant state fields.
        Interrupt events have type="interrupt" with interrupt info.
    """
    import uuid
    
    agent = create_training_agent()
    initial_state = _create_initial_state(goal, linked_datasets, user_model_preference)
    
    if not thread_id:
        thread_id = f"training-{uuid.uuid4().hex[:8]}"
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Define step order for progress calculation
    step_order = [
        "select_model",
        "data_collection", 
        "cleaning_and_standardization",
        "label_split_definition",
        "feature_selection_specification",
        "feature_engineering_executor",
        "training_approval",
        "training",
        "generate_report",
    ]
    
    yield {
        "type": "started",
        "node": "init",
        "progress": 0,
        "message": "Training agent started",
        "thread_id": thread_id,
    }
    
    # Stream with "updates" mode to see which node produced what
    for event in agent.stream(initial_state, config=config, stream_mode="updates"):
        # Check for interrupt
        if "__interrupt__" in event:
            interrupt_data = event["__interrupt__"]
            # Extract interrupt info from the Interrupt object
            if interrupt_data and len(interrupt_data) > 0:
                interrupt_obj = interrupt_data[0]
                interrupt_value = interrupt_obj.value if hasattr(interrupt_obj, 'value') else interrupt_obj
                
                # Get the node name and details from the interrupt value
                node_name = interrupt_value.get("node", "unknown") if isinstance(interrupt_value, dict) else "unknown"
                summary = interrupt_value.get("summary", "") if isinstance(interrupt_value, dict) else str(interrupt_value)
                message = interrupt_value.get("message", "Awaiting approval") if isinstance(interrupt_value, dict) else "Awaiting approval"
                state_snapshot = interrupt_value.get("state_snapshot", {}) if isinstance(interrupt_value, dict) else {}
                
                yield {
                    "type": "interrupt",
                    "node": node_name,
                    "summary": summary,
                    "message": message,
                    "state_snapshot": state_snapshot,
                    "thread_id": thread_id,
                }
            else:
                yield {
                    "type": "interrupt",
                    "node": "unknown",
                    "summary": "Step completed",
                    "message": "Awaiting human review",
                    "state_snapshot": {},
                    "thread_id": thread_id,
                }
            return
        
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
                
            elif node_name == "training_approval":
                training_plan = node_output.get("training_plan", {}) or {}
                hyperparams = training_plan.get("hyperparameters", {})
                data_summary = training_plan.get("data_summary", {})
                update["summary"] = {
                    "model_type": training_plan.get("model_type"),
                    "task_type": training_plan.get("task_type"),
                    "hyperparameters": hyperparams,
                    "class_weight": training_plan.get("class_weight"),
                    "max_iterations": training_plan.get("max_iterations"),
                    "strategy_notes": training_plan.get("strategy_notes"),
                    "expected_metrics": training_plan.get("expected_metrics"),
                    "data_summary": data_summary,
                }
                update["details"] = {
                    "title": "Training Configuration Approved",
                    "description": f"Training will use **{training_plan.get('model_type', 'unknown')}** with approved hyperparameters.",
                    "training_plan": training_plan,
                    "hyperparameters": hyperparams,
                    "data_summary": data_summary,
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


def stream_resume_training_agent_with_updates(
    decision: Any,
    thread_id: str,
) -> Generator[dict[str, Any], None, None]:
    """
    Resume streaming the training agent after an interrupt, yielding structured updates.
    
    Args:
        decision: The user's decision (same format as resume_training_agent)
        thread_id: The thread ID from the previous invocation
        
    Yields:
        Update dicts with type, node, progress, and relevant state fields.
        Interrupt events have type="interrupt" with interrupt info.
    """
    agent = create_training_agent()
    config = {"configurable": {"thread_id": thread_id}}
    
    # Define step order for progress calculation
    step_order = [
        "select_model",
        "data_collection", 
        "cleaning",
        "label_split_definition",
        "feature_selection_specification",
        "feature_engineering_executor",
        "training_approval",
        "training",
        "generate_report",
    ]
    
    # Stream with "updates" mode to see which node produced what
    for event in agent.stream(Command(resume=decision), config=config, stream_mode="updates"):
        # Check for interrupt
        if "__interrupt__" in event:
            interrupt_data = event["__interrupt__"]
            # Extract interrupt info
            if interrupt_data and len(interrupt_data) > 0:
                interrupt_value = interrupt_data[0].value if hasattr(interrupt_data[0], 'value') else interrupt_data[0]
                yield {
                    "type": "interrupt",
                    "node": interrupt_value.get("node", "unknown") if isinstance(interrupt_value, dict) else "unknown",
                    "summary": interrupt_value.get("summary", "") if isinstance(interrupt_value, dict) else str(interrupt_value),
                    "message": interrupt_value.get("message", "Awaiting approval") if isinstance(interrupt_value, dict) else "Awaiting approval",
                    "state_snapshot": interrupt_value.get("state_snapshot", {}) if isinstance(interrupt_value, dict) else {},
                    "thread_id": thread_id,
                }
            return
        
        # Process node updates
        for node_name, node_output in event.items():
            # Calculate progress
            try:
                step_idx = step_order.index(node_name)
                progress = int(((step_idx + 1) / len(step_order)) * 100)
            except ValueError:
                progress = 50
            
            # Basic update structure
            update = {
                "type": "node_complete",
                "node": node_name,
                "progress": progress,
                "state": node_output,
                "thread_id": thread_id,
            }
            
            yield update
    
    yield {
        "type": "completed",
        "node": "end",
        "progress": 100,
        "message": "Training completed",
        "thread_id": thread_id,
    }
