"""
State definitions and constants for the ML Training Agent.
"""

from typing import Any, Literal, Optional, TypedDict


# =============================================================================
# CONSTANTS
# =============================================================================

# Keys to include in state snapshots for HITL review
STATE_SNAPSHOT_KEYS = [
    "selected_model",
    "task_type",
    "model_explanation",
    "collected_dataset_ref",
    "cleaned_dataset_ref",
    "cleaning_summary",
    "label_definition",
    "feature_spec",
    "analysis_trace",
    "transformed_train_ref",
    "transformed_val_ref",
    "transformed_test_ref",
    "feature_validation_passed",
    "feature_pipeline_mode",
    "experiment_result",
    "feature_rankings",
    "audit_trace",
    "training_plan",
    "training_metrics",
    "report_path",
    "error",
    "plan",
    "plan_index",
    "plan_strategy",
]

# Step order for progress calculation and routing
STEP_ORDER = [
    "select_model",
    "data_collection",
    "cleaning",
    "label_split_definition",
    "feature_specification_and_engineering",
    "feature_experiment_runner",
    "evaluate_models",
    "training_approval",
    "training",
    "generate_report",
]


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

    features: list[
        dict[str, Any]
    ]  # Each feature has: name, formula, source_tables, window, grain, as_of_constraint, encoding


class TrainingAgentState(TypedDict):
    """Main state object passed through the graph"""

    # Inputs
    goal: str
    linked_datasets: Optional[list[str]]
    user_model_preference: Optional[str]
    # Multi-turn lab chat (user / agent) used by the planner to synthesize the run
    conversation_history: list[dict[str, str]]

    # Step 1: Model Family Selection (supervised / unsupervised / neural_networks)
    selected_model: Optional[str]
    model_explanation: Optional[str]
    model_regen_count: int

    # Derived from selected_model + goal: "classification" | "regression" | "unsupervised"
    task_type: Optional[str]

    # Pre-resolved inputs — set before graph starts, corresponding steps auto-skip
    resolved_dataset_ref: str | None
    resolved_model_type: str | None
    resolved_target_column: str | None

    # Step 2: Data Collection
    collected_dataset_ref: Optional[str]
    data_source: Optional[str]  # "local", "kaggle", "huggingface", "pre-registered"
    use_external_sources: bool

    # Step 3: Cleaning & Standardization
    cleaned_dataset_ref: Optional[str]
    cleaning_transformations: list[dict[str, Any]]
    cleaning_summary: Optional[str]  # Summary message from cleaning agent

    # Step 3.5: Label & Split Definition
    label_definition: Optional[LabelDefinition]
    split_indices: Optional[dict[str, Any]]  # Output of compute_split_indices
    train_dataset_ref: Optional[str]  # Registered train dataset
    val_dataset_ref: Optional[str]  # Registered val dataset
    test_dataset_ref: Optional[str]  # Registered test dataset

    # Step 4: Feature Selection & Specification
    feature_spec: Optional[FeatureSpec]
    analysis_trace: list[dict[str, Any]]

    # Step 5: Feature Engineering
    transformed_dataset_ref: Optional[str]  # Legacy - for backward compat
    transformed_train_ref: Optional[str]  # NEW: transformed train dataset
    transformed_val_ref: Optional[str]  # NEW: transformed val dataset
    transformed_test_ref: Optional[str]  # NEW: transformed test dataset
    feature_validation_passed: bool
    # "engineered" = spec executed; "passthrough" = unsupervised / use cleaned table as-is
    feature_pipeline_mode: Optional[str]

    # Step 5.5: Feature Experiment Runner
    experiment_result: Optional[dict[str, Any]]
    feature_rankings: Optional[dict[str, float]]
    experiment_grid_summary: Optional[list[dict[str, Any]]]

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

    # Agentic planner
    plan: Optional[list[dict[str, Any]]]
    plan_index: int
    plan_history: list[dict[str, Any]]
    plan_strategy: Optional[str]
    evaluator_decision: Optional[str]

    # Control flow
    current_step: str
    error: Optional[str]
    hitl_auto_approve: bool  # skip HITL when True (background async tasks)


# =============================================================================
# STATE FACTORY
# =============================================================================


def create_initial_state(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None,
    use_external_sources: bool = False,
    resolved_dataset_ref: str | None = None,
    resolved_model_type: str | None = None,
    resolved_target_column: str | None = None,
    conversation_history: Optional[list[dict[str, str]]] = None,
    hitl_auto_approve: bool = False,
) -> TrainingAgentState:
    """Create the initial state for the training agent."""
    return {
        # Inputs
        "goal": goal,
        "linked_datasets": linked_datasets,
        "user_model_preference": user_model_preference,
        "conversation_history": list(conversation_history or []),
        "use_external_sources": use_external_sources,
        # Pre-resolved inputs
        "resolved_dataset_ref": resolved_dataset_ref,
        "resolved_model_type": resolved_model_type,
        "resolved_target_column": resolved_target_column,
        # Initialize all other fields
        "selected_model": None,
        "model_explanation": None,
        "model_regen_count": 0,
        "task_type": None,
        "collected_dataset_ref": None,
        "data_source": None,
        "cleaned_dataset_ref": None,
        "cleaning_transformations": [],
        "cleaning_summary": None,
        "label_definition": None,
        "split_indices": None,
        "train_dataset_ref": None,
        "val_dataset_ref": None,
        "test_dataset_ref": None,
        "feature_spec": None,
        "analysis_trace": [],
        "transformed_dataset_ref": None,
        "transformed_train_ref": None,
        "transformed_val_ref": None,
        "transformed_test_ref": None,
        "feature_validation_passed": False,
        "feature_pipeline_mode": None,
        "experiment_result": None,
        "feature_rankings": None,
        "experiment_grid_summary": None,
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
        "plan": None,
        "plan_index": 0,
        "plan_history": [],
        "plan_strategy": None,
        "evaluator_decision": None,
        "current_step": "planner",
        "error": None,
        "hitl_auto_approve": hitl_auto_approve,
    }
