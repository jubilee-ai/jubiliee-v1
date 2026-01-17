"""
ML Model Training Agent - LangGraph Structure
Based on the architecture defined in README.md
"""

from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .cleaning_simple import run_cleaning_simple
from .data_collection import data_collection
from .feature_engineering_executor import execute_feature_spec
from .feature_engineering_simple import run_feature_engineering_simple
from .label_and_split import run_label_split_definition
# Import node implementations
from .select_model import select_model

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
    
    # Step 3.5: Label & Split Definition
    label_definition: Optional[LabelDefinition]
    
    # Step 4: Feature Selection & Specification
    feature_spec: Optional[FeatureSpec]
    analysis_trace: list[dict[str, Any]]
    
    # Step 5: Feature Engineering
    transformed_dataset_ref: Optional[str]
    feature_validation_passed: bool
    
    # Step 6: Human Confirmation
    human_confirmed: bool
    
    # Step 7: Training
    training_params: Optional[dict[str, Any]]
    model_weights_path: Optional[str]
    training_metrics: Optional[dict[str, Any]]
    training_iteration: int
    
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
        # Base of 15 iterations + 0.5 per column, capped at 50
        max_iters = min(50, max(15, 15 + int(num_columns * 0.5)))
    except Exception:
        max_iters = 30
    
    result = run_cleaning_simple(
        dataset_ref=state["collected_dataset_ref"],
        goal=state["goal"],
        max_iterations=max_iters,
    )
    
    return {
        **state,
        "cleaned_dataset_ref": result["cleaned_ref"],
        "current_step": "label_split_definition",
    }


def label_split_definition(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 3.5: Label + Split Definition
    
    Uses LLM to infer the 6 key parameters for supervised learning:
    1. Target column
    2. Prediction horizon
    3. Grain (what does one row represent?)
    4. As-of cutoff
    5. Split strategy (random / time-based / entity-based)
    6. Forbidden columns (not available at prediction time)
    
    If label_definition is already set in state, uses those values.
    Otherwise, LLM infers based on goal, model, and schema context.
    
    Locked definitions passed to downstream steps (4, 5, 7)
    """
    # Extract any pre-provided label definition values
    existing = state.get("label_definition") or {}
    
    result = run_label_split_definition(
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
    
    return {
        **state,
        "label_definition": result,
        "current_step": "feature_selection_specification",
    }


def feature_selection_specification(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 4: Feature Selection & Specification
    
    Uses the simple approach: run all analysis tools upfront, then one LLM call.
    
    Inputs from state:
    - cleaned_dataset_ref: The cleaned dataset from step 3
    - goal: The ML goal
    - label_definition: Contains target_column, grain, as_of_cutoff, forbidden_columns
    
    Output: feature_spec (structured contract for step 5)
    """
    # Get inputs from state
    dataset_ref = state.get("cleaned_dataset_ref")
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
    if not dataset_ref:
        raise ValueError("No cleaned_dataset_ref in state - step 3 must complete first")
    if not target_column:
        raise ValueError("No target_column in label_definition - step 3.5 must complete first")
    
    # Run feature engineering
    result = run_feature_engineering_simple(
        dataset_ref=dataset_ref,
        goal=goal,
        target_column=target_column,
        grain=grain,
        task_type=task_type,
        forbidden_columns=forbidden_columns,
        as_of_cutoff=as_of_cutoff,
        prediction_horizon=prediction_horizon,
    )
    
    # Extract feature_spec
    feature_spec = result.get("feature_spec")
    validation = result.get("validation", {})
    analysis_results = result.get("analysis_results", {})
    
    # Check validation
    if validation and not validation.get("valid", True):
        errors = validation.get("errors", [])
        print(f"[WARNING] Feature spec validation issues: {errors}")
    
    # Update state
    return {
        **state,
        "feature_spec": feature_spec,
        "analysis_trace": [
            {
                "step": "feature_selection_specification",
                "analysis_results": analysis_results,
                "validation": validation,
            }
        ],
    }


def feature_engineering_executor(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 5: Feature Engineering Executor (DETERMINISTIC - no LLM reasoning loop)
    
    Inputs from state:
    - cleaned_dataset_ref: The cleaned dataset from step 3
    - feature_spec: The feature specification from step 4
    - label_definition: Contains target_column, grain, as_of_cutoff
    
    Executes each feature in the spec using the appropriate transformation.
    No iterative LLM decision-making; execution follows the spec exactly.
    
    Output: transformed_dataset_ref with all features built
    """
    # Get inputs from state
    dataset_ref = state.get("cleaned_dataset_ref")
    feature_spec = state.get("feature_spec")
    label_def = state.get("label_definition") or {}
    
    target_column = label_def.get("target_column", "")
    grain = label_def.get("grain", "")
    as_of_cutoff = label_def.get("as_of_cutoff")
    
    # Validate inputs
    if not dataset_ref:
        raise ValueError("No cleaned_dataset_ref in state - step 3 must complete first")
    if not feature_spec:
        raise ValueError("No feature_spec in state - step 4 must complete first")
    if not target_column:
        raise ValueError("No target_column in label_definition")
    
    # Execute the feature spec
    result = execute_feature_spec(
        dataset_ref=dataset_ref,
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
    
    # Update state
    return {
        **state,
        "transformed_dataset_ref": result.get("transformed_dataset_ref"),
        "feature_validation_passed": validation_passed,
        "audit_trace": state.get("audit_trace", []) + [
            {
                "step": "feature_engineering_executor",
                "features_created": features_created,
                "errors": errors,
                "shape": result.get("shape"),
                "temporal_constraints_applied": result.get("temporal_constraints_applied", 0),
            }
        ],
    }


def human_confirmation(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 6: Show trace of everything and get human confirmation
    - Present full audit trace
    - Get human to confirm we can proceed to training
    """
    raise NotImplementedError("human_confirmation not implemented")


def training(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 7: Iterative Training Subagent
    Tools: glm + logistic_regression + random_forest + survival_analysis + 
           xgboost_model + model_storage
    
    Inputs from 3.5: split strategy, split indices (already locked)
    
    i. Run LLM analysis on data and goal to decide: parameters, model architecture, 
       type of learning and learning params
    ii. Training:
        a. Apply the split defined in 3.5
        b. Train on initial training set
        c. Evaluate on validation
        d. LLM decides to either:
           → Go back to i (HUMAN IN THE LOOP)
           → Continue to testing (HUMAN IN THE LOOP)
        e. Run model on test set
        f. Review test results and either:
           → Go back to i (HUMAN IN THE LOOP)
           → Complete (HUMAN IN THE LOOP)
    iii. Complete and output weights file + audit trace, explanations
    """
    raise NotImplementedError("training not implemented")


def generate_report(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 8: Generate Report
    - Generate a report on everything to accompany the weights file
    """
    raise NotImplementedError("generate_report not implemented")


# =============================================================================
# CONDITIONAL EDGE FUNCTIONS
# =============================================================================

def should_regen_model(state: TrainingAgentState) -> Literal["regen", "continue"]:
    """Check if user wants to regenerate model selection (max 3 times)"""
    raise NotImplementedError("should_regen_model not implemented")


def should_skip_label_definition(state: TrainingAgentState) -> Literal["skip", "define"]:
    """Check if label/split definition is relevant or should be skipped"""
    raise NotImplementedError("should_skip_label_definition not implemented")


def feature_validation_result(state: TrainingAgentState) -> Literal["passed", "failed"]:
    """Check if feature engineering validation passed or needs spec revision"""
    raise NotImplementedError("feature_validation_result not implemented")


def human_confirmed_proceed(state: TrainingAgentState) -> Literal["proceed", "abort"]:
    """Check if human confirmed to proceed to training"""
    raise NotImplementedError("human_confirmed_proceed not implemented")


def training_decision(state: TrainingAgentState) -> Literal["iterate", "complete"]:
    """Check if training should iterate or is complete"""
    raise NotImplementedError("training_decision not implemented")


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
    
    # Step 7 → Step 8 or iterate (training loop with human checkpoints)
    graph.add_conditional_edges(
        "training",
        training_decision,
        {
            "iterate": "training",  # Back to training for another iteration
            "complete": "generate_report"
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


def invoke_training_agent(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None
) -> TrainingAgentState:
    """
    Invoke the training agent with the given inputs.
    
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
    
    initial_state: TrainingAgentState = {
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
        "report_path": None,
        "audit_trace": [],
        "explanations": [],
        "current_step": "select_model",
        "error": None,
    }
    
    return agent.invoke(initial_state)
