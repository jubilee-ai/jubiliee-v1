"""
ML Model Training Agent - LangGraph Structure
Based on the architecture defined in README.md
"""

from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .cleaning_simple import run_cleaning_simple
from .data_collection import data_collection
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
    result = run_cleaning_simple(
        dataset_ref=state["collected_dataset_ref"],
        goal=state["goal"],
    )
    
    return {
        **state,
        "cleaned_dataset_ref": result["cleaned_ref"],
        "current_step": "label_split_definition",
    }


def label_split_definition(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 3.5: Label + Split Definition (One-time, Human-driven)
    - SKIP IF NOT RELEVANT
    
    Agent presents dataset schema and asks user to define:
    1. Target column
    2. Prediction horizon
    3. Grain (what does one row represent?)
    4. As-of cutoff
    5. Split strategy (random / time-based / entity-based)
    6. Forbidden columns (not available at prediction time)
    
    User can choose:
    - Manual mode: Answer each question directly
    - Auto-fill mode: Agent infers, user reviews and confirms
    
    Locked definitions passed to downstream steps (4, 5, 7)
    """
    raise NotImplementedError("label_split_definition not implemented")


def feature_selection_specification(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 4: Iterative Feature Selection & Specification Agent
    Tools: concentration_analysis + correlation_matrix + data_validation + 
           distribution_analysis + eda_report + feature_diagnostics + 
           group_summary + trend_analysis
    
    Inputs from 3.5: target column, as-of cutoff, forbidden columns list
    
    - Choose which features to use AND specify how to build them
    - Excludes forbidden columns (future-leaking features)
    - Access to all analysis tools for statistical analysis
    - Iterates until confident, outputs feature_spec
    - Human in the loop to verify or comment on features chosen
    
    Output: feature_spec (structured contract for step 5)
    """
    raise NotImplementedError("feature_selection_specification not implemented")


def feature_engineering_executor(state: TrainingAgentState) -> TrainingAgentState:
    """
    Step 5: Feature Engineering Executor (DETERMINISTIC - no LLM reasoning loop)
    Tools: feature_ops, agg_ops, row_ops, column_ops
    
    Inputs: feature_spec from step 4, as-of cutoff, grain, split indices from 3.5
    
    i. Parse the feature_spec
    ii. For each feature in spec, call appropriate tool with specified parameters
    iii. Apply transformations (respecting as-of dates to avoid leakage)
    iv. Run feature_diagnostics tests to validate
    v. If validation fails → return error to step 4 for spec revision
    vi. If validation passes → output transformed dataset
    
    Failures are spec failures, not execution failures
    """
    raise NotImplementedError("feature_engineering_executor not implemented")


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
