"""
Graph construction and compilation for the ML Training Agent.
"""

import uuid
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from .edges import (
    data_collection_result,
    feature_validation_result,
    should_regen_model,
    should_skip_label_definition,
    training_decision,
)
from ..steps.orchestrator import (
    cleaning_node,
    data_collection,
    feature_engineering_executor,
    feature_selection_specification,
    generate_report,
    label_split_definition,
    select_model,
    training,
    training_approval,
)
from .state import TrainingAgentState, create_initial_state


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
    graph.add_node("training_approval", training_approval)  # Approval before training
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
        {"regen": "select_model", "continue": "data_collection"},  # Loop back for regeneration
    )

    # Step 2 → Step 3 (only if data collection succeeded)
    graph.add_conditional_edges(
        "data_collection",
        data_collection_result,
        {"success": "cleaning", "retry": "data_collection"},
    )

    # Step 3 → Step 3.5 (cleaning handles its own iteration internally)
    graph.add_edge("cleaning", "label_split_definition")

    # Step 3.5 → Step 4 (with skip option)
    graph.add_conditional_edges(
        "label_split_definition",
        should_skip_label_definition,
        {
            "skip": "feature_selection_specification",
            "define": "feature_selection_specification",  # Both go to step 4, but with different state
        },
    )

    # Step 4 → Step 5
    graph.add_edge("feature_selection_specification", "feature_engineering_executor")

    # Step 5 → Step 6 (training_approval) or back to Step 4 (validation check)
    graph.add_conditional_edges(
        "feature_engineering_executor",
        feature_validation_result,
        {
            "passed": "training_approval",  # Go to approval step first
            "failed": "feature_selection_specification",  # Back to step 4 for spec revision
        },
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
            "redo_features": "feature_selection_specification",  # Loop back to feature engineering
        },
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
    agent = create_training_agent()
    initial_state = create_initial_state(goal, linked_datasets, user_model_preference)

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
