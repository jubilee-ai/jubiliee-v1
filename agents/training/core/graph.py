"""
Graph construction and compilation for the ML Training Agent.

Architecture: Planner → Dispatcher → Step → Evaluator loop.
The planner generates a dynamic execution plan, the dispatcher routes to
each step, and the evaluator decides whether to continue, amend, replan,
or finish.
"""

import uuid
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from .conversation_context import normalize_conversation_turns
from .dispatcher import dispatcher_node
from .edges import route_to_step, should_continue
from .evaluator import evaluator_node
from .planner import planner_node
from ..steps.orchestrator import (
    cleaning_node,
    data_collection,
    feature_engineering_executor,
    feature_experiment_runner,
    feature_selection_specification,
    generate_report,
    label_split_definition,
    select_model,
    training,
    training_approval,
)
from .state import TrainingAgentState, create_initial_state

ALL_STEP_NAMES = [
    "data_collection",
    "select_model",
    "cleaning",
    "label_split_definition",
    "feature_selection_specification",
    "feature_engineering_executor",
    "feature_experiment_runner",
    "training_approval",
    "training",
    "generate_report",
]


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def build_training_agent_graph() -> StateGraph:
    """
    Build the ML Training Agent graph with planner + executor + evaluator.

    Flow:
      START → planner (HITL) → dispatcher → [step node] (HITL) → evaluator
        ├─ continue → dispatcher (next step)
        ├─ replan  → planner (new plan)
        └─ done    → END
    """

    graph = StateGraph(TrainingAgentState)

    # -------------------------------------------------------------------------
    # ADD NODES
    # -------------------------------------------------------------------------

    # Agentic control nodes
    graph.add_node("planner", planner_node)
    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("evaluator", evaluator_node)

    # Pipeline step nodes (unchanged implementations)
    graph.add_node("data_collection", data_collection)
    graph.add_node("select_model", select_model)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("label_split_definition", label_split_definition)
    graph.add_node("feature_selection_specification", feature_selection_specification)
    graph.add_node("feature_engineering_executor", feature_engineering_executor)
    graph.add_node("feature_experiment_runner", feature_experiment_runner)
    graph.add_node("training_approval", training_approval)
    graph.add_node("training", training)
    graph.add_node("generate_report", generate_report)

    # -------------------------------------------------------------------------
    # ADD EDGES
    # -------------------------------------------------------------------------

    # Entry: always start with the planner
    graph.set_entry_point("planner")

    # Planner → Dispatcher (plan approved, start executing)
    graph.add_edge("planner", "dispatcher")

    # Dispatcher → step node (dynamic routing based on current_step)
    step_routing = {name: name for name in ALL_STEP_NAMES}
    step_routing["done"] = END
    graph.add_conditional_edges("dispatcher", route_to_step, step_routing)

    # Every step node → evaluator
    for step_name in ALL_STEP_NAMES:
        graph.add_edge(step_name, "evaluator")

    # Evaluator → dispatcher (continue), planner (replan), or END (done)
    graph.add_conditional_edges(
        "evaluator",
        should_continue,
        {
            "continue": "dispatcher",
            "replan": "planner",
            "done": END,
        },
    )

    return graph


# =============================================================================
# GRAPH COMPILATION & INVOCATION
# =============================================================================

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
    conversation: Optional[list[dict]] = None,
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
    g = (goal or "").strip() or "Training run"
    turns = normalize_conversation_turns(conversation, triggering_message=g)
    initial_state = create_initial_state(
        g,
        linked_datasets,
        user_model_preference,
        conversation_history=turns,
    )

    if not thread_id:
        thread_id = f"training-{uuid.uuid4().hex[:8]}"

    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(initial_state, config=config)

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

    if isinstance(result, dict):
        result["_thread_id"] = thread_id

    return result
