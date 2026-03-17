"""
Core graph infrastructure for the ML Training Agent.

This module contains:
- state.py: State definitions and constants
- graph.py: Graph construction and compilation (planner + executor + evaluator)
- edges.py: Conditional routing functions
- hitl.py: Human-in-the-loop utilities
- planner.py: LLM-based plan generation
- evaluator.py: LLM-based step evaluation and plan amendment
- dispatcher.py: Pure routing from plan to step nodes
"""

from .state import (
    STEP_ORDER,
    STATE_SNAPSHOT_KEYS,
    FeatureSpec,
    LabelDefinition,
    TrainingAgentState,
    create_initial_state,
)

from .hitl import (
    extract_state_snapshot,
    make_serializable,
    parse_decision,
    run_with_hitl,
)

from .edges import (
    route_to_step,
    should_continue,
)

__all__ = [
    # State
    "TrainingAgentState",
    "LabelDefinition",
    "FeatureSpec",
    "STEP_ORDER",
    "STATE_SNAPSHOT_KEYS",
    "create_initial_state",
    # HITL
    "run_with_hitl",
    "make_serializable",
    "parse_decision",
    "extract_state_snapshot",
    # Edges
    "route_to_step",
    "should_continue",
]
