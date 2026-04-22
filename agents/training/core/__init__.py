"""
Core infrastructure for the ML Training Agent.

This module contains:
- state.py: State definitions and constants
- graph.py: invoke_training_agent entrypoint (unified LLM executor in agent_simple)
- edges.py: Legacy conditional routing helpers
- hitl.py: Human-in-the-loop utilities
- planner.py: Legacy plan generation (not used by unified chat training)
- evaluator.py: Legacy step evaluation (not used by unified chat training)
- dispatcher.py: Legacy plan-index routing (not used by unified chat training)
"""

from .pipeline import (
    DEFAULT_TRAINING_RECAP,
    UNIFIED_PIPELINE_STEP_NAMES,
)
from .state import (
    STEP_ORDER,
    STATE_SNAPSHOT_KEYS,
    FeatureSpec,
    LabelDefinition,
    TrainingAgentState,
    create_initial_state,
)
from .task_inference import (
    infer_supervised_task_type_from_target_column,
    infer_task_type,
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
    # Pipeline
    "UNIFIED_PIPELINE_STEP_NAMES",
    "DEFAULT_TRAINING_RECAP",
    # Task inference
    "infer_task_type",
    "infer_supervised_task_type_from_target_column",
    # HITL
    "run_with_hitl",
    "make_serializable",
    "parse_decision",
    "extract_state_snapshot",
    # Edges
    "route_to_step",
    "should_continue",
]
