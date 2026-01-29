"""
Core graph infrastructure for the ML Training Agent.

This module contains:
- state.py: State definitions and constants
- graph.py: Graph construction and compilation
- edges.py: Conditional routing functions
- hitl.py: Human-in-the-loop utilities
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
    feature_validation_result,
    should_regen_model,
    should_skip_label_definition,
    training_decision,
)

# Note: graph module is imported lazily to avoid circular imports
# with steps.orchestrator. Use:
#   from agents.training.core.graph import create_training_agent
# Or import from agents.training.agent

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
    "should_regen_model",
    "should_skip_label_definition",
    "feature_validation_result",
    "training_decision",
]
