"""
ML Model Training Agent - LangGraph Structure
Based on the architecture defined in README.md

This module provides the main entry point for the training agent.
All implementation details are organized in submodules:
- core/state.py: State definitions and constants
- core/hitl.py: Human-in-the-loop utilities
- core/edges.py: Conditional routing functions
- core/graph.py: Graph construction and compilation
- steps/orchestrator.py: Node function implementations
- utils/streaming.py: Streaming functions for real-time updates
- utils/prompts.py: System prompts
"""

# =============================================================================
# STATE DEFINITIONS
# =============================================================================
from .core.state import (
    STEP_ORDER,
    STATE_SNAPSHOT_KEYS,
    FeatureSpec,
    LabelDefinition,
    TrainingAgentState,
    create_initial_state,
)

# =============================================================================
# HITL UTILITIES
# =============================================================================
from .core.hitl import (
    extract_state_snapshot,
    make_serializable,
    parse_decision,
    run_with_hitl,
)

# =============================================================================
# NODE FUNCTIONS
# Note: Imported lazily to avoid circular imports
# =============================================================================
def _import_nodes():
    """Lazy import node functions to avoid circular imports."""
    from .steps.orchestrator import (
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
    return {
        'cleaning_node': cleaning_node,
        'data_collection': data_collection,
        'feature_engineering_executor': feature_engineering_executor,
        'feature_selection_specification': feature_selection_specification,
        'generate_report': generate_report,
        'label_split_definition': label_split_definition,
        'select_model': select_model,
        'training': training,
        'training_approval': training_approval,
    }

_nodes = None
def _get_nodes():
    global _nodes
    if _nodes is None:
        _nodes = _import_nodes()
    return _nodes

def cleaning_node(*args, **kwargs):
    return _get_nodes()['cleaning_node'](*args, **kwargs)

def data_collection(*args, **kwargs):
    return _get_nodes()['data_collection'](*args, **kwargs)

def feature_engineering_executor(*args, **kwargs):
    return _get_nodes()['feature_engineering_executor'](*args, **kwargs)

def feature_selection_specification(*args, **kwargs):
    return _get_nodes()['feature_selection_specification'](*args, **kwargs)

def generate_report(*args, **kwargs):
    return _get_nodes()['generate_report'](*args, **kwargs)

def label_split_definition(*args, **kwargs):
    return _get_nodes()['label_split_definition'](*args, **kwargs)

def select_model(*args, **kwargs):
    return _get_nodes()['select_model'](*args, **kwargs)

def training(*args, **kwargs):
    return _get_nodes()['training'](*args, **kwargs)

def training_approval(*args, **kwargs):
    return _get_nodes()['training_approval'](*args, **kwargs)

# =============================================================================
# EDGE FUNCTIONS
# =============================================================================
from .core.edges import (
    feature_validation_result,
    should_regen_model,
    should_skip_label_definition,
    training_decision,
)

# =============================================================================
# GRAPH CONSTRUCTION & INVOCATION
# Note: Imported lazily to avoid circular imports
# =============================================================================
def _import_graph():
    """Lazy import graph functions to avoid circular imports."""
    from .core.graph import (
        build_training_agent_graph,
        create_training_agent,
        invoke_training_agent,
        resume_training_agent,
    )
    return {
        'build_training_agent_graph': build_training_agent_graph,
        'create_training_agent': create_training_agent,
        'invoke_training_agent': invoke_training_agent,
        'resume_training_agent': resume_training_agent,
    }

_graph = None
def _get_graph():
    global _graph
    if _graph is None:
        _graph = _import_graph()
    return _graph

def build_training_agent_graph(*args, **kwargs):
    return _get_graph()['build_training_agent_graph'](*args, **kwargs)

def create_training_agent(*args, **kwargs):
    return _get_graph()['create_training_agent'](*args, **kwargs)

def invoke_training_agent(*args, **kwargs):
    return _get_graph()['invoke_training_agent'](*args, **kwargs)

def resume_training_agent(*args, **kwargs):
    return _get_graph()['resume_training_agent'](*args, **kwargs)

# =============================================================================
# STREAMING FUNCTIONS
# Note: Imported after other modules to avoid circular import
# =============================================================================
def _import_streaming():
    """Lazy import streaming functions to avoid circular imports."""
    from .utils.streaming import (
        build_node_update,
        calculate_progress,
        extract_interrupt_info,
        stream_resume_training_agent,
        stream_resume_training_agent_with_updates,
        stream_training_agent,
        stream_training_agent_with_updates,
    )
    return {
        'build_node_update': build_node_update,
        'calculate_progress': calculate_progress,
        'extract_interrupt_info': extract_interrupt_info,
        'stream_resume_training_agent': stream_resume_training_agent,
        'stream_resume_training_agent_with_updates': stream_resume_training_agent_with_updates,
        'stream_training_agent': stream_training_agent,
        'stream_training_agent_with_updates': stream_training_agent_with_updates,
    }

# Import streaming functions lazily
_streaming = None
def _get_streaming():
    global _streaming
    if _streaming is None:
        _streaming = _import_streaming()
    return _streaming

def stream_training_agent(*args, **kwargs):
    return _get_streaming()['stream_training_agent'](*args, **kwargs)

def stream_resume_training_agent(*args, **kwargs):
    return _get_streaming()['stream_resume_training_agent'](*args, **kwargs)

def stream_training_agent_with_updates(*args, **kwargs):
    return _get_streaming()['stream_training_agent_with_updates'](*args, **kwargs)

def stream_resume_training_agent_with_updates(*args, **kwargs):
    return _get_streaming()['stream_resume_training_agent_with_updates'](*args, **kwargs)

def calculate_progress(*args, **kwargs):
    return _get_streaming()['calculate_progress'](*args, **kwargs)

def extract_interrupt_info(*args, **kwargs):
    return _get_streaming()['extract_interrupt_info'](*args, **kwargs)

def build_node_update(*args, **kwargs):
    return _get_streaming()['build_node_update'](*args, **kwargs)

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Alias for the old private function name
_make_serializable = make_serializable
_create_initial_state = create_initial_state


# =============================================================================
# PUBLIC API
# =============================================================================

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
    # Nodes
    "select_model",
    "data_collection",
    "cleaning_node",
    "label_split_definition",
    "feature_selection_specification",
    "feature_engineering_executor",
    "training_approval",
    "training",
    "generate_report",
    # Edges
    "should_regen_model",
    "should_skip_label_definition",
    "feature_validation_result",
    "training_decision",
    # Graph
    "build_training_agent_graph",
    "create_training_agent",
    "invoke_training_agent",
    "resume_training_agent",
    # Streaming
    "stream_training_agent",
    "stream_resume_training_agent",
    "stream_training_agent_with_updates",
    "stream_resume_training_agent_with_updates",
    "calculate_progress",
    "extract_interrupt_info",
    "build_node_update",
]
