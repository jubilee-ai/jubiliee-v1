"""
Conditional edge functions for the ML Training Agent graph.
These functions determine routing between nodes based on state.
"""

from typing import Literal

from .state import TrainingAgentState


def should_regen_model(state: TrainingAgentState) -> Literal["regen", "continue"]:
    """Check if user wants to regenerate model selection (max 3 times)"""
    # For automated runs, always continue with selected model
    regen_count = state.get("model_regen_count", 0)
    if regen_count >= 3:
        print("[should_regen_model] Max regen count reached, continuing...")
        return "continue"
    return "continue"


def data_collection_result(state: TrainingAgentState) -> Literal["success", "retry"]:
    """Route to cleaning only if data collection produced a dataset ref."""
    if state.get("collected_dataset_ref"):
        return "success"
    print("[data_collection_result] No collected_dataset_ref — routing back to data_collection")
    return "retry"


def should_skip_label_definition(state: TrainingAgentState) -> Literal["skip", "define"]:
    """Check if label/split definition is relevant or should be skipped"""
    # Always define labels for supervised learning
    return "define"


def feature_validation_result(state: TrainingAgentState) -> Literal["passed", "failed"]:
    """Check if feature engineering validation passed or needs spec revision"""
    if state.get("transformed_train_ref"):
        return "passed"
    return "failed"


def training_decision(
    state: TrainingAgentState,
) -> Literal["iterate", "complete", "redo_features"]:
    """Check if training should iterate, complete, or redo feature engineering."""

    # Check if feature engineering redo was requested
    if state.get("feature_redo_requested"):
        redo_iteration = state.get("feature_redo_iteration", 0)
        # Limit feature redo iterations to prevent infinite loops
        if redo_iteration >= 2:
            print(
                "[training_decision] Max feature redo iterations (2) reached, completing..."
            )
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
