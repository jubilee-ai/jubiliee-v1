"""
Training Agent - Step 7 of the ML Training Pipeline.

Uses LangChain's create_agent to orchestrate model training with:
- Train on training set
- Evaluate on validation set
- LLM decides to iterate or proceed to testing
- Final evaluation on test set

Tools: logistic_regression, random_forest, xgboost, glm, survival_analysis, model_storage
"""

import sys
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..utils.prompts import TRAINING_SYSTEM_PROMPT

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# TODO: More hyperparameters (e.g. sample weights... and for each model...)

# Add tools path
# Path: steps -> training -> agents -> root -> tools/models-tools/training
_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "models-tools" / "training"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# Add data-tools path for utils
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from glm import sklearn_glm_tool
# Import training tools
from logistic_regression import sklearn_logistic_regression_tool
from model_storage import (delete_model, evaluate_model_tool,
                           get_model_info_tool, list_models,
                           list_trained_models_tool, predict_with_model_tool)
from random_forest import sklearn_random_forest_tool
from survival_analysis import survival_analysis_tool
from utils import get_registered_dataset
from xgboost_model import xgboost_train_tool

# TODO: Human in the loop

# =============================================================================
# FEATURE ENGINEERING REDO TOOL
# =============================================================================


class FeatureRedoRequest(BaseModel):
    """Request to redo feature engineering with specific recommendations."""
    recommendation: str = Field(
        description="Specific recommendation for what to change in feature engineering. "
        "Be specific about which features to add, remove, or modify and why."
    )
    reason: str = Field(
        description="Why you believe the current features are limiting model performance. "
        "Include evidence from training results (e.g., specific metrics, patterns observed)."
    )
    suspected_issues: list[str] = Field(
        default_factory=list,
        description="List of suspected feature issues: "
        "'missing_interactions', 'high_cardinality', 'data_leakage', 'irrelevant_features', "
        "'missing_transformations', 'scale_issues', 'temporal_issues'"
    )


# Global variable to store feature redo request (set by tool, read by run_training_agent)
_feature_redo_request: Optional[FeatureRedoRequest] = None


def _get_and_clear_feature_redo_request() -> Optional[FeatureRedoRequest]:
    """Get the feature redo request and clear it."""
    global _feature_redo_request
    request = _feature_redo_request
    _feature_redo_request = None
    return request

@tool
def request_feature_engineering_redo_tool(
    recommendation: str,
    reason: str,
    suspected_issues: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Request to redo feature engineering with specific recommendations.
    
    IMPORTANT: Only call this tool when you have strong evidence that the current features
    are the bottleneck limiting model performance. You must have tried multiple models
    and hyperparameter configurations first.
    
    Valid reasons to call this tool:
    - All model types show similar poor performance despite tuning
    - Feature diagnostics show high correlation or potential leakage
    - Performance is far below expected baseline for the task
    - You've exhausted reasonable hyperparameter tuning
    
    Args:
        recommendation: Specific recommendation for what to change in feature engineering.
        reason: Why you believe features are limiting performance.
        suspected_issues: List of suspected issues like 'missing_interactions', 
                         'high_cardinality', 'irrelevant_features', etc.
    
    Returns:
        Confirmation that the request was registered.
    """

    global _feature_redo_request
    
    _feature_redo_request = FeatureRedoRequest(
        recommendation=recommendation,
        reason=reason,
        suspected_issues=suspected_issues or [],
    )
    
    return {
        "status": "registered",
        "message": "Feature engineering redo requested. Training will stop after this call.",
        "recommendation": recommendation,
        "reason": reason,
        "suspected_issues": suspected_issues or [],
    }


# =============================================================================
# TRAINING TOOLS
# =============================================================================

TRAINING_TOOLS = [
    sklearn_logistic_regression_tool,
    sklearn_random_forest_tool,
    xgboost_train_tool,
    sklearn_glm_tool,
    survival_analysis_tool,
    list_trained_models_tool,
    predict_with_model_tool,
    get_model_info_tool,
    evaluate_model_tool,
    request_feature_engineering_redo_tool,
]

# TODO: Add ML standard practice here or as a skill
# TODO: Only stop when optimized as much as you can... --> if we have blockers from data or features we'll loop back to previous steps


# =============================================================================
# TRAINING RESULT SCHEMA
# =============================================================================

class TrainingIteration(BaseModel):
    """A single training iteration attempt."""
    model_name: str = Field(description="Name of the model for this iteration")
    tool_used: str = Field(description="Training tool used (e.g., sklearn_logistic_regression)")
    hyperparams: dict = Field(default_factory=dict, description="Hyperparameters used")
    # Classification metrics
    train_accuracy: Optional[float] = Field(default=None, description="Training accuracy")
    val_accuracy: Optional[float] = Field(default=None, description="Validation accuracy from evaluate_model")
    val_roc_auc: Optional[float] = Field(default=None, description="Validation ROC-AUC from evaluate_model")
    # Regression metrics
    train_r2: Optional[float] = Field(default=None, description="Training R² score")
    val_r2: Optional[float] = Field(default=None, description="Validation R² score")
    val_rmse: Optional[float] = Field(default=None, description="Validation RMSE")
    val_mae: Optional[float] = Field(default=None, description="Validation MAE")
    test_r2: Optional[float] = Field(default=None, description="Test R² score")
    test_rmse: Optional[float] = Field(default=None, description="Test RMSE")
    test_mae: Optional[float] = Field(default=None, description="Test MAE")
    success: bool = Field(description="Whether this iteration succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class TrainingResult(BaseModel):
    """Structured output for training completion."""
    success: bool = Field(description="Whether training completed successfully")
    best_model_name: str = Field(description="Name of the best trained model")
    model_type: str = Field(description="Type of model trained (e.g., logistic_regression, random_forest)")
    
    # Classification metrics
    val_accuracy: Optional[float] = Field(default=None, description="Best model validation accuracy")
    val_roc_auc: Optional[float] = Field(default=None, description="Best model validation ROC-AUC")
    test_accuracy: Optional[float] = Field(default=None, description="Best model test accuracy")
    test_roc_auc: Optional[float] = Field(default=None, description="Best model test ROC-AUC")
    
    # Regression metrics
    train_r2: Optional[float] = Field(default=None, description="Training R² score")
    val_r2: Optional[float] = Field(default=None, description="Validation R² score")
    val_rmse: Optional[float] = Field(default=None, description="Validation RMSE")
    val_mae: Optional[float] = Field(default=None, description="Validation MAE")
    test_r2: Optional[float] = Field(default=None, description="Test R² score")
    test_rmse: Optional[float] = Field(default=None, description="Test RMSE")
    test_mae: Optional[float] = Field(default=None, description="Test MAE")
    
    # Iteration tracking
    iterations: list[TrainingIteration] = Field(default_factory=list, description="List of all training iterations attempted")
    num_iterations: int = Field(description="Total number of training iterations attempted")
    
    # Summary
    summary: str = Field(description="Summary of training process and results")
    recommendations: Optional[str] = Field(default=None, description="Recommendations for improvement")
    
    # Feature engineering redo request
    feature_redo_requested: bool = Field(
        default=False, 
        description="Whether a feature engineering redo was requested via request_feature_engineering_redo tool"
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _prepare_data_for_tool(df) -> list[dict]:
    """Convert DataFrame to list of dicts for tool input."""
    return df.to_dict(orient="records")


def _cleanup_intermediate_models(
    best_model_name: str,
    base_model_name: str,
    iteration_model_names: Optional[list[str]] = None,
) -> list[str]:
    """
    Clean up intermediate model versions, keeping only the best model.
    
    Deletes all models that:
    1. Match the base_model_name pattern (e.g., 'model_v1', 'model_v2')
    2. Are listed in iteration_model_names (models created during this training run)
    
    Args:
        best_model_name: The model to keep (e.g., 'logistic_regression_123_v2')
        base_model_name: The base name used for this training run (e.g., 'logistic_regression_123')
        iteration_model_names: Optional list of all model names created during training iterations
    
    Returns:
        List of deleted model names
    """
    import re
    
    deleted = []
    all_models = list_models()
    
    # Pattern to match versioned models: base_name_vN
    base_pattern = re.escape(base_model_name)
    version_pattern = rf"^{base_pattern}_v\d+$"
    
    # Set of models to potentially delete (from iterations, if provided)
    iteration_models_set = set(iteration_model_names) if iteration_model_names else set()
    
    for model_info in all_models:
        model_name = model_info["model_name"]
        
        # Skip the best model - never delete it
        if model_name == best_model_name:
            continue
        
        should_delete = False
        
        # Check if this model matches our base pattern (versioned)
        if re.match(version_pattern, model_name):
            should_delete = True
        
        # Check if this model was created during this training run
        if model_name in iteration_models_set:
            should_delete = True
        
        if should_delete:
            if delete_model(model_name):
                deleted.append(model_name)
                print(f"[training_agent] Cleaned up intermediate model: {model_name}")
    
    return deleted


def _get_task_type(selected_model: str, goal: str) -> Literal["classification", "regression"]:
    """Infer task type from model selection and goal."""
    goal_lower = goal.lower()
    model_lower = selected_model.lower()
    
    # Regression indicators
    if any(word in goal_lower for word in ["regress", "predict value", "forecast", "amount", "price", "cost"]):
        return "regression"
    if any(word in model_lower for word in ["glm", "regression"]) and "logistic" not in model_lower:
        return "regression"
    
    # Default to classification
    return "classification"


def _get_available_models(task_type: str) -> list[dict[str, str]]:
    """
    Get all models available for a given task type.
    
    Returns list of dicts with 'tool' name and 'label' for display.
    """
    classification_models = [
        {"tool": "sklearn_logistic_regression", "label": "Logistic Regression"},
        {"tool": "sklearn_random_forest", "label": "Random Forest"},
        {"tool": "xgboost_train", "label": "XGBoost"},
    ]
    
    regression_models = [
        {"tool": "sklearn_random_forest", "label": "Random Forest"},
        {"tool": "xgboost_train", "label": "XGBoost"},
        {"tool": "sklearn_glm", "label": "GLM"},
    ]
    
    if task_type == "classification":
        return classification_models
    return regression_models


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

# TODO: Go back to change feature engineering with comments if need be
# TODO: More evaluation metrics
# TODO: Polish the process to match best ML practices
def run_training_agent(
    train_ref: str,
    val_ref: str,
    test_ref: str,
    target_column: str,
    selected_model: str,
    goal: str,
    model_name: Optional[str] = None,
    max_iterations: int = 6,
    llm_model: str = "openai:gpt-5.1",
) -> dict[str, Any]:
    """
    Run the training agent to train and evaluate a model.
    
    Args:
        train_ref: Reference to training dataset
        val_ref: Reference to validation dataset
        test_ref: Reference to test dataset
        target_column: Column to predict
        selected_model: Model type to use (logistic_regression, random_forest, xgboost, glm)
        goal: The ML goal/objective
        model_name: Optional name for the model (auto-generated if not provided)
        max_iterations: Maximum training iterations
        llm_model: LLM to use for the agent
    
    Returns:
        Dict with training results
    """
    # Load datasets
    train_df = get_registered_dataset(train_ref)
    val_df = get_registered_dataset(val_ref)
    test_df = get_registered_dataset(test_ref)
    
    if train_df is None:
        raise ValueError(f"Training dataset not found: {train_ref}")
    if val_df is None:
        raise ValueError(f"Validation dataset not found: {val_ref}")
    if test_df is None:
        raise ValueError(f"Test dataset not found: {test_ref}")
    
    # Prepare data
    train_data = _prepare_data_for_tool(train_df)
    val_data = _prepare_data_for_tool(val_df)
    test_data = _prepare_data_for_tool(test_df)
    
    # Get task type and all available models
    task_type = _get_task_type(selected_model, goal)
    available_models = _get_available_models(task_type)
    
    # Generate model name if not provided
    if not model_name:
        import time
        model_name = f"{selected_model}_{int(time.time())}"
    
    # Get feature columns (all except target)
    feature_columns = [c for c in train_df.columns if c != target_column]
    
    print(f"[training_agent] Starting training...")
    print(f"  Model: {selected_model}")
    print(f"  Target: {target_column}")
    print(f"  Task type: {task_type}")
    print(f"  Train size: {len(train_data)}")
    print(f"  Val size: {len(val_data)}")
    print(f"  Test size: {len(test_data)}")
    print(f"  Features: {len(feature_columns)}")
    
    # Calculate class imbalance
    class_counts = train_df[target_column].value_counts().to_dict()
    total = sum(class_counts.values())
    minority_ratio = min(class_counts.values()) / total if total > 0 else 0
    is_imbalanced = minority_ratio < 0.3
    
    # Build model list for context
    models_str = "\n".join(
        f"- **{m['label']}** → tool: `{m['tool']}`" for m in available_models
    )
    imbalance_note = (
        f"⚠️ IMBALANCED DATA — minority class is {minority_ratio:.1%}. "
        "Use class_weight='balanced' (LR/RF) or scale_pos_weight (XGB)."
        if is_imbalanced else "✓ Balanced classes"
    )
    
    context = f"""## Goal
{goal}

## Available Models (use any, switch freely)
{models_str}

Start with **{selected_model}**, but switch to other models whenever you think it could improve performance. Try at least 2 different model types.

## Data
- Task type: {task_type}
- Target column: `{target_column}`
- Training: {len(train_data)} rows (ref: `{train_ref}`)
- Validation: {len(val_data)} rows (ref: `{val_ref}`)
- Test: {len(test_data)} rows (ref: `{test_ref}`)
- Features ({len(feature_columns)}): {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}

**Class distribution:** {class_counts}
{imbalance_note}

## Instructions
- Max iterations: {max_iterations}
- Name models descriptively: `lr_v1`, `rf_v1`, `xgb_v1`, `rf_v2`, etc.
- For training tools: `train_dataset_ref="{train_ref}"`, `target_column="{target_column}"`
- For evaluate_model: `dataset_ref="{val_ref}"` (validation) or `dataset_ref="{test_ref}"` (final test)
- **Optimize aggressively** — try different models and hyperparameters to get the best validation metrics before running the final test evaluation.

## Sample Data (first 3 rows)
{train_data[:3]}
"""
    
    # Create the agent
    llm = init_chat_model(llm_model)
    
    agent = create_agent(
        model=llm,
        tools=TRAINING_TOOLS,
        system_prompt=TRAINING_SYSTEM_PROMPT,
        # Use ToolStrategy to avoid strict schema requirements on other tools
        response_format=ToolStrategy(schema=TrainingResult),
    )
    
    # Run the agent
    messages = [{"role": "user", "content": context}]
    
    instruction_message = f"""Begin training now. Maximize validation performance by exploring different models and hyperparameters. Use the dataset refs above. Run final test evaluation on your best model before finishing."""
    
    messages.append({"role": "user", "content": instruction_message})
    
    try:
        # Invoke the agent with structured output
        result = agent.invoke({"messages": messages})
        
        # Extract the structured TrainingResult from the 'structured_response' key
        # (as per LangChain ToolStrategy docs)
        final_messages = result.get("messages", [])
        training_result: TrainingResult = result.get("structured_response")
        
        # If not in structured_response, try to find in messages
        if training_result is None:
            for msg in reversed(final_messages):
                if hasattr(msg, "content") and isinstance(msg.content, TrainingResult):
                    training_result = msg.content
                    break
                # Handle case where content is a dict (parsed structured output)
                elif hasattr(msg, "content") and isinstance(msg.content, dict):
                    try:
                        training_result = TrainingResult(**msg.content)
                        break
                    except Exception:
                        pass
        
        if training_result is None:
            # Fallback: try to parse from last message content as JSON
            for msg in reversed(final_messages):
                if hasattr(msg, "content") and msg.content:
                    content = msg.content
                    if isinstance(content, str):
                        try:
                            import json
                            data = json.loads(content)
                            training_result = TrainingResult(**data)
                            break
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
        
        if training_result is None:
            raise ValueError("Failed to extract TrainingResult from agent response")
        
        # Check if feature redo was requested via the tool
        feature_redo_request = _get_and_clear_feature_redo_request()
        feature_redo_requested = feature_redo_request is not None or training_result.feature_redo_requested
        
        # Clean up intermediate models - keep only the best
        if training_result.success and training_result.best_model_name:
            # Collect all model names from iterations
            iteration_model_names = [it.model_name for it in training_result.iterations if it.model_name]
            
            deleted_models = _cleanup_intermediate_models(
                best_model_name=training_result.best_model_name,
                base_model_name=model_name,
                iteration_model_names=iteration_model_names,
            )
            if deleted_models:
                print(f"\n[training_agent] Cleaned up {len(deleted_models)} intermediate models")
        
        # Log results
        print(f"\n[training_agent] Training complete!")
        print(f"  Success: {training_result.success}")
        print(f"  Iterations: {training_result.num_iterations}")
        print()
        print("  ITERATION LOG:")
        print("  " + "-" * 60)
        for i, it in enumerate(training_result.iterations, 1):
            status = "OK" if it.success else "FAIL"
            hp_str = ", ".join(f"{k}={v}" for k, v in it.hyperparams.items() if v is not None)
            print(f"  [{status}] Iter {i}: {it.model_name}")
            print(f"     Tool: {it.tool_used}")
            print(f"     Hyperparams: {hp_str[:60]}...")
            if it.val_accuracy is not None or it.val_roc_auc is not None:
                print(f"     Val Accuracy: {it.val_accuracy}, Val ROC-AUC: {it.val_roc_auc}")
            if it.error:
                print(f"     Error: {it.error}")
        print("  " + "-" * 60)
        print(f"  Best Model: {training_result.best_model_name}")
        print(f"  Val Accuracy: {training_result.val_accuracy}")
        print(f"  Val ROC-AUC: {training_result.val_roc_auc}")
        print(f"  Test Accuracy: {training_result.test_accuracy}")
        print(f"  Test ROC-AUC: {training_result.test_roc_auc}")
        print(f"\n  Summary: {training_result.summary}")
        if training_result.recommendations:
            print(f"  Recommendations: {training_result.recommendations}")
        
        # Convert iterations to dict format for backward compatibility
        iterations_dict = [
            {
                "model_name": it.model_name,
                "tool": it.tool_used,
                "hyperparams": it.hyperparams,
                "metrics": {
                    # Classification (kept for backward compat)
                    "train_accuracy": it.train_accuracy,
                    "val_accuracy": it.val_accuracy,
                    "roc_auc": it.val_roc_auc,
                },
                # Classification metrics (top level for UI access)
                "train_accuracy": it.train_accuracy,
                "val_accuracy": it.val_accuracy,
                "val_roc_auc": it.val_roc_auc,
                # Regression metrics (top level for easier access)
                "train_r2": it.train_r2,
                "val_r2": it.val_r2,
                "val_rmse": it.val_rmse,
                "val_mae": it.val_mae,
                "test_r2": it.test_r2,
                "test_rmse": it.test_rmse,
                "test_mae": it.test_mae,
                "success": it.success,
                "error": it.error,
            }
            for it in training_result.iterations
        ]
        
        # Find best iteration (highest ROC-AUC)
        best_iteration = None
        for it_dict in iterations_dict:
            if it_dict.get("success") and it_dict.get("metrics"):
                if best_iteration is None or (it_dict["metrics"].get("roc_auc") or 0) > (best_iteration["metrics"].get("roc_auc") or 0):
                    best_iteration = it_dict
        
        return {
            "success": training_result.success,
            "model_name": training_result.best_model_name,
            "model_type": training_result.model_type,
            "task_type": task_type,
            "target_column": target_column,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            # Classification metrics
            "val_accuracy": training_result.val_accuracy,
            "val_roc_auc": training_result.val_roc_auc,
            "test_accuracy": training_result.test_accuracy,
            "test_roc_auc": training_result.test_roc_auc,
            # Regression metrics
            "train_r2": training_result.train_r2,
            "val_r2": training_result.val_r2,
            "val_rmse": training_result.val_rmse,
            "val_mae": training_result.val_mae,
            "test_r2": training_result.test_r2,
            "test_rmse": training_result.test_rmse,
            "test_mae": training_result.test_mae,
            # Iterations
            "iterations": iterations_dict,
            "num_iterations": training_result.num_iterations,
            "best_iteration": best_iteration,
            "summary": training_result.summary,
            "recommendations": training_result.recommendations,
            "messages": final_messages,
            # Feature engineering redo request
            "feature_redo_requested": feature_redo_requested,
            "feature_redo_recommendation": feature_redo_request.recommendation if feature_redo_request else None,
            "feature_redo_reason": feature_redo_request.reason if feature_redo_request else None,
            "feature_redo_suspected_issues": feature_redo_request.suspected_issues if feature_redo_request else None,
        }
        
    except Exception as e:
        print(f"[training_agent] Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name,
            "model_type": selected_model,
        }


# =============================================================================
# SIMPLE TRAINING FUNCTION (No Agent, Direct Tool Call)
# =============================================================================

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_training_agent",
    "TrainingResult",
    "TrainingIteration",
    "TRAINING_TOOLS",
    "FeatureRedoRequest",
    "request_feature_engineering_redo_tool",
]
