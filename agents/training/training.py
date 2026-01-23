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
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# TODO: More hyperparameters (e.g. sample weights... and for each model...)

# Add tools path
_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "models-tools" / "training"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

# Add data-tools path for utils
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from glm import sklearn_glm_tool
# Import training tools
from logistic_regression import sklearn_logistic_regression_tool
from model_storage import (evaluate_model_tool, get_model_info_tool,
                           list_trained_models_tool, predict_with_model_tool)
from random_forest import sklearn_random_forest_tool
from survival_analysis import survival_analysis_tool
from utils import get_registered_dataset
from xgboost_model import xgboost_train_tool

# TODO: Human in the loop

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
]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

# TODO: Add ML standard practice here or as a skill
# TODO: Only stop when optimized as much as you can... --> if we have blockers from data or features we'll loop back to previous steps
TRAINING_SYSTEM_PROMPT = """You are an ML Training Agent that strategically trains and tunes models through iterative experimentation.

## Workflow

Each iteration: **Train → Evaluate → Decide**

1. **Train** a model on `train_dataset_ref`
2. **Evaluate** on validation using `evaluate_model`
3. **Decide** next action based on the decision logic below

## Decision Logic

After each evaluation, analyze results and choose ONE action:

### → STOP & TEST (metrics are good)
When: val_roc_auc ≥ 0.75 AND val_accuracy beats baseline (majority class rate)
Action: Run final `evaluate_model` on test data with best model

### → TUNE HYPERPARAMETERS (model shows promise but can improve)
When: Current model type is working (val_roc_auc > 0.6) but not optimal
Logic:
- If OVERFITTING (train_score >> val_score by >0.1):
  - LR: reduce C (0.1 → 0.01), try l1_ratio=0.5
  - RF: reduce max_depth (10 → 6 → 4), increase min_samples_leaf
  - XGB: increase reg_alpha/reg_lambda, reduce learning_rate, reduce max_depth
- If UNDERFITTING (both train and val scores low):
  - LR: increase C (1.0 → 10), reduce regularization
  - RF: increase max_depth, reduce min_samples_leaf
  - XGB: increase max_depth, reduce regularization

### → SWITCH MODELS (current architecture is inadequate)
When: val_roc_auc < 0.6 after tuning attempts, OR linear model on non-linear data
Logic:
- LR → RF: When LR performance plateaus and you suspect non-linear relationships
- RF → XGB: When RF overfits badly or you need better regularization control
- Any → XGB: When you need maximum performance and have tried simpler models
**Critical:** Carry forward imbalance handling:
- If previous used class_weight='balanced' → RF must use class_weight='balanced'
- If switching to XGB with imbalanced data → use scale_pos_weight = (neg_count / pos_count)

## First Iteration Strategy

Before training, analyze the data context:
1. **Check class balance:** If positive rate < 20%, use class_weight='balanced' or scale_pos_weight
2. **Check feature count:** If features > 20, consider regularization
3. **Check dataset size:** If < 500 rows, prefer simpler models (LR, shallow RF)

Start with the model specified in context. Use sensible defaults, but apply imbalance handling if needed.

## Tracking State

Keep mental track of:
- What you've tried (model types, key hyperparameters)
- What worked (which changes improved metrics)
- What didn't work (avoid repeating failed experiments)

Use this history to make informed next decisions. Don't try the same configuration twice.

## Model Naming

Use unique names reflecting the experiment: `{model}_v{n}` (e.g., `lr_v1`, `rf_v2`, `xgb_v1`)

## Output

Provide structured summary:
- `success`, `best_model_name`, `model_type`
- `val_accuracy`, `val_roc_auc`, `test_accuracy`, `test_roc_auc`
- `iterations`: list of attempts with model_name, tool_used, hyperparams, metrics
- `num_iterations`, `summary`, `recommendations`
"""


# =============================================================================
# TRAINING RESULT SCHEMA
# =============================================================================

class TrainingIteration(BaseModel):
    """A single training iteration attempt."""
    model_name: str = Field(description="Name of the model for this iteration")
    tool_used: str = Field(description="Training tool used (e.g., sklearn_logistic_regression)")
    hyperparams: dict = Field(default_factory=dict, description="Hyperparameters used")
    train_accuracy: Optional[float] = Field(default=None, description="Training accuracy")
    val_accuracy: Optional[float] = Field(default=None, description="Validation accuracy from evaluate_model")
    val_roc_auc: Optional[float] = Field(default=None, description="Validation ROC-AUC from evaluate_model")
    success: bool = Field(description="Whether this iteration succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class TrainingResult(BaseModel):
    """Structured output for training completion."""
    success: bool = Field(description="Whether training completed successfully")
    best_model_name: str = Field(description="Name of the best trained model")
    model_type: str = Field(description="Type of model trained (e.g., logistic_regression, random_forest)")
    
    # Best model metrics
    val_accuracy: Optional[float] = Field(default=None, description="Best model validation accuracy")
    val_roc_auc: Optional[float] = Field(default=None, description="Best model validation ROC-AUC")
    test_accuracy: Optional[float] = Field(default=None, description="Best model test accuracy")
    test_roc_auc: Optional[float] = Field(default=None, description="Best model test ROC-AUC")
    
    # Iteration tracking
    iterations: list[TrainingIteration] = Field(default_factory=list, description="List of all training iterations attempted")
    num_iterations: int = Field(description="Total number of training iterations attempted")
    
    # Summary
    summary: str = Field(description="Summary of training process and results")
    recommendations: Optional[str] = Field(default=None, description="Recommendations for improvement")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _prepare_data_for_tool(df) -> list[dict]:
    """Convert DataFrame to list of dicts for tool input."""
    return df.to_dict(orient="records")


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


def _get_alternative_models(selected_model: str, task_type: str) -> list[str]:
    """
    Get alternative models that can be tried if the selected model plateaus.
    
    Args:
        selected_model: The initially selected model
        task_type: Either 'classification' or 'regression'
    
    Returns:
        List of alternative model names (tool names) that are compatible with the task
    """
    # Define models by task type
    classification_models = [
        "sklearn_logistic_regression",
        "sklearn_random_forest", 
        "xgboost_train",
    ]
    
    regression_models = [
        "sklearn_random_forest",
        "xgboost_train",
        "sklearn_glm",
    ]
    
    # Map selected_model to tool name for comparison
    model_to_tool = {
        "logistic_regression": "sklearn_logistic_regression",
        "random_forest": "sklearn_random_forest",
        "xgboost": "xgboost_train",
        "glm": "sklearn_glm",
        "survival": "survival_analysis",
    }
    
    # Get the tool name for the selected model
    selected_tool = model_to_tool.get(selected_model.lower(), selected_model)
    
    # Get all compatible models for this task type
    if task_type == "classification":
        all_models = classification_models
    else:
        all_models = regression_models
    
    # Return alternatives (excluding the selected model)
    alternatives = [m for m in all_models if m != selected_tool]
    return alternatives


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

# TODO: Go back to change feature engineering with comments if need be
# --> Or just modify features in this loop
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
    llm_model: str = "openai:gpt-5-mini",
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
    
    # Get task type and alternative models
    task_type = _get_task_type(selected_model, goal)
    alternative_models = _get_alternative_models(selected_model, task_type)
    
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
    
    # Build context for the agent
    alt_models_str = ", ".join(alternative_models) if alternative_models else "None"
    
    context = f"""
## Training Context

**Goal:** {goal}

**Model Type:** {selected_model}
**Task Type:** {task_type}
**Target Column:** {target_column}
**Base Model Name:** {model_name} (append _v1, _v2, _v3 for iterations)
**Max Iterations:** {max_iterations}

**Alternative Models Available:** {alt_models_str}
(You may switch to any of these if you think a different model would perform better)

**Dataset Sizes:**
- Training: {len(train_data)} rows
- Validation: {len(val_data)} rows  
- Test: {len(test_data)} rows

**Dataset References:**
- Training: train_dataset_ref="{train_ref}"
- Validation: dataset_ref="{val_ref}"
- Test: dataset_ref="{test_ref}"

**Features ({len(feature_columns)}):** {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}

**Class Distribution (Training):**
{class_counts}
{"⚠️ IMBALANCED DATA - minority class is " + f"{minority_ratio:.1%}" + " - use class_weight='balanced'" if is_imbalanced else "✓ Balanced classes"}

## Iterative Training Instructions

1. **Train a model**: Start with {selected_model}
   - model_name="{model_name}_v1"
   - train_dataset_ref="{train_ref}"
   - target_column="{target_column}"
   {"- class_weight='balanced' (data is imbalanced)" if is_imbalanced else ""}
   
2. **Evaluate on validation** using evaluate_model tool:
   - dataset_ref="{val_ref}"
   - target_column="{target_column}"

3. **Decide next action** based on results:
   - Proceed to test if metrics are satisfactory
   - Tune hyperparameters if you think adjustments will help
   - Switch to a different model ({alt_models_str}) if you think it would perform better
   - Use unique model names for each attempt (e.g., {model_name}_v2, rf_v1, xgb_v1)

4. **Final test evaluation**: Use evaluate_model with dataset_ref="{test_ref}"

5. **Report**: Summarize all iterations, final metrics, and chosen model

## Sample Training Data (first 3 rows)
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
    
    # Simple instruction message (no raw data needed - tools load from registry)
    instruction_message = f"""
Now please train the model using these dataset references:

For training tools (sklearn_logistic_regression, sklearn_random_forest, xgboost_train, etc.):
- train_dataset_ref="{train_ref}"
- target_column="{target_column}"

For evaluate_model tool:
- Validation: dataset_ref="{val_ref}", target_column="{target_column}"
- Test: dataset_ref="{test_ref}", target_column="{target_column}"

Begin training now.
"""
    
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
                    "train_accuracy": it.train_accuracy,
                    "val_accuracy": it.val_accuracy,
                    "roc_auc": it.val_roc_auc,
                },
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
            "val_accuracy": training_result.val_accuracy,
            "val_roc_auc": training_result.val_roc_auc,
            "test_accuracy": training_result.test_accuracy,
            "test_roc_auc": training_result.test_roc_auc,
            "iterations": iterations_dict,
            "num_iterations": training_result.num_iterations,
            "best_iteration": best_iteration,
            "summary": training_result.summary,
            "recommendations": training_result.recommendations,
            "messages": final_messages,
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
]
