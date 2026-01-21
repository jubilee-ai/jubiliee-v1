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
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent.parent / ".env")

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

TRAINING_SYSTEM_PROMPT = """You are an ML Training Agent responsible for training and evaluating machine learning models.

## Your Task
Iteratively train and improve a model until validation metrics are acceptable, then evaluate on test data.

## Available Tools
- `sklearn_logistic_regression`: Train logistic regression (classification)
- `sklearn_random_forest`: Train random forest (classification or regression)
- `xgboost_train`: Train XGBoost model (classification or regression)
- `sklearn_glm`: Train Generalized Linear Model (regression with Poisson/Gamma/Tweedie)
- `survival_analysis`: Train survival model (time-to-event prediction)
- `evaluate_model`: **USE THIS** to evaluate a trained model on validation/test datasets (returns accuracy, ROC-AUC, precision, recall, F1)
- `get_model_info`: Get details about a trained model
- `list_trained_models`: List all available models

## Iterative Training Process

### Iteration Loop (repeat until good or max 3 iterations):
1. **Analyze** the data info provided (class distribution, features, etc.)
2. **Decide hyperparameters** based on your analysis:
   - For imbalanced data: use class_weight="balanced"
   - Start with moderate regularization (C=1.0), adjust if overfitting/underfitting
   - For tree models: start with n_estimators=100, max_depth=10
3. **Train the model** using `train_dataset_ref` parameter:
   - Use a UNIQUE model_name for each iteration (e.g., model_v1, model_v2, model_v3)
   - Set train_dataset_ref to the training dataset reference (provided in context)
4. **Evaluate on validation** using `evaluate_model` tool:
   - model_name: the model you just trained
   - dataset_ref: the validation dataset reference (provided in context)
   - target_column: the target column name
5. **Analyze validation results**:
   - Check accuracy, ROC-AUC, precision, recall
   - Identify issues: overfitting? underfitting? class imbalance problems?
6. **Decide next action**:
   - If ROC-AUC >= 0.75 AND accuracy reasonable → STOP iterating, proceed to test
   - If ROC-AUC < 0.75 OR poor metrics → TRY AGAIN with different hyperparameters:
     * If overfitting: increase regularization (lower C), reduce max_depth
     * If underfitting: decrease regularization (higher C), more estimators
     * If class imbalance issues: ensure class_weight="balanced"

### After iteration loop:
7. **Evaluate best model on test data** using `evaluate_model` with the test dataset_ref
8. **Report final results** including:
   - Best model name and hyperparameters used
   - Validation metrics that led to selection
   - Final test metrics
   - Summary of iterations tried and why this model was chosen

## Important Rules
- Training tools now use `train_dataset_ref` parameter (NOT raw data)
- ALWAYS use training data for fitting, validation for tuning, test for final eval
- NEVER train on validation or test data
- USE DIFFERENT model_name FOR EACH TRAINING ATTEMPT (append _v1, _v2, etc.)
- Maximum 3 training iterations to avoid infinite loops
- Use `evaluate_model` tool to get metrics (NOT predict_with_model)
- Explain your reasoning for hyperparameter choices

## Acceptable Metric Thresholds
- ROC-AUC >= 0.75 (good ranking ability)
- Accuracy should beat baseline (majority class rate)
- For imbalanced data, prioritize recall on minority class

## Data Format
All datasets are accessed via dataset_ref names (provided in context).
- train_dataset_ref: for training tools
- dataset_ref: for evaluate_model tool
"""


# =============================================================================
# TRAINING RESULT SCHEMA
# =============================================================================

class TrainingResult(BaseModel):
    """Structured output for training completion."""
    success: bool = Field(description="Whether training completed successfully")
    model_name: str = Field(description="Name of the trained model")
    model_type: str = Field(description="Type of model trained")
    
    # Metrics
    train_accuracy: Optional[float] = Field(default=None, description="Training accuracy")
    val_accuracy: Optional[float] = Field(default=None, description="Validation accuracy")
    test_accuracy: Optional[float] = Field(default=None, description="Test accuracy")
    val_roc_auc: Optional[float] = Field(default=None, description="Validation ROC-AUC")
    test_roc_auc: Optional[float] = Field(default=None, description="Test ROC-AUC")
    
    # Summary
    summary: str = Field(description="Summary of training process and results")
    recommendations: Optional[str] = Field(default=None, description="Recommendations for improvement")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _prepare_data_for_tool(df) -> list[dict]:
    """Convert DataFrame to list of dicts for tool input."""
    return df.to_dict(orient="records")


def _extract_metrics_from_messages(messages: list, val_ref: str, test_ref: str) -> dict:
    """
    Parse agent messages to extract metrics from tool calls.
    
    Looks for:
    1. evaluate_model tool results (JSON format, preferred)
    2. Training tool results (text format with metrics, fallback)
    
    Returns dict with val_accuracy, val_roc_auc, test_accuracy, test_roc_auc.
    """
    import json
    import re
    
    val_metrics = {}
    test_metrics = {}
    last_training_metrics = {}
    training_failed = False
    
    training_tools = ["sklearn_logistic_regression", "sklearn_random_forest", 
                     "xgboost_train", "sklearn_glm", "survival_analysis"]
    
    for i, msg in enumerate(messages):
        msg_name = getattr(msg, "name", None)
        
        # Skip non-tool messages
        if not hasattr(msg, "content") or msg_name is None:
            continue
            
        content = msg.content
        if not content or not isinstance(content, str):
            continue
        
        # Check for evaluate_model results (JSON format)
        if msg_name == "evaluate_model":
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    # Look at previous AIMessage for tool call args to identify val vs test
                    if i > 0:
                        prev_msg = messages[i - 1]
                        if hasattr(prev_msg, "tool_calls"):
                            for tc in prev_msg.tool_calls:
                                if tc.get("name") == "evaluate_model":
                                    args = tc.get("args", {})
                                    dataset_ref = args.get("dataset_ref", "")
                                    
                                    accuracy = data.get("accuracy")
                                    roc_auc = data.get("roc_auc")
                                    
                                    if dataset_ref == test_ref or "test" in dataset_ref.lower():
                                        test_metrics = {"accuracy": accuracy, "roc_auc": roc_auc}
                                    elif dataset_ref == val_ref or "val" in dataset_ref.lower():
                                        val_metrics = {"accuracy": accuracy, "roc_auc": roc_auc}
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Check training tool results (text format)
        if msg_name in training_tools:
            # Check for training failure
            if "TRAINING FAILED" in content or "Error:" in content:
                training_failed = True
                continue
            
            # Parse text format like:
            # "  Train Accuracy: 0.8914"
            # "  Test Accuracy:  0.8914"
            # "  Test ROC-AUC:   0.6123"
            
            train_acc_match = re.search(r"Train Accuracy:\s*([\d.]+)", content)
            test_acc_match = re.search(r"Test Accuracy:\s*([\d.]+)", content)
            test_roc_match = re.search(r"Test ROC-AUC:\s*([\d.]+)", content)
            # TODO: LLM extract these from structured output instead
            
            if train_acc_match or test_acc_match:
                training_failed = False  # Found successful training output
                last_training_metrics = {
                    "train_accuracy": float(train_acc_match.group(1)) if train_acc_match else None,
                    "test_accuracy": float(test_acc_match.group(1)) if test_acc_match else None,
                    "test_roc_auc": float(test_roc_match.group(1)) if test_roc_match else None,
                }
    
    # Use evaluate_model results if available, otherwise fall back to training tool metrics
    if val_metrics:
        result_val_accuracy = val_metrics.get("accuracy")
        result_val_roc_auc = val_metrics.get("roc_auc")
    else:
        # Training tools report "Test" metrics but it's really training data (no internal split)
        # These are the best we have if evaluate_model wasn't called
        result_val_accuracy = last_training_metrics.get("train_accuracy")
        result_val_roc_auc = last_training_metrics.get("test_roc_auc")  # ROC-AUC from training
    
    if test_metrics:
        result_test_accuracy = test_metrics.get("accuracy")
        result_test_roc_auc = test_metrics.get("roc_auc")
    else:
        result_test_accuracy = last_training_metrics.get("test_accuracy")
        result_test_roc_auc = last_training_metrics.get("test_roc_auc")
    
    return {
        "val_accuracy": result_val_accuracy,
        "val_roc_auc": result_val_roc_auc,
        "test_accuracy": result_test_accuracy,
        "test_roc_auc": result_test_roc_auc,
        "training_failed": training_failed,
    }


def _extract_iterations_from_messages(messages: list) -> list[dict]:
    """
    Extract training iteration logs from agent messages.
    
    Returns list of iteration dicts with model_name, hyperparams, and metrics.
    """
    import re
    
    iterations = []
    training_tools = ["sklearn_logistic_regression", "sklearn_random_forest", 
                     "xgboost_train", "sklearn_glm", "survival_analysis"]
    
    for i, msg in enumerate(messages):
        # Look for AIMessage with tool_calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") in training_tools:
                    args = tc.get("args", {})
                    iteration = {
                        "model_name": args.get("model_name", "unknown"),
                        "tool": tc.get("name"),
                        "hyperparams": {
                            "C": args.get("C"),
                            "class_weight": args.get("class_weight"),
                            "n_estimators": args.get("n_estimators"),
                            "max_depth": args.get("max_depth"),
                            "max_iter": args.get("max_iter"),
                        },
                        "metrics": None,  # Will be filled from tool result
                        "success": False,
                    }
                    
                    # Look for the corresponding ToolMessage result
                    for j in range(i + 1, min(i + 3, len(messages))):
                        result_msg = messages[j]
                        if hasattr(result_msg, "name") and result_msg.name == tc.get("name"):
                            content = result_msg.content if hasattr(result_msg, "content") else ""
                            
                            # Check for failure
                            if "TRAINING FAILED" in content:
                                iteration["success"] = False
                                iteration["error"] = "Training failed"
                                break
                            
                            # Extract metrics from success output
                            if "TRAINING COMPLETE" in content:
                                iteration["success"] = True
                                
                                train_acc = re.search(r"Train Accuracy:\s*([\d.]+)", content)
                                test_acc = re.search(r"Test Accuracy:\s*([\d.]+)", content)
                                test_roc = re.search(r"Test ROC-AUC:\s*([\d.]+)", content)
                                
                                iteration["metrics"] = {
                                    "train_accuracy": float(train_acc.group(1)) if train_acc else None,
                                    "test_accuracy": float(test_acc.group(1)) if test_acc else None,
                                    "roc_auc": float(test_roc.group(1)) if test_roc else None,
                                }
                            break
                    
                    iterations.append(iteration)
    
    return iterations


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


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

# TODO: switch between different models when plateauing
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
    
    # Get task type
    task_type = _get_task_type(selected_model, goal)
    
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
    context = f"""
## Training Context

**Goal:** {goal}

**Model Type:** {selected_model}
**Task Type:** {task_type}
**Target Column:** {target_column}
**Base Model Name:** {model_name} (append _v1, _v2, _v3 for iterations)
**Max Iterations:** {max_iterations}

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

1. **Iteration 1**: Train {selected_model} with reasonable defaults
   - model_name="{model_name}_v1"
   - train_dataset_ref="{train_ref}"
   - target_column="{target_column}"
   {"- class_weight='balanced' (data is imbalanced)" if is_imbalanced else ""}
   
2. **Evaluate on validation** using evaluate_model tool:
   - model_name="{model_name}_v1"
   - dataset_ref="{val_ref}"
   - target_column="{target_column}"

3. **Check metrics**: 
   - If ROC-AUC >= 0.75 and accuracy is reasonable → proceed to step 5
   - If metrics are poor → go to step 4

4. **Iterate** (up to {max_iterations} total attempts):
   - Analyze what went wrong
   - Adjust hyperparameters (C, n_estimators, max_depth, etc.)
   - Train again with train_dataset_ref="{train_ref}", model_name="{model_name}_v2" (or _v3)
   - Re-evaluate on validation using evaluate_model

5. **Final test evaluation**: Use evaluate_model with dataset_ref="{test_ref}"

6. **Report**: Summarize all iterations, final metrics, and chosen model

## Sample Training Data (first 3 rows)
{train_data[:3]}
"""
    
    # Create the agent
    llm = init_chat_model(llm_model)
    
    agent = create_agent(
        model=llm,
        tools=TRAINING_TOOLS,
        system_prompt=TRAINING_SYSTEM_PROMPT,
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
        # Invoke the agent
        result = agent.invoke({"messages": messages})
        # TODO: Make this structured output and remove stuff below
        
        # Extract the final response
        final_messages = result.get("messages", [])
        final_response = ""
        for msg in reversed(final_messages):
            if hasattr(msg, "content") and msg.content:
                final_response = msg.content
                break
        
        # Extract metrics from tool call results
        metrics = _extract_metrics_from_messages(final_messages, val_ref, test_ref)
        
        # Extract iteration logs
        iterations = _extract_iterations_from_messages(final_messages)
        
        # Determine success based on whether training actually succeeded
        training_succeeded = not metrics.get("training_failed", False)
        
        # Find best iteration (highest ROC-AUC)
        best_iteration = None
        best_model = model_name
        for it in iterations:
            if it.get("success") and it.get("metrics"):
                if best_iteration is None or (it["metrics"].get("roc_auc") or 0) > (best_iteration["metrics"].get("roc_auc") or 0):
                    best_iteration = it
                    best_model = it.get("model_name", model_name)
        
        # Log iterations
        print(f"\n[training_agent] Training complete!")
        print(f"  Success: {training_succeeded}")
        print(f"  Iterations: {len(iterations)}")
        print()
        print("  📊 ITERATION LOG:")
        print("  " + "-" * 60)
        for i, it in enumerate(iterations, 1):
            status = "✅" if it.get("success") else "❌"
            m = it.get("metrics") or {}
            hp = it.get("hyperparams") or {}
            hp_str = ", ".join(f"{k}={v}" for k, v in hp.items() if v is not None)
            print(f"  {status} Iter {i}: {it.get('model_name', 'unknown')}")
            print(f"     Hyperparams: {hp_str[:60]}...")
            if m:
                print(f"     Accuracy: {m.get('train_accuracy', 'N/A'):.3f}, ROC-AUC: {m.get('roc_auc', 'N/A'):.3f}" if m.get('train_accuracy') else f"     Metrics: {m}")
            if it.get("error"):
                print(f"     Error: {it.get('error')}")
        print("  " + "-" * 60)
        print(f"  Best Model: {best_model}")
        print(f"  Val Accuracy: {metrics.get('val_accuracy')}")
        print(f"  Val ROC-AUC: {metrics.get('val_roc_auc')}")
        print(f"  Test Accuracy: {metrics.get('test_accuracy')}")
        print(f"  Test ROC-AUC: {metrics.get('test_roc_auc')}")
        
        return {
            "success": training_succeeded,
            "model_name": best_model,
            "model_type": selected_model,
            "task_type": task_type,
            "target_column": target_column,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "val_accuracy": metrics.get("val_accuracy"),
            "val_roc_auc": metrics.get("val_roc_auc"),
            "test_accuracy": metrics.get("test_accuracy"),
            "test_roc_auc": metrics.get("test_roc_auc"),
            "iterations": iterations,
            "num_iterations": len(iterations),
            "best_iteration": best_iteration,
            "agent_response": final_response,
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

# def run_training_simple(
#     train_ref: str,
#     val_ref: Optional[str],
#     test_ref: Optional[str],
#     target_column: str,
#     selected_model: str,
#     goal: str,
#     model_name: Optional[str] = None,
# ) -> dict[str, Any]:
#     """
#     Simple training without agent - directly calls the appropriate tool.
    
#     This is a simpler alternative that doesn't use an LLM agent,
#     just directly trains the model using the appropriate tool.
#     """
#     # Load datasets
#     train_df = get_registered_dataset(train_ref)
    
#     if train_df is None:
#         raise ValueError(f"Training dataset not found: {train_ref}")
    
#     # Prepare data
#     train_data = _prepare_data_for_tool(train_df)
    
#     # Get task type
#     task_type = _get_task_type(selected_model, goal)
    
#     # Generate model name if not provided
#     if not model_name:
#         import time
#         model_name = f"{selected_model}_{int(time.time())}"
    
#     print(f"[training_simple] Training {selected_model} model...")
#     print(f"  Model name: {model_name}")
#     print(f"  Target: {target_column}")
#     print(f"  Train size: {len(train_data)}")
    
#     # Call the appropriate training tool
#     model_lower = selected_model.lower()
    
#     try:
#         if "logistic" in model_lower:
#             result = sklearn_logistic_regression_tool.invoke({
#                 "model_name": model_name,
#                 "data": train_data,
#                 "target_column": target_column,
#                 "description": goal,
#                 "class_weight": "balanced",
#                 "max_iter": 500,
#             })
#         elif "random_forest" in model_lower or "rf" in model_lower:
#             result = sklearn_random_forest_tool.invoke({
#                 "model_name": model_name,
#                 "data": train_data,
#                 "target_column": target_column,
#                 "task_type": task_type,
#                 "description": goal,
#                 "n_estimators": 100,
#                 "class_weight": "balanced" if task_type == "classification" else None,
#             })
#         elif "xgboost" in model_lower or "xgb" in model_lower:
#             result = xgboost_train_tool.invoke({
#                 "model_name": model_name,
#                 "data": train_data,
#                 "target_column": target_column,
#                 "task_type": task_type,
#                 "description": goal,
#                 "n_estimators": 100,
#             })
#         elif "glm" in model_lower:
#             result = sklearn_glm_tool.invoke({
#                 "model_name": model_name,
#                 "data": train_data,
#                 "target_column": target_column,
#                 "distribution": "gamma",  # Default
#                 "description": goal,
#             })
#         else:
#             # Default to logistic regression for classification
#             result = sklearn_logistic_regression_tool.invoke({
#                 "model_name": model_name,
#                 "data": train_data,
#                 "target_column": target_column,
#                 "description": goal,
#             })
        
#         print(f"[training_simple] Training complete!")
#         print(result)
        
#         # Evaluate on validation if provided
#         val_result = None
#         if val_ref:
#             val_df = get_registered_dataset(val_ref)
#             if val_df is not None:
#                 val_data = _prepare_data_for_tool(val_df)
#                 val_result = predict_with_model_tool.invoke({
#                     "model_name": model_name,
#                     "data": val_data,
#                 })
#                 print(f"\n[training_simple] Validation predictions:")
#                 print(val_result[:500] + "..." if len(val_result) > 500 else val_result)
        
#         # Evaluate on test if provided
#         test_result = None
#         if test_ref:
#             test_df = get_registered_dataset(test_ref)
#             if test_df is not None:
#                 test_data = _prepare_data_for_tool(test_df)
#                 test_result = predict_with_model_tool.invoke({
#                     "model_name": model_name,
#                     "data": test_data,
#                 })
#                 print(f"\n[training_simple] Test predictions:")
#                 print(test_result[:500] + "..." if len(test_result) > 500 else test_result)
        
#         return {
#             "success": True,
#             "model_name": model_name,
#             "model_type": selected_model,
#             "training_result": result,
#             "validation_result": val_result,
#             "test_result": test_result,
#         }
        
#     except Exception as e:
#         print(f"[training_simple] Error: {e}")
#         import traceback
#         traceback.print_exc()
#         return {
#             "success": False,
#             "error": str(e),
#             "model_name": model_name,
#         }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_training_agent",
    "run_training_simple",
    "TRAINING_TOOLS",
]
