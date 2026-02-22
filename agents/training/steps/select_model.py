"""
Step 1: Model Selection Node
Selects the appropriate ML model based on the user's goal.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# Load environment variables
# Path: steps -> training -> agents -> root
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

if TYPE_CHECKING:
    from ..core.state import TrainingAgentState


# =============================================================================
# AVAILABLE TRAINING MODELS
# =============================================================================

TRAINING_MODELS = {
    "glm": {
        "name": "glm",
        "description": "Generalized Linear Model - flexible regression for various response distributions",
        "when_to_use": [
            "Modeling non-normal response variables (counts, binary, proportions)",
            "When interpretability is important",
            "Poisson regression for count data",
            "Gamma regression for positive continuous data",
        ],
        "when_not_to_use": [
            "Complex non-linear relationships",
            "High-dimensional feature spaces",
        ],
    },
    "logistic_regression": {
        "name": "logistic_regression",
        "description": "Logistic Regression - binary/multiclass classification with interpretable coefficients",
        "when_to_use": [
            "Binary classification (yes/no, default/no-default)",
            "When you need interpretable feature weights",
            "Baseline model for classification",
            "Regulatory environments requiring explainability",
        ],
        "when_not_to_use": [
            "Non-linear decision boundaries",
            "Regression tasks with continuous targets",
        ],
    },
    "random_forest": {
        "name": "random_forest",
        "description": "Random Forest - ensemble of decision trees for robust predictions",
        "when_to_use": [
            "Both classification and regression tasks",
            "Handling missing values and outliers",
            "Feature importance ranking",
            "When accuracy matters more than interpretability",
        ],
        "when_not_to_use": [
            "Very high-dimensional sparse data",
            "Real-time inference with strict latency requirements",
        ],
    },
    "survival_analysis": {
        "name": "survival_analysis",
        "description": "Survival Analysis - time-to-event modeling with censoring support",
        "when_to_use": [
            "Time-to-event prediction (churn, default, failure)",
            "When data has censoring (incomplete observations)",
            "Customer lifetime value modeling",
            "Policy lapse prediction",
        ],
        "when_not_to_use": [
            "Standard classification without time component",
            "When there's no natural event/censoring structure",
        ],
    },
    "xgboost": {
        "name": "xgboost",
        "description": "XGBoost - gradient boosted trees for high-performance predictions",
        "when_to_use": [
            "Structured/tabular data with complex patterns",
            "When maximum predictive accuracy is needed",
            "Competitions and benchmarking",
            "Large datasets with many features",
        ],
        "when_not_to_use": [
            "Small datasets (may overfit)",
            "When full interpretability is required",
        ],
    },
}


def format_training_models_for_prompt() -> str:
    """Format training models for LLM selection prompt."""
    lines = []
    for model in TRAINING_MODELS.values():
        lines.append(f"**{model['name']}**")
        lines.append(f"Description: {model['description']}")
        lines.append(f"When to use: {', '.join(model['when_to_use'])}")
        lines.append(f"When NOT to use: {', '.join(model['when_not_to_use'])}")
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# STRUCTURED OUTPUT SCHEMA
# =============================================================================


class ModelSelectionOutput(BaseModel):
    """Structured output for model selection."""
    # TODO: Add more models
    # TODO: Add clarification
    selected_model: Literal["glm", "logistic_regression", "random_forest", "survival_analysis", "xgboost"] = Field(
        description="The selected model type for training"
    )
    explanation: str = Field(
        description="Explanation of why this model was selected based on the goal"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in the model selection"
    )
    alternative_models: list[str] = Field(
        default_factory=list,
        description="Alternative models that could also work for this goal"
    )


# =============================================================================
# PROMPT
# =============================================================================

MODEL_SELECTION_PROMPT = """You are an ML model selection expert. Based on the user's goal, select the most appropriate model for training.

## Available Training Models

{models}

## User's Goal

{goal}
{redo_section}
## Instructions

1. Analyze the goal to understand what kind of prediction/modeling is needed
2. Select the most appropriate model from the available options
3. Explain your reasoning
4. List any alternative models that could also work
"""


# =============================================================================
# NODE FUNCTION
# =============================================================================


def select_model(state: "TrainingAgentState") -> "TrainingAgentState":
    """
    Step 1: Select Model
    - Based on the goal
    - If user specifies a model, use that
    - Return an explanation if not given
    - User can comment and regenerate (3 total regens)
    """
    # If user specified a model preference, use it directly
    if state.get("user_model_preference"):
        model_name = state["user_model_preference"]
        
        # Validate the model exists
        if model_name not in TRAINING_MODELS:
            return {
                **state,
                "error": f"Unknown model: {model_name}. Available: {list(TRAINING_MODELS.keys())}",
                "current_step": "select_model",
            }
        
        model_info = TRAINING_MODELS[model_name]
        return {
            **state,
            "selected_model": model_name,
            "model_explanation": f"User specified {model_name}: {model_info['description']}",
            "audit_trace": [
                *state.get("audit_trace", []),
                {
                    "step": "select_model",
                    "action": "user_specified",
                    "model": model_name,
                },
            ],
            "explanations": [
                *state.get("explanations", []),
                f"Using user-specified model: {model_name}",
            ],
            "current_step": "select_model",
        }
    
    # Use LLM to select model based on goal
    llm = init_chat_model(model="gpt-5.1", temperature=0)
    structured_llm = llm.with_structured_output(ModelSelectionOutput)
    
    redo_hint = state.get("_select_model_redo_hint", "")
    redo_section = ""
    if redo_hint:
        redo_section = (
            f"\n## User Feedback (IMPORTANT — override your default choice)\n\n"
            f"The user rejected the previous model selection and said:\n"
            f'"{redo_hint}"\n\n'
            f"You MUST follow this feedback when choosing the model.\n"
        )

    prompt = MODEL_SELECTION_PROMPT.format(
        models=format_training_models_for_prompt(),
        goal=state.get("goal", ""),
        redo_section=redo_section,
    )
    
    result: ModelSelectionOutput = structured_llm.invoke(prompt)
    
    return {
        **state,
        "selected_model": result.selected_model,
        "model_explanation": result.explanation,
        "model_regen_count": state.get("model_regen_count", 0),
        "audit_trace": [
            *state.get("audit_trace", []),
            {
                "step": "select_model",
                "action": "llm_selected",
                "model": result.selected_model,
                "confidence": result.confidence,
                "alternatives": result.alternative_models,
            },
        ],
        "explanations": [
            *state.get("explanations", []),
            f"Selected model: {result.selected_model}. Reason: {result.explanation}",
        ],
        "current_step": "select_model",
    }
