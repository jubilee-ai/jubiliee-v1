"""
Step 1: Model Family Selection Node
Narrows the approach to a family of ML models based on the user's goal.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from agents.training.utils.graph_stream_hooks import emit_graph_stream

if TYPE_CHECKING:
    from ..core.state import TrainingAgentState


# =============================================================================
# MODEL FAMILIES
# =============================================================================

MODEL_FAMILIES = {
    "supervised": {
        "name": "supervised",
        "description": "Supervised learning — models that learn from labeled data to predict outcomes",
        "when_to_use": [
            "Classification tasks (binary or multiclass)",
            "Regression tasks with a clear target variable",
            "When labeled training data is available",
            "Predicting a known outcome (churn, default, price, etc.)",
            "When interpretability or feature importance matters",
        ],
        "when_not_to_use": [
            "No labeled target variable exists",
            "The goal is to discover hidden structure or groupings",
            "Primarily dimensionality reduction or anomaly detection without labels",
        ],
    },
    "unsupervised": {
        "name": "unsupervised",
        "description": "Unsupervised learning — models that find hidden patterns and structure in unlabeled data",
        "when_to_use": [
            "Clustering or segmentation (customer segments, groupings)",
            "Anomaly / outlier detection without labeled fraud/anomaly data",
            "Dimensionality reduction or feature extraction",
            "Exploratory data analysis to discover structure",
            "When no labeled target variable is available",
        ],
        "when_not_to_use": [
            "A clear labeled target variable exists and you want to predict it",
            "The task is straightforward classification or regression",
            "When you need well-calibrated probability estimates for a known outcome",
        ],
    },
    "neural_networks": {
        "name": "neural_networks",
        "description": "Neural networks — deep learning models for complex, high-dimensional, or unstructured data",
        "when_to_use": [
            "Image, text, audio, or other unstructured data",
            "Very large datasets (hundreds of thousands+ rows)",
            "Complex non-linear relationships that simpler models can't capture",
            "Sequence or time-series modeling with long-range dependencies",
            "Multi-modal inputs or representation learning",
        ],
        "when_not_to_use": [
            "Small to medium tabular datasets (supervised or unsupervised methods usually perform better)",
            "When interpretability is a hard requirement",
            "Limited compute resources or strict latency constraints",
            "When a simpler model can achieve comparable performance",
        ],
    },
}


def format_model_families_for_prompt() -> str:
    """Format model families for LLM selection prompt."""
    lines = []
    for family in MODEL_FAMILIES.values():
        lines.append(f"**{family['name']}**")
        lines.append(f"Description: {family['description']}")
        lines.append(f"When to use: {', '.join(family['when_to_use'])}")
        lines.append(f"When NOT to use: {', '.join(family['when_not_to_use'])}")
        lines.append("")
    return "\n".join(lines)


_FAMILY_ALIASES: dict[str, str] = {
    "supervised": "supervised",
    "supervised learning": "supervised",
    "classification": "supervised",
    "regression": "supervised",
    "unsupervised": "unsupervised",
    "unsupervised learning": "unsupervised",
    "clustering": "unsupervised",
    "cluster": "unsupervised",
    "segmentation": "unsupervised",
    "segment": "unsupervised",
    "anomaly detection": "unsupervised",
    "dimensionality reduction": "unsupervised",
    "neural network": "neural_networks",
    "neural networks": "neural_networks",
    "neural net": "neural_networks",
    "deep learning": "neural_networks",
    "deep neural": "neural_networks",
    "nn": "neural_networks",
    "dnn": "neural_networks",
    "cnn": "neural_networks",
    "rnn": "neural_networks",
    "transformer": "neural_networks",
    "lstm": "neural_networks",
}


def _extract_family_from_goal(goal: str) -> str | None:
    """Detect an explicit model-family request in the user's goal text.

    Returns the canonical family key if found, otherwise None.
    Longer alias strings are checked first to avoid partial matches.
    """
    goal_lower = goal.lower()
    for alias in sorted(_FAMILY_ALIASES, key=len, reverse=True):
        if alias in goal_lower:
            return _FAMILY_ALIASES[alias]
    return None


# =============================================================================
# STRUCTURED OUTPUT SCHEMA
# =============================================================================


class ModelFamilySelectionOutput(BaseModel):
    """Structured output for model family selection."""

    selected_family: Literal["supervised", "unsupervised", "neural_networks"] = Field(
        description="The selected model family for training"
    )
    explanation: str = Field(
        description=(
            "2–4 short sentences for a non-technical reader: what kind of problem this is, "
            "what the trained model will be used for, and why this broad approach fits. "
            "No estimator names, hyperparameters, formulas, or implementation detail."
        )
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in the model family selection"
    )
    alternative_families: list[str] = Field(
        default_factory=list,
        description="Alternative model families that could also work for this goal"
    )


# =============================================================================
# PROMPT
# =============================================================================

MODEL_FAMILY_SELECTION_PROMPT = """You are an ML model selection expert. Based on the user's goal, select the most appropriate family of models for training.

Your job is NOT to pick a specific algorithm — only to narrow the approach down to the right family so the training step can explore concrete models within that family.

## Available Model Families

{families}

## User's Goal

{goal}
{redo_section}
## Instructions

1. **If the user explicitly names a model family** (e.g. "use supervised learning", "cluster my customers", "deep learning"), you MUST select that family. The user's explicit request overrides your own preference.
2. Otherwise, analyze the goal to understand what kind of problem this is and select the most appropriate family.
3. In **`explanation`**, write for a **product or business reader**, not an ML engineer: what outcome they care about, what the model will support (e.g. prioritizing cases, forecasting demand), and why this **family** of methods fits — in plain language. **Do not** name specific libraries or estimators (no XGBoost, PyTorch, parameter names, formulas, or class-weight math). Keep **2–4 short sentences**; skip preamble; do not paste long “alternative families” essays (use **`alternative_families`** for those).
4. List any alternative families that could also work.
"""


# =============================================================================
# NODE FUNCTION
# =============================================================================


def _derive_task_type(family_key: str, goal: str) -> str:
    """Derive the task_type from the selected model family and goal text.

    Returns "unsupervised" for unsupervised families, otherwise infers
    "classification" vs "regression" from the goal.
    """
    if family_key == "unsupervised":
        return "unsupervised"
    goal_lower = goal.lower()
    regression_hints = [
        "regress", "predict value", "forecast", "amount", "price",
        "cost", "salary", "revenue", "income", "score", "continuous",
        "numeric target", "distress", "charges", "quantity", "duration",
        "rate", "ratio",
    ]
    if any(w in goal_lower for w in regression_hints):
        return "regression"
    classification_hints = [
        "classif", "churn", "fraud", "default", "spam", "diagnos",
        "detect", "binary", "multi-class", "category", "sentiment",
        "predict whether", "predict if",
    ]
    if any(w in goal_lower for w in classification_hints):
        return "classification"
    return "classification"


def select_model(state: "TrainingAgentState") -> "TrainingAgentState":
    """
    Step 1: Select Model Family
    - Based on the goal, choose supervised / unsupervised / neural_networks
    - If user specifies a family, use that
    - Return an explanation
    - User can comment and regenerate (3 total regens)
    - Sets task_type: "classification" | "regression" | "unsupervised"
    """
    resolved = state.get("resolved_model_type")
    if resolved:
        emit_graph_stream({"phase": "select_model", "message": "Continuing setup…"})
        return {
            **state,
            "selected_model": resolved,
            "model_explanation": "Configured from your preferences.",
            "task_type": state.get("task_type", "classification"),
            "current_step": "select_model",
        }

    explicit_pref = state.get("user_model_preference") or _extract_family_from_goal(
        state.get("goal", "")
    )

    if explicit_pref:
        family_key = explicit_pref

        if family_key not in MODEL_FAMILIES:
            return {
                **state,
                "error": f"Unknown model family: {family_key}. Available: {list(MODEL_FAMILIES.keys())}",
                "current_step": "select_model",
            }

        family_info = MODEL_FAMILIES[family_key]
        task_type = _derive_task_type(family_key, state.get("goal", ""))
        return {
            **state,
            "selected_model": family_key,
            "task_type": task_type,
            "model_explanation": "Configured from your goal.",
            "audit_trace": [
                *state.get("audit_trace", []),
                {
                    "step": "select_model",
                    "action": "user_specified",
                    "family": family_key,
                    "task_type": task_type,
                },
            ],
            "explanations": [
                *state.get("explanations", []),
                "Pipeline approach configured from your goal.",
            ],
            "current_step": "select_model",
        }

    llm = init_chat_model(model="gpt-5.1", temperature=0, streaming=True)
    structured_llm = llm.with_structured_output(ModelFamilySelectionOutput)

    redo_hint = state.get("_select_model_redo_hint", "")
    redo_section = ""
    if redo_hint:
        redo_section = (
            f"\n## User Feedback (IMPORTANT — override your default choice)\n\n"
            f"The user rejected the previous model family selection and said:\n"
            f'"{redo_hint}"\n\n'
            f"You MUST follow this feedback when choosing the model family.\n"
        )

    prompt = MODEL_FAMILY_SELECTION_PROMPT.format(
        families=format_model_families_for_prompt(),
        goal=state.get("goal", ""),
        redo_section=redo_section,
    )

    emit_graph_stream({
        "phase": "select_model",
        "message": "Continuing setup…",
    })
    result: ModelFamilySelectionOutput = structured_llm.invoke(prompt)
    task_type = _derive_task_type(result.selected_family, state.get("goal", ""))

    return {
        **state,
        "selected_model": result.selected_family,
        "task_type": task_type,
        "model_explanation": result.explanation.strip() or "Configured for your task.",
        "model_regen_count": state.get("model_regen_count", 0),
        "audit_trace": [
            *state.get("audit_trace", []),
            {
                "step": "select_model",
                "action": "llm_selected",
                "family": result.selected_family,
                "task_type": task_type,
                "confidence": result.confidence,
                "alternatives": result.alternative_families,
            },
        ],
        "explanations": [
            *state.get("explanations", []),
            "Pipeline approach configured for your task.",
        ],
        "current_step": "select_model",
    }
