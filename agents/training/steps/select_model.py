"""
Step 1: Model Selection Node
Selects the appropriate ML model based on the user's goal.

All model metadata is loaded dynamically from the skill registry —
no hard-coded model lists. Adding a new skill directory automatically
makes it available for selection.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field, field_validator

# Load environment variables
# Path: steps -> training -> agents -> root
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

from ..skill_registry import (
    build_alias_map,
    get_all_skill_names,
    get_all_skills_for_selection,
)

if TYPE_CHECKING:
    from ..core.state import TrainingAgentState


# =============================================================================
# DYNAMIC MODEL HELPERS (driven by skill metadata)
# =============================================================================


def format_training_models_for_prompt() -> str:
    """Format all discovered skills into a prompt-friendly string."""
    skills = get_all_skills_for_selection()
    lines: list[str] = []
    for meta in skills.values():
        lines.append(f"**{meta['name']}**")
        lines.append(f"Description: {meta.get('brief', '')}")
        when = meta.get("when_to_use", [])
        not_when = meta.get("when_not_to_use", [])
        if when:
            lines.append(f"When to use: {', '.join(when)}")
        if not_when:
            lines.append(f"When NOT to use: {', '.join(not_when)}")
        lines.append("")
    return "\n".join(lines)


_ALIAS_TO_ESTIMATOR: dict[str, str] = {
    "logistic regression": "LogisticRegression",
    "logistic_regression": "LogisticRegression",
    "logreg": "LogisticRegression",
    "random forest": "RandomForestClassifier",
    "random_forest": "RandomForestClassifier",
    "rf": "RandomForestClassifier",
    "gradient boosting": "GradientBoostingClassifier",
    "gradient_boosting": "GradientBoostingClassifier",
    "gbm": "GradientBoostingClassifier",
    "gradient boosted": "GradientBoostingClassifier",
    "xgboost": "GradientBoostingClassifier",
    "xgb": "GradientBoostingClassifier",
    "naive bayes": "GaussianNB",
    "naive_bayes": "GaussianNB",
    "nb": "GaussianNB",
    "svm": "SVC",
    "support vector machine": "SVC",
    "svc": "SVC",
    "svr": "SVR",
    "knn": "KNeighborsClassifier",
    "k-nearest neighbors": "KNeighborsClassifier",
    "nearest neighbors": "KNeighborsClassifier",
    "decision tree": "DecisionTreeClassifier",
    "dt": "DecisionTreeClassifier",
    "adaboost": "AdaBoostClassifier",
    "ada boost": "AdaBoostClassifier",
    "bagging": "BaggingClassifier",
    "extra trees": "ExtraTreesClassifier",
    "extremely randomized trees": "ExtraTreesClassifier",
    "mlp": "MLPClassifier",
    "neural network": "MLPClassifier",
    "multi-layer perceptron": "MLPClassifier",
    "ridge": "Ridge",
    "ridge classifier": "RidgeClassifier",
    "lasso": "Lasso",
    "elastic net": "ElasticNet",
    "elasticnet": "ElasticNet",
    "linear regression": "LinearRegression",
    "sgd": "SGDClassifier",
    "stochastic gradient descent": "SGDClassifier",
    "huber": "HuberRegressor",
    "huber regressor": "HuberRegressor",
}


def _resolve_alias(name: str, skills: dict) -> tuple[str, str | None]:
    """Resolve a model name/alias to (canonical_skill_name, estimator_hint).

    Handles underscores, spaces, and case variations so that
    "gradient_boosting", "Gradient Boosting", "GradientBoostingClassifier"
    all resolve to the correct skill (e.g. "sklearn_generic") AND return
    the specific sklearn estimator class name as a hint.

    Returns (skill_name, estimator_hint). estimator_hint is None if the
    name already matched a skill directly.
    """
    if name in skills:
        return name, None

    alias_map = build_alias_map()
    lower = name.lower()
    spaced = lower.replace("_", " ")

    def _find_estimator(key: str) -> str | None:
        """Look up the estimator class name for an alias."""
        return _ALIAS_TO_ESTIMATOR.get(key) or _ALIAS_TO_ESTIMATOR.get(key.replace("_", " "))

    # Direct lookup
    if lower in alias_map:
        return alias_map[lower], _find_estimator(lower)

    if spaced in alias_map:
        return alias_map[spaced], _find_estimator(spaced)

    # Substring match for compound names (e.g. "GradientBoostingClassifier")
    for alias in sorted(alias_map, key=len, reverse=True):
        if alias in lower or alias in spaced:
            return alias_map[alias], _find_estimator(alias)

    return name, None


def _extract_model_from_goal(goal: str) -> str | None:
    """Detect an explicit model request in the user's goal text.

    Returns the canonical skill name if found, otherwise None.
    Longer alias strings are checked first to avoid partial matches
    (e.g. "naive bayes" before "nb").
    """
    alias_map = build_alias_map()
    goal_lower = goal.lower()
    for alias in sorted(alias_map, key=len, reverse=True):
        if alias in goal_lower:
            return alias_map[alias]
    return None


# =============================================================================
# STRUCTURED OUTPUT SCHEMA
# =============================================================================


class ModelSelectionOutput(BaseModel):
    """Structured output for model selection."""

    selected_model: str = Field(
        description="The selected model type for training"
    )
    explanation: str = Field(
        description="Explanation of why this model was selected based on the goal"
    )
    confidence: str = Field(
        description="Confidence level in the model selection: high, medium, or low"
    )
    alternative_models: list[str] = Field(
        default_factory=list,
        description="Alternative models that could also work for this goal"
    )

    @field_validator("selected_model")
    @classmethod
    def must_be_known_skill(cls, v: str) -> str:
        known = get_all_skill_names()
        if v not in known:
            raise ValueError(
                f"Unknown model '{v}'. Must be one of: {sorted(known)}"
            )
        return v


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

1. **If the user explicitly names a model** (e.g. "train a naive bayes", "use xgboost", "logistic regression"), you MUST select that model. The user's explicit request overrides your own preference.
2. Otherwise, analyze the goal to understand what kind of prediction/modeling is needed and select the most appropriate model.
3. Explain your reasoning.
4. List any alternative models that could also work.
5. The selected_model value MUST be one of these exact names: {model_names}
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
    skills = get_all_skills_for_selection()

    # Resolve explicit preference: state field first, then parse goal text.
    explicit_pref = state.get("user_model_preference") or _extract_model_from_goal(state.get("goal", ""))

    if explicit_pref:
        # Resolve through alias map so that common names like
        # "gradient_boosting", "random_forest", "xgboost" etc. map to
        # their canonical skill (e.g. "sklearn_generic") AND get the
        # specific estimator class name hint.
        model_name, estimator_hint = _resolve_alias(explicit_pref, skills)

        if model_name not in skills:
            return {
                **state,
                "error": f"Unknown model: {model_name}. Available: {sorted(skills.keys())}",
                "current_step": "select_model",
            }

        model_info = skills[model_name]
        display_name = estimator_hint or model_name
        explanation = (
            f"User specified {display_name} (skill: {model_name}): "
            f"{model_info.get('brief', '')}"
        )
        return {
            **state,
            "selected_model": model_name,
            "estimator_hint": estimator_hint,
            "model_explanation": explanation,
            "audit_trace": [
                *state.get("audit_trace", []),
                {
                    "step": "select_model",
                    "action": "user_specified",
                    "model": model_name,
                    "estimator_hint": estimator_hint,
                },
            ],
            "explanations": [
                *state.get("explanations", []),
                f"Using user-specified model: {display_name}",
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
        model_names=", ".join(sorted(skills.keys())),
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
