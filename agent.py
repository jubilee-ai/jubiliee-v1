"""
Main Orchestrator Agent — Jubilee AI Chatbot

Uses langchain's create_agent to run a conversational loop that can spin up
two specialised sub-agents on demand:

  1. Analysis sub-agent  (agents/analysis_agent_v2)
     → statistical analysis, pretrained-model inference, data exploration

  2. Training sub-agent  (agents/training)
     → full ML pipeline: data collection → cleaning → feature eng → training → report

On top of that the orchestrator can:
  • List / inspect trained models from the registry
  • Run predictions against any trained model with user-supplied inputs
  • Clarify ambiguous requests before acting
  • Offer prediction sessions ("chat with the model") after training
"""

import json
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path / env bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env")

_MODEL_STORAGE_DIR = _ROOT / "tools" / "models-tools" / "training"
if str(_MODEL_STORAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_STORAGE_DIR))

from model_storage import (  # noqa: E402
    get_model_info,
    list_models,
    load_model,
)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-5.1", temperature=0)


# ============================================================================
# Tool 1 — Data Analysis (delegates to analysis_agent_v2)
# ============================================================================

class AnalyzeDataInput(BaseModel):
    question: str = Field(
        description="The analytical question to answer (e.g. 'What is the distribution of income in the loan dataset?')"
    )


@tool(args_schema=AnalyzeDataInput)
def analyze_data(question: str) -> str:
    """Run the data-analysis sub-agent to answer a question that requires
    data exploration, statistical analysis, or pretrained-model inference.

    USE FOR:
    - Statistical profiling, correlations, trends
    - Pretrained model inference (sentiment, credit-risk scoring, forecasting …)
    - Dataset exploration and lookup

    DO NOT USE FOR:
    - Training custom models  → use train_model
    - Predictions with user-trained models → use predict
    """
    from agents.analysis_agent_v2 import agent as analysis_agent

    result = analysis_agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "Analysis completed but no response was generated."


# ============================================================================
# Tool 2 — Model Training (delegates to the training LangGraph pipeline)
# ============================================================================

class TrainModelInput(BaseModel):
    goal: str = Field(
        description="Training objective — INCLUDE the model type the user wants "
        "(e.g. 'Train a GradientBoostingClassifier for loan-default prediction on the Loan_default dataset')"
    )
    linked_datasets: Optional[list[str]] = Field(
        default=None,
        description="Dataset references the agent should use (e.g. ['csv_Loan_default'])",
    )
    model_preference: Optional[str] = Field(
        default=None,
        description="Preferred algorithm. Common values: logistic_regression, random_forest, "
        "gradient_boosting, xgboost, naive_bayes, svm, knn, decision_tree, "
        "ridge, lasso, mlp, extra_trees, adaboost, sgd, glm. "
        "All sklearn estimators are supported via the sklearn_generic skill.",
    )


_last_training_state: dict[str, Any] = {}

# Queue that receives real-time training step events.  Set by the SSE
# generator in app.py before invoking the orchestrator, cleared after.
_training_step_sink: Optional[Any] = None  # queue.Queue | None

# Queue through which the frontend sends HITL decisions back to the
# blocked _run_training_to_completion function.
_training_decision_queue: Optional[Any] = None  # queue.Queue | None

_training_lock = threading.Lock()


def set_training_step_sink(q) -> None:
    """Set (or clear) the queue that receives training step events."""
    global _training_step_sink
    _training_step_sink = q


def set_training_decision_queue(q) -> None:
    """Set (or clear) the queue used to receive HITL decisions."""
    global _training_decision_queue
    _training_decision_queue = q


def submit_training_decision(decision: dict) -> None:
    """Submit a HITL decision for the in-flight chat training agent."""
    if _training_decision_queue is not None:
        _training_decision_queue.put(decision)


TOOL_TO_STEP = {
    "tool_data_collection": "data_collection",
    "tool_select_model": "select_model",
    "tool_cleaning": "cleaning",
    "tool_label_split_definition": "label_split_definition",
    "tool_feature_selection_specification": "feature_selection_specification",
    "tool_feature_engineering_executor": "feature_engineering_executor",
    "tool_training_approval": "training_approval",
    "tool_training": "training",
    "tool_generate_report": "generate_report",
}


def _run_training_to_completion(
    goal: str,
    linked_datasets: Optional[list[str]],
    model_preference: Optional[str],
) -> dict[str, Any]:
    """Run the simple training agent end-to-end.

    Uses ``agent.stream()`` so that step events can be pushed to
    ``_training_step_sink`` in real-time for the chat SSE generator.

    When a sink is connected (chat mode), HITL is enabled and LangGraph
    interrupts are forwarded to the frontend.  The function blocks on
    ``_training_decision_queue`` until the user responds, then resumes
    the agent.
    """
    if not _training_lock.acquire(blocking=False):
        return {
            "success": False,
            "error": "Another training run is already in progress. Please wait for it to finish.",
            "model_name": "none",
            "model_type": "none",
        }

    try:
        return _run_training_to_completion_locked(goal, linked_datasets, model_preference)
    finally:
        _training_lock.release()


def _run_training_to_completion_locked(
    goal: str,
    linked_datasets: Optional[list[str]],
    model_preference: Optional[str],
) -> dict[str, Any]:
    from agents.training.agent_simple import create_simple_training_agent
    from agents.training.utils.streaming import (
        build_node_update,
        calculate_progress,
        extract_interrupt_info,
    )
    from langgraph.types import Command

    sink = _training_step_sink
    decision_q = _training_decision_queue
    use_hitl = sink is not None and decision_q is not None

    thread_id = f"orch-{uuid.uuid4().hex[:8]}"
    agent, shared_state = create_simple_training_agent(
        goal=goal,
        linked_datasets=linked_datasets,
        user_model_preference=model_preference,
        hitl=use_hitl,
    )

    config = {"configurable": {"thread_id": thread_id}}
    emitted_starts: set[str] = set()
    emitted_steps: set[str] = set()

    if sink is not None:
        sink.put(("training_step", {"type": "training_started"}))

    # Drain any stale decisions that might have queued up
    if decision_q is not None:
        while not decision_q.empty():
            try:
                decision_q.get_nowait()
            except Exception:
                break

    input_data: Any = {"messages": [{"role": "user", "content": goal}]}

    while True:
        resume_with = None

        for event in agent.stream(input_data, config=config, stream_mode="updates"):
            # ── Interrupt (HITL) ──────────────────────────────────
            if "__interrupt__" in event:
                info = extract_interrupt_info(event["__interrupt__"])
                if sink is not None:
                    sink.put(("training_step", {"type": "interrupt", **info}))

                if decision_q is not None:
                    decision = decision_q.get()  # blocks until user responds
                    resume_with = Command(resume=decision)
                else:
                    resume_with = Command(resume={"approved": True})
                continue

            if sink is None:
                continue

            # ── node_started from model/agent tool_calls ──────────
            model_output = event.get("model") or event.get("agent")
            if model_output:
                for msg in model_output.get("messages", []):
                    for tc in getattr(msg, "tool_calls", None) or []:
                        tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                        if not tool_name:
                            continue
                        step_name = TOOL_TO_STEP.get(tool_name)
                        if not step_name or step_name in emitted_starts:
                            continue
                        emitted_starts.add(step_name)
                        sink.put(("training_step", {
                            "type": "node_started",
                            "node": step_name,
                            "progress": calculate_progress(step_name),
                        }))

            # ── node_complete from tool results ───────────────────
            if "tools" in event:
                tool_msgs = event["tools"].get("messages", [])
                if tool_msgs:
                    content = getattr(tool_msgs[0], "content", "")
                    if isinstance(content, str) and (
                        content.startswith("REJECTED") or content.startswith("SKIP:")
                    ):
                        continue
                    tool_name = getattr(tool_msgs[0], "name", "unknown")
                    step_name = TOOL_TO_STEP.get(tool_name)
                    if not step_name or step_name in emitted_steps:
                        continue
                    emitted_steps.add(step_name)
                    slim = {k: v for k, v in shared_state.items() if k != "split_indices"}
                    update = build_node_update(step_name, slim)
                    sink.put(("training_step", update))

        if resume_with is not None:
            input_data = resume_with
        else:
            break

    final_state = dict(shared_state)
    _last_training_state.clear()
    _last_training_state.update(final_state)

    if sink is not None:
        sink.put(("training_step", {"type": "training_completed"}))

    return final_state


@tool(args_schema=TrainModelInput)
def train_model(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    model_preference: Optional[str] = None,
) -> str:
    """Train a new ML model using the training sub-agent.

    Runs the full pipeline: model selection → data collection → cleaning →
    feature engineering → training → evaluation → report.

    Supports ALL scikit-learn estimators (GradientBoostingClassifier,
    RandomForestClassifier, LogisticRegression, SVC, XGBoost, etc.).
    Include the desired model type in the goal text.

    This tool may take several minutes.
    """
    try:
        result = _run_training_to_completion(goal, linked_datasets, model_preference)

        error = result.get("error")
        if error:
            return f"Training encountered an error: {error}"

        metrics = result.get("training_metrics", {})
        model_name = metrics.get("model_name") or result.get("model_weights_path", "unknown")
        model_type = metrics.get("model_type") or result.get("estimator_hint") or result.get("selected_model", "unknown")

        return (
            f"Training completed. Model: {model_name} ({model_type}). "
            f"The step-by-step results and report are shown above in the training panel. "
            f"The model is saved and ready for predictions."
        )

    except Exception as exc:
        return f"Training failed: {type(exc).__name__}: {exc}"


# ============================================================================
# Tool 3 — List Trained Models
# ============================================================================

class CheckModelsInput(BaseModel):
    model_type: Optional[str] = Field(
        default=None,
        description="Optional filter by model type (e.g. 'sklearn_logistic_regression')",
    )


@tool(args_schema=CheckModelsInput)
def check_trained_models(model_type: Optional[str] = None) -> str:
    """List all trained models in the registry.

    Use BEFORE training to avoid duplicates, and BEFORE predicting to find
    the right model name. Returns model names, types, target columns,
    feature lists, and key metrics.
    """
    models = list_models()
    if model_type:
        models = [m for m in models if m.get("model_type") == model_type]

    if not models:
        qualifier = f" of type '{model_type}'" if model_type else ""
        return f"No trained models found{qualifier}. Use `train_model` to create one."

    models.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
    lines = [f"Found {len(models)} trained model(s):\n"]
    for m in models:
        metrics = m.get("metrics", {})
        feat_names = m.get("feature_names", [])
        lines.append(f"• {m['model_name']}")
        lines.append(f"    Type        : {m['model_type']}")
        lines.append(f"    Description : {m.get('description', 'N/A')}")
        lines.append(f"    Target      : {m.get('target_column', 'N/A')}")
        lines.append(f"    Samples     : {m.get('training_samples', 'N/A')}")
        preview = ", ".join(feat_names[:6])
        if len(feat_names) > 6:
            preview += f" … (+{len(feat_names) - 6} more)"
        lines.append(f"    Features    : {preview}")

        acc = metrics.get("test_accuracy") or metrics.get("test_score")
        roc = metrics.get("test_roc_auc") or metrics.get("roc_auc")
        if acc is not None:
            lines.append(f"    Accuracy    : {acc:.4f}" if isinstance(acc, (int, float)) else f"    Accuracy    : {acc}")
        if roc is not None:
            lines.append(f"    ROC-AUC     : {roc:.4f}" if isinstance(roc, (int, float)) else f"    ROC-AUC     : {roc}")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# Tool 4 — Predict with a Trained Model
# ============================================================================

class PredictInput(BaseModel):
    model_name: str = Field(
        description="Exact name of the trained model (from check_trained_models)"
    )
    input_data: dict = Field(
        description="Feature values as a dict, e.g. {'age': 35, 'income': 50000, 'loan_amount': 20000}"
    )


@tool(args_schema=PredictInput)
def predict(model_name: str, input_data: dict) -> str:
    """Make a prediction with a user-trained model.

    Pass a dictionary of feature values. The tool maps them to the model's
    expected schema, runs inference, and returns the predicted class (with
    probabilities for classifiers) or predicted value (for regressors).
    """
    info = get_model_info(model_name)
    if info is None:
        available = [m["model_name"] for m in list_models()]
        return (
            f"Model '{model_name}' not found in registry.\n"
            f"Available models: {available if available else 'None — train one first.'}"
        )

    try:
        model = load_model(model_name)
    except Exception as exc:
        return f"Failed to load model '{model_name}': {exc}"

    feature_names: list[str] = info.get("feature_names", [])
    row = {}
    missing = []
    for f in feature_names:
        if f in input_data:
            row[f] = input_data[f]
        else:
            row[f] = 0
            missing.append(f)

    df = pd.DataFrame([row])

    try:
        pred = model.predict(df)
    except Exception as exc:
        return f"Prediction failed: {exc}"

    lines = [f"Prediction from '{model_name}':"]
    if missing:
        lines.append(f"  ⚠ Missing features (defaulted to 0): {missing}")

    is_classifier = hasattr(model, "predict_proba") and hasattr(model, "classes_")
    if is_classifier:
        probas = model.predict_proba(df)[0]
        classes = model.classes_
        lines.append(f"  Predicted class: {pred[0]}")
        for cls, prob in zip(classes, probas):
            lines.append(f"    P({cls}) = {prob:.4f}")
    else:
        val = pred[0]
        lines.append(
            f"  Predicted value: {val:.4f}" if isinstance(val, float) else f"  Predicted value: {val}"
        )

    return "\n".join(lines)


# ============================================================================
# Tool 5 — Get Model Details
# ============================================================================

class ModelDetailsInput(BaseModel):
    model_name: str = Field(description="Name of the model to inspect")


@tool(args_schema=ModelDetailsInput)
def get_model_details(model_name: str) -> str:
    """Return full metadata for a trained model — hyperparameters, feature list,
    metrics, and file path. Useful before making predictions so you know what
    features are expected.
    """
    info = get_model_info(model_name)
    if info is None:
        available = [m["model_name"] for m in list_models()]
        return f"Model '{model_name}' not found. Available: {available}"

    metrics = info.get("metrics", {})
    hp = info.get("hyperparameters", {})

    lines = [
        f"Model: {model_name}",
        f"  Type        : {info['model_type']}",
        f"  Description : {info.get('description', 'N/A')}",
        f"  Target      : {info.get('target_column', 'N/A')}",
        f"  Classes     : {info.get('classes', [])}",
        f"  Samples     : {info.get('training_samples', 'N/A')}",
        f"  Created     : {info.get('created_at', 'N/A')}",
        "",
        "  Features:",
    ]
    for feat in info.get("feature_names", []):
        lines.append(f"    - {feat}")

    lines.append("")
    lines.append("  Metrics:")
    for k, v in metrics.items():
        lines.append(f"    {k}: {v}")

    if hp:
        lines.append("")
        lines.append("  Hyperparameters:")
        for k, v in hp.items():
            lines.append(f"    {k}: {v}")

    return "\n".join(lines)


# ============================================================================
# System Prompt
# ============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are **Jubilee**, an AI assistant for data analysis and machine learning.

## Available Tools

| Tool | When to use |
|---|---|
| `analyze_data` | Analytical questions: statistics, trends, pretrained-model inference, data exploration |
| `train_model` | User explicitly wants to train / build a custom ML model |
| `check_trained_models` | See what models already exist (always check BEFORE training or predicting) |
| `predict` | Run a trained model on user-supplied inputs |
| `get_model_details` | Inspect a model's features, metrics, and hyperparameters |

## Decision Flow

1. **General / conversational question** → answer directly, no tool needed.
2. **Analytical question** (e.g. "what trends …", "analyze …", "what is the distribution …")
   → `analyze_data`
3. **User wants a prediction**
   a. `check_trained_models` to see if a suitable model exists.
   b. If a model exists → `get_model_details` (so you know the features) → `predict`.
   c. If NO model exists → tell the user no model is available and recommend they
      train one. If they mention a specific dataset, use it; otherwise the
      training pipeline can discover suitable data on its own.
4. **User wants to train a model** → ALWAYS use the `train_model` tool.
   The training pipeline supports ALL scikit-learn estimators including:
   LogisticRegression, RandomForestClassifier, GradientBoostingClassifier,
   ExtraTreesClassifier, SVC, KNeighborsClassifier, DecisionTreeClassifier,
   AdaBoostClassifier, MLPClassifier, Ridge, Lasso, and all their regression
   counterparts. Include the specific model type in the `goal` parameter.
   NEVER output raw Python code for the user to run — always delegate to `train_model`.
5. **After training** → Keep your response VERY brief (1-2 sentences).
   The training steps and report are already shown to the user in the training
   panel above your message. Just say something like "Training complete! You can
   check the detailed results in the report above. Want to make predictions?"
   Do NOT repeat metrics, steps, or details that are already visible in the panel.
6. **Ambiguous request** → If the user's intent to train is clear but details are
   missing, proceed with `train_model` using reasonable defaults rather than
   asking many clarifying questions. The pipeline handles dataset discovery,
   model selection, and hyperparameter tuning automatically.

## Rules
- ALWAYS call `check_trained_models` before recommending training — avoid duplicates.
- ALWAYS call `get_model_details` before `predict` so you pass the right features.
- NEVER output training code for the user to run manually. ALWAYS use `train_model`.
- When the user describes a scenario for prediction, extract feature values from
  their description and construct the `input_data` dict yourself.
- Explain predictions in plain language after showing the raw output.
- Be concise but thorough. Show your reasoning when it helps the user.\
"""


# ============================================================================
# Agent
# ============================================================================

TOOLS = [
    analyze_data,
    train_model,
    check_trained_models,
    predict,
    get_model_details,
]

# In-memory checkpointer keeps full message history per thread so each
# /api/chat call only needs to send the newest user message.
_checkpointer = MemorySaver()

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    checkpointer=_checkpointer,
)

__all__ = ["agent", "TOOLS", "ORCHESTRATOR_SYSTEM_PROMPT"]
