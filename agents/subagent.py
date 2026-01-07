"""
Subagent - A LangGraph subgraph that executes a single analysis subtask.

Flow: select_model → prepare_data → execute_model → reflect → check_complete
      ↑                                                          ↓
      └──────────────── iterate (if not complete) ───────────────┘
      
When complete: synthesize and return results.
"""

import json
import sys
from pathlib import Path
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Add model tools to path
_TOOLS_DIR = Path(__file__).parent.parent / "tools" / "models-tools" / "pretrained"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from .model_index import (format_models_for_prompt, get_model_by_name,
                          get_model_index)
from .prompts import (prepare_data_prompt, reflect_prompt, select_model_prompt,
                      subagent_synthesize_prompt)

# =============================================================================
# LLM Setup
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# =============================================================================
# Subagent State
# =============================================================================


class SubagentState(TypedDict):
    """State for a single subagent executing one subtask."""
    # Context from parent
    query: str
    user_data: dict | None
    
    # Subtask info
    goal: str
    
    # Execution state
    status: str  # "pending" | "in_progress" | "complete" | "failed"
    selected_model: str | None
    model_schema: dict | None
    prepared_data: dict | None
    result: str | None
    reflection: dict | None
    
    # Iteration tracking
    iteration_count: int
    max_iterations: int
    
    # Final output
    synthesis: dict | None


# =============================================================================
# Helper Functions
# =============================================================================


def parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    return json.loads(content)


def execute_model_impl(model_name: str, data: dict) -> str:
    """Execute a model and return the result string."""
    if model_name == "credit_card_risk_prediction":
        from credit_risk import CreditCardRiskInput, predict_credit_risk
        result = predict_credit_risk(CreditCardRiskInput(**data))
        return f"Risk: {result.risk}, Probabilities: {result.probabilities}"

    elif model_name == "loan_default_prediction":
        from loan_default_prediction import (LoanApplicantInput,
                                             predict_loan_default)
        result = predict_loan_default(LoanApplicantInput(**data))
        return f"Prediction: {result.prediction}, Default Probability: {result.default_probability:.1%}"

    elif model_name == "finbert_tone":
        from finbert_tone import FinBERTToneInput, analyze_tone
        result = analyze_tone(FinBERTToneInput(**data))
        return f"Sentiment: {result.sentiment}, Confidence: {result.confidence:.2%}"

    elif model_name == "finbert_sentiment":
        from prosus_finbert import FinBERTInput, analyze_sentiment
        result = analyze_sentiment(FinBERTInput(**data))
        return f"Sentiment: {result.sentiment}, Confidence: {result.confidence:.2%}"

    elif model_name == "claim_detection":
        from bert_finetuned_claim_detection import (ClaimDetectionInput,
                                                    detect_claim)
        result = detect_claim(ClaimDetectionInput(**data))
        return f"Classification: {result.label}, Confidence: {result.confidence:.2%}"

    elif model_name == "chronos2_forecast":
        from chronos_2 import Chronos2Input, forecast
        context_data = [{"item_id": "s1", "timestamp": i, "target": v} for i, v in enumerate(data["values"])]
        result = forecast(Chronos2Input(
            context_data=context_data,
            prediction_length=data.get("prediction_length", 12),
        ))
        return f"Forecast: {len(result.predictions)} steps predicted"

    elif model_name == "timesfm_forecast":
        from google_timesfm import TimesFMInput
        from google_timesfm import forecast as timesfm_forecast
        result = timesfm_forecast(TimesFMInput(inputs=[data["values"]], horizon=data.get("horizon", 12)))
        return f"Forecast: {len(result.point_forecast[0])} steps"

    else:
        return f"Model '{model_name}' execution not implemented"


# =============================================================================
# Subagent Nodes
# =============================================================================


def select_model(state: SubagentState) -> dict:
    """Select the best model for this subtask."""
    all_models = get_model_index()
    models_text = format_models_for_prompt(all_models)
    
    prompt = select_model_prompt(state["goal"], state.get("user_data"), models_text)
    response = llm.invoke(prompt)
    decision = parse_json_response(response.content)
    
    if decision["decision"] == "select":
        model = get_model_by_name(decision["model_name"])
        if model:
            return {
                "status": "in_progress",
                "selected_model": model["name"],
                "model_schema": model["schema"],
            }
        else:
            return {
                "status": "failed",
                "reflection": {"error": f"Model '{decision['model_name']}' not found"},
            }
    elif decision["decision"] == "train":
        return {
            "status": "failed",
            "reflection": {"error": "No suitable pretrained model", "recommendation": "Train custom model"},
        }
    else:  # skip
        return {
            "status": "in_progress",
            "selected_model": None,
            "model_schema": None,
        }


def prepare_data(state: SubagentState) -> dict:
    """Prepare data for model execution."""
    if state["status"] == "failed":
        return {}
    
    if not state.get("model_schema"):
        # No schema needed (skip case)
        return {"prepared_data": state.get("user_data")}
    
    prompt = prepare_data_prompt(state["goal"], state["model_schema"], state.get("user_data"))
    response = llm.invoke(prompt)
    prep_result = parse_json_response(response.content)
    
    if prep_result["data_ready"]:
        return {"prepared_data": prep_result["prepared_data"]}
    else:
        return {
            "status": "failed",
            "reflection": {
                "error": "Insufficient data",
                "missing_fields": prep_result.get("missing_fields", []),
                "notes": prep_result.get("notes", ""),
            },
        }


def execute_model(state: SubagentState) -> dict:
    """Execute the selected model on prepared data."""
    if state["status"] == "failed":
        return {}
    
    if not state.get("selected_model"):
        return {"result": "No model execution required for this subtask."}
    
    if not state.get("prepared_data"):
        return {
            "status": "failed",
            "reflection": {"error": "No prepared data available"},
        }
    
    try:
        result = execute_model_impl(state["selected_model"], state["prepared_data"])
        return {"result": result}
    except Exception as e:
        return {
            "status": "failed",
            "reflection": {"error": f"Model execution failed: {str(e)}"},
        }


def reflect(state: SubagentState) -> dict:
    """Generate structured reflection on the execution."""
    if state["status"] == "failed":
        return {}
    
    if state.get("reflection"):
        # Already has reflection (error case)
        return {}
    
    prompt = reflect_prompt(
        state["goal"],
        state.get("selected_model"),
        state.get("prepared_data"),
        state.get("result"),
    )
    response = llm.invoke(prompt)
    reflection = parse_json_response(response.content)
    
    return {"reflection": reflection}


def check_complete(state: SubagentState) -> dict:
    """Check if subtask is complete or needs iteration."""
    # Always mark as complete after hitting iteration limit
    if state["iteration_count"] >= state["max_iterations"]:
        return {"status": "complete" if state.get("result") else "failed"}
    
    if state["status"] == "failed":
        # Only retry if we haven't tried this model before
        # Don't retry same failure - move to complete/failed
        if state.get("reflection", {}).get("error"):
            # Has error - don't retry, just finish
            return {"status": "failed"}
        # Otherwise try once more
        return {
            "iteration_count": state["iteration_count"] + 1,
            "status": "pending",
            "selected_model": None,
            "model_schema": None,
            "prepared_data": None,
            "result": None,
        }
    
    # Check if we have a valid result
    if state.get("result") and state.get("reflection"):
        return {"status": "complete"}
    
    return {"status": "complete"}


def synthesize(state: SubagentState) -> dict:
    """Generate final synthesis for this subtask."""
    prompt = subagent_synthesize_prompt(
        query=state["query"],
        goal=state["goal"],
        model=state.get("selected_model"),
        data=state.get("prepared_data"),
        result=state.get("result"),
        reflection=state.get("reflection"),
    )
    response = llm.invoke(prompt)
    synthesis = parse_json_response(response.content)
    
    return {"synthesis": synthesis}


# =============================================================================
# Routing
# =============================================================================


def route_after_check(state: SubagentState) -> str:
    """Route after check_complete: iterate or synthesize."""
    if state["status"] == "pending":
        return "select_model"
    return "synthesize"


# =============================================================================
# Subgraph Builder
# =============================================================================


def build_subagent_graph() -> StateGraph:
    """Build the subagent LangGraph."""
    graph = StateGraph(SubagentState)
    
    # Add nodes
    graph.add_node("select_model", select_model)
    graph.add_node("prepare_data", prepare_data)
    graph.add_node("execute_model", execute_model)
    graph.add_node("reflect", reflect)
    graph.add_node("check_complete", check_complete)
    graph.add_node("synthesize", synthesize)
    
    # Set entry point
    graph.set_entry_point("select_model")
    
    # Linear flow through execution
    graph.add_edge("select_model", "prepare_data")
    graph.add_edge("prepare_data", "execute_model")
    graph.add_edge("execute_model", "reflect")
    graph.add_edge("reflect", "check_complete")
    
    # Conditional: iterate or complete
    graph.add_conditional_edges(
        "check_complete",
        route_after_check,
        {
            "select_model": "select_model",
            "synthesize": "synthesize",
        },
    )
    
    # End after synthesis
    graph.add_edge("synthesize", END)
    
    return graph


# =============================================================================
# Public API
# =============================================================================


def run_subagent(query: str, goal: str, user_data: dict | None, max_iterations: int = 3) -> dict:
    """
    Run the subagent graph for a single subtask.
    
    Args:
        query: The original user query (for context)
        goal: The specific goal for this subtask
        user_data: User-provided data
        max_iterations: Maximum retry attempts
    
    Returns:
        The synthesis dict from the subagent
    """
    graph = build_subagent_graph()
    app = graph.compile()
    
    initial_state: SubagentState = {
        "query": query,
        "user_data": user_data,
        "goal": goal,
        "status": "pending",
        "selected_model": None,
        "model_schema": None,
        "prepared_data": None,
        "result": None,
        "reflection": None,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "synthesis": None,
    }
    
    final_state = app.invoke(initial_state, {"recursion_limit": 30})
    
    return final_state.get("synthesis") or {
        "task_goal": goal,
        "status": "failed",
        "model_used": None,
        "key_findings": "Subagent failed to produce synthesis",
        "metrics": {},
        "confidence": "low",
        "limitations": ["Execution error"],
        "recommendation": "Investigate error",
    }


# Compile a reusable subgraph instance
_subagent_graph = None


def get_compiled_subagent():
    """Get or create a compiled subagent graph (for reuse)."""
    global _subagent_graph
    if _subagent_graph is None:
        _subagent_graph = build_subagent_graph().compile()
    return _subagent_graph

