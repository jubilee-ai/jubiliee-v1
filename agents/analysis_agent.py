"""
Analysis Agent - Orchestrates model selection, data retrieval, and execution.

Uses LangGraph for state management and OpenAI GPT-5-mini for LLM calls.
"""

import json
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Add model tools to path (handles hyphenated directory names)
_TOOLS_DIR = Path(__file__).parent.parent / "tools" / "models-tools" / "pretrained"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from .model_index import (format_models_for_prompt, get_model_by_name,
                          get_model_index)
from .prompts import (check_complete_prompt, decompose_query_prompt,
                      generate_report_prompt, prepare_data_prompt,
                      reflect_prompt, select_model_prompt, synthesize_prompt)

# =============================================================================
# LLM Setup
# =============================================================================

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)


# =============================================================================
# State Definition (TypedDict for LangGraph)
# =============================================================================


class Subtask(TypedDict):
    """A single subtask extracted from the user query."""
    goal: str
    status: str  # "pending" | "in_progress" | "complete" | "failed"
    selected_model: str | None
    model_schema: dict | None
    prepared_data: dict | None
    result: str | None
    reflection: dict | None
    iteration_count: int
    max_iterations: int


class AgentState(TypedDict):
    """Main state for the analysis agent."""
    # Input
    query: str
    user_data: dict | None
    
    # Decomposition
    subtasks: list[Subtask]
    current_subtask_idx: int
    
    # Synthesis
    all_reflections: list[dict]
    is_complete: bool
    final_report: str | None

# TODO: Context engineering
# TODO: have an initial 'planning' step which sets up a plan
# - Then delegate subagents to run in parallel (optionally)
# - Then have the main agent review the subagents 'synthesis' results
#   and decide if we're complete or need more.
#   - If more, then plan for next steps and do it again
#   - else, then generate the report

# TODO: Add in the model training agent
# - If we retrieve models ask the LLM:
# --> Do we have the right models or do we need to train a new one?
# --> Should be very based on if a model matches our schema or not and matches our goal


def make_subtask(goal: str) -> Subtask:
    """Create a new subtask with default values."""
    return Subtask(
        goal=goal,
        status="pending",
        selected_model=None,
        model_schema=None,
        prepared_data=None,
        result=None,
        reflection=None,
        iteration_count=0,
        max_iterations=3,
    )


# =============================================================================
# Node Functions
# =============================================================================


def decompose_query(state: AgentState) -> dict:
    """Break user query into independent subtasks."""
    prompt = decompose_query_prompt(state["query"], state.get("user_data"))
    response = llm.invoke(prompt)
    content = response.content.strip()

    # Parse JSON from response
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    subtask_data = json.loads(content)
    subtasks = [make_subtask(s["goal"]) for s in subtask_data]

    return {
        "subtasks": subtasks,
        "current_subtask_idx": 0,
        "all_reflections": [],
        "is_complete": False,
    }

# TODO: Do this with searching and finding the most relevant 3 models first.
# --> Then use an LLM to make the final decision similar to now.
def select_model(state: AgentState) -> dict:
    """Search for and select the best model for current subtask."""
    idx = state["current_subtask_idx"]
    subtasks = list(state["subtasks"])  # Copy for modification
    subtask = dict(subtasks[idx])  # Copy subtask
    
    subtask["status"] = "in_progress"

    # Get all models and do simple semantic matching via LLM
    all_models = get_model_index()
    models_text = format_models_for_prompt(all_models)

    prompt = select_model_prompt(subtask["goal"], state.get("user_data"), models_text)
    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    decision_data = json.loads(content)

    if decision_data["decision"] == "select":
        model = get_model_by_name(decision_data["model_name"])
        if model:
            subtask["selected_model"] = model["name"]
            subtask["model_schema"] = model["schema"]
    elif decision_data["decision"] == "train":
        subtask["status"] = "failed"
        subtask["reflection"] = {
            "error": "No suitable pretrained model found",
            "recommendation": "Train a custom model for this task",
        }

    subtasks[idx] = subtask
    return {"subtasks": subtasks}

# TODO: fix the prepare data. first add a node before to check if we need to call the data retrieval agrent or if the user input is enough. then it should have it's own iterative agent which is binded to the data -tools and iterates ubtil it ha the data it want to run the model on
def prepare_data(state: AgentState) -> dict:
    """Prepare data for model execution - either from user input or retrieval."""
    idx = state["current_subtask_idx"]
    subtasks = list(state["subtasks"])
    subtask = dict(subtasks[idx])

    if subtask["status"] == "failed":
        return {"subtasks": subtasks}

    if not subtask["model_schema"]:
        # No schema needed (skip case)
        subtask["prepared_data"] = state.get("user_data")
        subtasks[idx] = subtask
        return {"subtasks": subtasks}

    schema = subtask["model_schema"]

    prompt = prepare_data_prompt(subtask["goal"], schema, state.get("user_data"))
    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    prep_result = json.loads(content)

    if prep_result["data_ready"]:
        subtask["prepared_data"] = prep_result["prepared_data"]
    else:
        subtask["status"] = "failed"
        subtask["reflection"] = {
            "error": "Insufficient data",
            "missing_fields": prep_result.get("missing_fields", []),
            "notes": prep_result.get("notes", ""),
        }

    subtasks[idx] = subtask
    return {"subtasks": subtasks}


def execute_model(state: AgentState) -> dict:
    """Execute the selected model on prepared data."""
    idx = state["current_subtask_idx"]
    subtasks = list(state["subtasks"])
    subtask = dict(subtasks[idx])

    if subtask["status"] == "failed" or not subtask["prepared_data"]:
        return {"subtasks": subtasks}

    if not subtask["selected_model"]:
        subtask["result"] = "No model execution required for this subtask."
        subtasks[idx] = subtask
        return {"subtasks": subtasks}

    try:
        model_name = subtask["selected_model"]
        data = subtask["prepared_data"]

        if model_name == "credit_card_risk_prediction":
            from credit_risk import CreditCardRiskInput, predict_credit_risk
            result = predict_credit_risk(CreditCardRiskInput(**data))
            subtask["result"] = f"Risk: {result.risk}, Probabilities: {result.probabilities}"

        elif model_name == "loan_default_prediction":
            from loan_default_prediction import (LoanApplicantInput,
                                                 predict_loan_default)
            result = predict_loan_default(LoanApplicantInput(**data))
            subtask["result"] = f"Prediction: {result.prediction}, Default Probability: {result.default_probability:.1%}"

        elif model_name == "finbert_tone":
            from finbert_tone import FinBERTToneInput, analyze_tone
            result = analyze_tone(FinBERTToneInput(**data))
            subtask["result"] = f"Sentiment: {result.sentiment}, Confidence: {result.confidence:.2%}"

        elif model_name == "finbert_sentiment":
            from prosus_finbert import FinBERTInput, analyze_sentiment
            result = analyze_sentiment(FinBERTInput(**data))
            subtask["result"] = f"Sentiment: {result.sentiment}, Confidence: {result.confidence:.2%}"

        elif model_name == "claim_detection":
            from bert_finetuned_claim_detection import (ClaimDetectionInput,
                                                        detect_claim)
            result = detect_claim(ClaimDetectionInput(**data))
            subtask["result"] = f"Classification: {result.label}, Confidence: {result.confidence:.2%}"

        elif model_name == "chronos2_forecast":
            from chronos_2 import Chronos2Input, forecast
            context_data = [{"item_id": "s1", "timestamp": i, "target": v} for i, v in enumerate(data["values"])]
            result = forecast(Chronos2Input(
                context_data=context_data,
                prediction_length=data.get("prediction_length", 12),
            ))
            subtask["result"] = f"Forecast: {len(result.predictions)} steps predicted"

        elif model_name == "timesfm_forecast":
            from google_timesfm import TimesFMInput
            from google_timesfm import forecast as timesfm_forecast
            result = timesfm_forecast(TimesFMInput(inputs=[data["values"]], horizon=data.get("horizon", 12)))
            subtask["result"] = f"Forecast: {len(result.point_forecast[0])} steps"

        else:
            subtask["result"] = f"Model '{model_name}' execution not implemented"

    except Exception as e:
        subtask["status"] = "failed"
        subtask["reflection"] = {"error": f"Model execution failed: {str(e)}"}

    subtasks[idx] = subtask
    return {"subtasks": subtasks}


def reflect(state: AgentState) -> dict:
    """Generate structured reflection on subtask execution."""
    idx = state["current_subtask_idx"]
    subtasks = list(state["subtasks"])
    subtask = dict(subtasks[idx])

    if subtask.get("reflection"):  # Already has error reflection
        subtasks[idx] = subtask
        return {"subtasks": subtasks}

    prompt = reflect_prompt(
        subtask["goal"],
        subtask["selected_model"],
        subtask["prepared_data"],
        subtask["result"],
    )
    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    subtask["reflection"] = json.loads(content)
    subtasks[idx] = subtask
    return {"subtasks": subtasks}


def check_subtask_complete(state: AgentState) -> dict:
    """LLM reviews all work done so far and decides if goal is met or more work needed."""
    idx = state["current_subtask_idx"]
    subtasks = list(state["subtasks"])
    subtask = dict(subtasks[idx])
    all_reflections = list(state.get("all_reflections", []))

    # If execution failed, check if we should retry
    if subtask["status"] == "failed":
        if subtask["iteration_count"] < subtask["max_iterations"]:
            subtask["iteration_count"] += 1
            subtask["status"] = "pending"
            subtask["selected_model"] = None
            subtask["model_schema"] = None
            subtask["prepared_data"] = None
            subtask["result"] = None
            subtasks[idx] = subtask
            return {"subtasks": subtasks, "current_subtask_idx": idx}
        else:
            # Max retries reached - mark failed and move on
            all_reflections.append({
                "subtask_goal": subtask["goal"],
                "status": "failed",
                **(subtask.get("reflection") or {"error": "Max retries exceeded"}),
            })
            subtasks[idx] = subtask
            return {
                "subtasks": subtasks,
                "current_subtask_idx": idx + 1,
                "all_reflections": all_reflections,
            }

    # LLM reviews if the subtask goal has been accomplished
    prompt = check_complete_prompt(
        query=state["query"],
        goal=subtask["goal"],
        model=subtask.get("selected_model"),
        data=subtask.get("prepared_data"),
        result=subtask.get("result"),
        reflection=subtask.get("reflection"),
        iteration=subtask["iteration_count"],
        max_iterations=subtask["max_iterations"],
    )
    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    review = json.loads(content)

    if review["goal_accomplished"] and review["confidence"] != "low":
        # Goal accomplished - mark complete and move to next subtask
        subtask["status"] = "complete"
        all_reflections.append({
            "subtask_goal": subtask["goal"],
            "status": "complete",
            "result": subtask.get("result"),
            "confidence": review["confidence"],
            "reason": review["reason"],
            **(subtask.get("reflection") or {}),
        })
        subtasks[idx] = subtask
        return {
            "subtasks": subtasks,
            "current_subtask_idx": idx + 1,
            "all_reflections": all_reflections,
        }
    else:
        # Goal not accomplished - check if we can retry
        if subtask["iteration_count"] < subtask["max_iterations"]:
            subtask["iteration_count"] += 1
            subtask["status"] = "pending"
            # Keep the reflection with the suggestion for next iteration
            subtask["reflection"] = {
                "previous_attempt": subtask.get("result"),
                "issue": review["reason"],
                "suggestion": review.get("suggestion", ""),
                "next_action": review["next_action"],
            }
            # Reset for retry but keep reflection context
            subtask["selected_model"] = None
            subtask["model_schema"] = None
            subtask["prepared_data"] = None
            subtask["result"] = None
            subtasks[idx] = subtask
            return {"subtasks": subtasks, "current_subtask_idx": idx}
        else:
            # Max retries - move on with partial result
            subtask["status"] = "failed"
            all_reflections.append({
                "subtask_goal": subtask["goal"],
                "status": "incomplete",
                "result": subtask.get("result"),
                "reason": review["reason"],
                **(subtask.get("reflection") or {}),
            })
            subtasks[idx] = subtask
            return {
                "subtasks": subtasks,
                "current_subtask_idx": idx + 1,
                "all_reflections": all_reflections,
            }


def route_after_subtask(state: AgentState) -> str:
    """Determine next step after subtask completion check."""
    idx = state["current_subtask_idx"]
    subtasks = state["subtasks"]

    # Check if current subtask needs retry
    if idx < len(subtasks) and subtasks[idx]["status"] == "pending":
        return "select_model"

    # Check if more subtasks to process
    if idx < len(subtasks):
        return "select_model"

    return "synthesize"


def synthesize(state: AgentState) -> dict:
    """Synthesize all subtask results and determine if complete."""
    prompt = synthesize_prompt(state["query"], state["all_reflections"])
    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    synthesis = json.loads(content)
    return {"is_complete": synthesis["is_complete"]}


def generate_report(state: AgentState) -> dict:
    """Generate the final analysis report."""
    prompt = generate_report_prompt(state["query"], state["all_reflections"])
    response = llm.invoke(prompt)
    return {"final_report": response.content}


# =============================================================================
# Graph Construction
# =============================================================================


def build_analysis_graph() -> StateGraph:
    """Build the LangGraph for the analysis agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("decompose", decompose_query)
    graph.add_node("select_model", select_model)
    graph.add_node("prepare_data", prepare_data)
    graph.add_node("execute", execute_model)
    graph.add_node("reflect", reflect)
    graph.add_node("check_complete", check_subtask_complete)
    graph.add_node("synthesize", synthesize)
    graph.add_node("generate_report", generate_report)

    # Set entry point
    graph.set_entry_point("select_model")

    # Add edges for subtask loop
    # graph.add_edge("decompose", "select_model")
    graph.add_edge("select_model", "prepare_data")
    graph.add_edge("prepare_data", "execute")
    graph.add_edge("execute", "reflect")
    graph.add_edge("reflect", "check_complete")

    # Conditional routing after subtask completion
    graph.add_conditional_edges(
        "check_complete",
        route_after_subtask,
        {
            "select_model": "select_model",
            "synthesize": "synthesize",
        },
    )

    # After synthesis, go to report
    graph.add_edge("synthesize", "generate_report")

    # End after report
    graph.add_edge("generate_report", END)

    return graph


# =============================================================================
# Main Entry Point
# =============================================================================


def run_analysis(query: str, user_data: dict | None = None) -> str:
    """
    Run the analysis agent on a query.

    Args:
        query: The user's analysis request
        user_data: Optional dict with user-provided data

    Returns:
        The final analysis report as a string
    """
    graph = build_analysis_graph()
    app = graph.compile()

    initial_state: AgentState = {
        "query": query,
        "user_data": user_data,
        "subtasks": [],
        "current_subtask_idx": 0,
        "all_reflections": [],
        "is_complete": False,
        "final_report": None,
    }

    final_state = app.invoke(initial_state, {"recursion_limit": 50})
    return final_state.get("final_report") or "Analysis could not be completed."


# Example usage
if __name__ == "__main__":
    result = run_analysis(
        query="Assess credit risk for a 35-year-old married male with $75,000 income",
        user_data={
            "gender": "m",
            "marital": "married",
            "howpaid": "monthly",
            "mortgage": "n",
            "age": 35,
            "income": 75000,
            "numkids": 2,
            "numcards": 3,
            "storecar": 1,
            "loans": 1,
        },
    )
    print(result)
