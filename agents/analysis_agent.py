"""
Analysis Agent - Orchestrates model selection, data retrieval, and execution.

Uses LangGraph for state management with Send API for parallel subagent execution.
"""

import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from .prompts import generate_report_prompt, plan_analysis_prompt
from .subagent import SubagentState, build_subagent_graph

# =============================================================================
# LLM Setup
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# =============================================================================
# State Definition
# =============================================================================


class Subtask(TypedDict):
    """A subtask to be executed by a subagent."""
    goal: str
    type: str
    priority: int


class AgentState(TypedDict):
    """Main state for the analysis agent."""
    query: str
    user_data: dict | None
    subtasks: list[Subtask]
    # Use reducer to collect syntheses from parallel branches
    all_syntheses: Annotated[list[dict], operator.add]
    final_report: str | None


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


# =============================================================================
# Graph Nodes
# =============================================================================


def plan_analysis(state: AgentState) -> dict:
    """Create an execution plan by breaking query into subtasks."""
    prompt = plan_analysis_prompt(state["query"], state.get("user_data"))
    response = llm.invoke(prompt)
    plan = parse_json_response(response.content)
    
    # Sort tasks by priority
    tasks = sorted(plan.get("tasks", []), key=lambda t: t.get("priority", 1))
    
    subtasks = [
        Subtask(
            goal=task["goal"],
            type=task.get("type", "other"),
            priority=task.get("priority", 1),
        )
        for task in tasks
    ]
    
    return {"subtasks": subtasks}


def fan_out_subtasks(state: AgentState) -> list[Send]:
    """Fan out to run each subtask in parallel via Send API."""
    sends = []
    for subtask in state["subtasks"]:
        # Create initial state for each subagent
        subagent_state: SubagentState = {
            "query": state["query"],
            "user_data": state.get("user_data"),
            "goal": subtask["goal"],
            "status": "pending",
            "selected_model": None,
            "model_schema": None,
            "prepared_data": None,
            "result": None,
            "reflection": None,
            "iteration_count": 0,
            "max_iterations": 3,
            "synthesis": None,
        }
        sends.append(Send("run_subagent", subagent_state))
    return sends


def run_subagent_node(state: SubagentState) -> dict:
    """
    Run the subagent graph for a single subtask.
    This node runs the complete subagent workflow and returns synthesis.
    """
    # Build and run the subagent graph
    subagent = build_subagent_graph().compile()
    final_state = subagent.invoke(state, {"recursion_limit": 30})
    
    synthesis = final_state.get("synthesis") or {
        "task_goal": state["goal"],
        "status": "failed",
        "model_used": None,
        "key_findings": "Subagent failed to produce synthesis",
        "metrics": {},
        "confidence": "low",
        "limitations": ["Execution error"],
        "recommendation": "Investigate error",
    }
    
    # Return in format that the reducer can collect
    return {"all_syntheses": [synthesis]}


def generate_report(state: AgentState) -> dict:
    """Generate the final analysis report from all subagent syntheses."""
    prompt = generate_report_prompt(state["query"], state["all_syntheses"])
    response = llm.invoke(prompt)
    return {"final_report": response.content}


# =============================================================================
# Graph Construction
# =============================================================================


def build_analysis_graph() -> StateGraph:
    """
    Build the main analysis graph with parallel subagent execution.
    
    Flow:
        plan → [fan_out] → run_subagent (parallel) → generate_report → END
                              ↑ (one per subtask)
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("plan", plan_analysis)
    graph.add_node("run_subagent", run_subagent_node)
    graph.add_node("generate_report", generate_report)

    # Set entry point
    graph.set_entry_point("plan")

    # After plan, fan out to parallel subagents using Send
    graph.add_conditional_edges("plan", fan_out_subtasks, ["run_subagent"])
    
    # After all subagents complete, generate report
    graph.add_edge("run_subagent", "generate_report")
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
        "all_syntheses": [],
        "final_report": None,
    }

    final_state = app.invoke(initial_state)
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
