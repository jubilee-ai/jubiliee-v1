"""
Step 3: Cleaning + Standardization Agent (Iterative)
Runs EDA + validation, LLM reviews, applies transformations or marks complete.

Implemented as an internal LangGraph that is built and executed by cleaning_standardization().
"""

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Add data-tools to path
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

if TYPE_CHECKING:
    from .agent import TrainingAgentState

from analysis.data_validation import validate_dataset
from analysis.eda_report import EdaReportType, run_eda_report
from transformations.clean_ops import clean_tools
from transformations.column_ops import column_tools
from transformations.reshape_ops import reshape_tools
from transformations.row_ops import row_tools
from utils import generate_unique_id, get_registered_dataset, register_dataset

# =============================================================================
# CONSTANTS
# =============================================================================

MAX_CLEANING_ITERATIONS = 10
DATASET_REF_PATTERN = re.compile(r"→ `([^`]+)`")

# =============================================================================
# MARK COMPLETE TOOL
# =============================================================================

class MarkCleaningCompleteInput(BaseModel):
    dataset_ref: str = Field(description="The dataset reference to register as cleaned")
    reasoning: str = Field(description="Brief explanation of why the dataset is ready")


@tool(args_schema=MarkCleaningCompleteInput)
def mark_cleaning_complete(dataset_ref: str, reasoning: str) -> str:
    """
    Mark the dataset as cleaned and ready for feature engineering.
    Call this when the data is clean - no major nulls, outliers fixed, types correct.
    """
    df = get_registered_dataset(dataset_ref)
    if df is None:
        return f"✗ Dataset '{dataset_ref}' not found"
    
    cleaned_ref = generate_unique_id("cleaned")
    register_dataset(cleaned_ref, df)
    return f"CLEANING_COMPLETE|{cleaned_ref}|{reasoning}"


# =============================================================================
# TOOLS
# =============================================================================

CLEANING_TOOLS = [
    mark_cleaning_complete,
    *clean_tools,
    *row_tools,
    *column_tools,
    *reshape_tools,
]

_TOOL_MAP = {t.name: t for t in CLEANING_TOOLS}

# =============================================================================
# LLM (cached)
# =============================================================================

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = init_chat_model(model="gpt-5-mini").bind_tools(CLEANING_TOOLS)
    return _llm

# =============================================================================
# PROMPT
# =============================================================================

CLEANING_PROMPT = """You are a data cleaning agent. Review the EDA report and validation results, then decide what to do.

## Goal
{goal}

## EDA Report
{eda_report}

## Validation Results
{validation_results}

## Instructions
1. Review the EDA report and validation results
2. If there are data quality issues, use transformation tools to fix them:
   - High null percentages -> impute or drop_nulls
   - Outliers -> clip
   - Wrong dtypes -> cast  
   - Duplicates -> dedupe
   - Invalid values -> replace_values or filter
3. If the data is clean (no critical issues), call `mark_cleaning_complete` to finish

You MUST call either transformation tools OR `mark_cleaning_complete`. Never return without calling a tool.
Only fix real issues. Don't over-engineer. When ready, call mark_cleaning_complete.
"""

# =============================================================================
# INTERNAL GRAPH STATE
# =============================================================================

class CleaningGraphState(TypedDict):
    """Internal state for the cleaning graph."""
    # Inputs
    goal: str
    dataset_ref: str
    iteration: int
    
    # Analysis results
    eda_result: Optional[dict]
    validation_result: Optional[dict]
    
    # LLM response
    tool_calls: list[dict]
    
    # Outputs
    new_dataset_ref: str
    transformations: list[dict]
    cleaning_complete: bool
    completion_reason: str
    
    # Control
    status: str  # "continue", "done", "error", "no_tools"


# =============================================================================
# GRAPH NODES
# =============================================================================

def check_guards_node(state: CleaningGraphState) -> CleaningGraphState:
    """Check iteration limits and dataset availability."""
    if not state["dataset_ref"]:
        return {**state, "status": "error", "completion_reason": "No dataset available"}
    
    if state["iteration"] >= MAX_CLEANING_ITERATIONS:
        return {**state, "status": "done", "completion_reason": f"Max iterations ({MAX_CLEANING_ITERATIONS}) reached"}
    
    return {**state, "status": "continue"}


def run_analysis_node(state: CleaningGraphState) -> CleaningGraphState:
    """Run EDA report and validation."""
    dataset_ref = state["dataset_ref"]
    
    eda_result: EdaReportType = run_eda_report(dataset_ref=dataset_ref)
    column_names = [col["column"] for col in eda_result["schema"]]
    
    validation_result = validate_dataset(
        dataset_ref=dataset_ref,
        rules=[{"type": "not_null", "columns": column_names[:10]}],
        thresholds={"max_null_pct": 0.3},
    )
    
    return {
        **state,
        "eda_result": eda_result,
        "validation_result": validation_result,
    }


def llm_decide_node(state: CleaningGraphState) -> CleaningGraphState:
    """LLM reviews analysis and decides what tools to call."""
    prompt = CLEANING_PROMPT.format(
        goal=state["goal"],
        eda_report=json.dumps(state["eda_result"], indent=1),
        validation_results=json.dumps(state["validation_result"], indent=1),
    )
    
    response = _get_llm().invoke(prompt)
    tool_calls = getattr(response, "tool_calls", [])
    
    if not tool_calls:
        return {**state, "tool_calls": [], "status": "no_tools"}
    
    return {**state, "tool_calls": tool_calls, "status": "continue"}


def execute_tools_node(state: CleaningGraphState) -> CleaningGraphState:
    """Execute the tool calls from LLM."""
    new_dataset_ref = state["dataset_ref"]
    transformations = list(state["transformations"])
    
    for tc in state["tool_calls"]:
        tool_name, tool_args = tc["name"], tc["args"]
        tool_fn = _TOOL_MAP.get(tool_name)
        if not tool_fn:
            continue
        
        # Always use the latest dataset ref
        if "dataset_ref" in tool_args:
            tool_args["dataset_ref"] = new_dataset_ref
        
        result = tool_fn.invoke(tool_args)
        transformations.append({"tool": tool_name, "args": tool_args, "result": str(result)[:200]})
        
        # Check for completion
        if isinstance(result, str) and result.startswith("CLEANING_COMPLETE|"):
            _, cleaned_ref, reason = result.split("|", 2)
            return {
                **state,
                "new_dataset_ref": cleaned_ref,
                "transformations": transformations,
                "cleaning_complete": True,
                "completion_reason": reason,
                "status": "done",
            }
        
        # Extract new dataset ref from tool output
        if isinstance(result, str):
            match = DATASET_REF_PATTERN.search(result)
            if match:
                new_dataset_ref = match.group(1)
    
    # Transformations applied, need to iterate
    return {
        **state,
        "new_dataset_ref": new_dataset_ref,
        "transformations": transformations,
        "status": "iterate",
    }


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================

def after_guards(state: CleaningGraphState) -> Literal["run_analysis", "end"]:
    """Route after guard checks."""
    if state["status"] in ["error", "done"]:
        return "end"
    return "run_analysis"


def after_llm_decide(state: CleaningGraphState) -> Literal["execute_tools", "end"]:
    """Route after LLM decision."""
    if state["status"] == "no_tools":
        return "end"  # Will be handled as retry in parent
    return "execute_tools"


def after_execute_tools(state: CleaningGraphState) -> Literal["end"]:
    """Route after tool execution - always end (parent handles iteration)."""
    return "end"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def _build_cleaning_graph() -> StateGraph:
    """Build the internal cleaning graph."""
    graph = StateGraph(CleaningGraphState)
    
    # Add nodes
    graph.add_node("check_guards", check_guards_node)
    graph.add_node("run_analysis", run_analysis_node)
    graph.add_node("llm_decide", llm_decide_node)
    graph.add_node("execute_tools", execute_tools_node)
    
    # Set entry point
    graph.set_entry_point("check_guards")
    
    # Add edges
    graph.add_conditional_edges("check_guards", after_guards, {
        "run_analysis": "run_analysis",
        "end": END,
    })
    
    graph.add_edge("run_analysis", "llm_decide")
    
    graph.add_conditional_edges("llm_decide", after_llm_decide, {
        "execute_tools": "execute_tools",
        "end": END,
    })
    
    graph.add_conditional_edges("execute_tools", after_execute_tools, {
        "end": END,
    })
    
    return graph


# Cache the compiled graph
_cleaning_graph = None

def _get_cleaning_graph():
    """Get or create the compiled cleaning graph."""
    global _cleaning_graph
    if _cleaning_graph is None:
        _cleaning_graph = _build_cleaning_graph().compile()
    return _cleaning_graph


# =============================================================================
# MAIN NODE FUNCTION (for parent graph)
# =============================================================================

def cleaning_standardization(state: "TrainingAgentState") -> "TrainingAgentState":
    """
    Cleaning + Standardization node.
    Builds and executes the internal cleaning graph.
    """
    dataset_ref = state.get("cleaned_dataset_ref") or state.get("collected_dataset_ref")
    goal = state.get("goal", "")
    iteration = state.get("cleaning_iteration", 0)
    
    # Build initial state for internal graph
    internal_state: CleaningGraphState = {
        "goal": goal,
        "dataset_ref": dataset_ref,
        "iteration": iteration,
        "eda_result": None,
        "validation_result": None,
        "tool_calls": [],
        "new_dataset_ref": dataset_ref,
        "transformations": list(state.get("cleaning_transformations", [])),
        "cleaning_complete": False,
        "completion_reason": "",
        "status": "continue",
    }
    
    # Execute the internal graph
    result = _get_cleaning_graph().invoke(internal_state)
    
    # Map internal result back to parent state
    if result["status"] == "error":
        return {**state, "error": result["completion_reason"]}
    
    if result["status"] == "done":
        return {
            **state,
            "cleaned_dataset_ref": result["new_dataset_ref"],
            "cleaning_status": "done",
            "cleaning_transformations": result["transformations"],
            "cleaning_iteration": iteration + 1,
            "explanations": [*state.get("explanations", []), f"Cleaning complete: {result['completion_reason']}"],
        }
    
    if result["status"] == "no_tools":
        return {
            **state,
            "cleaning_iteration": iteration + 1,
            "cleaning_status": "iterate",
            "explanations": [*state.get("explanations", []), "LLM returned no actions, retrying"],
        }
    
    # status == "iterate"
    return {
        **state,
        "cleaned_dataset_ref": result["new_dataset_ref"],
        "cleaning_status": "iterate",
        "cleaning_transformations": result["transformations"],
        "cleaning_iteration": iteration + 1,
        "explanations": [*state.get("explanations", []), f"Applied {len(result['tool_calls'])} transformation(s)"],
    }


# =============================================================================
# CONDITIONAL EDGE (for parent graph)
# =============================================================================

def should_continue_cleaning(state: "TrainingAgentState") -> Literal["iterate", "done"]:
    """Check if cleaning loop should continue or exit."""
    return "done" if state.get("cleaning_status") == "done" else "iterate"
