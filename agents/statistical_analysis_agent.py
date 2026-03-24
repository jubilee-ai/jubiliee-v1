"""
Statistical Analysis Agent - Multi-turn agent for data retrieval and analysis.

Uses LangChain create_agent with data retrieval and analysis tools to find data,
perform validation, EDA, and feature diagnostics based on a given goal.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Import tools
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))
from analysis import analysis_tools
from data_loader import dataset_get_tool
from join_merge import join_merge_tool
from retrieval import catalog_search_tool, list_datasets_tool
from sql_query import get_sql_schema_tool, sql_query_tool

# Combine data retrieval and analysis tools
DATA_RETRIEVAL_TOOLS = [
    catalog_search_tool,
    list_datasets_tool,
    dataset_get_tool,
    get_sql_schema_tool,
    sql_query_tool,
    join_merge_tool,
]

ALL_TOOLS = DATA_RETRIEVAL_TOOLS + analysis_tools

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are a data analysis agent. You can find, retrieve, and analyze datasets to answer questions.

## Capabilities

**Data Retrieval**: Search the catalog, list datasets, load data, run SQL queries, join datasets.

**Statistical Analysis**: Profile data, validate quality, find correlations, analyze distributions, identify patterns.

## Approach

Use whatever tools make sense to accomplish the goal. If you need data, find and load it. If you need analysis, run it. Iterate as needed.

## Output

Provide a clear summary with:
- Key findings relevant to the goal
- Data/evidence supporting your conclusions  
- Recommendations for next steps (if applicable)
"""


# =============================================================================
# Agent Creation
# =============================================================================

def create_statistical_analysis_agent():
    """
    Create the statistical analysis agent with data retrieval and analysis tools.
    
    Returns:
        Compiled agent graph ready for invocation
    """
    agent = create_agent(
        model="openai:gpt-5.4",
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


# =============================================================================
# Main Entry Point
# =============================================================================

def run_statistical_analysis(goal: str, dataset_ref: str | None = None) -> str:
    """
    Run statistical analysis to accomplish a goal.
    
    Args:
        goal: The analysis goal to accomplish (e.g., "Analyze smoking impact on insurance costs",
              "Find features most predictive of default", "Profile the loan dataset")
        dataset_ref: Optional reference to the dataset. If not provided, the agent will
                     search for and load appropriate data.
    
    Returns:
        Analysis results as a string
    """
    agent = create_statistical_analysis_agent()
    
    # Construct the user message
    if dataset_ref:
        user_message = f"""Dataset: {dataset_ref}

Goal: {goal}

Analyze this dataset to accomplish the goal."""
    else:
        user_message = f"""Goal: {goal}

Find the appropriate data and analyze it to accomplish this goal."""

    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    
    # Extract the final response from messages
    messages = result.get("messages", [])
    if messages:
        final_message = messages[-1]
        if hasattr(final_message, "content"):
            return final_message.content
        return str(final_message)
    
    return "Analysis could not be completed."


# =============================================================================
# LangChain Tool
# =============================================================================

class StatisticalAnalysisInput(BaseModel):
    """Input schema for statistical_analysis_tool."""
    goal: str = Field(
        description="The analysis goal. Examples: "
        "'Analyze insurance costs by smoking status', "
        "'Find which features predict loan default', "
        "'Profile the credit risk dataset and check for quality issues', "
        "'Compare charges across regions in the insurance data'"
    )
    dataset_ref: str | None = Field(
        default=None,
        description="Optional dataset reference if already known. If not provided, "
        "the agent will search for and load appropriate data."
    )

@tool(args_schema=StatisticalAnalysisInput)
def statistical_analysis_tool(
    goal: str,
    dataset_ref: str | None = None,
) -> str:
    """
    Find data and run statistical analysis to answer a specific goal.
    
    USE THIS TOOL WHEN YOU NEED TO:
    - Analyze patterns in data (e.g., "How does smoking affect insurance costs?")
    - Profile a dataset (shape, types, distributions, correlations)
    - Validate data quality (nulls, invalid ranges, type mismatches)
    - Find feature-target relationships and key drivers
    - Compare metrics across groups/segments
    - Get actionable insights from data
    
    This tool can find and load data automatically if you don't have a dataset_ref.
    Just describe what you want to analyze in the goal.
    
    RETURNS: Analysis findings with supporting data and recommendations.
    """
    try:
        return run_statistical_analysis(goal=goal, dataset_ref=dataset_ref)
    except Exception as e:
        return f"✗ Statistical analysis failed: {type(e).__name__}: {e}"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "create_statistical_analysis_agent",
    "run_statistical_analysis",
    "statistical_analysis_tool",
]

