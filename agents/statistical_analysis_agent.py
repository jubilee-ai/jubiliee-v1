"""
Statistical Analysis Agent - Multi-turn agent for comprehensive data analysis.

Uses LangChain create_agent with analysis tools to perform data validation,
EDA, and feature diagnostics based on a given goal.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Import analysis tools
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))
from analysis import analysis_tools

# =============================================================================
# System Prompt
# =============================================================================

# TODO: Might need to give transformation tools to help it clean the data when there are issues.

SYSTEM_PROMPT = """You are a statistical analysis agent. Your job is to analyze a dataset
to accomplish a specific analysis goal.

RULES:
- Always use the provided dataset_ref for all tool calls
- Do NOT repeat the same tool call with identical parameters
- Focus on findings relevant to the analysis goal
- Stop when you have sufficient information to answer the goal

When done, provide a final summary with:
- Key findings relevant to the goal
- Data quality issues found (if any)
- Recommendations for next steps
"""


# =============================================================================
# Agent Creation
# =============================================================================

# TODO: Optimize the context. Create agent is bad. --> maybe claude sdk instead of manual
def create_statistical_analysis_agent():
    """
    Create the statistical analysis agent with bound tools.
    
    Returns:
        Compiled agent graph ready for invocation
    """
    agent = create_agent(
        model="openai:gpt-5-mini",
        tools=analysis_tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


# =============================================================================
# Main Entry Point
# =============================================================================

def run_statistical_analysis(goal: str, dataset_ref: str) -> str:
    """
    Run statistical analysis on a dataset to accomplish a goal.
    
    Args:
        goal: The analysis goal to accomplish (e.g., "Identify data quality issues",
              "Find features most predictive of default", "Profile the dataset")
        dataset_ref: Reference to the dataset (table name or path)
    
    Returns:
        Analysis results as a string
    """
    agent = create_statistical_analysis_agent()
    
    # Construct the user message with goal and dataset context
    user_message = f"""Dataset reference: {dataset_ref}

Analysis goal: {goal}

Analyze this dataset to accomplish the goal. Call each tool at most once with the same parameters.
Do not repeat tool calls you have already made.
Once you have gathered enough information, stop calling tools and provide your final answer."""

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
    dataset_ref: str = Field(
        description="Reference to the dataset to analyze. Use the dataset name, table name, "
        "or path from a previous data loading/transformation step."
    )
    goal: str = Field(
        description="The specific analysis goal. Examples: "
        "'Profile this dataset and identify data quality issues', "
        "'Find which features are most predictive of the target column X', "
        "'Check for class imbalance and recommend handling strategies', "
        "'Identify redundant or leaky features before model training', "
        "'Summarize distributions and correlations for numeric columns'"
    )

# TODO: Add more statistical analysis tools to this tool/agent.
@tool(args_schema=StatisticalAnalysisInput)
def statistical_analysis_tool(
    dataset_ref: str,
    goal: str,
) -> str:
    """
    Run comprehensive statistical analysis on a dataset to answer a specific goal.
    
    USE THIS TOOL WHEN YOU NEED TO:
    - Profile a dataset (shape, types, distributions, correlations)
    - Validate data quality (nulls, invalid ranges, type mismatches)
    - Analyze target variable (class imbalance, distribution)
    - Find feature-target relationships (correlations, AUC)
    - Identify problematic features (high cardinality, leakage, redundancy)
    - Compare metrics across groups/segments
    - Analyze time trends, growth rates, seasonality
    - Get actionable recommendations before model training
    
    DO NOT USE THIS TOOL IF:
    - You haven't loaded or referenced a dataset yet
    - You need to transform data (use transformation tools instead)
    - You need to train a model (use model tools instead)
    
    This tool runs a multi-turn analysis agent that selects and runs appropriate tools
    based on the goal, then synthesizes findings into actionable insights.
    
    RETURNS: A comprehensive analysis report with key findings, alerts, and recommendations.
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

