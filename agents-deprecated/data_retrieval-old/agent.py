"""
Data Retrieval Agent - Retrieves and prepares datasets based on user requests.

Uses LangChain create_agent to orchestrate data discovery, loading, and transformation.
Returns a dataset_ref that can be used by downstream agents.
"""

import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Add data-tools to path for imports
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from data_loader import dataset_get_tool
from join_merge import join_merge_tool
# Import tools from data-tools package
from retrieval import catalog_search_tool, list_datasets_tool
from sql_query import get_sql_schema_tool, sql_query_tool

# =============================================================================
# Configuration
# =============================================================================

SYSTEM_PROMPT = """You are a data retrieval agent. Your job is to find and prepare datasets based on user requests.

## Available Tools

1. **catalog_search_tool**: Search the data catalog to discover relevant datasets by name, description, columns, or use case.
2. **list_datasets_tool**: List all available datasets (SQL tables, registered datasets from previous operations).
3. **dataset_get_tool**: Load data from a catalog dataset by asset_id. Supports filtering and column selection.
4. **get_sql_schema_tool**: Get schema of SQL tables. CALL THIS before writing SQL queries.
5. **sql_query_tool**: Execute SQL SELECT queries against the data warehouse.
6. **join_merge_tool**: Join two datasets on key columns with diagnostics.

## Workflow

1. **Discover**: Use catalog_search_tool or list_datasets_tool to find relevant data.
2. **Inspect**: Use get_sql_schema_tool or dataset_get_tool (with small limit) to understand structure.
3. **Retrieve**: Load the data using dataset_get_tool or sql_query_tool.
4. **Transform** (if needed): Use join_merge_tool to combine datasets or sql_query_tool for complex filtering.

## Output Format

When you have successfully retrieved the data, respond with a summary containing:
- **dataset_ref**: The reference ID for the retrieved/transformed dataset
- **description**: Brief description of what the data contains
- **rows**: Number of rows retrieved
- **columns**: List of column names
- **source**: Where the data came from (asset_id, SQL query, join operation)

If the request cannot be fulfilled, explain why and suggest alternatives.

## Important Notes

- Always start by searching or listing available data before attempting to load.
- Use filters and limits appropriately to avoid loading excessive data.
- For joins, verify key columns exist in both datasets first.
- Return the dataset_ref so downstream agents can use the prepared data.
"""

# =============================================================================
# LLM Setup
# =============================================================================

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# =============================================================================
# Tools
# =============================================================================

DATA_RETRIEVAL_TOOLS = [
    catalog_search_tool,
    list_datasets_tool,
    dataset_get_tool,
    get_sql_schema_tool,
    sql_query_tool,
    join_merge_tool,
]

# =============================================================================
# Response Schema
# =============================================================================


class DataRetrievalResponse(BaseModel):
    """Structured response from the data retrieval agent."""

    dataset_ref: str = Field(
        description="Reference ID for the retrieved dataset (can be used in downstream operations)"
    )
    description: str = Field(description="Brief description of the dataset contents")
    rows: int = Field(description="Number of rows in the dataset")
    columns: list[str] = Field(description="List of column names")
    source: str = Field(
        description="Source of the data (asset_id, SQL query, or join operation)"
    )


# =============================================================================
# Agent Construction
# =============================================================================


def build_data_retrieval_agent():
    """
    Build the data retrieval agent using LangChain create_agent.

    Returns:
        CompiledStateGraph that can be invoked with messages.
    """
    return create_agent(
        model=llm,
        tools=DATA_RETRIEVAL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def retrieve_data(
    request: str,
    context: Optional[dict] = None,
) -> dict:
    """
    Run the data retrieval agent to find and prepare data.

    Args:
        request: Natural language description of the data needed.
        context: Optional context dict with additional information.

    Returns:
        Dict with dataset_ref and metadata, or error information.

    Example:
        >>> result = retrieve_data("Find loan default data with credit scores")
        >>> print(result["dataset_ref"])
        'Loan_default.csv'
    """
    agent = build_data_retrieval_agent()

    # Build the input message
    user_message = request
    if context:
        user_message = f"{request}\n\nContext: {context}"

    # Run the agent
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    # Extract the final response
    messages = result.get("messages", [])
    if messages:
        final_message = messages[-1]
        return {
            "success": True,
            "response": final_message.content if hasattr(final_message, "content") else str(final_message),
            "messages": messages,
        }

    return {
        "success": False,
        "error": "No response from agent",
        "messages": [],
    }


# =============================================================================
# Convenience function for subagent integration
# =============================================================================


def get_data_retrieval_tools() -> list:
    """Return the list of tools used by the data retrieval agent."""
    return DATA_RETRIEVAL_TOOLS


__all__ = [
    "build_data_retrieval_agent",
    "retrieve_data",
    "get_data_retrieval_tools",
    "DATA_RETRIEVAL_TOOLS",
    "DataRetrievalResponse",
]
