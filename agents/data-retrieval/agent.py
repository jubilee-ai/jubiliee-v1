"""
Data Retrieval Agent - Retrieves and prepares datasets based on user requests.

Uses LangChain create_agent to orchestrate data discovery, loading, and transformation.
Returns a dataset_ref that can be used by downstream agents.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
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

# Import tools from data-tools package
from data_loader import dataset_get_tool
from join_merge import join_merge_tool
from retrieval import catalog_search_tool, list_datasets_tool
from sql_query import get_sql_schema_tool, sql_query_tool
from utils import get_registered_dataset

# =============================================================================
# Configuration
# =============================================================================

SYSTEM_PROMPT = """You are a data retrieval agent. Find and prepare datasets based on user requests.

## Tools Available

- **catalog_search_tool**: Search datasets by name, description, columns, or use case
- **list_datasets_tool**: List all available datasets
- **dataset_get_tool**: Load data from catalog by asset_id
- **get_sql_schema_tool**: Get SQL table schemas
- **sql_query_tool**: Execute SQL SELECT queries
- **join_merge_tool**: Join datasets on key columns

Use whatever approach makes sense for the request. You can search, query, join, or combine tools as needed.

## Output Format

End your response with a JSON block:

```json
{
  "dataset_ref": "<asset_id, table name, or join reference>",
  "description": "<what the data contains>",
  "rows": <row count>,
  "columns": ["col1", "col2", ...],
  "source": "<where it came from>"
}
```

## Important Notes

- Search or list available data before attempting to load unfamiliar datasets.
- Use filters and limits to avoid loading excessive data.
- For joins, verify key columns exist in both datasets first.
- The dataset_ref you provide will be used by other agents to access the data.
"""

# =============================================================================
# LLM Setup
# =============================================================================

llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# =============================================================================
# Tools
# =============================================================================

# TODO: Semantic retrieval
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


class DataRetrievalResult(BaseModel):
    """Structured result from the data retrieval agent."""

    dataset_ref: str = Field(
        description="Reference ID for the retrieved dataset (used by downstream agents)"
    )
    description: str = Field(description="Brief description of the dataset contents")
    rows: int = Field(description="Number of rows in the dataset")
    columns: list[str] = Field(description="List of column names")
    source: str = Field(
        description="Source of the data (asset_id, SQL table, or join reference)"
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
# Response Parsing
# =============================================================================


def _extract_json_from_response(response: str) -> Optional[dict]:
    """Extract JSON block from agent response."""
    # Try to find JSON in code block
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r"\{[^{}]*\"dataset_ref\"[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# Main Entry Point
# =============================================================================


def retrieve_data(
    request: str,
    context: Optional[dict] = None,
) -> DataRetrievalResult | dict:
    """
    Run the data retrieval agent to find and prepare data.

    Args:
        request: Natural language description of the data needed.
        context: Optional context dict with additional information.

    Returns:
        DataRetrievalResult with dataset_ref and metadata, or error dict.

    Example:
        >>> result = retrieve_data("Find loan default data with credit scores")
        >>> print(result.dataset_ref)
        'loan_default'
        >>> df = get_dataset(result.dataset_ref)  # Get the actual data
    """
    agent = build_data_retrieval_agent()

    # Build the input message
    user_message = request
    if context:
        user_message = f"{request}\n\nContext: {json.dumps(context)}"

    # Run the agent
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    # Extract the final response
    messages = result.get("messages", [])
    if not messages:
        return {"success": False, "error": "No response from agent"}

    final_message = messages[-1]
    response_text = (
        final_message.content
        if hasattr(final_message, "content")
        else str(final_message)
    )

    # Parse the structured response
    parsed = _extract_json_from_response(response_text)
    if parsed:
        try:
            return DataRetrievalResult(**parsed)
        except Exception:
            pass

    # Fallback: return raw response
    return {
        "success": True,
        "response": response_text,
        "parsed": parsed,
        "messages": messages,
    }


# =============================================================================
# Dataset Access Functions (for consuming agents)
# =============================================================================


def get_dataset(dataset_ref: str) -> Optional[pd.DataFrame]:
    """
    Get a dataset by its reference ID.

    This is the main function for consuming agents to access retrieved data.

    Args:
        dataset_ref: The dataset reference returned by retrieve_data()

    Returns:
        pandas DataFrame or None if not found

    Example:
        >>> result = retrieve_data("Get insurance data for smokers")
        >>> df = get_dataset(result.dataset_ref)
        >>> print(df.head())
    """
    # First check the in-memory registry (for joins and derived datasets)
    df = get_registered_dataset(dataset_ref)
    if df is not None:
        return df

    # Try to load from SQL warehouse
    try:
        from sql_query import get_warehouse

        warehouse = get_warehouse()
        table_info = warehouse.get_table_info(dataset_ref)
        if table_info:
            df, _ = warehouse.execute_query(f"SELECT * FROM {table_info.name}")
            return df
    except Exception:
        pass

    # Try to load from catalog
    try:
        from data_loader import dataset_get

        result = dataset_get(asset_id=dataset_ref, limit=-1)
        return pd.DataFrame(result.data)
    except Exception:
        pass

    return None


def list_available_datasets() -> dict:
    """
    List all datasets available for retrieval.

    Returns:
        Dict with 'registered' (in-memory), 'sql_tables', and 'catalog' datasets.
    """
    from utils import get_all_available_datasets

    return get_all_available_datasets()


def is_dataset_available(dataset_ref: str) -> bool:
    """
    Check if a dataset reference is valid and accessible.

    Args:
        dataset_ref: The dataset reference to check

    Returns:
        True if the dataset can be accessed, False otherwise
    """
    return get_dataset(dataset_ref) is not None


# =============================================================================
# Convenience function for subagent integration
# =============================================================================


def get_data_retrieval_tools() -> list:
    """Return the list of tools used by the data retrieval agent."""
    return DATA_RETRIEVAL_TOOLS


__all__ = [
    # Agent
    "build_data_retrieval_agent",
    "retrieve_data",
    # Dataset access (for consuming agents)
    "get_dataset",
    "list_available_datasets",
    "is_dataset_available",
    # Tools
    "get_data_retrieval_tools",
    "DATA_RETRIEVAL_TOOLS",
    # Types
    "DataRetrievalResult",
]
