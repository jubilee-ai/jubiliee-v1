"""Data Retrieval Agent - finds and prepares datasets based on user requests."""

import importlib.util
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

# Setup
load_dotenv(Path(__file__).parent.parent.parent / ".env")
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"


def _load_module(name: str, file_path: Path):
    """Load a module from file path, handling relative imports."""
    full_name = f"data_tools.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def _setup_data_tools():
    """Setup data_tools package for imports."""
    if "data_tools" in sys.modules:
        return
    
    # Create fake package
    class FakePackage:
        pass
    
    sys.modules["data_tools"] = FakePackage()
    
    # Load core modules in dependency order
    _load_module("utils", _DATA_TOOLS_DIR / "utils.py")
    _load_module("embedder", _DATA_TOOLS_DIR / "embedder.py")
    _load_module("sql_query", _DATA_TOOLS_DIR / "sql_query.py")
    _load_module("data_loader", _DATA_TOOLS_DIR / "data_loader.py")
    _load_module("retrieval", _DATA_TOOLS_DIR / "retrieval.py")
    _load_module("join_merge", _DATA_TOOLS_DIR / "join_merge.py")


# Initialize data tools
_setup_data_tools()

# Get the tools we need
from data_tools.data_loader import dataset_get, dataset_get_tool
from data_tools.join_merge import join_merge_tool
from data_tools.retrieval import catalog_search_tool, list_datasets_tool
from data_tools.sql_query import (get_sql_schema_tool, get_warehouse,
                                  sql_query_tool)
from data_tools.utils import get_all_available_datasets, get_registered_dataset

# Config
SYSTEM_PROMPT = """You are a data retrieval agent. Your goal is to find and prepare the data the user needs.

## Data Sources

There are two types of data:
1. **SQL Tables** - Queryable with sql_query_tool. Examples: `loan_default`, `insurance`, `credit_card_risk`
2. **Catalog Assets** - Loaded with dataset_get_tool. Examples: `csv/insurance.csv`, HuggingFace datasets

## Tools

**Discovery:**
- `list_datasets_tool()` - See all available SQL tables and catalog assets. START HERE.
- `catalog_search_tool(query)` - Search catalog by keywords if you need to find specific data.
- `get_sql_schema_tool(table_name)` - Get columns/types for a SQL table. CALL BEFORE writing queries.

**Retrieval:**
- `sql_query_tool(query)` - Execute SELECT queries on SQL tables. Supports filtering, aggregation, etc.
- `dataset_get_tool(asset_id)` - Load data from catalog assets (CSV files, HuggingFace).

**Transformation:**
- `join_merge_tool(left_ref, right_ref, keys)` - Join two datasets. Works with SQL table names directly.

## Typical Workflows

**Simple SQL query:**
1. `list_datasets_tool()` → see available tables
2. `get_sql_schema_tool("table_name")` → see columns
3. `sql_query_tool("SELECT ... FROM table_name WHERE ...")` → get data

**Join two tables:**
1. `get_sql_schema_tool("table1")` → check columns
2. `get_sql_schema_tool("table2")` → find join key
3. `join_merge_tool("table1", "table2", {"col1": "col2"})` → join

## Tips
- Quote reserved words in SQL: `"Default"`, `"Order"`
- Use `dedupe_strategy='first'` in joins to avoid row explosion
- Join results become new SQL tables you can query
- Before joining, verify exact column names with `get_sql_schema_tool` since they are case-sensitive (e.g., `Age` vs `age`)

## Output
When done, return:
```json
{"dataset_ref": "<ref>", "description": "<desc>", "rows": <n>, "columns": [...], "source": "<src>"}
```
"""

DATA_RETRIEVAL_TOOLS = [
    catalog_search_tool,
    list_datasets_tool,
    dataset_get_tool,
    get_sql_schema_tool,
    sql_query_tool,
    join_merge_tool,
]


class DataRetrievalResult(BaseModel):
    """Result from the data retrieval agent."""
    dataset_ref: str = Field(description="Reference ID for the dataset")
    description: str = Field(description="Description of the dataset")
    rows: int = Field(description="Number of rows")
    columns: list[str] = Field(description="Column names")
    source: str = Field(description="Data source")


def build_data_retrieval_agent():
    """Build the data retrieval agent."""
    return create_agent(
        model=ChatOpenAI(model="gpt-5-mini", temperature=0),
        tools=DATA_RETRIEVAL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from response text."""
    for pattern in [r"```json\s*(\{.*?\})\s*```", r'\{\s*"dataset_ref"[^}]+\}']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if "```" in pattern else match.group(0))
            except json.JSONDecodeError:
                continue
    return None


def retrieve_data(request: str, context: Optional[dict] = None) -> DataRetrievalResult | dict:
    """Run the data retrieval agent."""
    agent = build_data_retrieval_agent()
    
    msg = f"{request}\n\nContext: {json.dumps(context)}" if context else request
    result = agent.invoke({"messages": [{"role": "user", "content": msg}]})
    
    messages = result.get("messages", [])
    if not messages:
        return {"success": False, "error": "No response from agent"}
    
    response = getattr(messages[-1], "content", str(messages[-1]))
    parsed = _extract_json(response)
    
    if parsed:
        try:
            return DataRetrievalResult(**parsed)
        except Exception:
            pass
    
    return {"success": True, "response": response, "parsed": parsed, "messages": messages}


def get_dataset(dataset_ref: str) -> Optional[pd.DataFrame]:
    """Get a dataset by reference ID. Checks registry, SQL warehouse, then catalog."""
    # Check in-memory registry
    df = get_registered_dataset(dataset_ref)
    if df is not None:
        return df
    
    # Check SQL warehouse
    warehouse = get_warehouse()
    for ref in [dataset_ref, dataset_ref.lower()]:
        table_info = warehouse.get_table_info(ref)
        if table_info:
            df, _ = warehouse.execute_query(f"SELECT * FROM {table_info.name}")
            return df
    
    # Check catalog
    try:
        result = dataset_get(asset_id=dataset_ref, limit=-1)
        return pd.DataFrame(result.data)
    except Exception:
        return None


def list_available_datasets() -> dict:
    """List all available datasets."""
    return get_all_available_datasets()


def is_dataset_available(dataset_ref: str) -> bool:
    """Check if a dataset reference is valid and accessible."""
    return get_dataset(dataset_ref) is not None


__all__ = [
    "build_data_retrieval_agent",
    "retrieve_data",
    "get_dataset",
    "list_available_datasets",
    "is_dataset_available",
    "DATA_RETRIEVAL_TOOLS",
    "DataRetrievalResult",
]
