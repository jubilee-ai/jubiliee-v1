"""
Dataset Curator Agent — finds, downloads, transforms, and exports
training-ready CSV datasets using the Kaggle MCP and local data tools.

Uses LangChain create_agent with:
- Kaggle MCP tools (search_datasets, get_dataset_info, list_dataset_files)
  for discovering datasets on Kaggle
- Local tools for downloading, joining, transforming, and exporting data
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from join_merge import join_merge_tool
from retrieval import list_datasets_tool
from transformations import apply_transformations_tool
from transformations.clean_ops import drop_nulls_tool, fill_null_tool
from transformations.column_ops import (
    cast_tool,
    drop_columns_tool,
    rename_columns_tool,
    select_columns_tool,
)
from transformations.row_ops import (
    dedupe_tool,
    filter_rows_tool,
    limit_rows_tool,
    sample_rows_tool,
)

from .tools import (
    curator_tools,
    download_hf_dataset,
    normalize_columns,
    profile_dataset,
    suggest_join_keys,
    validate_target,
)

# =============================================================================
# Configuration
# =============================================================================

SYSTEM_PROMPT = """\
You are a dataset curator agent. Build a training-ready CSV dataset based on \
the user's model training goal.

## Workflow

1. **Search** — Always search **both** Kaggle (`search_datasets`) and \
HuggingFace (`hub_repo_search` with repo_types=["dataset"]) in parallel. \
For NLP/text tasks, HuggingFace often has the canonical benchmark — prefer \
the original source over mirrors. For tabular ML, Kaggle is usually stronger.

2. **Compare** — **MANDATORY: call `get_dataset_info` or `hub_repo_details` \
on at least 2 candidates before downloading anything.** Never skip this step. \
For Kaggle check usability_rating (skip < 0.5), download_count, kernel_count. \
For HuggingFace check downloads, tags, description via `hub_repo_details`.

3. **Download** — Use `download_kaggle_dataset` or `download_hf_dataset`. \
Kaggle `ref` fields are `owner/slug` format — split on `/` for the tool args. \
For Kaggle, filter searches with `file_type: "DATASET_FILE_TYPE_GROUP_CSV"`.

4. **Profile** — Always run `profile_dataset` after download. If the data \
doesn't fit the goal (too few rows, missing features, >50% nulls), go back \
to step 1 with different terms or try the next candidate.

5. **Clean & transform** — Use the transformation tools as needed (the tool \
descriptions explain each one). If a download produced multiple files, use \
`suggest_join_keys` to check if they should be combined.

6. **Finalize** — Always run `normalize_columns`, then `validate_target`, \
then `export_csv`. This sequence is required before every export.

## Key Rules

- If the user requests N rows and the dataset has fewer, search for a larger \
  one. Only accept fewer rows after trying at least 2 other candidates.
- Use `sample_rows` (random strategy) to downsample large datasets.
- When multiple files share a key, prefer `inner` join for clean training data.

## Output

End with a JSON summary:
```json
{"csv_path": "...", "description": "...", "rows": 0, "columns": [], \
"target_column": "...", "kaggle_sources": [], "hf_sources": []}
```
"""

# =============================================================================
# LLM
# =============================================================================

llm = ChatOpenAI(model="gpt-5.4", temperature=0)

# =============================================================================
# Local tools (non-MCP)
# =============================================================================

LOCAL_TOOLS = [
    *curator_tools,
    join_merge_tool,
    list_datasets_tool,
    apply_transformations_tool,
    drop_nulls_tool,
    fill_null_tool,
    rename_columns_tool,
    select_columns_tool,
    drop_columns_tool,
    cast_tool,
    filter_rows_tool,
    dedupe_tool,
    sample_rows_tool,
    limit_rows_tool,
]

# =============================================================================
# Response schema
# =============================================================================


class DatasetCuratorResult(BaseModel):
    csv_path: str = Field(description="Path to the exported CSV file")
    description: str = Field(description="Description of the dataset contents")
    rows: int = Field(description="Number of rows")
    columns: list[str] = Field(description="Column names")
    target_column: Optional[str] = Field(
        default=None, description="Suggested target variable for model training"
    )
    kaggle_sources: list[str] = Field(
        default_factory=list, description="Kaggle dataset identifiers used"
    )
    hf_sources: list[str] = Field(
        default_factory=list, description="HuggingFace dataset identifiers used"
    )


# =============================================================================
# Agent construction (async — Kaggle MCP requires HTTP transport)
# =============================================================================


async def build_dataset_curator_agent():
    """
    Build the dataset curator agent with Kaggle + HuggingFace MCP tools + local tools.

    Returns a compiled agent that can be invoked with messages.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    kaggle_token = os.environ.get(
        "KAGGLE_MCP_TOKEN", os.environ.get("KAGGLE_API_KEY", "")
    )
    hf_token = os.environ.get("HUGGING_FACE_TOKEN", "")

    kaggle_headers = {}
    if kaggle_token:
        kaggle_headers["Authorization"] = f"Bearer {kaggle_token}"

    hf_headers = {}
    if hf_token:
        hf_headers["Authorization"] = f"Bearer {hf_token}"

    servers = {
        "kaggle": {
            "transport": "http",
            "url": "https://www.kaggle.com/mcp",
            **({"headers": kaggle_headers} if kaggle_headers else {}),
        },
    }
    if hf_token:
        servers["huggingface"] = {
            "transport": "http",
            "url": "https://huggingface.co/mcp",
            "headers": hf_headers,
        }

    client = MultiServerMCPClient(servers)
    mcp_tools = await client.get_tools()

    wanted_tools = {
        "search_datasets", "get_dataset_info", "list_dataset_files",
        "hub_repo_search", "hub_repo_details",
    }
    mcp_subset = [t for t in mcp_tools if t.name in wanted_tools]

    all_tools = mcp_subset + LOCAL_TOOLS

    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent, client


# =============================================================================
# Response parsing
# =============================================================================


def _extract_json_from_response(response: str) -> Optional[dict]:
    """Extract JSON block from agent response."""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    json_match = re.search(
        r"\{[^{}]*\"csv_path\"[^{}]*\}", response, re.DOTALL
    )
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# =============================================================================
# Main entry point
# =============================================================================


async def curate_dataset(
    goal: str,
    output_name: Optional[str] = None,
    desired_rows: Optional[int] = None,
) -> DatasetCuratorResult | dict:
    """
    Run the dataset curator agent to build a training-ready CSV.

    Args:
        goal: Natural language description of the model training objective.
              Example: "Build a dataset for predicting house prices with
              features like square footage, bedrooms, location."
        output_name: Optional name for the output CSV file.
        desired_rows: Optional target number of rows for the final dataset.
                      The agent will sample or limit the data accordingly.

    Returns:
        DatasetCuratorResult with csv_path and metadata, or a fallback dict.
    """
    agent, client = await build_dataset_curator_agent()

    user_message = goal
    if output_name:
        user_message += f"\n\nPlease name the output CSV: {output_name}"
    if desired_rows is not None:
        user_message += (
            f"\n\nThe final dataset should have approximately {desired_rows:,} rows. "
            f"Use sample_rows (random) to downsample if the source is larger, "
            f"or note if the source has fewer rows than requested."
        )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_message}]}
    )

    messages = result.get("messages", [])
    if not messages:
        return {"success": False, "error": "No response from agent"}

    final_message = messages[-1]
    response_text = (
        final_message.content
        if hasattr(final_message, "content")
        else str(final_message)
    )

    parsed = _extract_json_from_response(response_text)
    if parsed:
        try:
            return DatasetCuratorResult(**parsed)
        except Exception:
            pass

    return {
        "success": True,
        "response": response_text,
        "parsed": parsed,
    }


def curate_dataset_sync(
    goal: str,
    output_name: Optional[str] = None,
    desired_rows: Optional[int] = None,
) -> DatasetCuratorResult | dict:
    """Synchronous wrapper around curate_dataset."""
    return asyncio.run(curate_dataset(goal, output_name, desired_rows))


__all__ = [
    "build_dataset_curator_agent",
    "curate_dataset",
    "curate_dataset_sync",
    "DatasetCuratorResult",
    "LOCAL_TOOLS",
]
