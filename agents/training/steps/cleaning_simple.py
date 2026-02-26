"""
Simple Cleaning Agent using LangChain's create_agent.

A streamlined cleaning workflow:
1. Agent calls `run_clean_tests` to get EDA + validation results
2. Agent uses cleaning tools to fix issues OR calls `mark_cleaning_complete` if done
3. Repeat until complete or max iterations reached

Uses create_agent for a simple tool-calling loop without custom LangGraph.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add data-tools to path
# Path: steps -> training -> agents -> root -> tools/data-tools
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from analysis.data_validation import validate_dataset
from analysis.eda_report import run_eda_report
from transformations.clean_ops import clean_tools
from transformations.column_ops import cast_tool, drop_columns_tool
from transformations.row_ops import dedupe_tool, filter_rows_tool
from transformations.tool_utils import resolve_dataset
from utils import generate_unique_id, get_registered_dataset, register_dataset

# =============================================================================
# TOOLS
# =============================================================================

# TODO: Optimize it so that we have a single apply_transformations tool which takes the transformations as inputs.
# then we apply them in a loop, getting the new dataset_ref and running each next one on those
# --> Put this into the tools folder
# --> Currently we have it but needs improvement (NEED THIS NOW)
class RunCleanTestsInput(BaseModel):
    """Input for running EDA + validation tests."""
    dataset_ref: str = Field(description="Reference to the dataset to analyze")


@tool(args_schema=RunCleanTestsInput)
def run_clean_tests(dataset_ref: str) -> str:
    """
    Run EDA report and data validation on the dataset.
    
    ALWAYS call this tool FIRST to understand the data quality issues.
    Returns a summary of shape, nulls, outliers, skew, and violations.
    Use the output to decide which cleaning tools to apply.
    """
    try:
        # Run EDA
        eda = run_eda_report(dataset_ref=dataset_ref)
        
        # Build validation rules from schema
        column_names = [col["column"] for col in eda["schema"][:15]]
        validation = validate_dataset(
            dataset_ref=dataset_ref,
            rules=[{"type": "not_null", "columns": column_names}],
            thresholds={"max_null_pct": 0.3},
        )
        
        # Format compact output
        lines = []
        lines.append(f"# Dataset Analysis: {dataset_ref}")
        lines.append(f"**Shape:** {eda['shape']['rows']:,} rows × {eda['shape']['columns']} cols")
        lines.append("")
        
        # Schema with null info
        lines.append("## Columns")
        for col in eda["schema"][:15]:
            null_str = f" ({col['null_pct']:.0%} null)" if col["null_pct"] > 0 else ""
            lines.append(f"- `{col['column']}`: {col['dtype']}{null_str}")
        if len(eda["schema"]) > 15:
            lines.append(f"- ... +{len(eda['schema']) - 15} more")
        lines.append("")
        
        # Alerts (most important)
        if eda["alerts"]:
            lines.append("## ⚠️ Issues Found")
            for alert in eda["alerts"][:8]:
                if alert["type"] == "high_skew":
                    lines.append(f"- High skew on `{alert['column']}` (skew={alert['skew']})")
                elif alert["type"] == "high_nulls":
                    lines.append(f"- High nulls on `{alert['column']}` ({alert['null_pct']:.0%})")
                elif alert["type"] == "id_like":
                    lines.append(f"- ID-like column: `{alert['column']}`")
                elif alert["type"] == "redundant":
                    lines.append(f"- Redundant: `{alert['columns'][0]}` ↔ `{alert['columns'][1]}`")
                elif alert["type"] == "high_cardinality":
                    lines.append(f"- High cardinality: `{alert['column']}` ({alert['unique']} unique)")
            lines.append("")
        
        # Validation violations
        if not validation["passed"]:
            lines.append("## Validation Issues")
            for v in validation["violations"][:5]:
                lines.append(f"- {v['rule']} on `{v['column']}`: {v['count']:,} issues ({v['pct']:.1%})")
                fix = v["suggested_fix"]
                lines.append(f"  → Suggested: `{fix['action']}` {fix.get('params', {})}")
            lines.append("")
        
        # Recommendations
        if eda["recommendations"]:
            lines.append("## Recommended Actions")
            for rec in eda["recommendations"][:5]:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Summary status
        if not eda["alerts"] and validation["passed"]:
            lines.append("✅ **Data looks clean!** Call `mark_cleaning_complete` to finish.")
        else:
            lines.append("⚠️ **Issues found.** Apply cleaning tools, then call `run_clean_tests` again to verify.")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"✗ Analysis failed: {type(e).__name__}: {e}"


class MarkCleaningCompleteInput(BaseModel):
    """Input for marking cleaning complete."""
    dataset_ref: str = Field(description="The current dataset reference")
    reasoning: str = Field(description="Brief explanation of why the dataset is clean")


@tool(args_schema=MarkCleaningCompleteInput)
def mark_cleaning_complete(dataset_ref: str, reasoning: str) -> str:
    """
    Mark the dataset as cleaned and ready for feature engineering.
    
    Call this ONLY when:
    - No major null issues remain
    - Outliers are handled
    - Data types are correct
    - No critical validation failures
    
    This registers the cleaned dataset and ends the cleaning process.
    """
    try:
        df = resolve_dataset(dataset_ref)
    except ValueError:
        return f"✗ Dataset '{dataset_ref}' not found. Check the dataset reference."
    
    cleaned_ref = generate_unique_id("cleaned")
    register_dataset(cleaned_ref, df)
    
    return f"✅ CLEANING COMPLETE\nCleaned dataset: `{cleaned_ref}`\nRows: {len(df):,}\nColumns: {len(df.columns)}\nReason: {reasoning}"

# TODO: something more like this
# CLEANING_TOOLS = [apply_transformations_tool, run_clean_tests, mark_cleaning_complete]

# All tools for the cleaning agent
CLEANING_TOOLS = [
    run_clean_tests,
    mark_cleaning_complete,
    *clean_tools,  # drop_nulls, fill_null, impute, clip, replace_values, regex_replace
    drop_columns_tool,
    cast_tool,
    dedupe_tool,
    filter_rows_tool,
]

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

CLEANING_SYSTEM_PROMPT = """You are a data cleaning agent. Clean a dataset for machine learning.

## Workflow
1. Call `run_clean_tests` to see issues
2. Fix issues using cleaning tools (impute, clip, etc.)
3. When done, call `mark_cleaning_complete` and STOP IMMEDIATELY

## Tools
- `run_clean_tests`: Analyze dataset for issues (call first)
- `impute`: Fill nulls with mean/median/mode
- `fill_null`: Fill nulls with specific value
- `drop_nulls`: Remove rows with nulls
- `clip`: Cap outliers to a range
- `drop_columns`: Remove columns
- `dedupe`: Remove duplicate rows
- `mark_cleaning_complete`: Finish cleaning (call last, then STOP)

## Rules
- Fix nulls first, then outliers
- Use the dataset_ref from previous tool output
- Don't over-clean - minor issues are OK

## CRITICAL: After `mark_cleaning_complete`
When you call `mark_cleaning_complete`, you MUST stop immediately:
- Do NOT output any summary of what you did
- Do NOT offer next steps or suggestions
- Do NOT write anything after the tool call
- Just call the tool and end your turn with no additional text"""

# =============================================================================
# AGENT CREATION
# =============================================================================

def create_cleaning_agent(model: str = "openai:gpt-5.1"):
    """
    Create the cleaning agent using LangChain's create_agent.
    
    Args:
        model: Model identifier (default: openai:gpt-5.1)
    
    Returns:
        Compiled agent graph
    """
    return create_agent(
        model=model,
        tools=CLEANING_TOOLS,
        system_prompt=CLEANING_SYSTEM_PROMPT,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_cleaning_simple(
    dataset_ref: str,
    goal: str = "Clean the dataset for machine learning",
    model: str = "openai:gpt-5.1",
    max_iterations: int = 10,
):
    """
    Run the simple cleaning agent on a dataset.
    
    Args:
        dataset_ref: Reference to the dataset to clean
        goal: Description of what the dataset will be used for
        model: Model to use for the agent
        max_iterations: Maximum number of agent steps
    
    Returns:
        Dict with:
        - cleaned_ref: The cleaned dataset reference (or None if not complete)
        - messages: All agent messages for debugging
    """
    agent = create_cleaning_agent(model=model)
    
    # Initial message includes the goal and dataset
    initial_message = f"""Clean this dataset for the following goal:

**Goal:** {goal}

**Dataset:** `{dataset_ref}`

Start by calling `run_clean_tests` to analyze the dataset."""
    
    # Each tool call uses ~3 graph steps (model → middleware → tools),
    # so multiply max_iterations to get the actual recursion limit.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": initial_message}]},
        {"recursion_limit": max_iterations * 3},
    )
    
    # Extract the cleaned dataset reference, summary, and transformations from tool output
    cleaned_ref = None
    cleaning_summary = None
    cleaning_reason = None
    transformations = []
    
    # Tools that are actual transformations (not analysis tools)
    transformation_tools = {
        "drop_columns_tool", "drop_nulls", "fill_null", "impute", 
        "clip", "replace_values", "regex_replace", "cast_tool",
        "dedupe_tool", "filter_rows_tool", "drop_columns", "cast", "dedupe", "filter_rows"
    }
    
    for msg in result.get("messages", []):
        # Check for tool calls (AIMessage with tool_calls)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                if tool_name in transformation_tools or tool_name.startswith("drop_") or tool_name.startswith("fill_"):
                    transformations.append({
                        "tool": tool_name,
                        "args": tool_args,
                    })
        
        # Check for tool message with cleaning complete
        if hasattr(msg, "content") and isinstance(msg.content, str):
            # Check if this is a transformation tool result
            if hasattr(msg, "name") and msg.name in transformation_tools:
                # Update the last transformation with its result
                if transformations and transformations[-1].get("tool") == msg.name:
                    transformations[-1]["result"] = msg.content
            
            if "CLEANING COMPLETE" in msg.content and "Cleaned dataset:" in msg.content:
                # Store the full cleaning summary message
                cleaning_summary = msg.content
                
                # Extract the dataset ref from the tool output
                for line in msg.content.split("\n"):
                    if "Cleaned dataset:" in line:
                        # Format: "Cleaned dataset: `cleaned_xxx`"
                        cleaned_ref = line.split("`")[1] if "`" in line else None
                    if "Reason:" in line:
                        # Extract the reasoning
                        cleaning_reason = line.replace("Reason:", "").strip()
    
    if cleaned_ref is None:
        print(f"[cleaning_simple] WARNING: Could not extract cleaned_ref from agent output. "
              f"Falling back to original dataset_ref: {dataset_ref}")
        cleaned_ref = dataset_ref

    return {
        "cleaned_ref": cleaned_ref,
        "messages": result.get("messages", []),
        "cleaning_summary": cleaning_summary,
        "cleaning_reason": cleaning_reason,
        "transformations": transformations,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "create_cleaning_agent",
    "run_cleaning_simple",
    "run_clean_tests",
    "mark_cleaning_complete",
    "CLEANING_TOOLS",
]
