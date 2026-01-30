"""
Simple Cleaning Agent using LangChain's create_agent.

A streamlined cleaning workflow:
1. Agent calls `run_clean_tests` to get EDA + validation results
2. Agent calls `apply_transformations_tool` with ALL fixes batched in one call
3. Agent calls `run_clean_tests` again to verify fixes worked
4. Repeat if needed, then `mark_cleaning_complete` when done

Uses apply_transformations_tool to batch multiple cleaning operations,
reducing LLM round-trips and intermediate dataset registrations.
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
from transformations.apply_transformations import apply_transformations_tool
from utils import generate_unique_id, get_registered_dataset, list_registered_datasets, register_dataset

# =============================================================================
# TOOLS
# =============================================================================

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
        
        # Get column names for detecting binned columns
        all_columns = [col["column"] for col in eda["schema"]]
        
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
        
        # Filter alerts - skip skew warnings if a binned version exists
        critical_alerts = []
        skipped_alerts = []
        for alert in eda["alerts"][:8]:
            if alert["type"] == "high_skew":
                col = alert["column"]
                # Check if binned version exists (e.g., income_binned, income_bin)
                has_binned = any(c.startswith(f"{col}_bin") or c == f"{col}_binned" for c in all_columns)
                if has_binned:
                    skipped_alerts.append(f"high_skew on {col} (binned version exists)")
                    continue
            critical_alerts.append(alert)
        
        # Print decision logic
        print(f"\n[run_clean_tests] Dataset: {dataset_ref}")
        print(f"[run_clean_tests] Total alerts: {len(eda['alerts'])}, Critical: {len(critical_alerts)}, Skipped: {len(skipped_alerts)}")
        if skipped_alerts:
            print(f"[run_clean_tests] Skipped alerts: {skipped_alerts}")
        print(f"[run_clean_tests] Validation passed: {validation['passed']}")
        
        # Alerts (most important)
        if critical_alerts:
            lines.append("## ⚠️ Issues Found")
            for alert in critical_alerts:
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
        
        # Recommendations (only if there are critical issues)
        if critical_alerts and eda["recommendations"]:
            lines.append("## Recommended Actions")
            for rec in eda["recommendations"][:5]:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Summary status - use critical_alerts not eda["alerts"]
        is_clean = not critical_alerts and validation["passed"]
        print(f"[run_clean_tests] Is clean: {is_clean}")
        
        if is_clean:
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
    df = get_registered_dataset(dataset_ref)
    
    # If not found, try to find the latest transformed dataset
    if df is None:
        available = list_registered_datasets()
        # Look for transformed datasets (apply_transformations_tool creates these)
        transformed = sorted([r for r in available if r.startswith("transformed_")], reverse=True)
        if transformed:
            # Use the most recent transformed dataset
            actual_ref = transformed[0]
            df = get_registered_dataset(actual_ref)
            if df is not None:
                print(f"[mark_cleaning_complete] Using latest transformed dataset: {actual_ref}")
    
    if df is None:
        available = list_registered_datasets()
        return f"✗ Dataset '{dataset_ref}' not found. Available datasets: {available[:5]}"
    
    cleaned_ref = generate_unique_id("cleaned")
    register_dataset(cleaned_ref, df)
    
    return f"✅ CLEANING COMPLETE\nCleaned dataset: `{cleaned_ref}`\nRows: {len(df):,}\nColumns: {len(df.columns)}\nReason: {reasoning}"

# All tools for the cleaning agent
# Uses apply_transformations_tool to batch all cleaning operations in one call
CLEANING_TOOLS = [
    run_clean_tests,
    mark_cleaning_complete,
    apply_transformations_tool,
]

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

CLEANING_SYSTEM_PROMPT = """You are a data cleaning agent. Clean a dataset for machine learning.

## Workflow
1. Call `run_clean_tests` to see issues
2. Call `apply_transformations_tool` with ALL fixes at once (batch them!)
3. Call `run_clean_tests` again to verify fixes worked
4. Repeat if needed, then call `mark_cleaning_complete` and STOP

## Tools
- `run_clean_tests`: Analyze dataset for issues (call first)
- `apply_transformations_tool`: Apply multiple transformations in one call
- `mark_cleaning_complete`: Finish cleaning (call last, then STOP)

## Using apply_transformations_tool
Pass a list of transformations. Each is a dict with 'tool_name' and the tool's parameters.

Available tool_names:
- impute: Fill nulls (column, strategy: mean/median/mode/ffill/bfill)
- fill_null: Fill nulls with specific value (column, value)
- drop_nulls: Remove rows with nulls (subset: [columns])
- clip: Cap outliers (column, min_val, max_val)
- drop_columns: Remove columns (columns: [list])
- dedupe: Remove duplicate rows
- cast: Change column type (column, dtype)

Example call:
```
apply_transformations_tool(
    dataset_ref="ds_123",
    transformations=[
        {"tool_name": "impute", "column": "age", "strategy": "median"},
        {"tool_name": "impute", "column": "income", "strategy": "mean"},
        {"tool_name": "clip", "column": "income", "min_val": 0, "max_val": 500000},
        {"tool_name": "drop_columns", "columns": ["id", "timestamp"]}
    ],
    output_prefix="cleaned"
)
```

## Rules
- BATCH all transformations in ONE call to apply_transformations_tool
- Fix nulls first, then outliers in the same batch
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
    
    # Run the agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": initial_message}]},
        {"recursion_limit": max_iterations},
    )
    
    # Extract the cleaned dataset reference, summary, and transformations from tool output
    cleaned_ref = None
    cleaning_summary = None
    cleaning_reason = None
    transformations = []
    
    for msg in result.get("messages", []):
        # Check for tool calls (AIMessage with tool_calls)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                
                # Extract transformations from apply_transformations_tool calls
                if tool_name == "apply_transformations_tool":
                    batch_transforms = tool_args.get("transformations", [])
                    for t in batch_transforms:
                        transformations.append({
                            "tool": t.get("tool_name", "unknown"),
                            "args": {k: v for k, v in t.items() if k != "tool_name"},
                        })
        
        # Check for tool message with cleaning complete or transformation results
        if hasattr(msg, "content") and isinstance(msg.content, str):
            # Check if this is an apply_transformations_tool result
            if hasattr(msg, "name") and msg.name == "apply_transformations_tool":
                # Store result in last batch of transformations
                if transformations:
                    transformations[-1]["batch_result"] = msg.content
            
            # Check for mark_cleaning_complete errors
            if hasattr(msg, "name") and msg.name == "mark_cleaning_complete":
                if "not found" in msg.content.lower() or msg.content.startswith("✗"):
                    # Tool returned an error - the dataset ref was wrong
                    raise ValueError(
                        f"mark_cleaning_complete failed: {msg.content}. "
                        "The agent may be using the wrong dataset reference."
                    )
            
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
    
    # Validate that cleaning completed successfully
    if cleaned_ref is None:
        # Gather diagnostic info
        num_messages = len(result.get("messages", []))
        tool_calls_made = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_made.append(tc.get("name", "unknown"))
        
        # Check if mark_cleaning_complete was called but output wasn't parsed correctly
        found_cleaning_complete = any(
            hasattr(msg, "content") and isinstance(msg.content, str) and "CLEANING COMPLETE" in msg.content
            for msg in result.get("messages", [])
        )
        
        diagnostic = (
            f"Messages: {num_messages}, "
            f"Tool calls: {tool_calls_made[-10:] if tool_calls_made else 'none'}, "
            f"Found 'CLEANING COMPLETE': {found_cleaning_complete}"
        )
        
        if found_cleaning_complete:
            # The tool was called but parsing failed - likely format issue
            raise ValueError(
                f"Cleaning appeared to complete but the cleaned dataset reference could not be parsed. "
                f"Diagnostic: {diagnostic}"
            )
        else:
            raise ValueError(
                f"Cleaning did not complete: 'mark_cleaning_complete' was not called successfully. "
                f"The agent may have hit the iteration limit ({max_iterations}) or encountered an error. "
                f"Diagnostic: {diagnostic}"
            )
    
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
