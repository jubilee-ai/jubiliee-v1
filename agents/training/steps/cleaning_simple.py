"""
Simple Cleaning Agent using LangChain's create_agent.

1. Agent calls `run_clean_tests` to see what's wrong
2. Agent uses cleaning tools to fix issues
3. Agent calls `mark_cleaning_complete` when done (re-validates before accepting)

Uses create_agent for a simple tool-calling loop — no custom LangGraph.
"""

import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from analysis.data_validation import validate_dataset
from analysis.eda_report import run_eda_report
from transformations.clean_ops import clean_tools
from transformations.column_ops import (add_column_tool, cast_tool,
                                         drop_columns_tool,
                                         parse_datetime_tool,
                                         rename_columns_tool,
                                         select_columns_tool)
from transformations.row_ops import dedupe_tool, filter_rows_tool
from transformations.tool_utils import resolve_dataset
from utils import generate_unique_id, get_registered_dataset, register_dataset

from ..utils.graph_stream_hooks import emit_graph_stream

# =============================================================================
# TOOLS
# =============================================================================


class RunCleanTestsInput(BaseModel):
    dataset_ref: str = Field(description="Reference to the dataset to analyze")


def _build_run_clean_tests(
    target_col: Optional[str] = None,
    task_type: Optional[str] = None,
    protected_columns: Optional[set[str]] = None,
):
    _protected = protected_columns or set()

    @tool(args_schema=RunCleanTestsInput)
    def run_clean_tests(dataset_ref: str) -> str:
        """
        Run EDA report and data validation on the dataset.
        ALWAYS call this FIRST to understand data quality issues.
        Returns shape, nulls, outliers, skew, and violations.
        """
        try:
            eda = run_eda_report(
                dataset_ref=dataset_ref,
                target_col=target_col,
                task_type=task_type,
            )

            all_columns = [col["column"] for col in eda["schema"][:15]]
            validatable_columns = [c for c in all_columns if c not in _protected]

            validation = validate_dataset(
                dataset_ref=dataset_ref,
                rules=[{"type": "not_null", "columns": validatable_columns}] if validatable_columns else [],
                thresholds={"max_null_pct": 0.3},
            )

            lines = []
            lines.append(f"# Dataset Analysis: {dataset_ref}")
            lines.append(f"**Shape:** {eda['shape']['rows']:,} rows × {eda['shape']['columns']} cols")
            lines.append("")

            if _protected:
                lines.append("## Protected Columns (DO NOT MODIFY)")
                for col in sorted(_protected):
                    role = "target" if col == target_col else "forbidden"
                    lines.append(f"- `{col}` ({role})")
                lines.append("")

            if "target_analysis" in eda:
                ta = eda["target_analysis"]
                lines.append("## Target Analysis (informational — do not act on target column)")
                lines.append(f"**Target:** `{ta['column']}` ({ta['task']})")
                if ta["task"] == "classification":
                    lines.append(f"**Class counts:** {ta.get('class_counts', {})}")
                    if ta.get("imbalance_ratio", 1) > 3:
                        lines.append(f"**Imbalanced:** {ta['imbalance_ratio']:.1f}:1")
                else:
                    lines.append(f"**Mean:** {ta.get('mean', 0):,.2f}, Skew: {ta.get('skew', 0):.2f}")
                if ta.get("recommendation"):
                    lines.append(f"**Note:** {ta['recommendation']}")
                lines.append("")

            if eda.get("target_associations"):
                lines.append("## Top Feature-Target Associations")
                for assoc in eda["target_associations"][:8]:
                    direction = f" ({assoc.get('direction', '')})" if "direction" in assoc else ""
                    lines.append(f"- `{assoc['column']}`: {assoc['metric']}={assoc['value']:.3f}{direction}")
                lines.append("")

            lines.append("## Columns")
            for col in eda["schema"][:15]:
                null_str = f" ({col['null_pct']:.0%} null)" if col["null_pct"] > 0 else ""
                protected_marker = " [PROTECTED]" if col["column"] in _protected else ""
                lines.append(f"- `{col['column']}`: {col['dtype']}{null_str}{protected_marker}")
            if len(eda["schema"]) > 15:
                lines.append(f"- ... +{len(eda['schema']) - 15} more")
            lines.append("")

            actionable_alerts = [
                a for a in eda.get("alerts", [])
                if a.get("column") not in _protected
                and not any(c in _protected for c in a.get("columns", []))
            ]
            if actionable_alerts:
                lines.append("## Issues Found")
                for alert in actionable_alerts[:8]:
                    if alert["type"] == "high_skew":
                        lines.append(f"- High skew on `{alert['column']}` (skew={alert['skew']})")
                    elif alert["type"] == "high_nulls":
                        lines.append(f"- High nulls on `{alert['column']}` ({alert['null_pct']:.0%})")
                    elif alert["type"] == "id_like":
                        lines.append(f"- ID-like column: `{alert['column']}`")
                    elif alert["type"] == "redundant":
                        lines.append(f"- Redundant: `{alert['columns'][0]}` <-> `{alert['columns'][1]}`")
                    elif alert["type"] == "high_cardinality":
                        lines.append(f"- High cardinality: `{alert['column']}` ({alert['unique']} unique)")
                lines.append("")

            if not validation.get("passed", True):
                filtered_violations = [
                    v for v in validation.get("violations", [])
                    if v.get("column") not in _protected
                ]
                if filtered_violations:
                    lines.append("## Validation Issues")
                    for v in filtered_violations[:5]:
                        lines.append(f"- {v['rule']} on `{v['column']}`: {v['count']:,} issues ({v['pct']:.1%})")
                        fix = v["suggested_fix"]
                        lines.append(f"  Suggested: `{fix['action']}` {fix.get('params', {})}")
                    lines.append("")

            filtered_recs = [
                rec for rec in eda.get("recommendations", [])
                if not any(pc in rec for pc in _protected)
            ]
            if filtered_recs:
                lines.append("## Recommended Actions")
                for rec in filtered_recs[:5]:
                    lines.append(f"- {rec}")
                lines.append("")

            if not actionable_alerts and validation.get("passed", True):
                lines.append("**Data looks clean!** Call `mark_cleaning_complete` to finish.")
            else:
                lines.append("**Issues found.** Apply cleaning tools, then `run_clean_tests` again to verify.")

            return "\n".join(lines)

        except Exception as e:
            return f"Analysis failed: {type(e).__name__}: {e}"

    return run_clean_tests


class MarkCleaningCompleteInput(BaseModel):
    dataset_ref: str = Field(description="The current dataset reference")
    reasoning: str = Field(description="Brief explanation of why the dataset is clean")


_MIN_ROWS_DEFAULT = 20
_MAX_NULL_PCT_DEFAULT = 0.30


def _build_mark_cleaning_complete(
    protected_columns: Optional[set[str]] = None,
    min_rows: int = _MIN_ROWS_DEFAULT,
    max_null_pct: float = _MAX_NULL_PCT_DEFAULT,
):
    _protected = protected_columns or set()

    @tool(args_schema=MarkCleaningCompleteInput)
    def mark_cleaning_complete(dataset_ref: str, reasoning: str) -> str:
        """
        Mark the dataset as cleaned and ready for feature engineering.
        Re-validates before accepting. If critical issues remain, it rejects.
        """
        try:
            df = resolve_dataset(dataset_ref)
        except ValueError:
            return f"Dataset '{dataset_ref}' not found. Check the dataset reference."

        issues: list[str] = []

        if len(df) < min_rows:
            issues.append(f"Only {len(df)} rows remaining (minimum: {min_rows}). Over-filtered?")

        missing_protected = [c for c in _protected if c not in df.columns]
        if missing_protected:
            issues.append(f"Protected columns were dropped: {missing_protected}.")

        actionable_cols = [c for c in df.columns if c not in _protected]
        high_null_cols = [f"`{c}` ({df[c].isna().mean():.0%})" for c in actionable_cols if df[c].isna().mean() > max_null_pct]
        if high_null_cols:
            issues.append(f"Columns above {max_null_pct:.0%} null: {', '.join(high_null_cols)}")

        if len(df.columns) < 2:
            issues.append("Dataset has fewer than 2 columns.")

        if issues:
            body = "\n".join(f"- {i}" for i in issues)
            return f"Cannot mark complete — validation failed:\n{body}\n\nFix these, then try again."

        cleaned_ref = generate_unique_id("cleaned")
        register_dataset(cleaned_ref, df)

        return (
            f"CLEANING COMPLETE\n"
            f"Cleaned dataset: `{cleaned_ref}`\n"
            f"Rows: {len(df):,}\n"
            f"Columns: {len(df.columns)}\n"
            f"Reason: {reasoning}"
        )

    return mark_cleaning_complete


# Default module-level instances
mark_cleaning_complete = _build_mark_cleaning_complete()
run_clean_tests = _build_run_clean_tests()

CLEANING_TOOLS = [
    run_clean_tests,
    mark_cleaning_complete,
    *clean_tools,  # drop_nulls, fill_null, impute, clip, replace_values, regex_replace
    drop_columns_tool,
    select_columns_tool,
    rename_columns_tool,
    cast_tool,
    parse_datetime_tool,
    add_column_tool,
    dedupe_tool,
    filter_rows_tool,
]

# =============================================================================
# SYSTEM PROMPT
# =============================================================================


def _model_cleaning_guidance(selected_model: Optional[str]) -> str:
    if not selected_model:
        return ""
    m = selected_model.lower()
    if any(w in m for w in ["xgboost", "lightgbm", "catboost", "gradient_boost"]):
        return (
            "\n## Model: Tree-based\n"
            "- Trees handle nulls natively — only impute columns with >50% nulls\n"
            "- Don't clip outliers unless extreme — trees split around them\n"
            "- Focus on dropping ID-like/constant columns and deduplication"
        )
    if any(w in m for w in ["logistic", "linear", "svm", "elastic"]):
        return (
            "\n## Model: Linear\n"
            "- Linear models can't handle nulls — impute every column\n"
            "- Clip extreme outliers — they skew coefficients\n"
            "- Drop high-cardinality categoricals and ID columns"
        )
    if any(w in m for w in ["neural", "deep", "mlp", "network"]):
        return (
            "\n## Model: Neural Network\n"
            "- Impute all nulls — neural nets can't handle missing values\n"
            "- Clip extreme outliers — they destabilize training\n"
            "- Ensure consistent numeric types"
        )
    if any(w in m for w in ["random_forest", "decision_tree"]):
        return (
            "\n## Model: Tree-based\n"
            "- Trees tolerate some nulls — only impute columns with >30% nulls\n"
            "- Don't clip outliers — trees handle them well\n"
            "- Drop ID-like columns and deduplicate"
        )
    return ""


def _build_system_prompt(
    goal: str = "",
    target_col: Optional[str] = None,
    protected_columns: Optional[set[str]] = None,
    selected_model: Optional[str] = None,
) -> str:
    protected_section = ""
    if protected_columns:
        cols_list = ", ".join(
            f"`{c}`" + (" (target)" if c == target_col else "")
            for c in sorted(protected_columns)
        )
        protected_section = f"\n\n## Protected Columns — DO NOT MODIFY\n{cols_list}\nNever impute, clip, drop, or transform these."

    model_section = _model_cleaning_guidance(selected_model)

    goal_section = f"\n\n## Goal\n{goal}" if goal else ""

    return f"""You are a data cleaning agent. Prepare a dataset for ML training.
{goal_section}{protected_section}{model_section}
## Workflow
1. Call `run_clean_tests` to see the data and its issues
2. Fix issues using cleaning tools — work through these in order:
   - Drop ID-like columns, constant columns, and exact duplicates
   - Fix data types (cast numeric strings to numbers, parse dates)
   - Handle nulls (impute with median/mode, or drop if >70% null)
   - Clip extreme outliers in numeric columns
   - Fix inconsistent values (replace_values, regex_replace)
   - Filter irrelevant rows if clearly not useful for the goal
   - Create simple derived columns if obviously helpful (add_column)
3. Call `run_clean_tests` again to verify fixes
4. Call `mark_cleaning_complete` when done, then STOP immediately

## Tools
- `run_clean_tests` — Analyze dataset (call first + after changes)
- `impute` — Fill nulls with mean/median/mode
- `fill_null` — Fill nulls with specific value
- `drop_nulls` — Remove rows with nulls
- `clip` — Cap numeric values to a range
- `replace_values` — Map old values to new
- `regex_replace` — Regex pattern replacement
- `drop_columns` — Remove columns
- `select_columns` — Keep only specified columns
- `rename_columns` — Rename columns
- `cast` — Change column type
- `parse_datetime` — Parse strings to datetime
- `add_column` — Create column from expression
- `filter_rows` — Keep rows matching a predicate
- `dedupe` — Remove duplicate rows
- `mark_cleaning_complete` — Finalize (validates first)

## Rules
- Always use the dataset_ref from the MOST RECENT tool output
- Don't over-clean — minor issues are OK
- Never modify protected columns
- After `mark_cleaning_complete`, output NOTHING else"""


CLEANING_SYSTEM_PROMPT = _build_system_prompt()

# =============================================================================
# AGENT CREATION
# =============================================================================


def create_cleaning_agent(
    model: str = "openai:gpt-5.1",
    goal: str = "",
    target_col: Optional[str] = None,
    task_type: Optional[str] = None,
    forbidden_columns: Optional[list[str]] = None,
    selected_model: Optional[str] = None,
):
    protected_columns: set[str] = set()
    if target_col:
        protected_columns.add(target_col)
    if forbidden_columns:
        protected_columns.update(forbidden_columns)

    has_context = bool(protected_columns or selected_model or goal)

    if has_context:
        tests_tool = _build_run_clean_tests(target_col, task_type, protected_columns)
        complete_tool = _build_mark_cleaning_complete(protected_columns)
        tools = [
            tests_tool, complete_tool,
            *clean_tools,
            drop_columns_tool, select_columns_tool, rename_columns_tool,
            cast_tool, parse_datetime_tool, add_column_tool,
            dedupe_tool, filter_rows_tool,
        ]
        system_prompt = _build_system_prompt(goal, target_col, protected_columns, selected_model)
    else:
        tools = CLEANING_TOOLS
        system_prompt = CLEANING_SYSTEM_PROMPT

    return create_agent(model=model, tools=tools, system_prompt=system_prompt)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def run_cleaning_simple(
    dataset_ref: str,
    goal: str = "Clean the dataset for machine learning",
    model: str = "openai:gpt-5.1",
    max_iterations: int = 10,
    target_col: Optional[str] = None,
    task_type: Optional[str] = None,
    forbidden_columns: Optional[list[str]] = None,
    selected_model: Optional[str] = None,
):
    """Run the cleaning agent on a dataset. Returns dict with cleaned_ref, messages, etc."""
    # Pre-validation
    try:
        df = resolve_dataset(dataset_ref)
        if len(df.columns) < 2:
            return {
                "cleaned_ref": dataset_ref, "messages": [],
                "cleaning_summary": f"Skipped: only {len(df.columns)} column(s)",
                "cleaning_reason": None, "transformations": [],
            }
        if target_col and target_col not in df.columns:
            target_col = None
            task_type = None
    except Exception as e:
        print(f"[cleaning_simple] WARNING: pre-validation failed: {e}")

    emit_graph_stream({"type": "progress", "message": "Analyzing data for cleaning...", "phase": "cleaning"})

    agent = create_cleaning_agent(
        model=model, goal=goal, target_col=target_col,
        task_type=task_type, forbidden_columns=forbidden_columns,
        selected_model=selected_model,
    )

    context_parts = [f"**Goal:** {goal}", f"**Dataset:** `{dataset_ref}`"]
    if selected_model:
        context_parts.append(f"**Model:** {selected_model}")
    if target_col:
        context_parts.append(f"**Target column:** `{target_col}` ({task_type or 'unknown'})")

    initial_message = (
        f"Clean this dataset for training:\n\n"
        + "\n".join(context_parts)
        + "\n\nStart by calling `run_clean_tests`."
    )

    emit_graph_stream({"type": "progress", "message": f"Applying transformations...", "phase": "cleaning"})

    result = agent.invoke(
        {"messages": [{"role": "user", "content": initial_message}]},
        {"recursion_limit": max_iterations * 3},
    )

    emit_graph_stream({"type": "progress", "message": "Validating cleaned data...", "phase": "cleaning"})

    # Extract results from messages
    cleaned_ref = None
    cleaning_summary = None
    cleaning_reason = None
    transformations = []

    transformation_tools = {
        "drop_columns_tool", "drop_nulls", "fill_null", "impute",
        "clip", "replace_values", "regex_replace", "cast_tool",
        "dedupe_tool", "filter_rows_tool", "drop_columns", "cast", "dedupe",
        "filter_rows", "add_column_tool", "add_column", "select_columns_tool",
        "select_columns", "rename_columns_tool", "rename_columns",
        "parse_datetime_tool", "parse_datetime",
    }

    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                if name in transformation_tools or name.startswith(("drop_", "fill_", "add_", "select_", "rename_", "parse_")):
                    transformations.append({"tool": name, "args": tc.get("args", {})})

        if hasattr(msg, "content") and isinstance(msg.content, str):
            if hasattr(msg, "name") and msg.name in transformation_tools:
                if transformations and transformations[-1].get("tool") == msg.name:
                    transformations[-1]["result"] = msg.content

            if "CLEANING COMPLETE" in msg.content and "Cleaned dataset:" in msg.content:
                cleaning_summary = msg.content
                for line in msg.content.split("\n"):
                    if "Cleaned dataset:" in line and "`" in line:
                        cleaned_ref = line.split("`")[1]
                    if "Reason:" in line:
                        cleaning_reason = line.replace("Reason:", "").strip()

    if cleaned_ref is None:
        print(f"[cleaning_simple] WARNING: no cleaned_ref extracted, falling back to {dataset_ref}")
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
