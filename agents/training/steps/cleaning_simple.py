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
from typing import Optional

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


def _build_run_clean_tests(
    target_col: Optional[str] = None,
    task_type: Optional[str] = None,
    protected_columns: Optional[set[str]] = None,
):
    """Create a run_clean_tests tool, optionally with target/protected column awareness.

    When target_col is provided, the EDA report includes target analysis and
    feature-target associations. Alerts, violations, and recommendations for
    protected columns are filtered out so the LLM never sees them as actionable.
    """
    _protected = protected_columns or set()

    @tool(args_schema=RunCleanTestsInput)
    def run_clean_tests(dataset_ref: str) -> str:
        """
        Run EDA report and data validation on the dataset.

        ALWAYS call this tool FIRST to understand the data quality issues.
        Returns a summary of shape, nulls, outliers, skew, and violations.
        Use the output to decide which cleaning tools to apply.
        """
        try:
            eda = run_eda_report(
                dataset_ref=dataset_ref,
                target_col=target_col,
                task_type=task_type,
            )

            # Only validate columns the agent is allowed to act on
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

            # ------ Protected columns section ------
            if _protected:
                lines.append("## 🔒 Protected Columns (DO NOT MODIFY)")
                for col in sorted(_protected):
                    role = "target" if col == target_col else "forbidden"
                    lines.append(f"- `{col}` ({role})")
                lines.append("")

            # ------ Target analysis (informational, not actionable) ------
            if "target_analysis" in eda:
                ta = eda["target_analysis"]
                lines.append("## Target Analysis (informational — do not act on target column)")
                lines.append(f"**Target:** `{ta['column']}` ({ta['task']})")
                if ta["task"] == "classification":
                    lines.append(f"**Class counts:** {ta.get('class_counts', {})}")
                    if ta.get("imbalance_ratio", 1) > 3:
                        lines.append(
                            f"**⚠️ Imbalanced:** {ta['imbalance_ratio']:.1f}:1 "
                            f"— note for downstream training, but do NOT modify the target"
                        )
                else:
                    lines.append(f"**Mean:** {ta.get('mean', 0):,.2f}, Skew: {ta.get('skew', 0):.2f}")
                if ta.get("recommendation"):
                    lines.append(f"**Note:** {ta['recommendation']}")
                lines.append("")

            # ------ Feature-target associations (informational) ------
            if eda.get("target_associations"):
                lines.append("## Top Feature-Target Associations (informational)")
                for assoc in eda["target_associations"][:8]:
                    direction = f" ({assoc.get('direction', '')})" if "direction" in assoc else ""
                    lines.append(
                        f"- `{assoc['column']}`: {assoc['metric']}={assoc['value']:.3f}{direction}"
                    )
                lines.append("")

            # ------ Schema with null info ------
            lines.append("## Columns")
            for col in eda["schema"][:15]:
                null_str = f" ({col['null_pct']:.0%} null)" if col["null_pct"] > 0 else ""
                protected_marker = " 🔒" if col["column"] in _protected else ""
                lines.append(f"- `{col['column']}`: {col['dtype']}{null_str}{protected_marker}")
            if len(eda["schema"]) > 15:
                lines.append(f"- ... +{len(eda['schema']) - 15} more")
            lines.append("")

            # ------ Alerts — filter out protected columns ------
            actionable_alerts = []
            for alert in eda.get("alerts", []):
                alert_col = alert.get("column")
                alert_cols = alert.get("columns", [])
                if alert_col and alert_col in _protected:
                    continue
                if alert_cols and any(c in _protected for c in alert_cols):
                    continue
                actionable_alerts.append(alert)

            if actionable_alerts:
                lines.append("## ⚠️ Issues Found")
                for alert in actionable_alerts[:8]:
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

            # ------ Validation violations — filter out protected columns ------
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
                        lines.append(f"  → Suggested: `{fix['action']}` {fix.get('params', {})}")
                    lines.append("")

            # ------ Recommendations — filter out protected columns ------
            filtered_recs = [
                rec for rec in eda.get("recommendations", [])
                if not any(pc in rec for pc in _protected)
            ]
            if filtered_recs:
                lines.append("## Recommended Actions")
                for rec in filtered_recs[:5]:
                    lines.append(f"- {rec}")
                lines.append("")

            # ------ Summary status ------
            if not actionable_alerts and validation.get("passed", True):
                lines.append("✅ **Data looks clean!** Call `mark_cleaning_complete` to finish.")
            else:
                lines.append("⚠️ **Issues found.** Apply cleaning tools, then call `run_clean_tests` again to verify.")

            return "\n".join(lines)

        except Exception as e:
            return f"✗ Analysis failed: {type(e).__name__}: {e}"

    return run_clean_tests


class MarkCleaningCompleteInput(BaseModel):
    """Input for marking cleaning complete."""
    dataset_ref: str = Field(description="The current dataset reference")
    reasoning: str = Field(description="Brief explanation of why the dataset is clean")


_MIN_ROWS_DEFAULT = 20
_MAX_NULL_PCT_DEFAULT = 0.30


def _build_mark_cleaning_complete(
    protected_columns: Optional[set[str]] = None,
    min_rows: int = _MIN_ROWS_DEFAULT,
    max_null_pct: float = _MAX_NULL_PCT_DEFAULT,
):
    """Create a mark_cleaning_complete tool that re-validates before accepting."""
    _protected = protected_columns or set()

    @tool(args_schema=MarkCleaningCompleteInput)
    def mark_cleaning_complete(dataset_ref: str, reasoning: str) -> str:
        """
        Mark the dataset as cleaned and ready for feature engineering.

        Call this ONLY when:
        - No major null issues remain
        - Outliers are handled
        - Data types are correct
        - No critical validation failures

        This re-validates the dataset before accepting. If critical issues
        remain, it will reject and tell you what to fix.
        """
        try:
            df = resolve_dataset(dataset_ref)
        except ValueError:
            return f"✗ Dataset '{dataset_ref}' not found. Check the dataset reference."

        issues: list[str] = []

        if len(df) < min_rows:
            issues.append(
                f"Only {len(df)} rows remaining (minimum: {min_rows}). "
                f"You may have over-filtered — undo row removals or use imputation instead."
            )

        missing_protected = [c for c in _protected if c not in df.columns]
        if missing_protected:
            issues.append(
                f"Protected columns were dropped: {missing_protected}. "
                f"These must remain in the dataset."
            )

        actionable_cols = [c for c in df.columns if c not in _protected]
        high_null_cols = []
        for col in actionable_cols:
            null_pct = df[col].isna().mean()
            if null_pct > max_null_pct:
                high_null_cols.append(f"`{col}` ({null_pct:.0%} null)")
        if high_null_cols:
            issues.append(
                f"Columns still above {max_null_pct:.0%} null threshold: "
                + ", ".join(high_null_cols)
            )

        if len(df.columns) < 2:
            issues.append("Dataset has fewer than 2 columns — nothing to train on.")

        if issues:
            body = "\n".join(f"- {i}" for i in issues)
            return (
                f"✗ Cannot mark complete — validation failed:\n{body}\n\n"
                f"Fix these issues, then call `mark_cleaning_complete` again."
            )

        cleaned_ref = generate_unique_id("cleaned")
        register_dataset(cleaned_ref, df)

        return (
            f"✅ CLEANING COMPLETE\n"
            f"Cleaned dataset: `{cleaned_ref}`\n"
            f"Rows: {len(df):,}\n"
            f"Columns: {len(df.columns)}\n"
            f"Reason: {reasoning}"
        )

    return mark_cleaning_complete


# Default module-level instance (no validation context)
mark_cleaning_complete = _build_mark_cleaning_complete()

# TODO: something more like this
# CLEANING_TOOLS = [apply_transformations_tool, run_clean_tests, mark_cleaning_complete]

# Default module-level instances (target-unaware, for backward compatibility)
run_clean_tests = _build_run_clean_tests()

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


def _model_cleaning_guidance(selected_model: Optional[str]) -> str:
    """Return model-specific cleaning guidance for the system prompt."""
    if not selected_model:
        return ""
    m = selected_model.lower()
    if any(w in m for w in ["xgboost", "lightgbm", "catboost", "gradient_boost"]):
        return """
## Model-Specific Guidance (tree-based)
- Tree models handle nulls natively — only impute columns with >50% nulls
- Don't clip outliers unless extreme — trees split around them naturally
- Focus on removing ID-like columns and deduplication
- Prefer dropping rows only as a last resort"""
    if any(w in m for w in ["logistic", "linear", "svm", "elastic"]):
        return """
## Model-Specific Guidance (linear model)
- Linear models cannot handle nulls — impute every column with missing values
- Clip extreme outliers — they disproportionately skew linear coefficients
- Remove ID-like and high-cardinality categorical columns
- Highly skewed features hurt performance — note them for downstream feature engineering"""
    if any(w in m for w in ["naive_bayes", "bayes"]):
        return """
## Model-Specific Guidance (Naive Bayes)
- Naive Bayes cannot handle nulls — impute all missing values
- Outliers are less critical but clip if extreme
- Ensure correct data types (numeric vs categorical)"""
    if any(w in m for w in ["neural", "deep", "mlp", "network"]):
        return """
## Model-Specific Guidance (neural network)
- Neural nets cannot handle nulls — impute all missing values
- Clip extreme outliers — they destabilize gradient-based training
- Ensure consistent numeric types"""
    if any(w in m for w in ["survival", "cox"]):
        return """
## Model-Specific Guidance (survival analysis)
- Ensure event and duration columns are clean and non-null
- Do NOT remove censored observations
- Handle nulls in covariates via imputation"""
    if any(w in m for w in ["random_forest", "decision_tree"]):
        return """
## Model-Specific Guidance (tree-based)
- Random forests tolerate some nulls — only impute columns with >30% nulls
- Don't clip outliers — trees handle them well
- Remove ID-like columns and deduplicate"""
    return ""


def _build_system_prompt(
    target_col: Optional[str] = None,
    protected_columns: Optional[set[str]] = None,
    selected_model: Optional[str] = None,
) -> str:
    """Build cleaning system prompt with optional protected columns and model guidance."""
    protected_section = ""
    if protected_columns:
        cols_list = "\n".join(
            f"- `{c}`" + (" (target)" if c == target_col else " (forbidden)")
            for c in sorted(protected_columns)
        )
        protected_section = f"""

## 🔒 Protected Columns — DO NOT MODIFY
These columns must not be imputed, clipped, dropped, or transformed:
{cols_list}

If `run_clean_tests` reports an issue on a protected column, SKIP it.
Do NOT apply any cleaning tool to these columns."""

    model_section = _model_cleaning_guidance(selected_model)

    protected_rule = (
        "\n- NEVER modify, impute, clip, or drop protected columns (marked with 🔒)"
        if protected_columns else ""
    )

    return f"""You are a data cleaning agent. Clean a dataset for machine learning.

## Workflow
1. Call `run_clean_tests` to see issues
2. Fix issues using cleaning tools (impute, clip, etc.)
3. When done, call `mark_cleaning_complete` and STOP IMMEDIATELY
{protected_section}{model_section}
## Tools
- `run_clean_tests`: Analyze dataset for issues (call first)
- `impute`: Fill nulls with mean/median/mode
- `fill_null`: Fill nulls with specific value
- `drop_nulls`: Remove rows with nulls
- `clip`: Cap outliers to a range
- `drop_columns`: Remove columns
- `dedupe`: Remove duplicate rows
- `mark_cleaning_complete`: Finish cleaning — re-validates before accepting (call last, then STOP)

## Rules
- Fix nulls first, then outliers
- Use the dataset_ref from previous tool output
- Don't over-clean — minor issues are OK
- `mark_cleaning_complete` will REJECT if critical issues remain — fix them first{protected_rule}

## CRITICAL: After `mark_cleaning_complete`
When you call `mark_cleaning_complete`, you MUST stop immediately:
- Do NOT output any summary of what you did
- Do NOT offer next steps or suggestions
- Do NOT write anything after the tool call
- Just call the tool and end your turn with no additional text"""


CLEANING_SYSTEM_PROMPT = _build_system_prompt()

# =============================================================================
# AGENT CREATION
# =============================================================================

def create_cleaning_agent(
    model: str = "openai:gpt-5.4",
    target_col: Optional[str] = None,
    task_type: Optional[str] = None,
    forbidden_columns: Optional[list[str]] = None,
    selected_model: Optional[str] = None,
):
    """
    Create the cleaning agent using LangChain's create_agent.
    
    Args:
        model: LLM identifier (default: openai:gpt-5.1)
        target_col: Target column name — protected from modification
        task_type: 'classification' or 'regression' for target-aware EDA
        forbidden_columns: Additional columns that must not be modified
        selected_model: ML model type (e.g. 'xgboost') — adjusts cleaning strategy
    
    Returns:
        Compiled agent graph
    """
    protected_columns: set[str] = set()
    if target_col:
        protected_columns.add(target_col)
    if forbidden_columns:
        protected_columns.update(forbidden_columns)

    has_context = bool(protected_columns or selected_model)

    if has_context:
        tests_tool = _build_run_clean_tests(target_col, task_type, protected_columns)
        complete_tool = _build_mark_cleaning_complete(protected_columns)
        tools = [
            tests_tool,
            complete_tool,
            *clean_tools,
            drop_columns_tool,
            cast_tool,
            dedupe_tool,
            filter_rows_tool,
        ]
        system_prompt = _build_system_prompt(target_col, protected_columns, selected_model)
    else:
        tools = CLEANING_TOOLS
        system_prompt = CLEANING_SYSTEM_PROMPT

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_cleaning_simple(
    dataset_ref: str,
    goal: str = "Clean the dataset for machine learning",
    model: str = "openai:gpt-5.4",
    max_iterations: int = 10,
    target_col: Optional[str] = None,
    task_type: Optional[str] = None,
    forbidden_columns: Optional[list[str]] = None,
    selected_model: Optional[str] = None,
):
    """
    Run the simple cleaning agent on a dataset.
    
    Args:
        dataset_ref: Reference to the dataset to clean
        goal: Description of what the dataset will be used for
        model: LLM to use for the agent
        max_iterations: Maximum number of agent steps
        target_col: Target column name — will not be modified during cleaning
        task_type: 'classification' or 'regression' for target-aware EDA
        forbidden_columns: Additional columns that must not be modified
        selected_model: ML model type (e.g. 'xgboost') — adjusts cleaning strategy
    
    Returns:
        Dict with:
        - cleaned_ref: The cleaned dataset reference (or None if not complete)
        - messages: All agent messages for debugging
    """
    # --- Pre-validation gate ---
    try:
        df = resolve_dataset(dataset_ref)
        if len(df) < _MIN_ROWS_DEFAULT:
            print(f"[cleaning_simple] WARNING: Dataset has only {len(df)} rows (minimum: {_MIN_ROWS_DEFAULT})")
        if len(df.columns) < 2:
            print(f"[cleaning_simple] ERROR: Dataset has only {len(df.columns)} column(s) — cannot clean")
            return {
                "cleaned_ref": dataset_ref,
                "messages": [],
                "cleaning_summary": f"Skipped: dataset has only {len(df.columns)} column(s)",
                "cleaning_reason": None,
                "transformations": [],
            }
        if target_col and target_col not in df.columns:
            print(f"[cleaning_simple] WARNING: target_col '{target_col}' not found in dataset columns")
            target_col = None
            task_type = None
    except Exception as e:
        print(f"[cleaning_simple] WARNING: Could not pre-validate dataset: {e}")

    agent = create_cleaning_agent(
        model=model,
        target_col=target_col,
        task_type=task_type,
        forbidden_columns=forbidden_columns,
        selected_model=selected_model,
    )
    
    # Build a context-rich initial message
    context_parts = [f"**Goal:** {goal}", f"**Dataset:** `{dataset_ref}`"]
    if selected_model:
        context_parts.append(f"**Model to train:** {selected_model}")
    if target_col:
        context_parts.append(f"**Target column:** `{target_col}` ({task_type or 'unknown task type'})")
    context_block = "\n".join(context_parts)

    initial_message = f"""Clean this dataset for the following training task:

{context_block}

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
