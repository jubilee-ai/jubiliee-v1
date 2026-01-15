"""
Label and Split Definition for ML Training Agent.

Step 3.5: Defines the 6 key parameters for supervised learning:
1. Target column
2. Prediction horizon
3. Grain (what one row represents)
4. As-of cutoff column
5. Split strategy
6. Forbidden columns (not available at prediction time)

Takes optional user-provided values and uses LLM to infer missing ones
based on goal, schema, and model context.
"""

import json
import sys
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Add data-tools to path
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from utils import get_registered_dataset

# =============================================================================
# TYPES
# =============================================================================

SplitStrategy = Literal["random", "time_based", "entity_based"]


class LabelDefinition:
    """Label and split configuration."""
    target_column: str
    prediction_horizon: Optional[str]
    grain: str
    as_of_cutoff: Optional[str]
    split_strategy: SplitStrategy
    forbidden_columns: list[str]


# =============================================================================
# SCHEMA HELPER
# =============================================================================

def _get_schema_context(dataset_ref: str) -> str:
    """
    Build a schema summary for the LLM context.
    
    Returns a formatted string with column names, types, null percentages,
    and sample values to help the LLM understand the data structure.
    """
    df = get_registered_dataset(dataset_ref)
    if df is None:
        return f"Could not load dataset: {dataset_ref}"
    
    lines = []
    lines.append(f"**Dataset:** {dataset_ref}")
    lines.append(f"**Shape:** {len(df):,} rows × {len(df.columns)} columns")
    lines.append("")
    lines.append("**Columns:**")
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_pct = df[col].isna().mean()
        unique = df[col].nunique()
        
        # Get sample values (non-null, up to 3)
        samples = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(str(s)[:30] for s in samples)
        
        null_str = f" ({null_pct:.0%} null)" if null_pct > 0 else ""
        lines.append(f"- `{col}`: {dtype}{null_str}, {unique} unique, samples: [{sample_str}]")
    
    return "\n".join(lines)


# =============================================================================
# LLM CALL
# =============================================================================

SYSTEM_PROMPT = """You are an ML engineer helping define label and split configuration for a supervised learning task.

Given the goal, model type, and dataset schema, infer the 6 key parameters:

1. **target_column**: Which column is the prediction target? Look for binary flags, status columns, or outcome variables.

2. **prediction_horizon**: How far into the future are we predicting? (e.g., "30 days", "90 days", "at application time", null if not applicable)

3. **grain**: What does one row represent? (e.g., "one customer", "one policy", "one transaction", "one claim")

4. **as_of_cutoff**: Which column represents the observation/prediction timestamp? (null if no time component)

5. **split_strategy**: How should we split the data?
   - "random": Standard random split (for cross-sectional data)
   - "time_based": Split by time to avoid leakage (for time-series or temporal data)
   - "entity_based": Split by entity ID to avoid leakage (for panel/longitudinal data)

6. **forbidden_columns**: Which columns would NOT be available at prediction time? 
   - Include outcome-derived columns (e.g., final_status, total_claims, was_approved)
   - Include future-dated columns
   - Include post-event columns
   - DO NOT include the target column here (it's already separate)

Be conservative with forbidden_columns - only include columns that clearly leak future information.

Respond with ONLY a JSON object (no markdown, no explanation):
{
    "target_column": "column_name",
    "prediction_horizon": "30 days" or null,
    "grain": "one policy",
    "as_of_cutoff": "application_date" or null,
    "split_strategy": "random" or "time_based" or "entity_based",
    "forbidden_columns": ["col1", "col2"]
}"""

def _call_llm(
    goal: str,
    selected_model: Optional[str],
    model_explanation: Optional[str],
    schema_context: str,
    provided_values: dict,
    model: str = "gpt-5-mini",
) -> dict:
    """
    Call LLM to infer missing label/split definitions.
    
    Args:
        goal: Training goal from user
        selected_model: Model type selected in step 1
        model_explanation: Why that model was selected
        schema_context: Formatted schema string
        provided_values: User-provided values (may be partial)
        model: Model to use (default: gpt-5-mini)
    
    Returns:
        Complete label definition dict
    """
    llm = init_chat_model(model=model, temperature=0)
    
    # Build user message with all context
    user_lines = []
    user_lines.append("## Goal")
    user_lines.append(goal)
    user_lines.append("")
    
    if selected_model:
        user_lines.append("## Selected Model")
        user_lines.append(f"**Model:** {selected_model}")
        if model_explanation:
            user_lines.append(f"**Rationale:** {model_explanation}")
        user_lines.append("")
    
    user_lines.append("## Dataset Schema")
    user_lines.append(schema_context)
    user_lines.append("")
    
    # Include any user-provided values as constraints
    if any(v is not None for v in provided_values.values()):
        user_lines.append("## User-Provided Values (use these exactly)")
        for key, value in provided_values.items():
            if value is not None:
                user_lines.append(f"- **{key}**: {json.dumps(value)}")
        user_lines.append("")
    
    user_lines.append("Infer the complete label/split definition. Return only JSON.")
    
    user_message = "\n".join(user_lines)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # Parse JSON (handle potential markdown wrapping)
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()
    
    result = json.loads(content)
    
    # Override with user-provided values
    for key, value in provided_values.items():
        if value is not None:
            result[key] = value
    
    return result


# =============================================================================
# MAIN FUNCTION
# =============================================================================

# TODO: Don't call LLM if all values are provided
# Right now it validates, but we should have code that validates it without calling the LLM
def run_label_split_definition(
    dataset_ref: str,
    goal: str,
    selected_model: Optional[str] = None,
    model_explanation: Optional[str] = None,
    # Optional user-provided values for each of the 6
    target_column: Optional[str] = None,
    prediction_horizon: Optional[str] = None,
    grain: Optional[str] = None,
    as_of_cutoff: Optional[str] = None,
    split_strategy: Optional[SplitStrategy] = None,
    forbidden_columns: Optional[list[str]] = None,
    # LLM config
    model: str = "gpt-5-mini",
) -> dict:
    """
    Define label and split configuration for supervised learning.
    
    Takes optional user-provided values for any of the 6 parameters.
    Uses LLM to infer missing values based on goal, model, and schema context.
    
    Args:
        dataset_ref: Reference to the cleaned dataset
        goal: Training goal/objective
        selected_model: Model type selected in step 1
        model_explanation: Why that model was selected
        target_column: Optional - which column is the prediction target
        prediction_horizon: Optional - how far into the future (e.g., "30 days")
        grain: Optional - what one row represents (e.g., "one policy")
        as_of_cutoff: Optional - timestamp column for as-of logic
        split_strategy: Optional - "random", "time_based", or "entity_based"
        forbidden_columns: Optional - columns not available at prediction time
        model: OpenAI model to use for inference
    
    Returns:
        Dict with all 6 label/split definition fields
    """
    # Get schema context
    schema_context = _get_schema_context(dataset_ref)
    
    # Collect user-provided values (LLM will use these as constraints)
    provided_values = {
        "target_column": target_column,
        "prediction_horizon": prediction_horizon,
        "grain": grain,
        "as_of_cutoff": as_of_cutoff,
        "split_strategy": split_strategy,
        "forbidden_columns": forbidden_columns,
    }
    
    # Always call LLM to process and validate
    result = _call_llm(
        goal=goal,
        selected_model=selected_model,
        model_explanation=model_explanation,
        schema_context=schema_context,
        provided_values=provided_values,
        model=model,
    )
    print(f"LLM result: {result}")
    
    # Validate required fields
    if not result.get("target_column"):
        raise ValueError("LLM failed to infer target_column")
    if not result.get("grain"):
        raise ValueError("LLM failed to infer grain")
    if result.get("split_strategy") not in ("random", "time_based", "entity_based"):
        result["split_strategy"] = "random"  # Default fallback
    if result.get("forbidden_columns") is None:
        result["forbidden_columns"] = []
    
    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_label_split_definition",
    "LabelDefinition",
    "SplitStrategy",
]
