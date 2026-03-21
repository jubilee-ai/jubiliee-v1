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

Also provides functions to compute and apply the actual train/val/test split
based on the defined strategy.
"""

import json
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from sklearn.model_selection import train_test_split

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add data-tools to path
# Path: steps -> training -> agents -> root -> tools/data-tools
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
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
    model: str = "gpt-5.1",
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
# TARGET TRANSFORM
# =============================================================================


def _detect_target_transform(
    df: pd.DataFrame,
    target_column: str,
    goal: str,
) -> Optional[str]:
    """Recommend a target transform for skewed regression targets.

    Returns ``"log1p"`` when the target is continuous, positive, and highly
    skewed — which is common for price/cost/amount variables.  Returns
    ``None`` otherwise.
    """
    goal_lower = goal.lower()
    classification_hints = [
        "classif", "churn", "fraud", "default", "spam", "diagnos",
        "detect", "binary", "multi-class", "category", "sentiment",
    ]
    if any(h in goal_lower for h in classification_hints):
        return None

    if target_column not in df.columns:
        return None

    target = df[target_column]
    if not np.issubdtype(target.dtype, np.number):
        return None

    target = target.dropna()
    if len(target) < 30:
        return None

    # Few unique values → classification target, not a regression candidate
    if target.nunique() <= 20:
        return None

    skew = float(target.skew())
    if target.min() >= 0 and abs(skew) > 1.0:
        return "log1p"

    return None


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
    model: str = "gpt-5.1",
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
    if not dataset_ref:
        raise ValueError("dataset_ref is required but got None — cleaning must run first to produce a cleaned dataset.")

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

    # Detect target transform for skewed regression targets
    df = get_registered_dataset(dataset_ref)
    if df is not None:
        transform = _detect_target_transform(df, result["target_column"], goal)
        if transform:
            print(f"[label_split] Detected skewed target — will apply '{transform}' transform")
        result["target_transform"] = transform
    else:
        result["target_transform"] = None

    return result


# =============================================================================
# SPLITTING FUNCTIONS
# =============================================================================

def compute_split_indices(
    df: pd.DataFrame,
    label_definition: dict,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> dict:
    """
    Compute train/val/test indices based on the split strategy.
    
    Args:
        df: The DataFrame to split
        label_definition: Output from run_label_split_definition containing:
            - split_strategy: "random", "time_based", or "entity_based"
            - as_of_cutoff: Column name for time-based splits
            - grain: Column name for entity-based splits
        train_ratio: Proportion for training set (default 0.7)
        val_ratio: Proportion for validation set (default 0.15)
        test_ratio: Proportion for test set (default 0.15)
        random_state: Random seed for reproducibility
    
    Returns:
        Dict with:
            - train_idx: array of training indices
            - val_idx: array of validation indices
            - test_idx: array of test indices
            - split_strategy: the strategy used
            - ratios: {"train": 0.7, "val": 0.15, "test": 0.15}
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    strategy = label_definition.get("split_strategy", "random")
    n = len(df)
    indices = np.arange(n)
    
    if strategy == "random":
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=random_state,
        )
        # Split remaining into val and test
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size_adjusted,
            random_state=random_state,
        )
    
    elif strategy == "time_based":
        time_col = label_definition.get("as_of_cutoff")
        if not time_col:
            raise ValueError("time_based split requires 'as_of_cutoff' column")
        if time_col not in df.columns:
            raise ValueError(f"as_of_cutoff column '{time_col}' not found in DataFrame")
        
        # Sort by time and split by position
        sorted_indices = df[time_col].sort_values().index.to_numpy()
        # Map to positional indices
        sorted_positions = np.array([df.index.get_loc(i) for i in sorted_indices])
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_idx = sorted_positions[:train_end]
        val_idx = sorted_positions[train_end:val_end]
        test_idx = sorted_positions[val_end:]
    
    elif strategy == "entity_based":
        grain = label_definition.get("grain")
        if not grain:
            raise ValueError("entity_based split requires 'grain' to identify entity column")
        
        # Try to find an entity column - grain is descriptive ("one policy"), 
        # so we need to infer or require an entity_column field
        # For now, look for common ID patterns or use first column with "id" in name
        entity_col = _find_entity_column(df, grain)
        if not entity_col:
            raise ValueError(
                f"Could not find entity column for grain='{grain}'. "
                "Consider adding 'entity_column' to label_definition."
            )
        
        # Get unique entities and split them
        unique_entities = df[entity_col].unique()
        np.random.seed(random_state)
        np.random.shuffle(unique_entities)
        
        n_entities = len(unique_entities)
        train_end = int(n_entities * train_ratio)
        val_end = int(n_entities * (train_ratio + val_ratio))
        
        train_entities = set(unique_entities[:train_end])
        val_entities = set(unique_entities[train_end:val_end])
        test_entities = set(unique_entities[val_end:])
        
        train_idx = df.index[df[entity_col].isin(train_entities)].to_numpy()
        val_idx = df.index[df[entity_col].isin(val_entities)].to_numpy()
        test_idx = df.index[df[entity_col].isin(test_entities)].to_numpy()
        
        # Convert to positional indices if needed
        if not isinstance(df.index, pd.RangeIndex):
            train_idx = np.array([df.index.get_loc(i) for i in train_idx])
            val_idx = np.array([df.index.get_loc(i) for i in val_idx])
            test_idx = np.array([df.index.get_loc(i) for i in test_idx])
    
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    
    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "split_strategy": strategy,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
    }


def _find_entity_column(df: pd.DataFrame, grain: str) -> Optional[str]:
    """
    Try to find the entity column based on grain description.
    
    Looks for columns with 'id' in the name, or common patterns like
    'customer_id', 'policy_id', 'user_id', etc.
    """
    grain_lower = grain.lower()
    
    # Extract entity type from grain (e.g., "one policy" -> "policy")
    entity_keywords = []
    for word in ["customer", "policy", "user", "account", "loan", "claim", "transaction", "order"]:
        if word in grain_lower:
            entity_keywords.append(word)
    
    # Look for columns matching entity + id pattern
    for col in df.columns:
        col_lower = col.lower()
        for keyword in entity_keywords:
            if keyword in col_lower and "id" in col_lower:
                return col
    
    # Fallback: look for any column ending in '_id' or 'id'
    for col in df.columns:
        col_lower = col.lower()
        if col_lower.endswith("_id") or col_lower == "id":
            return col
    
    return None


def apply_split(
    df: pd.DataFrame,
    split_indices: dict,
    target_column: Optional[str] = None,
    target_transform: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply precomputed split indices to get train/val/test DataFrames.

    If *target_transform* is ``"log1p"``, the target column is transformed
    in-place on all three splits (fitting nothing — log1p is stateless).
    """
    train_df = df.iloc[split_indices["train_idx"]].copy()
    val_df = df.iloc[split_indices["val_idx"]].copy()
    test_df = df.iloc[split_indices["test_idx"]].copy()

    if target_transform == "log1p" and target_column and target_column in df.columns:
        for split_df in [train_df, val_df, test_df]:
            split_df[target_column] = np.log1p(split_df[target_column])
        print(f"[label_split] Applied log1p transform to '{target_column}'")

    return train_df, val_df, test_df


def add_split_column(
    df: pd.DataFrame,
    split_indices: dict,
    column_name: str = "_split",
) -> pd.DataFrame:
    """
    Add a split label column to the DataFrame instead of physically splitting.
    
    Useful if you want to keep data in one DataFrame but mark which rows
    belong to which split. Feature engineering can then filter by this column.
    
    Args:
        df: The DataFrame to label
        split_indices: Output from compute_split_indices
        column_name: Name for the split column (default "_split")
    
    Returns:
        DataFrame with new column containing "train", "val", or "test"
    """
    df = df.copy()
    df[column_name] = None
    df.iloc[split_indices["train_idx"], df.columns.get_loc(column_name)] = "train"
    df.iloc[split_indices["val_idx"], df.columns.get_loc(column_name)] = "val"
    df.iloc[split_indices["test_idx"], df.columns.get_loc(column_name)] = "test"
    
    return df


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_label_split_definition",
    "compute_split_indices",
    "apply_split",
    "add_split_column",
    "LabelDefinition",
    "SplitStrategy",
]
