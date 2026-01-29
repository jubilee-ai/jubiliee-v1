"""
Feature Engineering Executor - Step 5 of the training pipeline.

DETERMINISTIC executor (no LLM reasoning loop):
1. Parse the feature_spec from step 4
2. For each feature in spec, call the appropriate transformation
3. Apply transformations (respecting as-of dates to avoid leakage)
4. Validate the result
5. Return transformed dataset

Failures are spec failures, not execution failures.
"""

import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Add data-tools to path
# Path: steps -> training -> agents -> root -> tools/data-tools
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from transformations.tool_utils import resolve_dataset, save_result
from utils import register_dataset

# =============================================================================
# OPERATION EXECUTORS
# =============================================================================

def _execute_passthrough(df: pd.DataFrame, formula: dict, feature_name: str) -> pd.DataFrame:
    """Keep column as-is, optionally rename."""
    column = formula["column"]
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found for passthrough")
    
    if column != feature_name:
        df = df.copy()
        df[feature_name] = df[column]
    return df


def _execute_expression(df: pd.DataFrame, formula: dict, feature_name: str) -> pd.DataFrame:
    """Compute expression and add as new column."""
    expression = formula["expression"]
    source_columns = formula.get("source_columns", [])
    
    # Validate source columns exist
    for col in source_columns:
        if col not in df.columns:
            raise ValueError(f"Source column '{col}' not found for expression")
    
    df = df.copy()
    try:
        # Use pandas eval with local column references
        df[feature_name] = df.eval(expression)
    except Exception:
        # Fallback: try with explicit column references
        import numpy as np
        local_vars = {col: df[col] for col in df.columns}
        local_vars["log"] = np.log
        local_vars["sqrt"] = np.sqrt
        local_vars["abs"] = np.abs
        df[feature_name] = eval(expression, {"__builtins__": {}}, local_vars)
    
    return df


def _execute_bin(df: pd.DataFrame, formula: dict, feature_name: str) -> pd.DataFrame:
    """Discretize numeric column into bins."""
    column = formula["column"]
    bins = formula["bins"]
    strategy = formula.get("strategy", "quantile")
    labels = formula.get("labels")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found for binning")
    
    df = df.copy()
    
    if isinstance(bins, int):
        # Number of bins
        if strategy == "quantile":
            df[feature_name] = pd.qcut(df[column], q=bins, labels=labels, duplicates="drop")
        else:
            df[feature_name] = pd.cut(df[column], bins=bins, labels=labels)
    else:
        # Custom bin edges
        df[feature_name] = pd.cut(df[column], bins=bins, labels=labels)
    
    return df


def _execute_one_hot(df: pd.DataFrame, formula: dict, feature_name: str) -> pd.DataFrame:
    """One-hot encode categorical column."""
    column = formula["column"]
    drop_first = formula.get("drop_first", True)
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found for one-hot encoding")
    
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=feature_name, drop_first=drop_first)
    df = pd.concat([df, dummies], axis=1)
    
    return df


def _execute_ordinal(df: pd.DataFrame, formula: dict, feature_name: str) -> pd.DataFrame:
    """Ordinal encode categorical column with specified order."""
    column = formula["column"]
    order = formula["order"]
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found for ordinal encoding")
    
    df = df.copy()
    # Create mapping from category to ordinal value
    mapping = {cat: i for i, cat in enumerate(order)}
    df[feature_name] = df[column].map(mapping)
    
    return df


def _execute_group_agg(df: pd.DataFrame, formula: dict, feature_name: str) -> pd.DataFrame:
    """Compute aggregation by group and merge back."""
    column = formula["column"]
    agg = formula["agg"]
    group_by = formula["group_by"]
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found for group aggregation")
    for col in group_by:
        if col not in df.columns:
            raise ValueError(f"Group column '{col}' not found")
    
    df = df.copy()
    agg_df = df.groupby(group_by)[column].agg(agg).reset_index()
    agg_df = agg_df.rename(columns={column: feature_name})
    df = df.merge(agg_df, on=group_by, how="left")
    
    return df


def _execute_rolling(df: pd.DataFrame, formula: dict, feature_name: str) -> pd.DataFrame:
    """Compute rolling window aggregation."""
    column = formula["column"]
    agg = formula["agg"]
    window = formula["window"]
    order_by = formula["order_by"]
    partition_by = formula.get("partition_by")
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found for rolling")
    if order_by not in df.columns:
        raise ValueError(f"Order column '{order_by}' not found")
    
    df = df.copy()
    df = df.sort_values(order_by)
    
    agg_map = {"mean": "mean", "sum": "sum", "min": "min", "max": "max", "std": "std"}
    
    if partition_by:
        # Grouped rolling
        df[feature_name] = df.groupby(partition_by)[column].transform(
            lambda x: getattr(x.rolling(window, min_periods=1), agg_map[agg])()
        )
    else:
        # Simple rolling
        df[feature_name] = getattr(df[column].rolling(window, min_periods=1), agg_map[agg])()
    
    return df


def _execute_date_extract(df: pd.DataFrame, formula: dict, feature_name: str) -> pd.DataFrame:
    """Extract datetime component."""
    column = formula["column"]
    part = formula["part"]
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found for date extract")
    
    df = df.copy()
    dt_col = pd.to_datetime(df[column])
    
    part_map = {
        "year": lambda dt: dt.dt.year,
        "month": lambda dt: dt.dt.month,
        "day": lambda dt: dt.dt.day,
        "dayofweek": lambda dt: dt.dt.dayofweek,
        "hour": lambda dt: dt.dt.hour,
        "quarter": lambda dt: dt.dt.quarter,
        "weekofyear": lambda dt: dt.dt.isocalendar().week,
    }
    
    if part not in part_map:
        raise ValueError(f"Unknown date part '{part}'")
    
    df[feature_name] = part_map[part](dt_col)
    
    return df


def _execute_date_diff(df: pd.DataFrame, formula: dict, feature_name: str) -> pd.DataFrame:
    """Compute difference between two date columns."""
    start_column = formula["start_column"]
    end_column = formula["end_column"]
    unit = formula.get("unit", "days")
    
    if start_column not in df.columns:
        raise ValueError(f"Start column '{start_column}' not found")
    if end_column not in df.columns:
        raise ValueError(f"End column '{end_column}' not found")
    
    df = df.copy()
    start_dt = pd.to_datetime(df[start_column])
    end_dt = pd.to_datetime(df[end_column])
    
    diff = end_dt - start_dt
    
    if unit == "days":
        df[feature_name] = diff.dt.days
    elif unit == "hours":
        df[feature_name] = diff.dt.total_seconds() / 3600
    elif unit == "months":
        df[feature_name] = diff.dt.days / 30.44  # Average month length
    elif unit == "years":
        df[feature_name] = diff.dt.days / 365.25
    else:
        df[feature_name] = diff.dt.days
    
    return df


# Operation dispatcher
OPERATION_EXECUTORS = {
    "passthrough": _execute_passthrough,
    "expression": _execute_expression,
    "bin": _execute_bin,
    "one_hot": _execute_one_hot,
    "ordinal": _execute_ordinal,
    "group_agg": _execute_group_agg,
    "rolling": _execute_rolling,
    "date_extract": _execute_date_extract,
    "date_diff": _execute_date_diff,
}


# =============================================================================
# MAIN EXECUTOR
# =============================================================================

def _apply_as_of_constraint(
    df: pd.DataFrame,
    feature_name: str,
    as_of_constraint: Optional[dict],
    as_of_cutoff: Optional[str],
) -> pd.DataFrame:
    """
    Apply temporal constraint to prevent data leakage.
    
    For rows where the source_date_column >= as_of_cutoff,
    set the feature value to NaN (it wouldn't be known at prediction time).
    """
    if not as_of_constraint or not as_of_cutoff:
        return df
    
    source_col = as_of_constraint.get("source_date_column")
    operator = as_of_constraint.get("operator", "<")
    
    if not source_col or source_col not in df.columns:
        return df
    
    df = df.copy()
    source_dt = pd.to_datetime(df[source_col])
    cutoff_dt = pd.to_datetime(as_of_cutoff)
    
    # Mask values that violate the constraint (would be future data)
    if operator == "<":
        future_mask = source_dt >= cutoff_dt
    else:  # "<="
        future_mask = source_dt > cutoff_dt
    
    if future_mask.any() and feature_name in df.columns:
        df.loc[future_mask, feature_name] = None
    
    return df


def execute_feature_spec(
    dataset_ref: str,
    feature_spec: dict,
    target_column: str,
    grain: str,
    as_of_cutoff: Optional[str] = None,
) -> dict[str, Any]:
    """
    Execute a feature specification to transform a dataset.
    
    Args:
        dataset_ref: Reference to the source dataset
        feature_spec: The feature specification from step 4
        target_column: Target column to keep
        grain: Grain column(s) to keep
        as_of_cutoff: Optional cutoff date for temporal constraints
    
    Returns:
        Dict with:
        - transformed_dataset_ref: Reference to the transformed dataset
        - features_created: List of features successfully created
        - errors: List of any errors encountered
        - temporal_constraints_applied: Number of features with as_of constraints
    """
    df = resolve_dataset(dataset_ref)
    features = feature_spec.get("features", [])
    
    features_created = []
    errors = []
    temporal_constraints_applied = 0
    
    # Track columns to keep at the end
    columns_to_keep = set()
    
    # Always keep grain and target
    if isinstance(grain, str):
        columns_to_keep.add(grain)
    elif isinstance(grain, list):
        columns_to_keep.update(grain)
    columns_to_keep.add(target_column)
    
    # Execute each feature
    for feature in features:
        feature_name = feature.get("name", "unknown")
        formula = feature.get("formula", {})
        op = formula.get("op")
        as_of_constraint = feature.get("as_of_constraint")
        
        if op not in OPERATION_EXECUTORS:
            errors.append(f"Unknown operation '{op}' for feature '{feature_name}'")
            continue
        
        try:
            executor = OPERATION_EXECUTORS[op]
            df = executor(df, formula, feature_name)
            
            # Apply temporal constraint if present
            if as_of_constraint:
                df = _apply_as_of_constraint(df, feature_name, as_of_constraint, as_of_cutoff)
                temporal_constraints_applied += 1
            
            # Track created columns
            if op == "one_hot":
                # One-hot creates multiple columns
                prefix = feature_name
                new_cols = [c for c in df.columns if c.startswith(prefix + "_")]
                columns_to_keep.update(new_cols)
                features_created.extend(new_cols)
            else:
                columns_to_keep.add(feature_name)
                features_created.append(feature_name)
                
        except Exception as e:
            errors.append(f"Feature '{feature_name}' ({op}): {str(e)}")
    
    # Select only the columns we want to keep
    final_columns = [c for c in df.columns if c in columns_to_keep]
    df_final = df[final_columns]
    
    # Register the transformed dataset
    output_ref = f"{dataset_ref}_features"
    register_dataset(output_ref, df_final)
    
    return {
        "transformed_dataset_ref": output_ref,
        "features_created": features_created,
        "errors": errors,
        "shape": df_final.shape,
        "temporal_constraints_applied": temporal_constraints_applied,
    }


# =============================================================================
# SPLIT-AWARE EXECUTOR (FIT ON TRAIN, TRANSFORM ALL)
# =============================================================================

def _fit_and_transform_bin(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
    formula: dict,
    feature_name: str,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fit bin edges on training data, apply to all sets.
    """
    column = formula["column"]
    bins = formula["bins"]
    strategy = formula.get("strategy", "quantile")
    labels = formula.get("labels")
    
    train_df = train_df.copy()
    
    if isinstance(bins, int):
        if strategy == "quantile":
            # Compute quantile edges from training data
            _, bin_edges = pd.qcut(train_df[column], q=bins, retbins=True, duplicates="drop")
        else:
            _, bin_edges = pd.cut(train_df[column], bins=bins, retbins=True)
        
        # Extend edges to handle values outside training range
        bin_edges[0] = float('-inf')
        bin_edges[-1] = float('inf')
        
        # Apply same edges to all sets
        actual_labels = labels if labels else False
        train_df[feature_name] = pd.cut(train_df[column], bins=bin_edges, labels=actual_labels)
        
        if val_df is not None:
            val_df = val_df.copy()
            val_df[feature_name] = pd.cut(val_df[column], bins=bin_edges, labels=actual_labels)
        
        if test_df is not None:
            test_df = test_df.copy()
            test_df[feature_name] = pd.cut(test_df[column], bins=bin_edges, labels=actual_labels)
    else:
        # Custom bin edges provided - apply directly
        train_df[feature_name] = pd.cut(train_df[column], bins=bins, labels=labels)
        if val_df is not None:
            val_df = val_df.copy()
            val_df[feature_name] = pd.cut(val_df[column], bins=bins, labels=labels)
        if test_df is not None:
            test_df = test_df.copy()
            test_df[feature_name] = pd.cut(test_df[column], bins=bins, labels=labels)
    
    return train_df, val_df, test_df


def _fit_and_transform_one_hot(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
    formula: dict,
    feature_name: str,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], list[str]]:
    """
    Fit one-hot encoding on training data (get categories), apply to all sets.
    Returns the list of created column names.
    """
    column = formula["column"]
    drop_first = formula.get("drop_first", True)
    
    # Get categories from training data
    train_categories = train_df[column].unique().tolist()
    
    train_df = train_df.copy()
    train_dummies = pd.get_dummies(train_df[column], prefix=feature_name, drop_first=drop_first)
    created_columns = train_dummies.columns.tolist()
    train_df = pd.concat([train_df, train_dummies], axis=1)
    
    # Apply same categories to val/test (handle unseen categories)
    if val_df is not None:
        val_df = val_df.copy()
        val_dummies = pd.get_dummies(val_df[column], prefix=feature_name, drop_first=drop_first)
        # Add missing columns with zeros
        for col in created_columns:
            if col not in val_dummies.columns:
                val_dummies[col] = 0
        # Keep only columns from training (drop extra categories)
        val_dummies = val_dummies[[c for c in created_columns if c in val_dummies.columns]]
        val_df = pd.concat([val_df, val_dummies[created_columns]], axis=1)
    
    if test_df is not None:
        test_df = test_df.copy()
        test_dummies = pd.get_dummies(test_df[column], prefix=feature_name, drop_first=drop_first)
        for col in created_columns:
            if col not in test_dummies.columns:
                test_dummies[col] = 0
        test_dummies = test_dummies[[c for c in created_columns if c in test_dummies.columns]]
        test_df = pd.concat([test_df, test_dummies[created_columns]], axis=1)
    
    return train_df, val_df, test_df, created_columns


def _fit_and_transform_group_agg(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: Optional[pd.DataFrame],
    formula: dict,
    feature_name: str,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Compute aggregation on training data, merge to all sets.
    """
    column = formula["column"]
    agg = formula["agg"]
    group_by = formula["group_by"]
    
    # Compute aggregation from training data only
    agg_df = train_df.groupby(group_by)[column].agg(agg).reset_index()
    agg_df = agg_df.rename(columns={column: feature_name})
    
    # Merge to all sets
    train_df = train_df.copy()
    train_df = train_df.merge(agg_df, on=group_by, how="left")
    
    if val_df is not None:
        val_df = val_df.copy()
        val_df = val_df.merge(agg_df, on=group_by, how="left")
    
    if test_df is not None:
        test_df = test_df.copy()
        test_df = test_df.merge(agg_df, on=group_by, how="left")
    
    return train_df, val_df, test_df


def execute_feature_spec_split(
    train_ref: str,
    val_ref: Optional[str],
    test_ref: Optional[str],
    feature_spec: dict,
    target_column: str,
    grain: str,
    as_of_cutoff: Optional[str] = None,
) -> dict[str, Any]:
    """
    Execute a feature specification on train/val/test sets.
    
    IMPORTANT: Transformations that require fitting (binning, one-hot, group_agg)
    are fit on training data and applied to all sets to prevent data leakage.
    
    Args:
        train_ref: Reference to the training dataset
        val_ref: Reference to the validation dataset (optional)
        test_ref: Reference to the test dataset (optional)
        feature_spec: The feature specification from step 4
        target_column: Target column to keep
        grain: Grain column(s) to keep
        as_of_cutoff: Optional cutoff date for temporal constraints
    
    Returns:
        Dict with:
        - train_ref: Reference to transformed training dataset
        - val_ref: Reference to transformed validation dataset (if provided)
        - test_ref: Reference to transformed test dataset (if provided)
        - features_created: List of features successfully created
        - errors: List of any errors encountered
    """
    train_df = resolve_dataset(train_ref)
    val_df = resolve_dataset(val_ref) if val_ref else None
    test_df = resolve_dataset(test_ref) if test_ref else None
    
    features = feature_spec.get("features", [])
    
    features_created = []
    errors = []
    temporal_constraints_applied = 0
    
    # Track columns to keep at the end
    columns_to_keep = set()
    
    # Always keep grain and target
    if isinstance(grain, str):
        columns_to_keep.add(grain)
    elif isinstance(grain, list):
        columns_to_keep.update(grain)
    columns_to_keep.add(target_column)
    
    # Execute each feature
    for feature in features:
        feature_name = feature.get("name", "unknown")
        formula = feature.get("formula", {})
        op = formula.get("op")
        as_of_constraint = feature.get("as_of_constraint")
        
        try:
            # Operations that need fit-on-train logic
            if op == "bin":
                train_df, val_df, test_df = _fit_and_transform_bin(
                    train_df, val_df, test_df, formula, feature_name
                )
                columns_to_keep.add(feature_name)
                features_created.append(feature_name)
            
            elif op == "one_hot":
                train_df, val_df, test_df, created_cols = _fit_and_transform_one_hot(
                    train_df, val_df, test_df, formula, feature_name
                )
                columns_to_keep.update(created_cols)
                features_created.extend(created_cols)
            
            elif op == "group_agg":
                train_df, val_df, test_df = _fit_and_transform_group_agg(
                    train_df, val_df, test_df, formula, feature_name
                )
                columns_to_keep.add(feature_name)
                features_created.append(feature_name)
            
            # Operations that don't need fitting (apply same to all)
            elif op in OPERATION_EXECUTORS:
                executor = OPERATION_EXECUTORS[op]
                train_df = executor(train_df, formula, feature_name)
                if val_df is not None:
                    val_df = executor(val_df, formula, feature_name)
                if test_df is not None:
                    test_df = executor(test_df, formula, feature_name)
                columns_to_keep.add(feature_name)
                features_created.append(feature_name)
            
            else:
                errors.append(f"Unknown operation '{op}' for feature '{feature_name}'")
                continue
            
            # Apply temporal constraint if present (to all sets)
            if as_of_constraint:
                train_df = _apply_as_of_constraint(train_df, feature_name, as_of_constraint, as_of_cutoff)
                if val_df is not None:
                    val_df = _apply_as_of_constraint(val_df, feature_name, as_of_constraint, as_of_cutoff)
                if test_df is not None:
                    test_df = _apply_as_of_constraint(test_df, feature_name, as_of_constraint, as_of_cutoff)
                temporal_constraints_applied += 1
                
        except Exception as e:
            errors.append(f"Feature '{feature_name}' ({op}): {str(e)}")
    
    # Select only the columns we want to keep
    final_columns_train = [c for c in train_df.columns if c in columns_to_keep]
    train_df_final = train_df[final_columns_train]
    
    # Register transformed datasets
    output_train_ref = f"{train_ref}_features"
    register_dataset(output_train_ref, train_df_final)
    
    output_val_ref = None
    output_test_ref = None
    
    if val_df is not None:
        final_columns_val = [c for c in val_df.columns if c in columns_to_keep]
        val_df_final = val_df[final_columns_val]
        output_val_ref = f"{val_ref}_features"
        register_dataset(output_val_ref, val_df_final)
    
    if test_df is not None:
        final_columns_test = [c for c in test_df.columns if c in columns_to_keep]
        test_df_final = test_df[final_columns_test]
        output_test_ref = f"{test_ref}_features"
        register_dataset(output_test_ref, test_df_final)
    
    return {
        "train_ref": output_train_ref,
        "val_ref": output_val_ref,
        "test_ref": output_test_ref,
        "features_created": features_created,
        "errors": errors,
        "shapes": {
            "train": train_df_final.shape,
            "val": val_df_final.shape if val_df is not None else None,
            "test": test_df_final.shape if test_df is not None else None,
        },
        "temporal_constraints_applied": temporal_constraints_applied,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "execute_feature_spec",
    "execute_feature_spec_split",
]
