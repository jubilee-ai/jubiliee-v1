"""
Tests for the Feature Engineering Executor (Step 5).

Tests each operation type and validates the deterministic execution.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from agents.training.feature_engineering_executor import execute_feature_spec
from utils import clear_registry, get_registered_dataset, register_dataset


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the dataset registry before each test."""
    clear_registry()
    yield
    clear_registry()


# =============================================================================
# TEST 1: Passthrough Operation
# =============================================================================

def test_passthrough_operation():
    """Test that passthrough keeps column as-is."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "age": [25, 35, 45],
        "income": [50000, 75000, 100000],
        "target": [0, 1, 0],
    })
    register_dataset("test_passthrough", df)
    
    feature_spec = {
        "features": [
            {"name": "age", "formula": {"op": "passthrough", "column": "age"}, "grain": "id"},
            {"name": "income", "formula": {"op": "passthrough", "column": "income"}, "grain": "id"},
        ],
        "reasoning": "Keep numeric features as-is",
        "excluded_columns": [],
    }
    
    result = execute_feature_spec(
        dataset_ref="test_passthrough",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    assert result["errors"] == []
    assert set(result["features_created"]) == {"age", "income"}
    
    df_out = get_registered_dataset(result["transformed_dataset_ref"])
    assert list(df_out["age"]) == [25, 35, 45]
    assert list(df_out["income"]) == [50000, 75000, 100000]
    
    print("\n=== TEST 1: Passthrough ===")
    print(f"Input: {df.shape}, Output: {df_out.shape}")
    print(f"Features: {result['features_created']}")
    print(df_out.to_string())


# =============================================================================
# TEST 2: Expression Operation
# =============================================================================

def test_expression_operation():
    """Test computed expressions."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "loan_amount": [10000, 20000, 30000],
        "income": [50000, 40000, 60000],
        "target": [0, 1, 0],
    })
    register_dataset("test_expression", df)
    
    feature_spec = {
        "features": [
            {
                "name": "loan_to_income",
                "formula": {
                    "op": "expression",
                    "expression": "loan_amount / income",
                    "source_columns": ["loan_amount", "income"],
                },
                "grain": "id",
            },
        ],
        "reasoning": "Compute loan-to-income ratio",
        "excluded_columns": [],
    }
    
    result = execute_feature_spec(
        dataset_ref="test_expression",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    assert result["errors"] == []
    assert "loan_to_income" in result["features_created"]
    
    df_out = get_registered_dataset(result["transformed_dataset_ref"])
    expected = [10000/50000, 20000/40000, 30000/60000]
    np.testing.assert_array_almost_equal(df_out["loan_to_income"].values, expected)
    
    print("\n=== TEST 2: Expression ===")
    print(f"Expression: loan_amount / income")
    print(f"Expected: {expected}")
    print(f"Got: {list(df_out['loan_to_income'])}")


# =============================================================================
# TEST 3: Binning Operation
# =============================================================================

def test_bin_operation():
    """Test binning continuous variables."""
    df = pd.DataFrame({
        "id": range(1, 11),
        "age": [22, 28, 35, 42, 48, 55, 62, 38, 31, 45],
        "target": [0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
    })
    register_dataset("test_bin", df)
    
    feature_spec = {
        "features": [
            {
                "name": "age_group",
                "formula": {
                    "op": "bin",
                    "column": "age",
                    "bins": 3,
                    "strategy": "quantile",
                },
                "grain": "id",
            },
        ],
        "reasoning": "Bin age into 3 quantile groups",
        "excluded_columns": [],
    }
    
    result = execute_feature_spec(
        dataset_ref="test_bin",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    assert result["errors"] == []
    assert "age_group" in result["features_created"]
    
    df_out = get_registered_dataset(result["transformed_dataset_ref"])
    # Should have 3 distinct bins
    assert df_out["age_group"].nunique() <= 3
    
    print("\n=== TEST 3: Binning ===")
    print(f"Input ages: {list(df['age'])}")
    print(f"Binned: {list(df_out['age_group'])}")
    print(f"Unique bins: {df_out['age_group'].nunique()}")


# =============================================================================
# TEST 4: One-Hot Encoding
# =============================================================================

def test_one_hot_encoding():
    """Test one-hot encoding categorical variables."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "region": ["north", "south", "east", "west"],
        "target": [0, 1, 0, 1],
    })
    register_dataset("test_one_hot", df)
    
    feature_spec = {
        "features": [
            {
                "name": "region_ohe",
                "formula": {
                    "op": "one_hot",
                    "column": "region",
                    "drop_first": True,
                },
                "grain": "id",
            },
        ],
        "reasoning": "One-hot encode region",
        "excluded_columns": [],
    }
    
    result = execute_feature_spec(
        dataset_ref="test_one_hot",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    assert result["errors"] == []
    # Should create 3 columns (4 categories - 1 for drop_first)
    assert len(result["features_created"]) == 3
    
    df_out = get_registered_dataset(result["transformed_dataset_ref"])
    
    print("\n=== TEST 4: One-Hot Encoding ===")
    print(f"Input categories: {list(df['region'])}")
    print(f"Created columns: {result['features_created']}")
    print(df_out.to_string())


# =============================================================================
# TEST 5: Ordinal Encoding
# =============================================================================

def test_ordinal_encoding():
    """Test ordinal encoding with specified order."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "education": ["high_school", "masters", "bachelors", "phd"],
        "target": [0, 1, 0, 1],
    })
    register_dataset("test_ordinal", df)
    
    feature_spec = {
        "features": [
            {
                "name": "education_level",
                "formula": {
                    "op": "ordinal",
                    "column": "education",
                    "order": ["high_school", "bachelors", "masters", "phd"],
                },
                "grain": "id",
            },
        ],
        "reasoning": "Ordinal encode education",
        "excluded_columns": [],
    }
    
    result = execute_feature_spec(
        dataset_ref="test_ordinal",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    assert result["errors"] == []
    assert "education_level" in result["features_created"]
    
    df_out = get_registered_dataset(result["transformed_dataset_ref"])
    # high_school=0, masters=2, bachelors=1, phd=3
    expected = [0, 2, 1, 3]
    assert list(df_out["education_level"]) == expected
    
    print("\n=== TEST 5: Ordinal Encoding ===")
    print(f"Input: {list(df['education'])}")
    print(f"Order: ['high_school', 'bachelors', 'masters', 'phd']")
    print(f"Expected: {expected}")
    print(f"Got: {list(df_out['education_level'])}")


# =============================================================================
# TEST 6: Group Aggregation
# =============================================================================

def test_group_agg_operation():
    """Test group aggregation features."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "customer_id": ["A", "A", "A", "B", "B", "B"],
        "amount": [100, 200, 300, 400, 500, 600],
        "target": [0, 0, 1, 0, 1, 1],
    })
    register_dataset("test_group_agg", df)
    
    feature_spec = {
        "features": [
            {
                "name": "customer_avg_amount",
                "formula": {
                    "op": "group_agg",
                    "column": "amount",
                    "agg": "mean",
                    "group_by": ["customer_id"],
                },
                "grain": "id",
            },
        ],
        "reasoning": "Average amount per customer",
        "excluded_columns": [],
    }
    
    result = execute_feature_spec(
        dataset_ref="test_group_agg",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    assert result["errors"] == []
    assert "customer_avg_amount" in result["features_created"]
    
    df_out = get_registered_dataset(result["transformed_dataset_ref"])
    # Customer A: mean(100,200,300) = 200
    # Customer B: mean(400,500,600) = 500
    expected = [200, 200, 200, 500, 500, 500]
    assert list(df_out["customer_avg_amount"]) == expected
    
    print("\n=== TEST 6: Group Aggregation ===")
    print(f"Input amounts: {list(df['amount'])}")
    print(f"Customer IDs: {list(df['customer_id'])}")
    print(f"Expected avg: {expected}")
    print(f"Got: {list(df_out['customer_avg_amount'])}")


# =============================================================================
# TEST 7: Rolling Window Aggregation
# =============================================================================

def test_rolling_operation():
    """Test rolling window aggregation."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "date": pd.date_range("2023-01-01", periods=5, freq="D"),
        "sales": [100, 150, 200, 180, 220],
        "target": [0, 0, 1, 0, 1],
    })
    register_dataset("test_rolling", df)
    
    feature_spec = {
        "features": [
            {
                "name": "rolling_avg_sales",
                "formula": {
                    "op": "rolling",
                    "column": "sales",
                    "agg": "mean",
                    "window": 3,
                    "order_by": "date",
                },
                "grain": "id",
            },
        ],
        "reasoning": "3-day rolling average of sales",
        "excluded_columns": [],
    }
    
    result = execute_feature_spec(
        dataset_ref="test_rolling",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    assert result["errors"] == []
    assert "rolling_avg_sales" in result["features_created"]
    
    df_out = get_registered_dataset(result["transformed_dataset_ref"])
    # Rolling mean with min_periods=1:
    # [100, 125, 150, 176.67, 200]
    rolling_values = df_out["rolling_avg_sales"].values
    assert rolling_values[0] == 100  # First value
    assert abs(rolling_values[2] - 150) < 1  # (100+150+200)/3
    
    print("\n=== TEST 7: Rolling Window ===")
    print(f"Input sales: {list(df['sales'])}")
    print(f"Rolling mean (window=3): {list(np.round(rolling_values, 2))}")


# =============================================================================
# TEST 8: Date Extraction
# =============================================================================

def test_date_extract_operation():
    """Test extracting date parts."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "order_date": pd.to_datetime(["2023-01-15", "2023-06-22", "2023-12-01", "2023-03-10"]),
        "target": [0, 1, 0, 1],
    })
    register_dataset("test_date_extract", df)
    
    feature_spec = {
        "features": [
            {
                "name": "order_month",
                "formula": {
                    "op": "date_extract",
                    "column": "order_date",
                    "part": "month",
                },
                "grain": "id",
            },
            {
                "name": "order_dayofweek",
                "formula": {
                    "op": "date_extract",
                    "column": "order_date",
                    "part": "dayofweek",
                },
                "grain": "id",
            },
        ],
        "reasoning": "Extract month and day of week",
        "excluded_columns": [],
    }
    
    result = execute_feature_spec(
        dataset_ref="test_date_extract",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    assert result["errors"] == []
    assert "order_month" in result["features_created"]
    assert "order_dayofweek" in result["features_created"]
    
    df_out = get_registered_dataset(result["transformed_dataset_ref"])
    expected_months = [1, 6, 12, 3]
    assert list(df_out["order_month"]) == expected_months
    
    print("\n=== TEST 8: Date Extraction ===")
    print(f"Input dates: {list(df['order_date'].dt.strftime('%Y-%m-%d'))}")
    print(f"Extracted months: {list(df_out['order_month'])}")
    print(f"Extracted dayofweek: {list(df_out['order_dayofweek'])}")


# =============================================================================
# TEST 9: Date Difference
# =============================================================================

def test_date_diff_operation():
    """Test computing date differences."""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "signup_date": pd.to_datetime(["2023-01-01", "2023-03-15", "2023-06-01"]),
        "purchase_date": pd.to_datetime(["2023-01-10", "2023-03-20", "2023-07-15"]),
        "target": [0, 1, 0],
    })
    register_dataset("test_date_diff", df)
    
    feature_spec = {
        "features": [
            {
                "name": "days_to_purchase",
                "formula": {
                    "op": "date_diff",
                    "start_column": "signup_date",
                    "end_column": "purchase_date",
                    "unit": "days",
                },
                "grain": "id",
            },
        ],
        "reasoning": "Days between signup and purchase",
        "excluded_columns": [],
    }
    
    result = execute_feature_spec(
        dataset_ref="test_date_diff",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    assert result["errors"] == []
    assert "days_to_purchase" in result["features_created"]
    
    df_out = get_registered_dataset(result["transformed_dataset_ref"])
    expected_days = [9, 5, 44]
    assert list(df_out["days_to_purchase"]) == expected_days
    
    print("\n=== TEST 9: Date Difference ===")
    print(f"Signup: {list(df['signup_date'].dt.strftime('%Y-%m-%d'))}")
    print(f"Purchase: {list(df['purchase_date'].dt.strftime('%Y-%m-%d'))}")
    print(f"Days diff: {list(df_out['days_to_purchase'])}")


# =============================================================================
# TEST 10: As-Of Constraint (Temporal Masking)
# =============================================================================

def test_as_of_constraint():
    """Test temporal masking with as_of_constraint."""
    df = pd.DataFrame({
        "id": range(1, 11),
        "transaction_date": pd.date_range("2023-01-01", periods=10, freq="D"),
        "amount": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "target": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })
    register_dataset("test_as_of", df)
    
    feature_spec = {
        "features": [
            {
                "name": "amount",
                "formula": {"op": "passthrough", "column": "amount"},
                "grain": "id",
                "as_of_constraint": {
                    "source_date_column": "transaction_date",
                    "operator": "<",
                },
            },
        ],
        "reasoning": "Amount with temporal constraint",
        "excluded_columns": [],
    }
    
    # Cutoff at Jan 6 - should mask Jan 6 onwards
    as_of_cutoff = "2023-01-06"
    
    result = execute_feature_spec(
        dataset_ref="test_as_of",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
        as_of_cutoff=as_of_cutoff,
    )
    
    assert result["errors"] == []
    assert result["temporal_constraints_applied"] == 1
    
    df_out = get_registered_dataset(result["transformed_dataset_ref"])
    
    # First 5 rows (before cutoff) should have values
    assert df_out["amount"].iloc[:5].notna().all()
    # Last 5 rows (on/after cutoff) should be NaN
    assert df_out["amount"].iloc[5:].isna().all()
    
    print("\n=== TEST 10: As-Of Constraint ===")
    print(f"Cutoff: {as_of_cutoff}")
    print(f"Temporal constraints applied: {result['temporal_constraints_applied']}")
    print(f"Before cutoff (should have values): {list(df_out['amount'].iloc[:5])}")
    print(f"After cutoff (should be NaN): {list(df_out['amount'].iloc[5:])}")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])
