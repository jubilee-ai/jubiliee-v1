"""Tests for feature engineering tools (bin_column, agg_feature, extract_datetime)."""

import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from transformations.feature_ops import (
    bin_column_tool,
    agg_feature_tool,
    multi_agg_feature_tool,
    extract_datetime_tool,
    date_diff_tool,
)
from utils import register_dataset, get_registered_dataset, clear_dataset_registry


def get_ref(result: str) -> str:
    """Extract dataset ref from tool output."""
    match = re.search(r"`([a-z]+_[a-f0-9]+_[a-f0-9]+)`", result)
    return match.group(1) if match else None


@pytest.fixture(autouse=True)
def clear_registry():
    clear_dataset_registry()
    yield
    clear_dataset_registry()


# =============================================================================
# BIN COLUMN TESTS
# =============================================================================

class TestBinColumn:
    """Test bin_column_tool."""
    
    def test_uniform_bins_int(self):
        """Bin into N equal-width bins with auto-generated labels."""
        df = pd.DataFrame({
            "age": [20, 30, 40, 50, 60, 70, 80],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: uniform bins (int) ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = bin_column_tool.invoke({
            "dataset_ref": "test",
            "column": "age",
            "bins": 3,
            "strategy": "uniform",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert "age_binned" in out_df.columns
        assert out_df["age_binned"].nunique() == 3
        # Check that we get clean labels, not intervals
        assert "bin_1" in out_df["age_binned"].values
        assert "bin_2" in out_df["age_binned"].values
        assert "bin_3" in out_df["age_binned"].values
    
    def test_quantile_bins(self):
        """Bin into quantiles (equal-frequency)."""
        # Skewed distribution
        df = pd.DataFrame({
            "income": [10000, 20000, 30000, 40000, 50000, 100000, 200000, 500000],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: quantile bins ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = bin_column_tool.invoke({
            "dataset_ref": "test",
            "column": "income",
            "bins": 4,
            "strategy": "quantile",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Each quartile should have ~2 values
        counts = out_df["income_binned"].value_counts()
        print(f"BIN COUNTS: {counts.to_dict()}")
        assert all(c >= 1 for c in counts)
    
    def test_custom_edges_with_labels(self):
        """Bin with custom edges and labels."""
        df = pd.DataFrame({
            "credit_score": [550, 620, 680, 720, 780, 820],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: custom edges with labels ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = bin_column_tool.invoke({
            "dataset_ref": "test",
            "column": "credit_score",
            "bins": [300, 580, 670, 740, 800, 850],
            "labels": ["poor", "fair", "good", "very_good", "excellent"],
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Check specific mappings
        assert out_df[out_df["credit_score"] == 550]["credit_score_binned"].iloc[0] == "poor"
        assert out_df[out_df["credit_score"] == 680]["credit_score_binned"].iloc[0] == "good"
        assert out_df[out_df["credit_score"] == 820]["credit_score_binned"].iloc[0] == "excellent"
    
    def test_error_non_numeric(self):
        """Error when column is not numeric."""
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        register_dataset("test", df)
        
        result = bin_column_tool.invoke({
            "dataset_ref": "test",
            "column": "name",
            "bins": 3,
        })
        
        print(f"\nNON-NUMERIC ERROR: {result}")
        assert "✗" in result
        assert "not numeric" in result


# =============================================================================
# AGG FEATURE TESTS
# =============================================================================

class TestAggFeature:
    """Test agg_feature_tool."""
    
    def test_mean_by_group(self):
        """Create mean feature by group."""
        df = pd.DataFrame({
            "zip_code": ["10001", "10001", "10001", "10002", "10002"],
            "income": [50000, 60000, 70000, 100000, 120000],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: mean by group ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = agg_feature_tool.invoke({
            "dataset_ref": "test",
            "column": "income",
            "agg": "mean",
            "group_by": "zip_code",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert "mean_income_by_zip_code" in out_df.columns
        
        # Check values
        zip1_mean = out_df[out_df["zip_code"] == "10001"]["mean_income_by_zip_code"].iloc[0]
        zip2_mean = out_df[out_df["zip_code"] == "10002"]["mean_income_by_zip_code"].iloc[0]
        assert zip1_mean == 60000  # (50k+60k+70k)/3
        assert zip2_mean == 110000  # (100k+120k)/2
    
    def test_sum_by_multiple_groups(self):
        """Create sum feature by multiple groups."""
        df = pd.DataFrame({
            "state": ["NY", "NY", "CA", "CA"],
            "city": ["NYC", "NYC", "LA", "SF"],
            "sales": [100, 200, 300, 400],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: sum by multiple groups ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = agg_feature_tool.invoke({
            "dataset_ref": "test",
            "column": "sales",
            "agg": "sum",
            "group_by": ["state", "city"],
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert "sum_sales_by_state_city" in out_df.columns
        
        # NYC has 2 rows that sum to 300
        nyc_sum = out_df[out_df["city"] == "NYC"]["sum_sales_by_state_city"].iloc[0]
        assert nyc_sum == 300
    
    def test_count_by_group(self):
        """Create count feature by group."""
        df = pd.DataFrame({
            "customer_id": ["A", "A", "A", "B", "B"],
            "order_id": [1, 2, 3, 4, 5],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: count by group ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = agg_feature_tool.invoke({
            "dataset_ref": "test",
            "column": "order_id",
            "agg": "count",
            "group_by": "customer_id",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Customer A has 3 orders
        a_count = out_df[out_df["customer_id"] == "A"]["count_order_id_by_customer_id"].iloc[0]
        assert a_count == 3
    
    def test_median_by_group(self):
        """Test median aggregation (uses merge pattern)."""
        df = pd.DataFrame({
            "group": ["A", "A", "A", "B", "B"],
            "value": [10, 20, 30, 100, 200],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: median by group ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = agg_feature_tool.invoke({
            "dataset_ref": "test",
            "column": "value",
            "agg": "median",
            "group_by": "group",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Median of [10, 20, 30] = 20
        a_median = out_df[out_df["group"] == "A"]["median_value_by_group"].iloc[0]
        assert a_median == 20.0
        # Median of [100, 200] = 150
        b_median = out_df[out_df["group"] == "B"]["median_value_by_group"].iloc[0]
        assert b_median == 150.0
    
    def test_nunique_by_group(self):
        """Test nunique aggregation (uses merge pattern)."""
        df = pd.DataFrame({
            "store": ["NYC", "NYC", "NYC", "LA", "LA"],
            "product": ["A", "A", "B", "A", "B"],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: nunique by group ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = agg_feature_tool.invoke({
            "dataset_ref": "test",
            "column": "product",
            "agg": "nunique",
            "group_by": "store",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # NYC has 2 unique products (A, B)
        nyc_nunique = out_df[out_df["store"] == "NYC"]["nunique_product_by_store"].iloc[0]
        assert nyc_nunique == 2


class TestMultiAggFeature:
    """Test multi_agg_feature_tool."""
    
    def test_multiple_aggs(self):
        """Create multiple agg features at once."""
        df = pd.DataFrame({
            "zip_code": ["10001", "10001", "10002", "10002"],
            "income": [50000, 60000, 100000, 120000],
            "age": [25, 35, 45, 55],
            "debt": [10000, 20000, 30000, 40000],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: multiple aggs ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = multi_agg_feature_tool.invoke({
            "dataset_ref": "test",
            "aggs": {"income": "mean", "age": "mean", "debt": "sum"},
            "group_by": "zip_code",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert "mean_income_by_zip_code" in out_df.columns
        assert "mean_age_by_zip_code" in out_df.columns
        assert "sum_debt_by_zip_code" in out_df.columns


# =============================================================================
# EXTRACT DATETIME TESTS
# =============================================================================

class TestExtractDatetime:
    """Test extract_datetime_tool."""
    
    def test_basic_extraction(self):
        """Extract basic datetime features."""
        df = pd.DataFrame({
            "date": ["2024-01-15", "2024-06-20", "2024-12-25"],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: basic datetime extraction ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = extract_datetime_tool.invoke({
            "dataset_ref": "test",
            "column": "date",
            "features": ["year", "month", "day", "quarter"],
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert out_df["date_year"].tolist() == [2024, 2024, 2024]
        assert out_df["date_month"].tolist() == [1, 6, 12]
        assert out_df["date_day"].tolist() == [15, 20, 25]
        assert out_df["date_quarter"].tolist() == [1, 2, 4]
    
    def test_weekday_features(self):
        """Extract day of week and weekend flag."""
        df = pd.DataFrame({
            "date": [
                "2024-01-15",  # Monday
                "2024-01-20",  # Saturday
                "2024-01-21",  # Sunday
            ],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: weekday features ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = extract_datetime_tool.invoke({
            "dataset_ref": "test",
            "column": "date",
            "features": ["dayofweek", "is_weekend"],
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert out_df["date_dayofweek"].tolist() == [0, 5, 6]  # Mon=0, Sat=5, Sun=6
        assert out_df["date_is_weekend"].tolist() == [0, 1, 1]
    
    def test_custom_prefix(self):
        """Use custom prefix for output columns."""
        df = pd.DataFrame({
            "application_date": ["2024-03-15"],
        })
        register_dataset("test", df)
        
        result = extract_datetime_tool.invoke({
            "dataset_ref": "test",
            "column": "application_date",
            "features": ["month", "year"],
            "prefix": "app",
        })
        
        print(f"\nCUSTOM PREFIX: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        
        assert "app_month" in out_df.columns
        assert "app_year" in out_df.columns


class TestDateDiff:
    """Test date_diff_tool."""
    
    def test_days_diff(self):
        """Calculate days between two dates."""
        df = pd.DataFrame({
            "signup_date": ["2024-01-01", "2024-01-15", "2024-02-01"],
            "purchase_date": ["2024-01-10", "2024-01-15", "2024-03-01"],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: days diff ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = date_diff_tool.invoke({
            "dataset_ref": "test",
            "start_column": "signup_date",
            "end_column": "purchase_date",
            "unit": "days",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        diff_col = "days_signup_date_to_purchase_date"
        assert diff_col in out_df.columns
        assert out_df[diff_col].tolist() == [9.0, 0.0, 29.0]
    
    def test_hours_diff(self):
        """Calculate hours between timestamps."""
        df = pd.DataFrame({
            "start": ["2024-01-01 10:00:00", "2024-01-01 00:00:00"],
            "end": ["2024-01-01 14:30:00", "2024-01-02 12:00:00"],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: hours diff ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = date_diff_tool.invoke({
            "dataset_ref": "test",
            "start_column": "start",
            "end_column": "end",
            "unit": "hours",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        hours = out_df["hours_start_to_end"].tolist()
        assert hours[0] == 4.5  # 10:00 to 14:30 = 4.5 hours
        assert hours[1] == 36.0  # 00:00 to next day 12:00 = 36 hours


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error cases across all tools."""
    
    def test_missing_column_bin(self):
        """bin_column with missing column."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        register_dataset("test", df)
        
        result = bin_column_tool.invoke({
            "dataset_ref": "test",
            "column": "missing",
            "bins": 3,
        })
        
        print(f"\nMISSING COL (bin): {result}")
        assert "✗" in result
        assert "not found" in result
    
    def test_missing_group_col_agg(self):
        """agg_feature with missing group column."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        register_dataset("test", df)
        
        result = agg_feature_tool.invoke({
            "dataset_ref": "test",
            "column": "a",
            "agg": "mean",
            "group_by": "missing",
        })
        
        print(f"\nMISSING GROUP (agg): {result}")
        assert "✗" in result
        assert "not found" in result
    
    def test_invalid_datetime_feature(self):
        """extract_datetime with invalid feature - Pydantic rejects at schema level."""
        from pydantic import ValidationError
        
        df = pd.DataFrame({"date": ["2024-01-01"]})
        register_dataset("test", df)
        
        # Pydantic's Literal type validates before our code runs
        with pytest.raises(ValidationError) as exc_info:
            extract_datetime_tool.invoke({
                "dataset_ref": "test",
                "column": "date",
                "features": ["invalid_feature"],
            })
        
        print(f"\nINVALID FEATURE (Pydantic): {exc_info.value}")
        assert "literal_error" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

