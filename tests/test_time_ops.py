"""Tests for time/window operation tools."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from transformations import (
    time_bucket_tool,
    lag_tool,
    lead_tool,
    rolling_mean_tool,
    rolling_sum_tool,
    row_number_tool,
)
from utils import register_dataset, get_registered_dataset, clear_dataset_registry
import re


def get_ref(result: str) -> str:
    """Extract dataset ref from tool output."""
    match = re.search(r"`([a-z]{2,3}_[a-f0-9]+_[a-f0-9]+)`", result)
    return match.group(1) if match else None


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear dataset registry before each test."""
    clear_dataset_registry()
    yield
    clear_dataset_registry()


class TestTimeBucket:
    """Tests for time_bucket_tool."""
    
    def test_bucket_by_day(self):
        """Bucket timestamps into daily periods."""
        # INPUT
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "timestamp": pd.to_datetime([
                "2024-01-15 10:30:00",
                "2024-01-15 14:45:00", 
                "2024-01-16 09:00:00",
                "2024-01-17 16:30:00",
            ]),
            "value": [100, 200, 300, 400],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: time_bucket by day ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        # EXECUTE
        result = time_bucket_tool.invoke({
            "dataset_ref": "test",
            "column": "timestamp",
            "granularity": "day",
        })
        
        print(f"\nOUTPUT: {result}")
        
        # VERIFY
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert "timestamp_day" in out_df.columns
        assert out_df["timestamp_day"].iloc[0] == pd.Timestamp("2024-01-15")
        assert out_df["timestamp_day"].iloc[1] == pd.Timestamp("2024-01-15")  # Same day
        assert out_df["timestamp_day"].iloc[2] == pd.Timestamp("2024-01-16")
    
    def test_bucket_by_week(self):
        """Bucket timestamps into weekly periods."""
        df = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-01",  # Week 1
                "2024-01-05",  # Week 1
                "2024-01-08",  # Week 2
                "2024-01-15",  # Week 3
            ]),
            "sales": [100, 200, 150, 300],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: time_bucket by week ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = time_bucket_tool.invoke({
            "dataset_ref": "test",
            "column": "date",
            "granularity": "week",
            "output_column": "week_start",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert "week_start" in out_df.columns
        # First two should be same week
        assert out_df["week_start"].iloc[0] == out_df["week_start"].iloc[1]
    
    def test_bucket_invalid_granularity(self):
        """Test error handling for invalid granularity."""
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"])})
        register_dataset("test", df)
        
        result = time_bucket_tool.invoke({
            "dataset_ref": "test",
            "column": "ts",
            "granularity": "invalid",
        })
        
        print(f"\nINVALID GRANULARITY OUTPUT: {result}")
        assert "✗" in result
        assert "Unknown granularity" in result


class TestLag:
    """Tests for lag_tool."""
    
    def test_simple_lag(self):
        """Get previous row value."""
        df = pd.DataFrame({
            "day": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: simple lag ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = lag_tool.invoke({
            "dataset_ref": "test",
            "column": "value",
            "periods": 1,
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert pd.isna(out_df["value_lag_1"].iloc[0])  # First row has no previous
        assert out_df["value_lag_1"].iloc[1] == 10
        assert out_df["value_lag_1"].iloc[2] == 20
    
    def test_lag_with_partition(self):
        """Get previous value within each user."""
        df = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2, 2],
            "timestamp": pd.to_datetime([
                "2024-01-01", "2024-01-02", "2024-01-03",
                "2024-01-01", "2024-01-02", "2024-01-03",
            ]),
            "amount": [100, 150, 200, 50, 75, 100],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: lag with partition ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = lag_tool.invoke({
            "dataset_ref": "test",
            "column": "amount",
            "partition_by": "user_id",
            "order_by": "timestamp",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Each user's first row should have NaN
        user1_first = out_df[out_df["user_id"] == 1].iloc[0]
        user2_first = out_df[out_df["user_id"] == 2].iloc[0]
        assert pd.isna(user1_first["amount_lag_1"])
        assert pd.isna(user2_first["amount_lag_1"])


class TestLead:
    """Tests for lead_tool."""
    
    def test_simple_lead(self):
        """Get next row value."""
        df = pd.DataFrame({
            "day": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: simple lead ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = lead_tool.invoke({
            "dataset_ref": "test",
            "column": "value",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert out_df["value_lead_1"].iloc[0] == 20
        assert out_df["value_lead_1"].iloc[3] == 50
        assert pd.isna(out_df["value_lead_1"].iloc[4])  # Last row has no next
    
    def test_lead_with_partition(self):
        """Get next value within each user."""
        df = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "seq": [1, 2, 1, 2],
            "value": [100, 200, 50, 75],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: lead with partition ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = lead_tool.invoke({
            "dataset_ref": "test",
            "column": "value",
            "partition_by": "user_id",
            "order_by": "seq",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result


class TestRollingMean:
    """Tests for rolling_mean_tool."""
    
    def test_rolling_mean_simple(self):
        """Compute 3-row moving average."""
        df = pd.DataFrame({
            "day": [1, 2, 3, 4, 5, 6],
            "value": [10, 20, 30, 40, 50, 60],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: rolling mean (window=3) ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = rolling_mean_tool.invoke({
            "dataset_ref": "test",
            "column": "value",
            "window": 3,
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Row 3 (index 2): mean of [10, 20, 30] = 20
        assert out_df["value_rmean_3"].iloc[2] == 20.0
        # Row 4 (index 3): mean of [20, 30, 40] = 30
        assert out_df["value_rmean_3"].iloc[3] == 30.0
    
    def test_rolling_mean_with_partition(self):
        """Compute rolling mean per user."""
        df = pd.DataFrame({
            "user_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "day": [1, 2, 3, 4, 1, 2, 3, 4],
            "sales": [100, 200, 300, 400, 10, 20, 30, 40],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: rolling mean with partition ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = rolling_mean_tool.invoke({
            "dataset_ref": "test",
            "column": "sales",
            "window": 2,
            "partition_by": "user_id",
            "order_by": "day",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert "sales_rmean_2" in out_df.columns


class TestRollingSum:
    """Tests for rolling_sum_tool."""
    
    def test_rolling_sum_simple(self):
        """Compute 3-row rolling sum."""
        df = pd.DataFrame({
            "day": [1, 2, 3, 4, 5],
            "amount": [10, 20, 30, 40, 50],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: rolling sum (window=3) ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = rolling_sum_tool.invoke({
            "dataset_ref": "test",
            "column": "amount",
            "window": 3,
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Row 3 (index 2): sum of [10, 20, 30] = 60
        assert out_df["amount_rsum_3"].iloc[2] == 60.0
        # Row 4 (index 3): sum of [20, 30, 40] = 90
        assert out_df["amount_rsum_3"].iloc[3] == 90.0
        # Row 5 (index 4): sum of [30, 40, 50] = 120
        assert out_df["amount_rsum_3"].iloc[4] == 120.0


class TestRowNumber:
    """Tests for row_number_tool."""
    
    def test_row_number_simple(self):
        """Add sequential row numbers."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "value": [100, 200, 150, 300],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: simple row number ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = row_number_tool.invoke({
            "dataset_ref": "test",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert list(out_df["row_num"]) == [1, 2, 3, 4]
    
    def test_row_number_with_partition_and_order(self):
        """Rank within each group by value descending."""
        df = pd.DataFrame({
            "dept": ["Sales", "Sales", "Sales", "Eng", "Eng", "Eng"],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
            "score": [85, 92, 78, 95, 88, 91],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: row number with partition ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = row_number_tool.invoke({
            "dataset_ref": "test",
            "partition_by": "dept",
            "order_by": "score",
            "ascending": False,
            "output_column": "rank",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Check Sales rankings: Bob (92) should be rank 1
        sales_df = out_df[out_df["dept"] == "Sales"]
        bob_row = sales_df[sales_df["name"] == "Bob"]
        assert bob_row["rank"].values[0] == 1
        
        # Check Eng rankings: Diana (95) should be rank 1
        eng_df = out_df[out_df["dept"] == "Eng"]
        diana_row = eng_df[eng_df["name"] == "Diana"]
        assert diana_row["rank"].values[0] == 1


class TestEdgeCases:
    """Test error handling and edge cases."""
    
    def test_missing_column(self):
        """Test error when column doesn't exist."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        register_dataset("test", df)
        
        result = lag_tool.invoke({
            "dataset_ref": "test",
            "column": "nonexistent",
        })
        
        print(f"\nMISSING COLUMN OUTPUT: {result}")
        assert "✗" in result
        assert "not found" in result
    
    def test_missing_dataset(self):
        """Test error when dataset doesn't exist."""
        result = time_bucket_tool.invoke({
            "dataset_ref": "nonexistent",
            "column": "ts",
            "granularity": "day",
        })
        
        print(f"\nMISSING DATASET OUTPUT: {result}")
        assert "✗" in result
        assert "not found" in result
    
    def test_rolling_mean_non_numeric(self):
        """Test error when rolling mean on non-numeric column."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "category": ["A", "B", "C"],
        })
        register_dataset("test", df)
        
        result = rolling_mean_tool.invoke({
            "dataset_ref": "test",
            "column": "category",
            "window": 2,
        })
        
        print(f"\nNON-NUMERIC COLUMN OUTPUT: {result}")
        assert "✗" in result
        assert "not numeric" in result
    
    def test_rolling_sum_non_numeric(self):
        """Test error when rolling sum on non-numeric column."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "text": ["hello", "world", "test"],
        })
        register_dataset("test", df)
        
        result = rolling_sum_tool.invoke({
            "dataset_ref": "test",
            "column": "text",
            "window": 2,
        })
        
        print(f"\nNON-NUMERIC COLUMN OUTPUT: {result}")
        assert "✗" in result
        assert "not numeric" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

