"""Tests for new tools: rolling_min, rolling_max, rank, pivot, unpivot, union."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from transformations import (
    rolling_min_tool,
    rolling_max_tool,
    rank_tool,
    pivot_tool,
    unpivot_tool,
    union_tool,
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


class TestRollingMin:
    """Tests for rolling_min_tool."""
    
    def test_rolling_min_simple(self):
        """Compute 3-row rolling minimum."""
        df = pd.DataFrame({
            "day": [1, 2, 3, 4, 5],
            "price": [50, 40, 60, 30, 70],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: rolling_min (window=3) ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = rolling_min_tool.invoke({
            "dataset_ref": "test",
            "column": "price",
            "window": 3,
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Row 3: min(50, 40, 60) = 40
        assert out_df["price_rmin_3"].iloc[2] == 40.0
        # Row 4: min(40, 60, 30) = 30
        assert out_df["price_rmin_3"].iloc[3] == 30.0
        # Row 5: min(60, 30, 70) = 30
        assert out_df["price_rmin_3"].iloc[4] == 30.0
    
    def test_rolling_min_with_partition(self):
        """Compute rolling min per group."""
        df = pd.DataFrame({
            "stock": ["AAPL", "AAPL", "AAPL", "GOOG", "GOOG", "GOOG"],
            "day": [1, 2, 3, 1, 2, 3],
            "price": [150, 145, 155, 100, 95, 105],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: rolling_min with partition ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = rolling_min_tool.invoke({
            "dataset_ref": "test",
            "column": "price",
            "window": 2,
            "partition_by": "stock",
            "order_by": "day",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result


class TestRollingMax:
    """Tests for rolling_max_tool."""
    
    def test_rolling_max_simple(self):
        """Compute 3-row rolling maximum."""
        df = pd.DataFrame({
            "day": [1, 2, 3, 4, 5],
            "price": [50, 40, 60, 30, 70],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: rolling_max (window=3) ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = rolling_max_tool.invoke({
            "dataset_ref": "test",
            "column": "price",
            "window": 3,
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Row 3: max(50, 40, 60) = 60
        assert out_df["price_rmax_3"].iloc[2] == 60.0
        # Row 4: max(40, 60, 30) = 60
        assert out_df["price_rmax_3"].iloc[3] == 60.0
        # Row 5: max(60, 30, 70) = 70
        assert out_df["price_rmax_3"].iloc[4] == 70.0


class TestRank:
    """Tests for rank_tool."""
    
    def test_rank_with_ties(self):
        """Test rank with tied values using average method."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "score": [85, 92, 85, 78],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: rank with ties (average method) ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = rank_tool.invoke({
            "dataset_ref": "test",
            "column": "score",
            "ascending": False,
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Bob (92) is highest → rank 1
        assert out_df[out_df["name"] == "Bob"]["score_rank"].values[0] == 1.0
        # Alice & Charlie (85) tied → average of 2,3 = 2.5
        assert out_df[out_df["name"] == "Alice"]["score_rank"].values[0] == 2.5
        assert out_df[out_df["name"] == "Charlie"]["score_rank"].values[0] == 2.5
        # Diana (78) is lowest → rank 4
        assert out_df[out_df["name"] == "Diana"]["score_rank"].values[0] == 4.0
    
    def test_rank_dense(self):
        """Test rank with dense method (no gaps)."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "score": [85, 92, 85, 78],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: rank dense method ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = rank_tool.invoke({
            "dataset_ref": "test",
            "column": "score",
            "ascending": False,
            "method": "dense",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Dense: Bob→1, Alice/Charlie→2, Diana→3 (no gap)
        assert out_df[out_df["name"] == "Bob"]["score_rank"].values[0] == 1.0
        assert out_df[out_df["name"] == "Alice"]["score_rank"].values[0] == 2.0
        assert out_df[out_df["name"] == "Diana"]["score_rank"].values[0] == 3.0
    
    def test_rank_with_partition(self):
        """Test rank within groups."""
        df = pd.DataFrame({
            "dept": ["Sales", "Sales", "Eng", "Eng"],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "score": [85, 92, 88, 95],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: rank with partition ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = rank_tool.invoke({
            "dataset_ref": "test",
            "column": "score",
            "partition_by": "dept",
            "ascending": False,
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Bob is #1 in Sales, Diana is #1 in Eng
        assert out_df[out_df["name"] == "Bob"]["score_rank"].values[0] == 1.0
        assert out_df[out_df["name"] == "Diana"]["score_rank"].values[0] == 1.0


class TestPivot:
    """Tests for pivot_tool."""
    
    def test_pivot_simple(self):
        """Pivot sales by region and product."""
        df = pd.DataFrame({
            "region": ["East", "East", "West", "West"],
            "product": ["A", "B", "A", "B"],
            "sales": [100, 150, 200, 250],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: pivot (region × product) ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = pivot_tool.invoke({
            "dataset_ref": "test",
            "index": "region",
            "columns": "product",
            "values": "sales",
            "agg_func": "sum",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert len(out_df) == 2  # East, West
        # Check East row
        east = out_df[out_df["region"] == "East"].iloc[0]
        assert east["sales_A"] == 100
        assert east["sales_B"] == 150
    
    def test_pivot_with_aggregation(self):
        """Pivot with mean aggregation on duplicates."""
        df = pd.DataFrame({
            "region": ["East", "East", "East", "West"],
            "product": ["A", "A", "B", "A"],
            "sales": [100, 120, 150, 200],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: pivot with mean aggregation ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = pivot_tool.invoke({
            "dataset_ref": "test",
            "index": "region",
            "columns": "product",
            "values": "sales",
            "agg_func": "mean",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # East×A = mean(100, 120) = 110
        east = out_df[out_df["region"] == "East"].iloc[0]
        assert east["sales_A"] == 110.0


class TestUnpivot:
    """Tests for unpivot_tool."""
    
    def test_unpivot_simple(self):
        """Unpivot quarterly sales columns."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob"],
            "q1": [100, 150],
            "q2": [120, 180],
            "q3": [140, 200],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: unpivot quarterly data ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = unpivot_tool.invoke({
            "dataset_ref": "test",
            "id_cols": "name",
            "var_name": "quarter",
            "value_name": "sales",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert len(out_df) == 6  # 2 people × 3 quarters
        # Check Alice's q1
        alice_q1 = out_df[(out_df["name"] == "Alice") & (out_df["quarter"] == "q1")]
        assert alice_q1["sales"].values[0] == 100
    
    def test_unpivot_specific_cols(self):
        """Unpivot only specified columns."""
        df = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "jan": [100, 150],
            "feb": [110, 160],
            "total": [210, 310],  # Should not be unpivoted
        })
        register_dataset("test", df)
        
        print("\n=== TEST: unpivot specific columns ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = unpivot_tool.invoke({
            "dataset_ref": "test",
            "id_cols": ["id", "name"],
            "value_cols": ["jan", "feb"],
            "var_name": "month",
            "value_name": "amount",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert len(out_df) == 4  # 2 people × 2 months
        assert "total" not in out_df.columns


class TestUnion:
    """Tests for union_tool."""
    
    def test_union_by_name(self):
        """Union two datasets by column name."""
        df1 = pd.DataFrame({"id": [1, 2], "value": [100, 200]})
        df2 = pd.DataFrame({"id": [3, 4], "value": [300, 400]})
        register_dataset("ds1", df1)
        register_dataset("ds2", df2)
        
        print("\n=== TEST: union by name ===")
        print("INPUT ds1:")
        print(df1.to_string(index=False))
        print("INPUT ds2:")
        print(df2.to_string(index=False))
        
        result = union_tool.invoke({
            "dataset_refs": ["ds1", "ds2"],
            "mode": "by_name",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert len(out_df) == 4
        assert list(out_df["id"]) == [1, 2, 3, 4]
    
    def test_union_mismatched_columns(self):
        """Union with different columns - fills with NaN."""
        df1 = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
        df2 = pd.DataFrame({"id": [3, 4], "b": [30, 40]})
        register_dataset("ds1", df1)
        register_dataset("ds2", df2)
        
        print("\n=== TEST: union with mismatched columns ===")
        print("INPUT ds1:")
        print(df1.to_string(index=False))
        print("INPUT ds2:")
        print(df2.to_string(index=False))
        
        result = union_tool.invoke({
            "dataset_refs": ["ds1", "ds2"],
            "mode": "by_name",
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert "⚠" in result  # Should have warning
        assert len(out_df) == 4
        assert "a" in out_df.columns
        assert "b" in out_df.columns
    
    def test_union_three_datasets(self):
        """Union three datasets."""
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        df3 = pd.DataFrame({"x": [5, 6]})
        register_dataset("d1", df1)
        register_dataset("d2", df2)
        register_dataset("d3", df3)
        
        print("\n=== TEST: union three datasets ===")
        print("INPUT d1:", df1["x"].tolist())
        print("INPUT d2:", df2["x"].tolist())
        print("INPUT d3:", df3["x"].tolist())
        
        result = union_tool.invoke({
            "dataset_refs": ["d1", "d2", "d3"],
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert len(out_df) == 6
        assert list(out_df["x"]) == [1, 2, 3, 4, 5, 6]


class TestEdgeCases:
    """Test error handling."""
    
    def test_rolling_min_non_numeric(self):
        """Error on non-numeric column."""
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        register_dataset("test", df)
        
        result = rolling_min_tool.invoke({
            "dataset_ref": "test",
            "column": "a",
            "window": 2,
        })
        
        print(f"\nNON-NUMERIC: {result}")
        assert "✗" in result
        assert "not numeric" in result
    
    def test_pivot_missing_column(self):
        """Error on missing column."""
        df = pd.DataFrame({"a": [1, 2]})
        register_dataset("test", df)
        
        result = pivot_tool.invoke({
            "dataset_ref": "test",
            "index": "a",
            "columns": "missing",
            "values": "a",
        })
        
        print(f"\nMISSING COLUMN: {result}")
        assert "✗" in result
        assert "not found" in result
    
    def test_union_single_dataset(self):
        """Error when only one dataset provided."""
        df = pd.DataFrame({"a": [1]})
        register_dataset("test", df)
        
        result = union_tool.invoke({
            "dataset_refs": ["test"],
        })
        
        print(f"\nSINGLE DATASET: {result}")
        assert "✗" in result
        assert "at least 2" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


