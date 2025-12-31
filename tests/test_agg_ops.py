"""Tests for groupby_agg tool."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from transformations import groupby_agg_tool
from utils import register_dataset, get_registered_dataset, clear_dataset_registry
import re


def get_ref(result: str) -> str:
    """Extract dataset ref from tool output."""
    match = re.search(r"`([a-z]{2,3}_[a-f0-9]+_[a-f0-9]+)`", result)
    return match.group(1) if match else None


@pytest.fixture(autouse=True)
def clear_registry():
    clear_dataset_registry()
    yield
    clear_dataset_registry()


class TestBasicAggregations:
    """Test standard aggregation functions."""
    
    def test_sum_and_count(self):
        """Group by region, compute sum and count."""
        df = pd.DataFrame({
            "region": ["East", "East", "West", "West", "West"],
            "sales": [100, 150, 200, 250, 300],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: sum and count ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = groupby_agg_tool.invoke({
            "dataset_ref": "test",
            "keys": "region",
            "aggs": [
                {"col": "sales", "fn": "sum", "output": "total_sales"},
                {"col": "sales", "fn": "count", "output": "num_orders"},
            ]
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        east = out_df[out_df["region"] == "East"].iloc[0]
        west = out_df[out_df["region"] == "West"].iloc[0]
        assert east["total_sales"] == 250  # 100 + 150
        assert east["num_orders"] == 2
        assert west["total_sales"] == 750  # 200 + 250 + 300
        assert west["num_orders"] == 3
    
    def test_avg_min_max(self):
        """Test avg, min, max aggregations."""
        df = pd.DataFrame({
            "category": ["A", "A", "A", "B", "B"],
            "value": [10, 20, 30, 100, 200],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: avg, min, max ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = groupby_agg_tool.invoke({
            "dataset_ref": "test",
            "keys": "category",
            "aggs": [
                {"col": "value", "fn": "avg"},
                {"col": "value", "fn": "min"},
                {"col": "value", "fn": "max"},
            ]
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        cat_a = out_df[out_df["category"] == "A"].iloc[0]
        assert cat_a["value_avg"] == 20.0  # (10+20+30)/3
        assert cat_a["value_min"] == 10
        assert cat_a["value_max"] == 30
    
    def test_count_distinct(self):
        """Test count_distinct (nunique)."""
        df = pd.DataFrame({
            "store": ["NYC", "NYC", "NYC", "LA", "LA"],
            "product": ["A", "A", "B", "A", "B"],
            "sales": [1, 2, 3, 4, 5],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: count_distinct ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = groupby_agg_tool.invoke({
            "dataset_ref": "test",
            "keys": "store",
            "aggs": [
                {"col": "product", "fn": "count_distinct", "output": "unique_products"},
            ]
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        nyc = out_df[out_df["store"] == "NYC"].iloc[0]
        assert nyc["unique_products"] == 2  # A and B


class TestMultipleKeys:
    """Test grouping by multiple columns."""
    
    def test_two_keys(self):
        """Group by two columns."""
        df = pd.DataFrame({
            "region": ["East", "East", "East", "West"],
            "product": ["A", "A", "B", "A"],
            "revenue": [100, 200, 150, 300],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: multiple keys ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = groupby_agg_tool.invoke({
            "dataset_ref": "test",
            "keys": ["region", "product"],
            "aggs": [
                {"col": "revenue", "fn": "sum", "output": "total"},
                {"col": "revenue", "fn": "count", "output": "orders"},
            ]
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert len(out_df) == 3  # East/A, East/B, West/A
        east_a = out_df[(out_df["region"] == "East") & (out_df["product"] == "A")].iloc[0]
        assert east_a["total"] == 300  # 100 + 200
        assert east_a["orders"] == 2


class TestAdvancedAggregations:
    """Test quantile and count_if."""
    
    def test_quantile(self):
        """Test quantile aggregation."""
        df = pd.DataFrame({
            "group": ["A"] * 5,
            "value": [10, 20, 30, 40, 50],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: quantile ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = groupby_agg_tool.invoke({
            "dataset_ref": "test",
            "keys": "group",
            "aggs": [
                {"col": "value", "fn": "quantile:0.5", "output": "median"},
                {"col": "value", "fn": "quantile:0.9", "output": "p90"},
            ]
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert out_df["median"].iloc[0] == 30.0  # Median of 10,20,30,40,50
    
    def test_count_if(self):
        """Test count_if with expression."""
        df = pd.DataFrame({
            "dept": ["Sales", "Sales", "Sales", "Eng", "Eng"],
            "status": ["active", "inactive", "active", "active", "active"],
            "score": [80, 60, 90, 85, 95],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: count_if ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = groupby_agg_tool.invoke({
            "dataset_ref": "test",
            "keys": "dept",
            "aggs": [
                {"col": "status", "fn": "count", "output": "total"},
                {"col": "score", "fn": "count_if:score >= 80", "output": "high_scorers"},
            ]
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        # Sales: 3 total, 2 high scorers (80, 90)
        sales = out_df[out_df["dept"] == "Sales"].iloc[0]
        assert sales["total"] == 3
        assert sales["high_scorers"] == 2
        # Eng: 2 total, 2 high scorers (85, 95)
        eng = out_df[out_df["dept"] == "Eng"].iloc[0]
        assert eng["total"] == 2
        assert eng["high_scorers"] == 2


class TestStatisticalAggregations:
    """Test std, var, median."""
    
    def test_std_var_median(self):
        """Test statistical aggregations."""
        df = pd.DataFrame({
            "group": ["A", "A", "A", "A", "A"],
            "value": [10, 20, 30, 40, 50],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: std, var, median ===")
        print("INPUT:")
        print(df.to_string(index=False))
        
        result = groupby_agg_tool.invoke({
            "dataset_ref": "test",
            "keys": "group",
            "aggs": [
                {"col": "value", "fn": "std"},
                {"col": "value", "fn": "var"},
                {"col": "value", "fn": "median"},
            ]
        })
        
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT DATA:")
        print(out_df.to_string(index=False))
        
        assert "✓" in result
        assert out_df["value_median"].iloc[0] == 30.0
        assert abs(out_df["value_std"].iloc[0] - 15.81) < 0.1  # ~15.81


class TestErrorHandling:
    """Test error cases."""
    
    def test_missing_key_column(self):
        """Error when key column doesn't exist."""
        df = pd.DataFrame({"a": [1, 2]})
        register_dataset("test", df)
        
        result = groupby_agg_tool.invoke({
            "dataset_ref": "test",
            "keys": "missing",
            "aggs": [{"col": "a", "fn": "sum"}]
        })
        
        print(f"\nMISSING KEY: {result}")
        assert "✗" in result
        assert "not found" in result
    
    def test_missing_agg_column(self):
        """Error when aggregation column doesn't exist."""
        df = pd.DataFrame({"a": [1, 2]})
        register_dataset("test", df)
        
        result = groupby_agg_tool.invoke({
            "dataset_ref": "test",
            "keys": "a",
            "aggs": [{"col": "missing", "fn": "sum"}]
        })
        
        print(f"\nMISSING AGG COL: {result}")
        assert "✗" in result
        assert "not found" in result
    
    def test_unknown_function(self):
        """Error on unknown aggregation function."""
        df = pd.DataFrame({"a": [1, 2]})
        register_dataset("test", df)
        
        result = groupby_agg_tool.invoke({
            "dataset_ref": "test",
            "keys": "a",
            "aggs": [{"col": "a", "fn": "unknown_fn"}]
        })
        
        print(f"\nUNKNOWN FN: {result}")
        assert "✗" in result
        assert "Unknown" in result
    
    def test_empty_aggs(self):
        """Error when no aggregations provided."""
        df = pd.DataFrame({"a": [1, 2]})
        register_dataset("test", df)
        
        result = groupby_agg_tool.invoke({
            "dataset_ref": "test",
            "keys": "a",
            "aggs": []
        })
        
        print(f"\nEMPTY AGGS: {result}")
        assert "✗" in result
        assert "No aggregations" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

