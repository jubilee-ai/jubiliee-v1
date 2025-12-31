"""Tests for data cleaning tools."""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from transformations import (
    clip_tool,
    drop_nulls_tool,
    fill_null_tool,
    impute_tool,
    regex_replace_tool,
    replace_values_tool,
)
from utils import clear_dataset_registry, get_registered_dataset, register_dataset


def get_ref(result: str) -> str:
    match = re.search(r"`([a-z]{2,3}_[a-f0-9]+_[a-f0-9]+)`", result)
    return match.group(1) if match else None


@pytest.fixture(autouse=True)
def clear_registry():
    clear_dataset_registry()
    yield
    clear_dataset_registry()


class TestDropNulls:
    def test_drop_any(self):
        """Drop rows where any column is null."""
        df = pd.DataFrame({
            "a": [1, 2, None, 4],
            "b": [10, None, 30, 40],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: drop_nulls (any) ===")
        print("INPUT:")
        print(df.to_string())
        
        result = drop_nulls_tool.invoke({"dataset_ref": "test"})
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT:")
        print(out_df.to_string())
        
        assert "✓" in result
        assert len(out_df) == 2  # Only rows 0 and 3 have no nulls
    
    def test_drop_all(self):
        """Drop only rows where ALL columns are null."""
        df = pd.DataFrame({
            "a": [1, None, None],
            "b": [10, 20, None],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: drop_nulls (all) ===")
        print("INPUT:")
        print(df.to_string())
        
        result = drop_nulls_tool.invoke({
            "dataset_ref": "test",
            "how": "all",
        })
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT:")
        print(out_df.to_string())
        
        assert "✓" in result
        assert len(out_df) == 2  # Only row 2 has all nulls
    
    def test_drop_subset(self):
        """Drop rows where specific column is null."""
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [None, 20, 30],
        })
        register_dataset("test", df)
        
        result = drop_nulls_tool.invoke({
            "dataset_ref": "test",
            "columns": "a",
        })
        print(f"\nSUBSET: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        
        assert len(out_df) == 2  # Only row 1 has null in 'a'


class TestFillNull:
    def test_fill_numeric(self):
        """Fill nulls with a numeric value."""
        df = pd.DataFrame({"age": [25, None, 30, None]})
        register_dataset("test", df)
        
        print("\n=== TEST: fill_null (numeric) ===")
        print("INPUT:")
        print(df.to_string())
        
        result = fill_null_tool.invoke({
            "dataset_ref": "test",
            "column": "age",
            "value": 0,
        })
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT:")
        print(out_df.to_string())
        
        assert "✓" in result
        assert out_df["age"].isna().sum() == 0
        assert list(out_df["age"]) == [25.0, 0.0, 30.0, 0.0]
    
    def test_fill_string(self):
        """Fill nulls with a string value."""
        df = pd.DataFrame({"status": ["active", None, "inactive"]})
        register_dataset("test", df)
        
        result = fill_null_tool.invoke({
            "dataset_ref": "test",
            "column": "status",
            "value": "unknown",
        })
        print(f"\nFILL STRING: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        
        assert list(out_df["status"]) == ["active", "unknown", "inactive"]


class TestImpute:
    def test_impute_mean(self):
        """Impute with mean."""
        df = pd.DataFrame({"value": [10, 20, None, 40]})
        register_dataset("test", df)
        
        print("\n=== TEST: impute (mean) ===")
        print("INPUT:")
        print(df.to_string())
        
        result = impute_tool.invoke({
            "dataset_ref": "test",
            "column": "value",
            "strategy": "mean",
        })
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT:")
        print(out_df.to_string())
        
        assert "✓" in result
        # Mean of [10, 20, 40] = 23.33
        assert abs(out_df["value"].iloc[2] - 23.33) < 0.1
    
    def test_impute_median(self):
        """Impute with median."""
        df = pd.DataFrame({"value": [10, 20, None, 100]})
        register_dataset("test", df)
        
        result = impute_tool.invoke({
            "dataset_ref": "test",
            "column": "value",
            "strategy": "median",
        })
        print(f"\nMEDIAN: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        
        # Median of [10, 20, 100] = 20
        assert out_df["value"].iloc[2] == 20.0
    
    def test_impute_mode(self):
        """Impute with mode (most frequent)."""
        df = pd.DataFrame({"cat": ["A", "A", None, "B"]})
        register_dataset("test", df)
        
        result = impute_tool.invoke({
            "dataset_ref": "test",
            "column": "cat",
            "strategy": "mode",
        })
        print(f"\nMODE: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        
        # Mode is 'A' (appears twice)
        assert out_df["cat"].iloc[2] == "A"
    
    def test_impute_ffill(self):
        """Forward fill."""
        df = pd.DataFrame({"value": [10, None, None, 40]})
        register_dataset("test", df)
        
        result = impute_tool.invoke({
            "dataset_ref": "test",
            "column": "value",
            "strategy": "ffill",
        })
        print(f"\nFFILL: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        
        assert list(out_df["value"]) == [10.0, 10.0, 10.0, 40.0]
    
    def test_impute_group_median(self):
        """Impute with group median."""
        df = pd.DataFrame({
            "dept": ["A", "A", "A", "B", "B"],
            "salary": [50000, None, 60000, 80000, None],
        })
        register_dataset("test", df)
        
        print("\n=== TEST: impute (group_median) ===")
        print("INPUT:")
        print(df.to_string())
        
        result = impute_tool.invoke({
            "dataset_ref": "test",
            "column": "salary",
            "strategy": "group_median",
            "group_by": "dept",
        })
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT:")
        print(out_df.to_string())
        
        assert "✓" in result
        # Dept A: median of [50000, 60000] = 55000
        # Dept B: median of [80000] = 80000
        assert out_df["salary"].iloc[1] == 55000.0
        assert out_df["salary"].iloc[4] == 80000.0


class TestClip:
    def test_clip_both(self):
        """Clip to both min and max."""
        df = pd.DataFrame({"score": [-10, 50, 150, 80]})
        register_dataset("test", df)
        
        print("\n=== TEST: clip ===")
        print("INPUT:")
        print(df.to_string())
        
        result = clip_tool.invoke({
            "dataset_ref": "test",
            "column": "score",
            "min_val": 0,
            "max_val": 100,
        })
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT:")
        print(out_df.to_string())
        
        assert "✓" in result
        assert list(out_df["score"]) == [0, 50, 100, 80]
    
    def test_clip_min_only(self):
        """Clip with only min."""
        df = pd.DataFrame({"amount": [-100, -50, 0, 100]})
        register_dataset("test", df)
        
        result = clip_tool.invoke({
            "dataset_ref": "test",
            "column": "amount",
            "min_val": 0,
        })
        print(f"\nMIN ONLY: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        
        assert list(out_df["amount"]) == [0, 0, 0, 100]


class TestReplaceValues:
    def test_replace_codes(self):
        """Replace code values."""
        df = pd.DataFrame({"gender": ["M", "F", "M", "F"]})
        register_dataset("test", df)
        
        print("\n=== TEST: replace_values ===")
        print("INPUT:")
        print(df.to_string())
        
        result = replace_values_tool.invoke({
            "dataset_ref": "test",
            "column": "gender",
            "mapping": {"M": "Male", "F": "Female"},
        })
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT:")
        print(out_df.to_string())
        
        assert "✓" in result
        assert list(out_df["gender"]) == ["Male", "Female", "Male", "Female"]
    
    def test_replace_sentinel(self):
        """Replace sentinel values with None."""
        df = pd.DataFrame({"age": [25, -1, 30, 999]})
        register_dataset("test", df)
        
        result = replace_values_tool.invoke({
            "dataset_ref": "test",
            "column": "age",
            "mapping": {-1: None, 999: None},
        })
        print(f"\nSENTINEL: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        
        assert out_df["age"].isna().sum() == 2


class TestRegexReplace:
    def test_remove_non_digits(self):
        """Remove non-digit characters."""
        df = pd.DataFrame({"phone": ["(123) 456-7890", "555.123.4567"]})
        register_dataset("test", df)
        
        print("\n=== TEST: regex_replace ===")
        print("INPUT:")
        print(df.to_string())
        
        result = regex_replace_tool.invoke({
            "dataset_ref": "test",
            "column": "phone",
            "pattern": r"[^0-9]",
            "replacement": "",
        })
        print(f"\nOUTPUT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        print("\nRESULT:")
        print(out_df.to_string())
        
        assert "✓" in result
        assert list(out_df["phone"]) == ["1234567890", "5551234567"]
    
    def test_extract_domain(self):
        """Extract email domain."""
        df = pd.DataFrame({"email": ["user@gmail.com", "admin@company.org"]})
        register_dataset("test", df)
        
        result = regex_replace_tool.invoke({
            "dataset_ref": "test",
            "column": "email",
            "pattern": r".*@(.+)",
            "replacement": r"\1",
        })
        print(f"\nEXTRACT: {result}")
        
        ref = get_ref(result)
        out_df = get_registered_dataset(ref)
        
        assert list(out_df["email"]) == ["gmail.com", "company.org"]


class TestErrorHandling:
    def test_missing_column(self):
        """Error on missing column."""
        df = pd.DataFrame({"a": [1]})
        register_dataset("test", df)
        
        result = fill_null_tool.invoke({
            "dataset_ref": "test",
            "column": "missing",
            "value": 0,
        })
        print(f"\nMISSING: {result}")
        assert "✗" in result
        assert "not found" in result
    
    def test_clip_no_bounds(self):
        """Error when no bounds provided."""
        df = pd.DataFrame({"a": [1]})
        register_dataset("test", df)
        
        result = clip_tool.invoke({
            "dataset_ref": "test",
            "column": "a",
        })
        print(f"\nNO BOUNDS: {result}")
        assert "✗" in result
        assert "min_val" in result or "max_val" in result
    
    def test_group_median_no_group(self):
        """Error when group_median without group_by."""
        df = pd.DataFrame({"a": [1, None]})
        register_dataset("test", df)
        
        result = impute_tool.invoke({
            "dataset_ref": "test",
            "column": "a",
            "strategy": "group_median",
        })
        print(f"\nNO GROUP: {result}")
        assert "✗" in result
        assert "group_by" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

