"""
Tests for group_summary tool.

Tests cover:
- Basic grouping by single column
- Multiple metrics and aggregations
- Real dataset testing (insurance, loan_default)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from analysis.group_summary import group_summary_tool, compute_group_summary
from utils import clear_dataset_registry, register_dataset


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def clear_registry():
    clear_dataset_registry()
    yield
    clear_dataset_registry()


@pytest.fixture
def insurance_df():
    """Load insurance.csv dataset."""
    csv_path = Path(__file__).parent.parent / "datasets" / "csv" / "insurance.csv"
    if not csv_path.exists():
        pytest.skip(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


@pytest.fixture
def loan_default_df():
    """Load Loan_default.csv dataset."""
    csv_path = Path(__file__).parent.parent / "datasets" / "csv" / "Loan_default.csv"
    if not csv_path.exists():
        pytest.skip(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


# =============================================================================
# TEST 1: Insurance Dataset - Charges by Smoker Status
# =============================================================================

class TestInsuranceBySmokingStatus:
    """Test group summary on insurance dataset grouped by smoker status."""
    
    def test_charges_by_smoker(self, insurance_df):
        """Test average charges differ significantly between smokers and non-smokers."""
        register_dataset("insurance", insurance_df)
        
        result = compute_group_summary(
            dataset_ref="insurance",
            group_by="smoker",
            metrics=["charges", "bmi", "age"],
            agg_funcs=["mean", "median", "count"],
        )
        
        print("\n" + "="*70)
        print("TEST 1: Insurance Charges by Smoker Status")
        print("="*70)
        print(f"Input: insurance.csv ({insurance_df.shape[0]} rows)")
        print(f"Group by: 'smoker' (yes/no)")
        print(f"Metrics: charges, bmi, age")
        print(f"\nResults:")
        for group in result["groups"]:
            print(f"  {group['smoker']}: charges_mean=${group['charges_mean']:,.2f}, "
                  f"bmi_mean={group['bmi_mean']:.1f}, count={group['charges_count']}")
        
        # Assertions
        assert result["n_groups"] == 2, "Expected 2 groups (yes/no)"
        
        # Find smoker vs non-smoker
        smoker_yes = next(g for g in result["groups"] if g["smoker"] == "yes")
        smoker_no = next(g for g in result["groups"] if g["smoker"] == "no")
        
        # Smokers should have much higher charges
        ratio = smoker_yes["charges_mean"] / smoker_no["charges_mean"]
        print(f"\nSmoker/Non-smoker charge ratio: {ratio:.2f}x")
        
        assert ratio > 2, f"Expected smokers to have >2x higher charges, got {ratio:.2f}x"
        
        # Test tool output
        output = group_summary_tool.invoke({
            "dataset_ref": "insurance",
            "group_by": "smoker",
            "metrics": ["charges"],
        })
        print(f"\nTool output:\n{output}")
        
        assert "smoker" in output.lower()
        assert "charges" in output.lower()
        
        print("\n✓ TEST PASSED: Correctly identified smokers have higher charges")


# =============================================================================
# TEST 2: Loan Default Dataset - Default Rate by Education
# =============================================================================

class TestLoanDefaultByEducation:
    """Test group summary on loan default dataset grouped by education level."""
    
    def test_default_rate_by_education(self, loan_default_df):
        """Test default rate varies by education level."""
        register_dataset("loans", loan_default_df)
        
        result = compute_group_summary(
            dataset_ref="loans",
            group_by="Education",
            metrics=["Default", "Income", "LoanAmount"],
            agg_funcs=["mean", "sum", "count"],
        )
        
        print("\n" + "="*70)
        print("TEST 2: Loan Default Rate by Education Level")
        print("="*70)
        print(f"Input: Loan_default.csv ({loan_default_df.shape[0]:,} rows)")
        print(f"Group by: 'Education'")
        print(f"Metrics: Default (rate), Income, LoanAmount")
        print(f"\nDefault rates by education:")
        for group in result["groups"]:
            default_rate = group["Default_mean"] * 100
            print(f"  {group['Education']}: {default_rate:.1f}% default rate, "
                  f"avg income=${group['Income_mean']:,.0f}, n={group['Default_count']:,}")
        
        # Assertions
        assert result["n_groups"] >= 2, "Expected at least 2 education levels"
        assert len(result["groups"]) > 0, "Expected groups to be populated"
        
        # Check that Default_mean is between 0 and 1 (it's a rate)
        for group in result["groups"]:
            assert 0 <= group["Default_mean"] <= 1, \
                f"Default rate should be between 0-1, got {group['Default_mean']}"
        
        # Test tool output
        output = group_summary_tool.invoke({
            "dataset_ref": "loans",
            "group_by": "Education",
            "metrics": ["Default", "Income"],
        })
        print(f"\nTool output preview:\n{output[:800]}...")
        
        assert "Education" in output
        assert "Default" in output
        
        print("\n✓ TEST PASSED: Correctly computed default rates by education level")


# =============================================================================
# TEST 3: Insurance Dataset - Multiple Group Columns
# =============================================================================

class TestMultipleGroupColumns:
    """Test grouping by multiple columns simultaneously."""
    
    def test_charges_by_smoker_and_sex(self, insurance_df):
        """Test grouping by both smoker status and sex."""
        register_dataset("insurance", insurance_df)
        
        result = compute_group_summary(
            dataset_ref="insurance",
            group_by=["smoker", "sex"],
            metrics=["charges"],
            agg_funcs=["mean", "count"],
        )
        
        print("\n" + "="*70)
        print("TEST 3: Insurance Charges by Smoker + Sex (Multi-column grouping)")
        print("="*70)
        print(f"Input: insurance.csv ({insurance_df.shape[0]} rows)")
        print(f"Group by: ['smoker', 'sex']")
        print(f"Metrics: charges")
        print(f"\nResults ({result['n_groups']} groups):")
        for group in result["groups"]:
            print(f"  {group['smoker']}/{group['sex']}: "
                  f"charges_mean=${group['charges_mean']:,.2f}, n={group['charges_count']}")
        
        # Assertions
        assert result["n_groups"] == 4, f"Expected 4 groups (2x2), got {result['n_groups']}"
        
        # Verify all combinations exist
        combinations = {(g["smoker"], g["sex"]) for g in result["groups"]}
        expected = {("yes", "male"), ("yes", "female"), ("no", "male"), ("no", "female")}
        assert combinations == expected, f"Missing group combinations: {expected - combinations}"
        
        # Test tool output
        output = group_summary_tool.invoke({
            "dataset_ref": "insurance",
            "group_by": ["smoker", "sex"],
            "metrics": ["charges", "bmi"],
        })
        print(f"\nTool output:\n{output}")
        
        assert "smoker" in output.lower()
        assert "sex" in output.lower()
        assert "charges" in output.lower()
        
        # Verify the highest charges group is male smokers (typically true in this dataset)
        male_smoker = next(g for g in result["groups"] 
                          if g["smoker"] == "yes" and g["sex"] == "male")
        female_nonsmoker = next(g for g in result["groups"] 
                               if g["smoker"] == "no" and g["sex"] == "female")
        
        ratio = male_smoker["charges_mean"] / female_nonsmoker["charges_mean"]
        print(f"\nMale smoker vs Female non-smoker ratio: {ratio:.2f}x")
        
        assert ratio > 2, "Expected significant difference between extreme groups"
        
        print("\n✓ TEST PASSED: Multi-column grouping works correctly")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

