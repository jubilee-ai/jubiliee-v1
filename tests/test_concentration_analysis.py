"""
Tests for concentration_analysis_tool using real datasets.

Tests cover:
1. Insurance charges - natural concentration in healthcare costs
2. Loan amounts - concentration by loan purpose
3. Income distribution - concentration in applicant incomes
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))
from utils import clear_dataset_registry, register_dataset
from analysis.concentration_analysis import concentration_analysis_tool, analyze_concentration


class TestInsuranceConcentration:
    """Test concentration analysis on insurance charges."""
    
    @pytest.fixture
    def insurance_df(self):
        """Load insurance dataset."""
        df = pd.read_csv(Path(__file__).parent.parent / "datasets/csv/insurance.csv")
        clear_dataset_registry()
        register_dataset("insurance", df)
        return df
    
    def test_charges_concentration(self, insurance_df):
        """Test: Analyze concentration of insurance charges across policyholders."""
        print("\n" + "="*70)
        print("TEST 1: Insurance Charges Concentration")
        print("="*70)
        
        print(f"\nInput Dataset:")
        print(f"  - Shape: {insurance_df.shape}")
        print(f"  - Charges range: ${insurance_df['charges'].min():,.2f} to ${insurance_df['charges'].max():,.2f}")
        print(f"  - Charges mean: ${insurance_df['charges'].mean():,.2f}")
        print(f"  - Charges median: ${insurance_df['charges'].median():,.2f}")
        
        result = concentration_analysis_tool.invoke({
            "dataset_ref": "insurance",
            "value_col": "charges",
        })
        
        print(f"\nTool Output:")
        print(result)
        
        # Get raw result for assertions
        raw = analyze_concentration("insurance", "charges")
        
        # Assertions
        assert "Concentration Analysis" in result
        assert "Gini" in result
        assert raw["gini_coefficient"] > 0.3, "Insurance charges should show meaningful concentration"
        assert raw["pareto"]["pct_entities_for_80pct_value"] < 60, "Should show some Pareto effect"
        
        print(f"\n✓ Gini = {raw['gini_coefficient']} (expected: moderate-to-high concentration)")
        print(f"✓ Pareto: {raw['pareto']['pct_entities_for_80pct_value']}% drive 80% of charges")


class TestLoanConcentration:
    """Test concentration analysis on loan data."""
    
    @pytest.fixture
    def loan_df(self):
        """Load loan default dataset."""
        df = pd.read_csv(Path(__file__).parent.parent / "datasets/csv/Loan_default.csv")
        clear_dataset_registry()
        register_dataset("loans", df)
        return df
    
    def test_loan_amount_by_purpose(self, loan_df):
        """Test: Analyze concentration of loan amounts by loan purpose."""
        print("\n" + "="*70)
        print("TEST 2: Loan Amount Concentration by Purpose")
        print("="*70)
        
        # Show input summary
        purpose_summary = loan_df.groupby("LoanPurpose")["LoanAmount"].agg(["sum", "count", "mean"])
        print(f"\nInput Dataset:")
        print(f"  - Shape: {loan_df.shape}")
        print(f"  - Unique purposes: {loan_df['LoanPurpose'].nunique()}")
        print(f"  - Purpose breakdown:")
        for purpose, row in purpose_summary.iterrows():
            print(f"    {purpose}: {row['count']:,} loans, ${row['sum']:,.0f} total")
        
        result = concentration_analysis_tool.invoke({
            "dataset_ref": "loans",
            "value_col": "LoanAmount",
            "entity_col": "LoanPurpose",
        })
        
        print(f"\nTool Output:")
        print(result)
        
        # Get raw result
        raw = analyze_concentration("loans", "LoanAmount", "LoanPurpose")
        
        # Assertions - with few categories, expect low concentration
        assert "Concentration Analysis" in result
        assert raw["n_entities"] == loan_df["LoanPurpose"].nunique()
        
        print(f"\n✓ Analyzed {raw['n_entities']} loan purposes")
        print(f"✓ Gini = {raw['gini_coefficient']} (expected: low with few equal categories)")
    
    def test_income_concentration(self, loan_df):
        """Test: Analyze concentration of applicant incomes."""
        print("\n" + "="*70)
        print("TEST 3: Applicant Income Concentration")
        print("="*70)
        
        print(f"\nInput Dataset:")
        print(f"  - Shape: {loan_df.shape}")
        print(f"  - Income range: ${loan_df['Income'].min():,} to ${loan_df['Income'].max():,}")
        print(f"  - Income mean: ${loan_df['Income'].mean():,.0f}")
        print(f"  - Income median: ${loan_df['Income'].median():,.0f}")
        
        result = concentration_analysis_tool.invoke({
            "dataset_ref": "loans",
            "value_col": "Income",
        })
        
        print(f"\nTool Output:")
        print(result)
        
        # Get raw result
        raw = analyze_concentration("loans", "Income")
        
        # Assertions
        assert "Concentration Analysis" in result
        assert "Gini" in result
        assert raw["n_positive"] > 0
        
        # Check top-N makes sense
        top_20 = raw["top_n_contribution"]["top_20pct"]["pct_of_total"]
        assert top_20 > 20, "Top 20% should have more than 20% of income (some concentration)"
        
        print(f"\n✓ Gini = {raw['gini_coefficient']}")
        print(f"✓ Top 20% of applicants have {top_20}% of total income")
        print(f"✓ Pareto: {raw['pareto']['pct_entities_for_80pct_value']}% drive 80% of income")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

