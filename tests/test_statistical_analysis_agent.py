"""
Tests for statistical_analysis_agent - runs the agent on real datasets.

Note: These tests require network access to call the LLM.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))
from utils import clear_dataset_registry, register_dataset

# Import agent
sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))
from statistical_analysis_agent import statistical_analysis_tool


class TestStatisticalAnalysisAgent:
    """Test the statistical analysis agent on real datasets."""
    
    @pytest.fixture
    def insurance_df(self):
        """Load insurance dataset."""
        df = pd.read_csv(Path(__file__).parent.parent / "datasets/csv/insurance.csv")
        clear_dataset_registry()
        register_dataset("insurance", df)
        return df
    
    @pytest.fixture
    def loan_df(self):
        """Load loan default dataset."""
        df = pd.read_csv(Path(__file__).parent.parent / "datasets/csv/Loan_default.csv")
        clear_dataset_registry()
        register_dataset("loans", df)
        return df
    
    def test_insurance_data_quality(self, insurance_df):
        """Test: Analyze insurance dataset for data quality issues."""
        print("\n" + "="*70)
        print("TEST 1: Insurance Data Quality Analysis")
        print("="*70)
        
        print(f"\nInput:")
        print(f"  - Dataset: insurance")
        print(f"  - Shape: {insurance_df.shape}")
        print(f"  - Columns: {insurance_df.columns.tolist()}")
        print(f"  - Goal: 'Identify any data quality issues in this dataset'")
        
        result = statistical_analysis_tool.invoke({
            "dataset_ref": "insurance",
            "goal": "Identify any data quality issues in this dataset. Check for missing values, outliers, and data type issues.",
        })
        
        print(f"\nAgent Output:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        
        # Basic assertions
        assert result is not None
        assert len(result) > 100, "Expected substantive analysis output"
    
    def test_loan_feature_analysis(self, loan_df):
        """Test: Analyze loan dataset features for prediction."""
        print("\n" + "="*70)
        print("TEST 2: Loan Default Feature Analysis")
        print("="*70)
        
        print(f"\nInput:")
        print(f"  - Dataset: loans")
        print(f"  - Shape: {loan_df.shape}")
        print(f"  - Target: Default column")
        print(f"  - Goal: 'Find which features are most predictive of Default'")
        
        result = statistical_analysis_tool.invoke({
            "dataset_ref": "loans",
            "goal": "Analyze which features are most predictive of the Default column. Focus on correlations and feature importance.",
        })
        
        print(f"\nAgent Output:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        
        assert result is not None
        assert len(result) > 100
    
    def test_insurance_charges_distribution(self, insurance_df):
        """Test: Analyze distribution of insurance charges."""
        print("\n" + "="*70)
        print("TEST 3: Insurance Charges Distribution Analysis")
        print("="*70)
        
        print(f"\nInput:")
        print(f"  - Dataset: insurance")
        print(f"  - Focus: charges column")
        print(f"  - Goal: 'Analyze the distribution of charges and identify segments'")
        
        result = statistical_analysis_tool.invoke({
            "dataset_ref": "insurance",
            "goal": "Analyze the distribution of the charges column. Is it normally distributed? Are there distinct segments? How concentrated is it?",
        })
        
        print(f"\nAgent Output:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        
        assert result is not None
        assert len(result) > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

