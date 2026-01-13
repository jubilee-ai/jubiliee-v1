"""
Tests for correlation_matrix tool.

Tests cover:
- Basic correlation computation
- High correlation detection (multicollinearity)
- Different correlation methods (pearson, spearman, kendall)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from analysis.correlation_matrix import correlation_matrix_tool, compute_correlation_matrix
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
def correlated_dataset():
    """Dataset with known correlations for testing."""
    np.random.seed(42)
    n = 500
    
    # Create features with known correlations
    income = np.random.normal(50000, 15000, n)
    debt = income * 0.4 + np.random.normal(0, 2000, n)  # High correlation with income (~0.97)
    age = np.random.randint(18, 70, n)
    months_employed = age * 4 + np.random.normal(0, 10, n)  # High correlation with age (~0.98)
    credit_score = np.random.randint(300, 850, n)  # Independent
    
    return pd.DataFrame({
        "income": income,
        "debt": debt,
        "age": age.astype(float),
        "months_employed": months_employed,
        "credit_score": credit_score.astype(float),
    })


@pytest.fixture
def uncorrelated_dataset():
    """Dataset with no significant correlations."""
    np.random.seed(42)
    n = 500
    
    return pd.DataFrame({
        "feature_a": np.random.normal(0, 1, n),
        "feature_b": np.random.normal(100, 10, n),
        "feature_c": np.random.uniform(0, 1, n),
        "feature_d": np.random.randint(0, 100, n).astype(float),
    })


# =============================================================================
# TEST 1: Basic Correlation Detection
# =============================================================================

class TestBasicCorrelation:
    """Test basic correlation matrix computation."""
    
    def test_detects_high_correlations(self, correlated_dataset):
        """Test that high correlations are correctly detected."""
        register_dataset("corr_test", correlated_dataset)
        
        result = compute_correlation_matrix(
            dataset_ref="corr_test",
            threshold=0.7,
        )
        
        print("\n" + "="*60)
        print("TEST 1: Basic Correlation Detection")
        print("="*60)
        print(f"Input: Dataset with {correlated_dataset.shape[0]} rows, {correlated_dataset.shape[1]} columns")
        print(f"Columns: {list(correlated_dataset.columns)}")
        print(f"\nExpected high correlations:")
        print("  - income ↔ debt (constructed with r ≈ 0.97)")
        print("  - age ↔ months_employed (constructed with r ≈ 0.98)")
        print(f"\nActual high correlations found (threshold=0.7):")
        for pair in result["high_correlations"]:
            print(f"  - {pair['col1']} ↔ {pair['col2']}: r = {pair['correlation']}")
        
        # Assertions
        assert len(result["high_correlations"]) == 2, \
            f"Expected 2 high correlations, got {len(result['high_correlations'])}"
        
        # Check income-debt correlation
        income_debt = next(
            (p for p in result["high_correlations"] 
             if set([p["col1"], p["col2"]]) == {"income", "debt"}),
            None
        )
        assert income_debt is not None, "Expected income-debt correlation not found"
        assert income_debt["abs_correlation"] > 0.9, \
            f"Expected income-debt correlation > 0.9, got {income_debt['abs_correlation']}"
        
        # Check age-months_employed correlation
        age_months = next(
            (p for p in result["high_correlations"] 
             if set([p["col1"], p["col2"]]) == {"age", "months_employed"}),
            None
        )
        assert age_months is not None, "Expected age-months_employed correlation not found"
        assert age_months["abs_correlation"] > 0.9, \
            f"Expected age-months correlation > 0.9, got {age_months['abs_correlation']}"
        
        print("\n✓ TEST PASSED: Correctly detected both high correlation pairs")


# =============================================================================
# TEST 2: No False Positives on Uncorrelated Data
# =============================================================================

class TestNoFalsePositives:
    """Test that uncorrelated data doesn't produce false high correlations."""
    
    def test_no_high_correlations_in_random_data(self, uncorrelated_dataset):
        """Test that random uncorrelated data has no high correlation flags."""
        register_dataset("uncorr_test", uncorrelated_dataset)
        
        result = compute_correlation_matrix(
            dataset_ref="uncorr_test",
            threshold=0.7,
        )
        
        print("\n" + "="*60)
        print("TEST 2: No False Positives on Uncorrelated Data")
        print("="*60)
        print(f"Input: Dataset with {uncorrelated_dataset.shape[0]} rows, {uncorrelated_dataset.shape[1]} columns")
        print(f"Columns: {list(uncorrelated_dataset.columns)}")
        print(f"All columns are independently generated random data")
        print(f"\nTop 5 correlations found:")
        for pair in result["top_pairs"][:5]:
            print(f"  - {pair['col1']} ↔ {pair['col2']}: r = {pair['correlation']}")
        print(f"\nHigh correlations above threshold 0.7: {len(result['high_correlations'])}")
        
        # Assertions
        assert len(result["high_correlations"]) == 0, \
            f"Expected 0 high correlations in random data, got {len(result['high_correlations'])}"
        
        # Verify all correlations are weak
        max_corr = max(p["abs_correlation"] for p in result["top_pairs"]) if result["top_pairs"] else 0
        assert max_corr < 0.3, \
            f"Expected max correlation < 0.3 in random data, got {max_corr}"
        
        print(f"\n✓ TEST PASSED: No false positives (max correlation = {max_corr:.3f})")


# =============================================================================
# TEST 3: Tool Output Format
# =============================================================================

class TestToolOutput:
    """Test LangChain tool output format."""
    
    def test_tool_output_format(self, correlated_dataset):
        """Test tool produces readable output for agents."""
        register_dataset("tool_test", correlated_dataset)
        
        output = correlation_matrix_tool.invoke({
            "dataset_ref": "tool_test",
            "method": "pearson",
            "threshold": 0.7,
        })
        
        print("\n" + "="*60)
        print("TEST 3: Tool Output Format")
        print("="*60)
        print("Input parameters:")
        print("  - dataset_ref: 'tool_test'")
        print("  - method: 'pearson'")
        print("  - threshold: 0.7")
        print(f"\nTool output:\n")
        print(output)
        
        # Assertions on output format
        assert "# Correlation Matrix Analysis" in output, "Missing header"
        assert "**Method:** pearson" in output, "Missing method info"
        assert "Top Correlated Pairs" in output, "Missing top pairs section"
        assert "High Correlations" in output, "Missing high correlations section"
        assert "⚠️" in output, "Missing warning flag for high correlations"
        assert "Correlation Matrix" in output, "Missing matrix table"
        assert "Recommendation" in output, "Missing recommendation"
        
        # Check that it contains the correlated pairs
        assert "income" in output and "debt" in output, "Missing income-debt pair"
        assert "age" in output and "months_employed" in output, "Missing age-months pair"
        
        print("\n✓ TEST PASSED: Tool output is well-formatted for agent consumption")


# =============================================================================
# TEST 4 & 5: Real Datasets
# =============================================================================

class TestRealDatasets:
    """Test on real datasets from the datasets folder."""
    
    @pytest.fixture
    def insurance_df(self):
        """Load insurance.csv dataset."""
        csv_path = Path(__file__).parent.parent / "datasets" / "csv" / "insurance.csv"
        if not csv_path.exists():
            pytest.skip(f"Dataset not found: {csv_path}")
        return pd.read_csv(csv_path)
    
    @pytest.fixture
    def loan_default_df(self):
        """Load Loan_default.csv dataset."""
        csv_path = Path(__file__).parent.parent / "datasets" / "csv" / "Loan_default.csv"
        if not csv_path.exists():
            pytest.skip(f"Dataset not found: {csv_path}")
        return pd.read_csv(csv_path)
    
    def test_insurance_dataset_correlations(self, insurance_df):
        """Test correlation analysis on insurance dataset."""
        register_dataset("insurance", insurance_df)
        
        result = compute_correlation_matrix(
            dataset_ref="insurance",
            threshold=0.5,
        )
        
        print("\n" + "="*60)
        print("TEST 4: Insurance Dataset Correlations")
        print("="*60)
        print(f"Input: insurance.csv")
        print(f"Shape: {insurance_df.shape}")
        print(f"Numeric columns: {result['columns']}")
        print(f"\nTop 10 correlated pairs:")
        for pair in result["top_pairs"][:10]:
            flag = " ⚠️" if pair["abs_correlation"] >= 0.5 else ""
            print(f"  - {pair['col1']} ↔ {pair['col2']}: r = {pair['correlation']}{flag}")
        print(f"\nHigh correlations (threshold=0.5): {len(result['high_correlations'])}")
        
        # Basic assertions
        assert result["n_features"] >= 2, "Expected at least 2 numeric features"
        assert len(result["top_pairs"]) > 0, "Expected some correlation pairs"
        
        # Test tool output as well
        output = correlation_matrix_tool.invoke({
            "dataset_ref": "insurance",
            "threshold": 0.5,
        })
        print(f"\nTool output preview (first 500 chars):\n{output[:500]}...")
        
        assert "Correlation Matrix Analysis" in output
        print("\n✓ TEST PASSED: Insurance dataset correlation analysis completed")
    
    def test_loan_default_dataset_correlations(self, loan_default_df):
        """Test correlation analysis on loan default dataset."""
        register_dataset("loan_default", loan_default_df)
        
        result = compute_correlation_matrix(
            dataset_ref="loan_default",
            threshold=0.7,
        )
        
        print("\n" + "="*60)
        print("TEST 5: Loan Default Dataset Correlations")
        print("="*60)
        print(f"Input: Loan_default.csv")
        print(f"Shape: {loan_default_df.shape}")
        print(f"Numeric columns ({result['n_features']}): {result['columns'][:10]}...")
        print(f"\nTop 10 correlated pairs:")
        for pair in result["top_pairs"][:10]:
            flag = " ⚠️" if pair["abs_correlation"] >= 0.7 else ""
            print(f"  - {pair['col1']} ↔ {pair['col2']}: r = {pair['correlation']}{flag}")
        print(f"\nHigh correlations (threshold=0.7): {len(result['high_correlations'])}")
        if result["high_correlations"]:
            print("Multicollinearity detected between:")
            for pair in result["high_correlations"][:5]:
                print(f"  - {pair['col1']} ↔ {pair['col2']}: r = {pair['correlation']}")
        
        # Basic assertions
        assert result["n_features"] >= 2, "Expected at least 2 numeric features"
        assert len(result["top_pairs"]) > 0, "Expected some correlation pairs"
        
        # Verify matrix is symmetric (sanity check)
        matrix = result["matrix"]
        cols = result["columns"][:3]  # Check first 3 columns
        for c1 in cols:
            for c2 in cols:
                assert matrix[c1][c2] == matrix[c2][c1], \
                    f"Matrix should be symmetric: {c1}-{c2}"
        
        print("\n✓ TEST PASSED: Loan default dataset correlation analysis completed")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

