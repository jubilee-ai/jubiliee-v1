"""
Tests for trend_analysis_tool using real and synthetic datasets.

Tests cover:
1. Financial Distress dataset - panel data with Time periods
2. Synthetic time series derived from Insurance dataset (monthly premiums)
3. Synthetic time series derived from Loan Default dataset (loan originations)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))
from utils import clear_dataset_registry, register_dataset
from analysis.trend_analysis import trend_analysis_tool, analyze_trend


class TestFinancialDistressDataset:
    """Test trend analysis on Financial Distress panel data."""
    
    @pytest.fixture
    def financial_distress_df(self):
        """Load Financial Distress dataset and prepare for trend analysis."""
        df = pd.read_csv(Path(__file__).parent.parent / "datasets/csv/Financial Distress.csv")
        
        # Aggregate Financial Distress by Time period across all companies
        # This creates a time series of average financial distress
        agg_df = df.groupby("Time").agg({
            "Financial Distress": "mean",
            "Company": "count"
        }).reset_index()
        agg_df.columns = ["period", "avg_distress", "n_companies"]
        
        # Convert period to date (assume quarterly starting from 2010)
        agg_df["date"] = pd.to_datetime("2010-01-01") + pd.to_timedelta(agg_df["period"] * 91, unit="D")
        
        clear_dataset_registry()
        register_dataset("financial_distress", agg_df)
        return agg_df
    
    def test_financial_distress_trend(self, financial_distress_df):
        """Test: Analyze average financial distress trend over time."""
        print("\n" + "="*70)
        print("TEST 1: Financial Distress Trend Over Time")
        print("="*70)
        
        print(f"\nInput Dataset:")
        print(f"  - Shape: {financial_distress_df.shape}")
        print(f"  - Columns: {financial_distress_df.columns.tolist()}")
        print(f"  - Date range: {financial_distress_df['date'].min()} to {financial_distress_df['date'].max()}")
        print(f"  - Avg distress range: {financial_distress_df['avg_distress'].min():.3f} to {financial_distress_df['avg_distress'].max():.3f}")
        
        result = trend_analysis_tool.invoke({
            "dataset_ref": "financial_distress",
            "date_col": "date",
            "metric_col": "avg_distress",
            "freq": "QE",
            "agg_func": "mean",
        })
        
        print(f"\nTool Output:")
        print(result)
        
        # Assertions
        assert "Trend Analysis" in result
        assert "avg_distress" in result
        assert "INCREASING" in result or "DECREASING" in result or "FLAT" in result
        assert "R²" in result


class TestInsuranceDataset:
    """Test trend analysis on synthetic time series from Insurance data."""
    
    @pytest.fixture
    def insurance_timeseries_df(self):
        """Create monthly insurance charges time series."""
        df = pd.read_csv(Path(__file__).parent.parent / "datasets/csv/insurance.csv")
        
        np.random.seed(42)
        n_months = 36
        dates = pd.date_range("2022-01-01", periods=n_months, freq="ME")
        
        # Simulate monthly premium collections with seasonality + growth
        base_charges = df["charges"].mean()
        growth = np.linspace(1.0, 1.3, n_months)  # 30% growth over 3 years
        seasonality = 1 + 0.15 * np.sin(np.arange(n_months) * np.pi / 6)  # Semi-annual cycle
        noise = np.random.normal(1, 0.05, n_months)
        
        monthly_charges = base_charges * growth * seasonality * noise * 100  # Scale up
        
        ts_df = pd.DataFrame({
            "month": dates,
            "total_charges": monthly_charges,
            "avg_charge": base_charges * growth * noise,
            "policy_count": np.random.randint(80, 150, n_months),
        })
        
        clear_dataset_registry()
        register_dataset("insurance_monthly", ts_df)
        return ts_df
    
    def test_insurance_monthly_charges(self, insurance_timeseries_df):
        """Test: Analyze monthly insurance charges trend."""
        print("\n" + "="*70)
        print("TEST 2: Insurance Monthly Charges Trend")
        print("="*70)
        
        print(f"\nInput Dataset (synthetic from insurance.csv):")
        print(f"  - Shape: {insurance_timeseries_df.shape}")
        print(f"  - Date range: {insurance_timeseries_df['month'].min()} to {insurance_timeseries_df['month'].max()}")
        print(f"  - Charges range: ${insurance_timeseries_df['total_charges'].min():,.0f} to ${insurance_timeseries_df['total_charges'].max():,.0f}")
        
        result = trend_analysis_tool.invoke({
            "dataset_ref": "insurance_monthly",
            "date_col": "month",
            "metric_col": "total_charges",
            "freq": "ME",
            "agg_func": "sum",
            "periods_for_ma": 3,
        })
        
        print(f"\nTool Output:")
        print(result)
        
        # Assertions - should detect increasing trend (we built in 30% growth)
        assert "Trend Analysis" in result
        assert "INCREASING" in result, "Should detect increasing trend (30% growth built in)"
        assert "R²" in result
        
        # Check for seasonality detection
        raw_result = analyze_trend(
            "insurance_monthly", "month", "total_charges", "ME", "sum", 3
        )
        print(f"\nRaw result seasonality: {raw_result['seasonality']}")


class TestLoanDefaultDataset:
    """Test trend analysis on synthetic loan origination time series."""
    
    @pytest.fixture
    def loan_timeseries_df(self):
        """Create weekly loan origination time series."""
        df = pd.read_csv(Path(__file__).parent.parent / "datasets/csv/Loan_default.csv")
        
        np.random.seed(123)
        n_weeks = 52
        dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W")
        
        # Simulate weekly loan volume with declining trend + noise
        base_volume = 5000
        decline = np.linspace(1.0, 0.7, n_weeks)  # 30% decline
        noise = np.random.normal(1, 0.1, n_weeks)
        
        weekly_volume = base_volume * decline * noise
        avg_loan_amount = df["LoanAmount"].mean()
        avg_income = df["Income"].mean()
        
        ts_df = pd.DataFrame({
            "week": dates,
            "loan_count": (weekly_volume).astype(int),
            "total_amount": weekly_volume * avg_loan_amount,
            "avg_applicant_income": avg_income * np.random.normal(1, 0.05, n_weeks),
        })
        
        clear_dataset_registry()
        register_dataset("loan_weekly", ts_df)
        return ts_df
    
    def test_loan_weekly_volume_decline(self, loan_timeseries_df):
        """Test: Detect declining trend in weekly loan originations."""
        print("\n" + "="*70)
        print("TEST 3: Weekly Loan Origination Volume (Declining Trend)")
        print("="*70)
        
        print(f"\nInput Dataset (synthetic from Loan_default.csv):")
        print(f"  - Shape: {loan_timeseries_df.shape}")
        print(f"  - Date range: {loan_timeseries_df['week'].min()} to {loan_timeseries_df['week'].max()}")
        print(f"  - Loan count range: {loan_timeseries_df['loan_count'].min():,} to {loan_timeseries_df['loan_count'].max():,}")
        
        result = trend_analysis_tool.invoke({
            "dataset_ref": "loan_weekly",
            "date_col": "week",
            "metric_col": "loan_count",
            "freq": "W",
            "agg_func": "sum",
        })
        
        print(f"\nTool Output:")
        print(result)
        
        # Assertions - should detect decreasing trend (we built in 30% decline)
        assert "Trend Analysis" in result
        assert "DECREASING" in result, "Should detect decreasing trend (30% decline built in)"
        
        # Get raw result to check R² - expect moderate fit due to noise
        raw_result = analyze_trend("loan_weekly", "week", "loan_count", "W", "sum", 3)
        print(f"\nTrend details: {raw_result['trend']}")
        assert raw_result["trend"]["r_squared"] > 0.3, "R² should show meaningful linear relationship"
        assert raw_result["trend"]["total_change_pct"] < -20, "Should show significant decline"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

