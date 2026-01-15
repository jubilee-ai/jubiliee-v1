"""
Test for the label and split definition function.
Run with: python tests/test_label_and_split.py
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

import json

import numpy as np
import pandas as pd
from utils import clear_registry, register_dataset


def test_label_split_auto_infer():
    """Test LLM auto-inference of all 6 parameters."""
    
    print("\n" + "="*70)
    print("TEST 1: Auto-infer all 6 parameters (no user input)")
    print("="*70)
    
    clear_registry()
    
    # Create a realistic loan dataset
    np.random.seed(42)
    n_rows = 100
    
    df = pd.DataFrame({
        "loan_id": range(1, n_rows + 1),
        "customer_id": np.random.randint(1000, 9999, n_rows),
        "application_date": pd.date_range("2023-01-01", periods=n_rows, freq="D"),
        "loan_amount": np.random.uniform(5000, 50000, n_rows).round(2),
        "interest_rate": np.random.uniform(5, 15, n_rows).round(2),
        "term_months": np.random.choice([12, 24, 36, 48, 60], n_rows),
        "credit_score": np.random.randint(550, 850, n_rows),
        "annual_income": np.random.uniform(30000, 150000, n_rows).round(2),
        "employment_length": np.random.randint(0, 20, n_rows),
        "home_ownership": np.random.choice(["OWN", "MORTGAGE", "RENT"], n_rows),
        "purpose": np.random.choice(["debt_consolidation", "home_improvement", "major_purchase"], n_rows),
        "default": np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
        "days_past_due": np.where(np.random.random(n_rows) > 0.9, np.random.randint(1, 90, n_rows), 0),
        "total_payments_made": np.random.randint(0, 36, n_rows),
    })
    
    dataset_ref = "loan_data_test"
    register_dataset(dataset_ref, df)
    
    print("\n[INPUT]")
    print("-"*70)
    print(f"Dataset: {dataset_ref}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nGoal: 'Predict which loans will default within 90 days of origination'")
    print(f"Selected Model: 'xgboost'")
    print(f"Model Explanation: 'XGBoost chosen for tabular classification with mixed feature types'")
    print(f"\nUser-provided values: NONE (all will be inferred)")
    
    # Run the function
    print("\n[CALLING run_label_split_definition...]")
    print("-"*70)
    
    from agents.training.label_and_split import run_label_split_definition
    
    result = run_label_split_definition(
        dataset_ref=dataset_ref,
        goal="Predict which loans will default within 90 days of origination",
        selected_model="xgboost",
        model_explanation="XGBoost chosen for tabular classification with mixed feature types",
        # No user-provided values - all will be inferred
    )
    
    print("\n[OUTPUT]")
    print("-"*70)
    print(json.dumps(result, indent=2))
    
    # Validate output structure
    print("\n[VALIDATION]")
    print("-"*70)
    assert "target_column" in result, "Missing target_column"
    assert "prediction_horizon" in result, "Missing prediction_horizon"
    assert "grain" in result, "Missing grain"
    assert "as_of_cutoff" in result, "Missing as_of_cutoff"
    assert "split_strategy" in result, "Missing split_strategy"
    assert "forbidden_columns" in result, "Missing forbidden_columns"
    assert result["split_strategy"] in ("random", "time_based", "entity_based"), "Invalid split_strategy"
    assert isinstance(result["forbidden_columns"], list), "forbidden_columns must be a list"
    
    print("✅ All fields present and valid")
    print(f"✅ target_column: {result['target_column']}")
    print(f"✅ split_strategy: {result['split_strategy']}")
    print(f"✅ forbidden_columns: {result['forbidden_columns']}")
    
    return result


def test_label_split_partial_user_input():
    """Test with some user-provided values (partial input)."""
    
    print("\n" + "="*70)
    print("TEST 2: Partial user input (target + split provided, rest inferred)")
    print("="*70)
    
    clear_registry()
    
    # Create insurance dataset
    np.random.seed(123)
    n_rows = 80
    
    df = pd.DataFrame({
        "policy_id": [f"POL-{i:05d}" for i in range(n_rows)],
        "customer_age": np.random.randint(25, 70, n_rows),
        "policy_start_date": pd.date_range("2022-01-01", periods=n_rows, freq="5D"),
        "premium": np.random.uniform(500, 5000, n_rows).round(2),
        "coverage_amount": np.random.choice([50000, 100000, 250000, 500000], n_rows),
        "region": np.random.choice(["Northeast", "Southeast", "Midwest", "West"], n_rows),
        "claim_count": np.random.poisson(0.5, n_rows),
        "total_claim_amount": np.random.exponential(2000, n_rows).round(2),
        "is_lapsed": np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
    })
    
    dataset_ref = "insurance_test"
    register_dataset(dataset_ref, df)
    
    print("\n[INPUT]")
    print("-"*70)
    print(f"Dataset: {dataset_ref}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nGoal: 'Predict policy lapse'")
    print(f"\nUser-provided values:")
    print(f"  - target_column: 'is_lapsed'")
    print(f"  - split_strategy: 'time_based'")
    print(f"  - (rest will be inferred)")
    
    from agents.training.label_and_split import run_label_split_definition
    
    result = run_label_split_definition(
        dataset_ref=dataset_ref,
        goal="Predict policy lapse",
        # User provides some values
        target_column="is_lapsed",
        split_strategy="time_based",
        # Rest inferred
    )
    
    print("\n[OUTPUT]")
    print("-"*70)
    print(json.dumps(result, indent=2))
    
    print("\n[VALIDATION]")
    print("-"*70)
    assert result["target_column"] == "is_lapsed", "User-provided target_column not preserved"
    assert result["split_strategy"] == "time_based", "User-provided split_strategy not preserved"
    print("✅ User-provided values preserved")
    print(f"✅ grain (inferred): {result['grain']}")
    print(f"✅ as_of_cutoff (inferred): {result['as_of_cutoff']}")
    print(f"✅ forbidden_columns (inferred): {result['forbidden_columns']}")
    
    return result


def test_label_split_all_provided():
    """Test with all values provided (LLM validates and returns them)."""
    
    print("\n" + "="*70)
    print("TEST 3: All values provided (LLM validates)")
    print("="*70)
    
    clear_registry()
    
    # Simple dataset
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "feature": [10, 20, 30],
        "target": [0, 1, 0],
    })
    
    dataset_ref = "simple_test"
    register_dataset(dataset_ref, df)
    
    print("\n[INPUT]")
    print("-"*70)
    print("All 6 values provided by user - LLM validates and returns them")
    
    from agents.training.label_and_split import run_label_split_definition
    
    result = run_label_split_definition(
        dataset_ref=dataset_ref,
        goal="Test",
        target_column="target",
        prediction_horizon="30 days",
        grain="one transaction",
        as_of_cutoff=None,  # Explicitly None
        split_strategy="random",
        forbidden_columns=["id"],
    )
    
    print("\n[OUTPUT]")
    print("-"*70)
    print(json.dumps(result, indent=2, default=str))
    
    print("\n[VALIDATION]")
    print("-"*70)
    assert result["target_column"] == "target"
    assert result["prediction_horizon"] == "30 days"
    assert result["grain"] == "one transaction"
    assert result["split_strategy"] == "random"
    assert result["forbidden_columns"] == ["id"]
    print("✅ All user-provided values returned exactly as provided")
    
    return result


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LABEL AND SPLIT DEFINITION TESTS")
    print("="*70)
    
    test_label_split_auto_infer()
    test_label_split_partial_user_input()
    test_label_split_all_provided()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70 + "\n")
