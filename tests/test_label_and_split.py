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


# =============================================================================
# SPLITTING FUNCTION TESTS
# =============================================================================

def test_compute_split_indices_random():
    """Test random split strategy."""
    
    print("\n" + "="*70)
    print("TEST 4: compute_split_indices - Random Strategy")
    print("="*70)
    
    clear_registry()
    
    # Create a simple dataset
    np.random.seed(42)
    n_rows = 1000
    
    df = pd.DataFrame({
        "id": range(n_rows),
        "feature1": np.random.randn(n_rows),
        "feature2": np.random.randn(n_rows),
        "target": np.random.choice([0, 1], n_rows),
    })
    
    print(f"\n[INPUT] DataFrame with {n_rows} rows")
    
    from agents.training.label_and_split import compute_split_indices, apply_split
    
    label_def = {
        "target_column": "target",
        "split_strategy": "random",
        "grain": "one row",
    }
    
    split_indices = compute_split_indices(
        df=df,
        label_definition=label_def,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )
    
    print("\n[OUTPUT]")
    print(f"  train_idx: {len(split_indices['train_idx'])} samples")
    print(f"  val_idx: {len(split_indices['val_idx'])} samples")
    print(f"  test_idx: {len(split_indices['test_idx'])} samples")
    print(f"  split_strategy: {split_indices['split_strategy']}")
    
    # Validate
    print("\n[VALIDATION]")
    
    # Check sizes match expected ratios (within 5%)
    expected_train = int(n_rows * 0.7)
    expected_val = int(n_rows * 0.15)
    expected_test = n_rows - expected_train - expected_val
    
    assert abs(len(split_indices['train_idx']) - expected_train) <= 50, "Train size mismatch"
    assert abs(len(split_indices['val_idx']) - expected_val) <= 50, "Val size mismatch"
    print("✅ Split sizes match expected ratios (70/15/15)")
    
    # Check no overlap
    train_set = set(split_indices['train_idx'])
    val_set = set(split_indices['val_idx'])
    test_set = set(split_indices['test_idx'])
    
    assert len(train_set & val_set) == 0, "Train and val overlap!"
    assert len(train_set & test_set) == 0, "Train and test overlap!"
    assert len(val_set & test_set) == 0, "Val and test overlap!"
    print("✅ No overlap between splits")
    
    # Check all indices accounted for
    all_indices = train_set | val_set | test_set
    assert len(all_indices) == n_rows, "Not all indices accounted for"
    print("✅ All indices accounted for")
    
    # Test apply_split
    train_df, val_df, test_df = apply_split(df, split_indices)
    
    assert len(train_df) == len(split_indices['train_idx']), "Train df size mismatch"
    assert len(val_df) == len(split_indices['val_idx']), "Val df size mismatch"
    assert len(test_df) == len(split_indices['test_idx']), "Test df size mismatch"
    print("✅ apply_split returns correct sized DataFrames")
    
    return split_indices


def test_compute_split_indices_time_based():
    """Test time-based split strategy."""
    
    print("\n" + "="*70)
    print("TEST 5: compute_split_indices - Time-Based Strategy")
    print("="*70)
    
    clear_registry()
    
    # Create a dataset with dates
    np.random.seed(42)
    n_rows = 1000
    
    df = pd.DataFrame({
        "id": range(n_rows),
        "event_date": pd.date_range("2023-01-01", periods=n_rows, freq="D"),
        "feature1": np.random.randn(n_rows),
        "target": np.random.choice([0, 1], n_rows),
    })
    
    # Shuffle the DataFrame (to test that time-based split still works correctly)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    
    print(f"\n[INPUT] DataFrame with {n_rows} rows, shuffled")
    print(f"  Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    
    from agents.training.label_and_split import compute_split_indices, apply_split
    
    label_def = {
        "target_column": "target",
        "split_strategy": "time_based",
        "as_of_cutoff": "event_date",
        "grain": "one row",
    }
    
    split_indices = compute_split_indices(
        df=df,
        label_definition=label_def,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    # Apply split
    train_df, val_df, test_df = apply_split(df, split_indices)
    
    print("\n[OUTPUT]")
    print(f"  Train: {len(train_df)} rows, dates: {train_df['event_date'].min().date()} to {train_df['event_date'].max().date()}")
    print(f"  Val: {len(val_df)} rows, dates: {val_df['event_date'].min().date()} to {val_df['event_date'].max().date()}")
    print(f"  Test: {len(test_df)} rows, dates: {test_df['event_date'].min().date()} to {test_df['event_date'].max().date()}")
    
    # Validate
    print("\n[VALIDATION]")
    
    # Check that train dates < val dates < test dates (no overlap in time)
    train_max = train_df['event_date'].max()
    val_min = val_df['event_date'].min()
    val_max = val_df['event_date'].max()
    test_min = test_df['event_date'].min()
    
    assert train_max <= val_min, f"Train max ({train_max}) should be <= val min ({val_min})"
    assert val_max <= test_min, f"Val max ({val_max}) should be <= test min ({test_min})"
    print("✅ Temporal ordering: train < val < test")
    
    # Check no overlap
    train_set = set(split_indices['train_idx'])
    val_set = set(split_indices['val_idx'])
    test_set = set(split_indices['test_idx'])
    
    assert len(train_set & val_set) == 0, "Train and val overlap!"
    assert len(train_set & test_set) == 0, "Train and test overlap!"
    assert len(val_set & test_set) == 0, "Val and test overlap!"
    print("✅ No overlap between splits")
    
    return split_indices


def test_compute_split_indices_entity_based():
    """Test entity-based split strategy."""
    
    print("\n" + "="*70)
    print("TEST 6: compute_split_indices - Entity-Based Strategy")
    print("="*70)
    
    clear_registry()
    
    # Create a dataset with multiple rows per entity
    np.random.seed(42)
    n_entities = 200
    rows_per_entity = 5
    
    rows = []
    for entity_id in range(n_entities):
        for _ in range(rows_per_entity):
            rows.append({
                "customer_id": f"CUST_{entity_id:04d}",
                "transaction_id": len(rows),
                "amount": np.random.exponential(100),
                "target": np.random.choice([0, 1]),
            })
    
    df = pd.DataFrame(rows)
    
    print(f"\n[INPUT] DataFrame with {len(df)} rows, {n_entities} unique entities")
    print(f"  Rows per entity: {rows_per_entity}")
    
    from agents.training.label_and_split import compute_split_indices, apply_split
    
    label_def = {
        "target_column": "target",
        "split_strategy": "entity_based",
        "grain": "one customer",  # This should find customer_id
    }
    
    split_indices = compute_split_indices(
        df=df,
        label_definition=label_def,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )
    
    # Apply split
    train_df, val_df, test_df = apply_split(df, split_indices)
    
    print("\n[OUTPUT]")
    print(f"  Train: {len(train_df)} rows, {train_df['customer_id'].nunique()} unique entities")
    print(f"  Val: {len(val_df)} rows, {val_df['customer_id'].nunique()} unique entities")
    print(f"  Test: {len(test_df)} rows, {test_df['customer_id'].nunique()} unique entities")
    
    # Validate
    print("\n[VALIDATION]")
    
    # Check that no entity appears in multiple splits
    train_entities = set(train_df['customer_id'].unique())
    val_entities = set(val_df['customer_id'].unique())
    test_entities = set(test_df['customer_id'].unique())
    
    assert len(train_entities & val_entities) == 0, "Same entity in train and val!"
    assert len(train_entities & test_entities) == 0, "Same entity in train and test!"
    assert len(val_entities & test_entities) == 0, "Same entity in val and test!"
    print("✅ No entity appears in multiple splits")
    
    # Check all entities accounted for
    all_entities = train_entities | val_entities | test_entities
    assert len(all_entities) == n_entities, f"Expected {n_entities} entities, got {len(all_entities)}"
    print("✅ All entities accounted for")
    
    # Check entity counts roughly match ratios
    expected_train_entities = int(n_entities * 0.7)
    assert abs(len(train_entities) - expected_train_entities) <= 20, "Train entity count off"
    print("✅ Entity counts match expected ratios")
    
    return split_indices


def test_add_split_column():
    """Test add_split_column function."""
    
    print("\n" + "="*70)
    print("TEST 7: add_split_column")
    print("="*70)
    
    clear_registry()
    
    # Create a simple dataset
    df = pd.DataFrame({
        "id": range(100),
        "feature": np.random.randn(100),
        "target": np.random.choice([0, 1], 100),
    })
    
    from agents.training.label_and_split import compute_split_indices, add_split_column
    
    label_def = {
        "target_column": "target",
        "split_strategy": "random",
        "grain": "one row",
    }
    
    split_indices = compute_split_indices(df, label_def)
    
    # Add split column
    df_with_split = add_split_column(df, split_indices, column_name="_split")
    
    print("\n[OUTPUT]")
    print(f"  New column added: '_split'")
    print(f"  Value counts:\n{df_with_split['_split'].value_counts()}")
    
    # Validate
    print("\n[VALIDATION]")
    
    assert "_split" in df_with_split.columns, "Split column not added"
    print("✅ Split column added")
    
    assert set(df_with_split['_split'].unique()) == {"train", "val", "test"}, "Invalid split values"
    print("✅ Split values are 'train', 'val', 'test'")
    
    # Check counts match indices
    assert len(df_with_split[df_with_split['_split'] == 'train']) == len(split_indices['train_idx'])
    assert len(df_with_split[df_with_split['_split'] == 'val']) == len(split_indices['val_idx'])
    assert len(df_with_split[df_with_split['_split'] == 'test']) == len(split_indices['test_idx'])
    print("✅ Split counts match indices")
    
    return df_with_split


def test_full_label_split_pipeline():
    """Test the full pipeline: define labels → compute split → apply split."""
    
    print("\n" + "="*70)
    print("TEST 8: Full Label + Split Pipeline")
    print("="*70)
    
    clear_registry()
    
    # Create loan dataset
    np.random.seed(42)
    n_rows = 500
    
    df = pd.DataFrame({
        "loan_id": range(1, n_rows + 1),
        "application_date": pd.date_range("2023-01-01", periods=n_rows, freq="D"),
        "loan_amount": np.random.uniform(5000, 50000, n_rows).round(2),
        "credit_score": np.random.randint(550, 850, n_rows),
        "income": np.random.uniform(30000, 150000, n_rows).round(2),
        "default": np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
    })
    
    dataset_ref = "loan_pipeline_test"
    register_dataset(dataset_ref, df)
    
    print(f"\n[1] Created dataset: {dataset_ref} ({n_rows} rows)")
    
    # Step 1: Define labels
    print("\n[2] Running label/split definition...")
    from agents.training.label_and_split import (
        run_label_split_definition,
        compute_split_indices,
        apply_split,
    )
    
    label_def = run_label_split_definition(
        dataset_ref=dataset_ref,
        goal="Predict loan default",
        target_column="default",
        split_strategy="time_based",
    )
    
    print(f"  Label definition: {json.dumps(label_def, indent=2, default=str)}")
    
    # Step 2: Compute split
    print("\n[3] Computing split indices...")
    split_indices = compute_split_indices(
        df=df,
        label_definition=label_def,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    print(f"  Train: {len(split_indices['train_idx'])} rows")
    print(f"  Val: {len(split_indices['val_idx'])} rows")
    print(f"  Test: {len(split_indices['test_idx'])} rows")
    
    # Step 3: Apply split
    print("\n[4] Applying split...")
    train_df, val_df, test_df = apply_split(df, split_indices)
    
    # Register split datasets
    register_dataset(f"{dataset_ref}_train", train_df)
    register_dataset(f"{dataset_ref}_val", val_df)
    register_dataset(f"{dataset_ref}_test", test_df)
    
    print(f"  Registered: {dataset_ref}_train ({len(train_df)} rows)")
    print(f"  Registered: {dataset_ref}_val ({len(val_df)} rows)")
    print(f"  Registered: {dataset_ref}_test ({len(test_df)} rows)")
    
    # Validate
    print("\n[VALIDATION]")
    
    # Check temporal ordering (since we used time_based)
    if label_def.get("split_strategy") == "time_based":
        train_max = train_df['application_date'].max()
        val_min = val_df['application_date'].min()
        val_max = val_df['application_date'].max()
        test_min = test_df['application_date'].min()
        
        assert train_max <= val_min, "Temporal ordering violated: train vs val"
        assert val_max <= test_min, "Temporal ordering violated: val vs test"
        print("✅ Temporal ordering preserved (train < val < test)")
    
    # Check no data loss
    total_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_rows == n_rows, f"Data loss: expected {n_rows}, got {total_rows}"
    print("✅ No data loss")
    
    # Check target distribution in each split
    print(f"  Train default rate: {train_df['default'].mean():.2%}")
    print(f"  Val default rate: {val_df['default'].mean():.2%}")
    print(f"  Test default rate: {test_df['default'].mean():.2%}")
    
    print("\n" + "="*70)
    print("TEST 8 PASSED ✅")
    print("="*70)
    
    return {
        "label_def": label_def,
        "split_indices": split_indices,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LABEL AND SPLIT DEFINITION TESTS")
    print("="*70)
    
    results = {}
    
    # Original tests (require LLM)
    try:
        test_label_split_auto_infer()
        results["test_1_auto_infer"] = "PASSED"
    except Exception as e:
        results["test_1_auto_infer"] = f"FAILED: {e}"
        print(f"❌ Test 1 failed: {e}")
    
    try:
        test_label_split_partial_user_input()
        results["test_2_partial_input"] = "PASSED"
    except Exception as e:
        results["test_2_partial_input"] = f"FAILED: {e}"
        print(f"❌ Test 2 failed: {e}")
    
    try:
        test_label_split_all_provided()
        results["test_3_all_provided"] = "PASSED"
    except Exception as e:
        results["test_3_all_provided"] = f"FAILED: {e}"
        print(f"❌ Test 3 failed: {e}")
    
    # New splitting tests (no LLM required)
    try:
        test_compute_split_indices_random()
        results["test_4_random_split"] = "PASSED"
    except Exception as e:
        results["test_4_random_split"] = f"FAILED: {e}"
        print(f"❌ Test 4 failed: {e}")
    
    try:
        test_compute_split_indices_time_based()
        results["test_5_time_split"] = "PASSED"
    except Exception as e:
        results["test_5_time_split"] = f"FAILED: {e}"
        print(f"❌ Test 5 failed: {e}")
    
    try:
        test_compute_split_indices_entity_based()
        results["test_6_entity_split"] = "PASSED"
    except Exception as e:
        results["test_6_entity_split"] = f"FAILED: {e}"
        print(f"❌ Test 6 failed: {e}")
    
    try:
        test_add_split_column()
        results["test_7_add_split_column"] = "PASSED"
    except Exception as e:
        results["test_7_add_split_column"] = f"FAILED: {e}"
        print(f"❌ Test 7 failed: {e}")
    
    try:
        test_full_label_split_pipeline()
        results["test_8_full_pipeline"] = "PASSED"
    except Exception as e:
        results["test_8_full_pipeline"] = f"FAILED: {e}"
        print(f"❌ Test 8 failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v == "PASSED")
    total = len(results)
    
    for test_name, status in results.items():
        icon = "✓" if status == "PASSED" else "✗"
        print(f"  {icon} {test_name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("="*70 + "\n")
