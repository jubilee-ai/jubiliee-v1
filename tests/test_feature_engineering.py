"""
Tests for the Feature Engineering Agent.
Run with: python -m pytest tests/test_feature_engineering.py -v -s
Or directly: python tests/test_feature_engineering.py
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

import numpy as np
import pandas as pd
from utils import clear_registry, get_registered_dataset, register_dataset

# Valid operation types in the DSL
VALID_OPS = {"passthrough", "expression", "bin", "one_hot", "ordinal", "group_agg", "rolling", "date_extract", "date_diff"}


def get_source_columns_from_formula(formula: dict) -> set[str]:
    """Extract source columns from a formula operation."""
    op = formula.get("op")
    cols = set()
    
    if op == "passthrough":
        cols.add(formula.get("column", ""))
    elif op == "expression":
        cols.update(formula.get("source_columns", []))
    elif op == "bin":
        cols.add(formula.get("column", ""))
    elif op == "one_hot":
        cols.add(formula.get("column", ""))
    elif op == "ordinal":
        cols.add(formula.get("column", ""))
    elif op == "group_agg":
        cols.add(formula.get("column", ""))
        cols.update(formula.get("group_by", []))
    elif op == "rolling":
        cols.add(formula.get("column", ""))
        cols.add(formula.get("order_by", ""))
        cols.update(formula.get("partition_by") or [])
    elif op == "date_extract":
        cols.add(formula.get("column", ""))
    elif op == "date_diff":
        cols.add(formula.get("start_column", ""))
        cols.add(formula.get("end_column", ""))
    
    # Remove empty strings
    cols.discard("")
    return cols


def test_feature_engineering_loan_default():
    """
    Test 1: Feature engineering for loan default prediction using REAL data.
    
    Uses the Loan_default.csv dataset which has:
    - Real correlations (credit score, income affect default)
    - Multiple categoricals (Education, EmploymentType, etc.)
    - Varying distributions worth investigating
    """
    print("\n" + "=" * 70)
    print("TEST 1: Feature Engineering - Real Loan Default Dataset")
    print("=" * 70)
    
    # Clear registry
    clear_registry()
    
    # Load real loan default dataset
    print("\n[1] Loading real loan default dataset...")
    csv_path = Path(__file__).parent.parent / "datasets" / "csv" / "Loan_default.csv"
    df = pd.read_csv(csv_path)
    
    # Sample to reasonable size for testing
    df = df.sample(n=2000, random_state=42).reset_index(drop=True)
    
    # Rename for clarity
    df = df.rename(columns={"Default": "default", "LoanID": "loan_id"})
    
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Target distribution: {df['default'].value_counts().to_dict()}")
    print(f"   Credit score range: [{df['CreditScore'].min()}, {df['CreditScore'].max()}]")
    print(f"   Income range: [{df['Income'].min():,.0f}, {df['Income'].max():,.0f}]")
    
    # Register dataset
    dataset_ref = "test_loan_default"
    register_dataset(dataset_ref, df)
    print(f"\n[2] Registered dataset as: `{dataset_ref}`")
    
    # Run feature engineering
    print("\n[3] Running feature engineering agent...")
    print("-" * 70)
    
    from agents.training.feature_engineering import run_feature_engineering
    
    result = run_feature_engineering(
        dataset_ref=dataset_ref,
        goal="Predict whether a loan will default",
        target_column="default",
        grain="loan_id",
        task_type="classification",
        forbidden_columns=["loan_id"],  # Only ID is forbidden in real dataset
        max_iterations=15,
    )
    
    # Validate results
    print("\n[4] Validating results...")
    print("-" * 70)
    
    feature_spec = result.get("feature_spec")
    
    # Check that we got a feature_spec
    assert feature_spec is not None, "feature_spec should not be None"
    print("✓ feature_spec is not None")
    
    # Check required keys
    assert "features" in feature_spec, "feature_spec should have 'features' key"
    assert "reasoning" in feature_spec, "feature_spec should have 'reasoning' key"
    print("✓ feature_spec has required keys (features, reasoning)")
    
    # Check features list
    features = feature_spec["features"]
    assert len(features) > 0, "Should have at least one feature"
    print(f"✓ Got {len(features)} features")
    
    # Validate each feature has required fields and valid formula
    for i, feat in enumerate(features):
        assert "name" in feat, f"Feature {i} missing 'name'"
        assert "formula" in feat, f"Feature {i} missing 'formula'"
        assert "grain" in feat, f"Feature {i} missing 'grain'"
        
        formula = feat["formula"]
        assert isinstance(formula, dict), f"Feature {i} formula should be a dict"
        assert "op" in formula, f"Feature {i} formula missing 'op'"
        assert formula["op"] in VALID_OPS, f"Feature {i} has invalid op: {formula['op']}"
    print("✓ All features have valid structured formulas")
    
    # Check that forbidden columns are not used in any formula
    forbidden = {"loan_id"}
    all_source_cols = set()
    for feat in features:
        source_cols = get_source_columns_from_formula(feat["formula"])
        all_source_cols.update(source_cols)
        overlap = source_cols & forbidden
        assert len(overlap) == 0, f"Feature '{feat['name']}' uses forbidden columns: {overlap}"
    print("✓ No forbidden columns used in features")
    
    # Check validation results
    validation = result.get("validation")
    assert validation is not None, "validation should not be None"
    print(f"✓ Validation: {validation['valid_features']}/{validation['total_features']} features valid")
    
    if validation["errors"]:
        print(f"  ⚠ Validation errors: {validation['errors']}")
    if validation["warnings"]:
        print(f"  ⚠ Validation warnings: {validation['warnings']}")
    
    assert validation["valid"], f"Feature spec should be valid. Errors: {validation['errors']}"
    print("✓ All features pass validation (columns exist, no forbidden columns)")
    
    # Print the feature spec
    print("\n[5] Feature Specification:")
    print("-" * 70)
    print(f"Reasoning: {feature_spec.get('reasoning', 'N/A')[:200]}...")
    print(f"\nFeatures ({len(features)}):")
    for feat in features:
        op = feat["formula"]["op"]
        print(f"  - {feat['name']}: op={op}, formula={feat['formula']}")
    
    if feature_spec.get("excluded_columns"):
        print(f"\nExcluded columns: {feature_spec['excluded_columns']}")
    
    print("\n" + "=" * 70)
    print("TEST 1 PASSED ✓")
    print("=" * 70 + "\n")
    
    return result


def test_feature_engineering_insurance_pricing():
    """
    Test 2: Feature engineering with TEMPORAL transaction data.
    
    Tests that the agent handles:
    - Transaction history requiring aggregation
    - Temporal constraints (as_of_cutoff)
    - Potential for rolling window features
    """
    print("\n" + "=" * 70)
    print("TEST 2: Feature Engineering - Customer Transactions (Temporal)")
    print("=" * 70)
    
    # Clear registry
    clear_registry()
    
    # Create customer + transaction dataset
    print("\n[1] Creating customer transaction dataset...")
    np.random.seed(123)
    n_customers = 500
    
    # Customer attributes (static at signup)
    signup_dates = pd.date_range("2022-01-01", periods=n_customers, freq="D")
    
    # Generate transaction history for each customer
    transactions = []
    for cust_id in range(1, n_customers + 1):
        signup = signup_dates[cust_id - 1]
        # Each customer has 5-30 transactions
        n_txns = np.random.randint(5, 30)
        for _ in range(n_txns):
            txn_date = signup + pd.Timedelta(days=np.random.randint(1, 365))
            transactions.append({
                "customer_id": cust_id,
                "transaction_date": txn_date,
                "amount": np.random.exponential(50) + 10,  # Skewed distribution
                "category": np.random.choice(["grocery", "restaurant", "online", "gas", "other"]),
                "is_fraud": 0,
            })
    
    # Add some fraud transactions with different patterns
    fraud_customers = np.random.choice(range(1, n_customers + 1), size=50, replace=False)
    for cust_id in fraud_customers:
        # Fraudulent transactions: higher amounts, clustered in time
        fraud_date = signup_dates[cust_id - 1] + pd.Timedelta(days=np.random.randint(100, 300))
        for i in range(np.random.randint(3, 8)):
            transactions.append({
                "customer_id": cust_id,
                "transaction_date": fraud_date + pd.Timedelta(hours=i * 2),
                "amount": np.random.uniform(200, 1000),  # Higher amounts
                "category": np.random.choice(["online", "other"]),
                "is_fraud": 1,
            })
    
    df = pd.DataFrame(transactions)
    df = df.sort_values(["customer_id", "transaction_date"]).reset_index(drop=True)
    
    # Add customer-level attributes
    customer_attrs = pd.DataFrame({
        "customer_id": range(1, n_customers + 1),
        "signup_date": signup_dates,
        "customer_age": np.random.randint(18, 70, n_customers),
        "credit_limit": np.random.choice([1000, 2500, 5000, 10000, 25000], n_customers),
        "account_type": np.random.choice(["basic", "premium", "platinum"], n_customers, p=[0.6, 0.3, 0.1]),
    })
    
    df = df.merge(customer_attrs, on="customer_id")
    
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"   Amount distribution: mean={df['amount'].mean():.2f}, median={df['amount'].median():.2f}, max={df['amount'].max():.2f}")
    print(f"   Date range: {df['transaction_date'].min().date()} to {df['transaction_date'].max().date()}")
    
    # Register dataset
    dataset_ref = "test_transactions"
    register_dataset(dataset_ref, df)
    print(f"\n[2] Registered dataset as: `{dataset_ref}`")
    
    # Run feature engineering
    print("\n[3] Running feature engineering agent...")
    print("-" * 70)
    
    from agents.training.feature_engineering import run_feature_engineering
    
    result = run_feature_engineering(
        dataset_ref=dataset_ref,
        goal="Predict whether a transaction is fraudulent",
        target_column="is_fraud",
        grain="customer_id",  # Features at customer level
        task_type="classification",
        forbidden_columns=["customer_id"],
        as_of_cutoff="transaction_date",  # Temporal constraint
        max_iterations=15,
    )
    
    # Validate results
    print("\n[4] Validating results...")
    print("-" * 70)
    
    feature_spec = result.get("feature_spec")
    
    assert feature_spec is not None, "feature_spec should not be None"
    print("✓ feature_spec is not None")
    
    features = feature_spec.get("features", [])
    assert len(features) > 0, "Should have at least one feature"
    print(f"✓ Got {len(features)} features")
    
    # Validate all features have structured formulas
    for feat in features:
        assert "formula" in feat, f"Feature {feat.get('name')} missing formula"
        assert isinstance(feat["formula"], dict), "Formula should be a dict"
        assert "op" in feat["formula"], "Formula missing op"
    print("✓ All features have structured formula DSL")
    
    # Check that reasoning exists
    reasoning = feature_spec.get("reasoning", "")
    assert len(reasoning) > 50, "Reasoning should be substantive"
    print(f"✓ Reasoning provided ({len(reasoning)} chars)")
    
    # Check for temporal awareness - with transaction data, agent should consider temporal features
    has_temporal = any(
        feat["formula"]["op"] in ("rolling", "group_agg", "date_extract") or feat.get("as_of_constraint")
        for feat in features
    )
    if has_temporal:
        print("✓ Agent used temporal-aware features (rolling/group_agg/date_extract or as_of_constraint)")
    else:
        print("  Note: Agent did not use temporal features (may be valid choice)")
    
    # Print the feature spec
    print("\n[5] Feature Specification:")
    print("-" * 70)
    print(f"Reasoning: {reasoning[:300]}...")
    print(f"\nFeatures ({len(features)}):")
    for feat in features:
        op = feat["formula"]["op"]
        as_of = feat.get("as_of_constraint") or "none"
        print(f"  - {feat['name']}: op={op}, as_of={as_of}")
    
    print("\n" + "=" * 70)
    print("TEST 2 PASSED ✓")
    print("=" * 70 + "\n")
    
    return result


def test_feature_engineering_forbidden_columns_validation():
    """
    Test 3: Validate that forbidden columns are properly excluded.
    
    Creates a dataset with obvious leaky features and verifies
    the agent identifies and excludes them.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Feature Engineering - Forbidden Columns Validation")
    print("=" * 70)
    
    # Clear registry
    clear_registry()
    
    # Create a dataset with intentionally leaky columns
    print("\n[1] Creating dataset with leaky columns...")
    np.random.seed(456)
    n_rows = 150
    
    # The target
    churn = np.random.choice([0, 1], n_rows, p=[0.7, 0.3])
    
    df = pd.DataFrame({
        "customer_id": range(1, n_rows + 1),
        # Good features
        "tenure_months": np.random.randint(1, 72, n_rows),
        "monthly_charges": np.random.uniform(20, 100, n_rows),
        "total_charges": np.random.uniform(100, 5000, n_rows),
        "contract_type": np.random.choice(["month-to-month", "one_year", "two_year"], n_rows),
        "payment_method": np.random.choice(["credit_card", "bank_transfer", "check"], n_rows),
        "num_support_tickets": np.random.randint(0, 10, n_rows),
        # Target
        "churned": churn,
        # LEAKY: These are derived from the outcome
        "churn_date": [pd.Timestamp("2024-06-01") if c == 1 else pd.NaT for c in churn],
        "exit_reason": [np.random.choice(["price", "service", "competitor"]) if c == 1 else None for c in churn],
        "final_invoice_amount": [np.random.uniform(0, 200) if c == 1 else None for c in churn],
    })
    
    print(f"   Shape: {df.shape}")
    print(f"   All columns: {list(df.columns)}")
    print(f"   Leaky columns: churn_date, exit_reason, final_invoice_amount")
    
    # Register dataset
    dataset_ref = "test_churn_leaky"
    register_dataset(dataset_ref, df)
    print(f"\n[2] Registered dataset as: `{dataset_ref}`")
    
    # Run feature engineering with forbidden columns
    print("\n[3] Running feature engineering agent with forbidden columns...")
    print("-" * 70)
    
    from agents.training.feature_engineering import run_feature_engineering
    
    forbidden = ["customer_id", "churn_date", "exit_reason", "final_invoice_amount"]
    
    result = run_feature_engineering(
        dataset_ref=dataset_ref,
        goal="Predict customer churn before it happens",
        target_column="churned",
        grain="customer_id",
        task_type="classification",
        forbidden_columns=forbidden,
        max_iterations=20,
    )
    
    # Validate results
    print("\n[4] Validating forbidden columns are excluded...")
    print("-" * 70)
    
    feature_spec = result.get("feature_spec")
    assert feature_spec is not None, "feature_spec should not be None"
    
    features = feature_spec.get("features", [])
    assert len(features) > 0, "Should have at least one feature"
    
    # Collect all source columns used
    all_source_cols = set()
    for feat in features:
        source_cols = get_source_columns_from_formula(feat["formula"])
        all_source_cols.update(source_cols)
    
    print(f"   All source columns used: {all_source_cols}")
    print(f"   Forbidden columns: {set(forbidden)}")
    
    # Check no forbidden columns are used
    forbidden_set = set(forbidden)
    used_forbidden = all_source_cols & forbidden_set
    
    assert len(used_forbidden) == 0, f"FAIL: Forbidden columns used: {used_forbidden}"
    print("✓ No forbidden columns used in any feature")
    
    # Check that the agent used valid columns
    valid_features = {"tenure_months", "monthly_charges", "total_charges", 
                      "contract_type", "payment_method", "num_support_tickets"}
    used_valid = all_source_cols & valid_features
    assert len(used_valid) > 0, "Should use at least some valid features"
    print(f"✓ Valid features used: {used_valid}")
    
    # Check excluded_columns in output
    excluded = set(feature_spec.get("excluded_columns", []))
    print(f"   Agent reported excluded: {excluded}")
    
    # Validate all formulas are structured
    for feat in features:
        assert "formula" in feat, f"Feature {feat.get('name')} missing formula"
        assert isinstance(feat["formula"], dict), "Formula should be a structured dict"
        assert feat["formula"].get("op") in VALID_OPS, f"Invalid op: {feat['formula'].get('op')}"
    print("✓ All features have valid structured formulas")
    
    # Print summary
    print("\n[5] Feature Specification Summary:")
    print("-" * 70)
    print(f"Features selected: {len(features)}")
    for feat in features:
        op = feat["formula"]["op"]
        cols = get_source_columns_from_formula(feat["formula"])
        print(f"  - {feat['name']}: op={op}, sources={cols}")
    
    print("\n" + "=" * 70)
    print("TEST 3 PASSED ✓")
    print("=" * 70 + "\n")
    
    return result


# =============================================================================
# SPLIT-AWARE FEATURE ENGINEERING TESTS
# =============================================================================

def test_execute_feature_spec_split_binning():
    """
    Test 4: Feature executor with split data - binning should fit on train.
    
    Verifies that bin edges are computed from training data only and
    applied consistently to val/test.
    """
    print("\n" + "=" * 70)
    print("TEST 4: execute_feature_spec_split - Binning (Fit on Train)")
    print("=" * 70)
    
    clear_registry()
    
    # Create datasets with different distributions
    np.random.seed(42)
    
    # Training data: values 0-100
    train_df = pd.DataFrame({
        "id": range(100),
        "amount": np.random.uniform(0, 100, 100),
        "target": np.random.choice([0, 1], 100),
    })
    
    # Validation data: same range
    val_df = pd.DataFrame({
        "id": range(100, 150),
        "amount": np.random.uniform(0, 100, 50),
        "target": np.random.choice([0, 1], 50),
    })
    
    # Test data: includes values OUTSIDE training range (0-150)
    test_df = pd.DataFrame({
        "id": range(150, 200),
        "amount": np.random.uniform(50, 150, 50),  # Some values > 100!
        "target": np.random.choice([0, 1], 50),
    })
    
    print(f"\n[INPUT]")
    print(f"  Train: {len(train_df)} rows, amount range: [{train_df['amount'].min():.1f}, {train_df['amount'].max():.1f}]")
    print(f"  Val: {len(val_df)} rows, amount range: [{val_df['amount'].min():.1f}, {val_df['amount'].max():.1f}]")
    print(f"  Test: {len(test_df)} rows, amount range: [{test_df['amount'].min():.1f}, {test_df['amount'].max():.1f}]")
    print(f"  Note: Test has values > 100, outside training range!")
    
    # Register datasets
    register_dataset("test_train", train_df)
    register_dataset("test_val", val_df)
    register_dataset("test_test", test_df)
    
    # Create feature spec with binning
    feature_spec = {
        "features": [
            {
                "name": "amount_binned",
                "formula": {
                    "op": "bin",
                    "column": "amount",
                    "bins": 5,
                    "strategy": "quantile",
                },
                "grain": "id",
            },
        ],
        "reasoning": "Test binning",
    }
    
    print("\n[RUNNING execute_feature_spec_split...]")
    
    from agents.training.feature_engineering_executor import \
        execute_feature_spec_split
    
    result = execute_feature_spec_split(
        train_ref="test_train",
        val_ref="test_val",
        test_ref="test_test",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    print("\n[OUTPUT]")
    print(f"  train_ref: {result['train_ref']}")
    print(f"  val_ref: {result['val_ref']}")
    print(f"  test_ref: {result['test_ref']}")
    print(f"  features_created: {result['features_created']}")
    print(f"  errors: {result['errors']}")
    
    # Validate
    print("\n[VALIDATION]")
    
    # Load transformed datasets
    train_transformed = get_registered_dataset(result['train_ref'])
    val_transformed = get_registered_dataset(result['val_ref'])
    test_transformed = get_registered_dataset(result['test_ref'])
    
    assert "amount_binned" in train_transformed.columns, "Binned column not in train"
    assert "amount_binned" in val_transformed.columns, "Binned column not in val"
    assert "amount_binned" in test_transformed.columns, "Binned column not in test"
    print("✅ Binned column created in all splits")
    
    # Check that test values outside train range still get binned (to edge bins)
    assert test_transformed["amount_binned"].notna().all(), "Test should have no NaN bins (edges extended)"
    print("✅ Test values outside train range handled correctly")
    
    # Check no errors
    assert len(result['errors']) == 0, f"Errors: {result['errors']}"
    print("✅ No errors")
    
    print("\n" + "=" * 70)
    print("TEST 4 PASSED ✓")
    print("=" * 70 + "\n")
    
    return result


def test_execute_feature_spec_split_one_hot():
    """
    Test 5: Feature executor with split data - one-hot should use train categories.
    
    Verifies that categories are learned from training data and unseen
    categories in val/test are handled gracefully.
    """
    print("\n" + "=" * 70)
    print("TEST 5: execute_feature_spec_split - One-Hot (Train Categories)")
    print("=" * 70)
    
    clear_registry()
    
    # Training data: has categories A, B, C
    train_df = pd.DataFrame({
        "id": range(100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "target": np.random.choice([0, 1], 100),
    })
    
    # Validation data: same categories
    val_df = pd.DataFrame({
        "id": range(100, 150),
        "category": np.random.choice(["A", "B", "C"], 50),
        "target": np.random.choice([0, 1], 50),
    })
    
    # Test data: includes category D (unseen in train!)
    test_df = pd.DataFrame({
        "id": range(150, 200),
        "category": np.random.choice(["A", "B", "C", "D"], 50),  # D is new!
        "target": np.random.choice([0, 1], 50),
    })
    
    print(f"\n[INPUT]")
    print(f"  Train categories: {sorted(train_df['category'].unique())}")
    print(f"  Val categories: {sorted(val_df['category'].unique())}")
    print(f"  Test categories: {sorted(test_df['category'].unique())} (includes 'D' unseen in train!)")
    
    # Register datasets
    register_dataset("onehot_train", train_df)
    register_dataset("onehot_val", val_df)
    register_dataset("onehot_test", test_df)
    
    # Create feature spec with one-hot
    feature_spec = {
        "features": [
            {
                "name": "cat",
                "formula": {
                    "op": "one_hot",
                    "column": "category",
                    "drop_first": True,
                },
                "grain": "id",
            },
        ],
        "reasoning": "Test one-hot",
    }
    
    print("\n[RUNNING execute_feature_spec_split...]")
    
    from agents.training.feature_engineering_executor import \
        execute_feature_spec_split
    
    result = execute_feature_spec_split(
        train_ref="onehot_train",
        val_ref="onehot_val",
        test_ref="onehot_test",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    print("\n[OUTPUT]")
    print(f"  features_created: {result['features_created']}")
    
    # Validate
    print("\n[VALIDATION]")
    
    # Load transformed datasets
    train_transformed = get_registered_dataset(result['train_ref'])
    val_transformed = get_registered_dataset(result['val_ref'])
    test_transformed = get_registered_dataset(result['test_ref'])
    
    # Get one-hot columns
    train_oh_cols = [c for c in train_transformed.columns if c.startswith("cat_")]
    val_oh_cols = [c for c in val_transformed.columns if c.startswith("cat_")]
    test_oh_cols = [c for c in test_transformed.columns if c.startswith("cat_")]
    
    print(f"  Train one-hot columns: {train_oh_cols}")
    print(f"  Val one-hot columns: {val_oh_cols}")
    print(f"  Test one-hot columns: {test_oh_cols}")
    
    # Check same columns in all splits
    assert set(train_oh_cols) == set(val_oh_cols), "Val should have same columns as train"
    assert set(train_oh_cols) == set(test_oh_cols), "Test should have same columns as train"
    print("✅ Same one-hot columns in all splits")
    
    # Check that 'D' (unseen category) did NOT create a new column
    assert "cat_D" not in test_oh_cols, "Unseen category D should NOT create new column"
    print("✅ Unseen category 'D' handled correctly (no new column)")
    
    print("\n" + "=" * 70)
    print("TEST 5 PASSED ✓")
    print("=" * 70 + "\n")
    
    return result


def test_execute_feature_spec_split_group_agg():
    """
    Test 6: Feature executor with split data - group_agg should use train stats.
    
    Verifies that group aggregations are computed from training data only
    and merged to val/test.
    """
    print("\n" + "=" * 70)
    print("TEST 6: execute_feature_spec_split - Group Agg (Train Stats)")
    print("=" * 70)
    
    clear_registry()
    
    np.random.seed(42)
    
    # Training data: region means are known
    train_df = pd.DataFrame({
        "id": range(300),
        "region": np.random.choice(["East", "West", "North"], 300),
        "amount": np.random.exponential(100, 300),
        "target": np.random.choice([0, 1], 300),
    })
    
    # Validation data: same regions
    val_df = pd.DataFrame({
        "id": range(300, 400),
        "region": np.random.choice(["East", "West", "North"], 100),
        "amount": np.random.exponential(100, 100),
        "target": np.random.choice([0, 1], 100),
    })
    
    # Test data: includes region "South" (unseen in train!)
    test_df = pd.DataFrame({
        "id": range(400, 500),
        "region": np.random.choice(["East", "West", "North", "South"], 100),
        "amount": np.random.exponential(100, 100),
        "target": np.random.choice([0, 1], 100),
    })
    
    # Compute expected means from training data
    train_means = train_df.groupby("region")["amount"].mean()
    
    print(f"\n[INPUT]")
    print(f"  Train region means: {train_means.to_dict()}")
    print(f"  Test includes 'South' region (unseen in train)")
    
    # Register datasets
    register_dataset("agg_train", train_df)
    register_dataset("agg_val", val_df)
    register_dataset("agg_test", test_df)
    
    # Create feature spec with group_agg
    feature_spec = {
        "features": [
            {
                "name": "region_avg_amount",
                "formula": {
                    "op": "group_agg",
                    "column": "amount",
                    "agg": "mean",
                    "group_by": ["region"],
                },
                "grain": "id",
            },
        ],
        "reasoning": "Test group agg",
    }
    
    print("\n[RUNNING execute_feature_spec_split...]")
    
    from agents.training.feature_engineering_executor import \
        execute_feature_spec_split
    
    result = execute_feature_spec_split(
        train_ref="agg_train",
        val_ref="agg_val",
        test_ref="agg_test",
        feature_spec=feature_spec,
        target_column="target",
        grain="id",
    )
    
    print("\n[OUTPUT]")
    print(f"  features_created: {result['features_created']}")
    
    # Validate
    print("\n[VALIDATION]")
    
    # Load transformed datasets
    train_transformed = get_registered_dataset(result['train_ref'])
    val_transformed = get_registered_dataset(result['val_ref'])
    test_transformed = get_registered_dataset(result['test_ref'])
    
    assert "region_avg_amount" in train_transformed.columns, "Feature not in train"
    assert "region_avg_amount" in val_transformed.columns, "Feature not in val"
    assert "region_avg_amount" in test_transformed.columns, "Feature not in test"
    print("✅ Group agg feature created in all splits")
    
    # Check that val uses train means by looking up the original val_df to find East rows
    # Then check that their region_avg_amount matches the train mean
    val_east_rows = val_df[val_df["region"] == "East"].index
    if len(val_east_rows) > 0:
        # Get the first East row's position in the val split
        first_east_idx = val_east_rows[0]
        # Find this in val_transformed - use the id column to match
        val_east_id = val_df.loc[first_east_idx, "id"]
        val_east_row = val_transformed[val_transformed["id"] == val_east_id]
        if len(val_east_row) > 0:
            val_east_mean = val_east_row["region_avg_amount"].iloc[0]
            expected_east_mean = train_means["East"]
            
            assert abs(val_east_mean - expected_east_mean) < 0.01, \
                f"Val should use train mean for East: got {val_east_mean}, expected {expected_east_mean}"
            print(f"✅ Val uses train means (East: {val_east_mean:.2f} == {expected_east_mean:.2f})")
    
    # Check that unseen region "South" gets NaN by looking up test_df
    south_ids = test_df[test_df["region"] == "South"]["id"].tolist()
    if south_ids:
        south_rows_transformed = test_transformed[test_transformed["id"].isin(south_ids)]
        if len(south_rows_transformed) > 0:
            assert south_rows_transformed["region_avg_amount"].isna().all(), \
                "Unseen region 'South' should have NaN (no train data)"
            print("✅ Unseen region 'South' correctly has NaN values")
    
    print("\n" + "=" * 70)
    print("TEST 6 PASSED ✓")
    print("=" * 70 + "\n")
    
    return result


def test_feature_engineering_simple_with_split():
    """
    Test 7: Full feature engineering simple with split data (requires LLM).
    
    Tests that run_feature_engineering_simple correctly accepts split refs
    and runs analysis only on training data.
    """
    print("\n" + "=" * 70)
    print("TEST 7: run_feature_engineering_simple with Split Data")
    print("=" * 70)
    
    clear_registry()
    
    # Create and split a dataset
    np.random.seed(42)
    
    # Load real loan default dataset
    csv_path = Path(__file__).parent.parent / "datasets" / "csv" / "Loan_default.csv"
    df = pd.read_csv(csv_path)
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    df = df.rename(columns={"Default": "default", "LoanID": "loan_id"})
    
    # Split the data
    from agents.training.label_and_split import (apply_split,
                                                 compute_split_indices)
    
    label_def = {
        "target_column": "default",
        "split_strategy": "random",
        "grain": "one loan",
    }
    
    split_indices = compute_split_indices(df, label_def, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df = apply_split(df, split_indices)
    
    print(f"\n[INPUT]")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    
    # Register datasets
    register_dataset("fe_train", train_df)
    register_dataset("fe_val", val_df)
    register_dataset("fe_test", test_df)
    
    print("\n[RUNNING run_feature_engineering_simple with split data...]")
    
    from agents.training.feature_engineering_simple import \
        run_feature_engineering_simple
    
    result = run_feature_engineering_simple(
        train_ref="fe_train",
        goal="Predict loan default",
        target_column="default",
        grain="loan_id",
        val_ref="fe_val",
        test_ref="fe_test",
        task_type="classification",
        forbidden_columns=["loan_id"],
    )
    
    print("\n[OUTPUT]")
    print(f"  feature_spec: {len(result['feature_spec'].get('features', []))} features")
    print(f"  dataset_refs: {result['dataset_refs']}")
    
    # Validate
    print("\n[VALIDATION]")
    
    feature_spec = result.get("feature_spec")
    assert feature_spec is not None, "feature_spec should not be None"
    print("✅ feature_spec generated")
    
    assert result["dataset_refs"]["train"] == "fe_train", "Train ref not preserved"
    assert result["dataset_refs"]["val"] == "fe_val", "Val ref not preserved"
    assert result["dataset_refs"]["test"] == "fe_test", "Test ref not preserved"
    print("✅ Dataset refs preserved in output")
    
    features = feature_spec.get("features", [])
    assert len(features) > 0, "Should have at least one feature"
    print(f"✅ Got {len(features)} features")
    
    # Check all features have valid formulas
    for feat in features:
        assert "formula" in feat, f"Feature {feat.get('name')} missing formula"
        assert feat["formula"].get("op") in VALID_OPS, f"Invalid op: {feat['formula'].get('op')}"
    print("✅ All features have valid structured formulas")
    
    print("\n" + "=" * 70)
    print("TEST 7 PASSED ✓")
    print("=" * 70 + "\n")
    
    return result


def test_full_split_feature_pipeline():
    """
    Test 8: Full pipeline - split → feature spec → execute with split.
    
    Integration test for the complete workflow.
    """
    print("\n" + "=" * 70)
    print("TEST 8: Full Split + Feature Engineering Pipeline")
    print("=" * 70)
    
    clear_registry()
    
    # Create dataset
    np.random.seed(42)
    n_rows = 500
    
    df = pd.DataFrame({
        "customer_id": range(n_rows),
        "age": np.random.randint(18, 70, n_rows),
        "income": np.random.exponential(50000, n_rows),
        "region": np.random.choice(["East", "West", "North", "South"], n_rows),
        "tenure_months": np.random.randint(1, 120, n_rows),
        "churned": np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
    })
    
    print(f"\n[1] Created dataset with {len(df)} rows")
    
    # Step 1: Split the data
    print("\n[2] Splitting data...")
    from agents.training.label_and_split import (apply_split,
                                                 compute_split_indices)
    
    label_def = {
        "target_column": "churned",
        "split_strategy": "random",
        "grain": "one customer",
    }
    
    split_indices = compute_split_indices(df, label_def)
    train_df, val_df, test_df = apply_split(df, split_indices)
    
    register_dataset("pipeline_train", train_df)
    register_dataset("pipeline_val", val_df)
    register_dataset("pipeline_test", test_df)
    
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Step 2: Create a feature spec (manually for this test, to avoid LLM)
    print("\n[3] Creating feature spec...")
    feature_spec = {
        "features": [
            {
                "name": "age",
                "formula": {"op": "passthrough", "column": "age"},
                "grain": "customer_id",
            },
            {
                "name": "income_binned",
                "formula": {"op": "bin", "column": "income", "bins": 4, "strategy": "quantile"},
                "grain": "customer_id",
            },
            {
                "name": "region_encoded",
                "formula": {"op": "one_hot", "column": "region", "drop_first": True},
                "grain": "customer_id",
            },
            {
                "name": "region_avg_income",
                "formula": {"op": "group_agg", "column": "income", "agg": "mean", "group_by": ["region"]},
                "grain": "customer_id",
            },
        ],
        "reasoning": "Test pipeline",
    }
    
    print(f"  Feature spec: {len(feature_spec['features'])} features")
    
    # Step 3: Execute feature spec with split
    print("\n[4] Executing feature spec with split data...")
    from agents.training.feature_engineering_executor import \
        execute_feature_spec_split
    
    result = execute_feature_spec_split(
        train_ref="pipeline_train",
        val_ref="pipeline_val",
        test_ref="pipeline_test",
        feature_spec=feature_spec,
        target_column="churned",
        grain="customer_id",
    )
    
    print(f"  Train output: {result['train_ref']}, shape: {result['shapes']['train']}")
    print(f"  Val output: {result['val_ref']}, shape: {result['shapes']['val']}")
    print(f"  Test output: {result['test_ref']}, shape: {result['shapes']['test']}")
    print(f"  Features created: {result['features_created']}")
    print(f"  Errors: {result['errors']}")
    
    # Validate
    print("\n[VALIDATION]")
    
    train_out = get_registered_dataset(result['train_ref'])
    val_out = get_registered_dataset(result['val_ref'])
    test_out = get_registered_dataset(result['test_ref'])
    
    # Check all expected columns present
    expected_cols = {"customer_id", "churned", "age", "income_binned", "region_avg_income"}
    # Plus one-hot columns
    oh_cols = [c for c in train_out.columns if c.startswith("region_encoded_")]
    
    assert expected_cols.issubset(set(train_out.columns)), f"Missing columns in train: {expected_cols - set(train_out.columns)}"
    assert len(oh_cols) > 0, "No one-hot columns created"
    print(f"✅ All expected columns present (including {len(oh_cols)} one-hot columns)")
    
    # Check same columns across splits
    assert set(train_out.columns) == set(val_out.columns), "Val has different columns than train"
    assert set(train_out.columns) == set(test_out.columns), "Test has different columns than train"
    print("✅ Same columns across all splits")
    
    # Check no errors
    assert len(result['errors']) == 0, f"Errors: {result['errors']}"
    print("✅ No errors in execution")
    
    # Check data integrity
    assert len(train_out) == len(train_df), "Train row count changed"
    assert len(val_out) == len(val_df), "Val row count changed"
    assert len(test_out) == len(test_df), "Test row count changed"
    print("✅ Row counts preserved")
    
    print("\n" + "=" * 70)
    print("TEST 8 PASSED ✓")
    print("=" * 70 + "\n")
    
    return result


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "=" * 70)
    print("RUNNING ALL FEATURE ENGINEERING TESTS")
    print("=" * 70)
    
    results = {}
    
    # Original tests (require LLM)
    try:
        test_feature_engineering_loan_default()
        results["test_1_loan_default"] = "PASSED"
    except Exception as e:
        results["test_1_loan_default"] = f"FAILED: {e}"
        print(f"\n❌ Test 1 failed: {e}")
    
    try:
        test_feature_engineering_insurance_pricing()
        results["test_2_insurance_pricing"] = "PASSED"
    except Exception as e:
        results["test_2_insurance_pricing"] = f"FAILED: {e}"
        print(f"\n❌ Test 2 failed: {e}")
    
    try:
        test_feature_engineering_forbidden_columns_validation()
        results["test_3_forbidden_columns"] = "PASSED"
    except Exception as e:
        results["test_3_forbidden_columns"] = f"FAILED: {e}"
        print(f"\n❌ Test 3 failed: {e}")
    
    # New split-aware tests (tests 4-6 don't require LLM)
    try:
        test_execute_feature_spec_split_binning()
        results["test_4_split_binning"] = "PASSED"
    except Exception as e:
        results["test_4_split_binning"] = f"FAILED: {e}"
        print(f"\n❌ Test 4 failed: {e}")
    
    try:
        test_execute_feature_spec_split_one_hot()
        results["test_5_split_one_hot"] = "PASSED"
    except Exception as e:
        results["test_5_split_one_hot"] = f"FAILED: {e}"
        print(f"\n❌ Test 5 failed: {e}")
    
    try:
        test_execute_feature_spec_split_group_agg()
        results["test_6_split_group_agg"] = "PASSED"
    except Exception as e:
        results["test_6_split_group_agg"] = f"FAILED: {e}"
        print(f"\n❌ Test 6 failed: {e}")
    
    # Test 7 requires LLM
    try:
        test_feature_engineering_simple_with_split()
        results["test_7_simple_with_split"] = "PASSED"
    except Exception as e:
        results["test_7_simple_with_split"] = f"FAILED: {e}"
        print(f"\n❌ Test 7 failed: {e}")
    
    # Test 8 doesn't require LLM
    try:
        test_full_split_feature_pipeline()
        results["test_8_full_pipeline"] = "PASSED"
    except Exception as e:
        results["test_8_full_pipeline"] = f"FAILED: {e}"
        print(f"\n❌ Test 8 failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v == "PASSED")
    total = len(results)
    
    for test_name, status in results.items():
        icon = "✓" if status == "PASSED" else "✗"
        print(f"  {icon} {test_name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 70 + "\n")
    
    return results


if __name__ == "__main__":
    import sys

    # Check if running a specific test
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "1" or test_name == "loan":
            test_feature_engineering_loan_default()
        elif test_name == "2" or test_name == "insurance":
            test_feature_engineering_insurance_pricing()
        elif test_name == "3" or test_name == "forbidden":
            test_feature_engineering_forbidden_columns_validation()
        else:
            print(f"Unknown test: {test_name}")
            print("Usage: python test_feature_engineering.py [1|2|3|loan|insurance|forbidden]")
    else:
        # Run all tests
        run_all_tests()
