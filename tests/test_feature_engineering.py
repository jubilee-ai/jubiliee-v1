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


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "=" * 70)
    print("RUNNING ALL FEATURE ENGINEERING TESTS")
    print("=" * 70)
    
    results = {}
    
    # Test 1
    try:
        test_feature_engineering_loan_default()
        results["test_1_loan_default"] = "PASSED"
    except Exception as e:
        results["test_1_loan_default"] = f"FAILED: {e}"
        print(f"\n❌ Test 1 failed: {e}")
    
    # Test 2
    try:
        test_feature_engineering_insurance_pricing()
        results["test_2_insurance_pricing"] = "PASSED"
    except Exception as e:
        results["test_2_insurance_pricing"] = f"FAILED: {e}"
        print(f"\n❌ Test 2 failed: {e}")
    
    # Test 3
    try:
        test_feature_engineering_forbidden_columns_validation()
        results["test_3_forbidden_columns"] = "PASSED"
    except Exception as e:
        results["test_3_forbidden_columns"] = f"FAILED: {e}"
        print(f"\n❌ Test 3 failed: {e}")
    
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
