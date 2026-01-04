"""
Comprehensive tests for data_validation tool.

Tests cover:
- All 6 rule types with edge cases
- Realistic financial/insurance datasets
- Large datasets with sampling
- Severity ranking and fix suggestions
- Tool output format for agent consumption
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from analysis.data_validation import data_validation_tool, validate_dataset
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
def loan_dataset():
    """Realistic loan dataset with various data quality issues."""
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        # IDs - some duplicates
        "loan_id": [f"LOAN-{i:05d}" for i in range(n)],
        "customer_id": np.random.choice([f"CUST-{i:04d}" for i in range(800)], n),
        
        # Numeric - some out of range
        "age": np.clip(np.random.normal(45, 15, n), 18, 100).astype(int),
        "income": np.random.lognormal(10.5, 0.8, n),
        "loan_amount": np.random.uniform(5000, 500000, n),
        "credit_score": np.random.normal(680, 80, n).astype(int),
        "dti_ratio": np.random.uniform(0, 0.8, n),
        
        # Categorical - some invalid values
        "loan_status": np.random.choice(
            ["active", "closed", "default", "PENDING", "unknown", ""],
            n, p=[0.4, 0.3, 0.1, 0.1, 0.05, 0.05]
        ),
        "employment_type": np.random.choice(
            ["full_time", "part_time", "self_employed", "retired", "N/A"],
            n, p=[0.5, 0.2, 0.15, 0.1, 0.05]
        ),
        
        # Dates as strings - some malformed
        "application_date": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        
        # Contact info - for regex testing
        "email": [
            f"user{i}@example.com" if np.random.random() > 0.1 
            else f"invalid-email-{i}" 
            for i in range(n)
        ],
        "phone": [
            f"555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}"
            if np.random.random() > 0.15
            else "INVALID"
            for i in range(n)
        ],
    })
    
    # Inject specific issues
    # 1. Some negative ages (data entry errors)
    df.loc[df.index[:5], "age"] = [-1, -5, 0, 150, 200]
    
    # 2. Some extreme credit scores
    df.loc[df.index[10:15], "credit_score"] = [900, 950, 100, 50, 1000]
    
    # 3. Add nulls
    null_idx = np.random.choice(n, size=int(n * 0.08), replace=False)
    df.loc[null_idx, "income"] = np.nan
    
    null_idx2 = np.random.choice(n, size=int(n * 0.12), replace=False)
    df.loc[null_idx2, "credit_score"] = np.nan
    
    null_idx3 = np.random.choice(n, size=int(n * 0.03), replace=False)
    df.loc[null_idx3, "age"] = np.nan
    
    # 4. Duplicate loan IDs
    df.loc[df.index[100:110], "loan_id"] = "LOAN-00001"
    
    return df


@pytest.fixture
def insurance_dataset():
    """Insurance claims dataset with quality issues."""
    np.random.seed(123)
    n = 500
    
    df = pd.DataFrame({
        "claim_id": [f"CLM{i:06d}" for i in range(n)],
        "policy_holder_age": np.random.randint(18, 85, n),
        "claim_amount": np.random.exponential(5000, n),
        "bmi": np.random.normal(28, 6, n),
        "smoker": np.random.choice(["yes", "no", "unknown", "Y", "N"], n, p=[0.2, 0.6, 0.1, 0.05, 0.05]),
        "region": np.random.choice(["northeast", "northwest", "southeast", "southwest", "INVALID"], n),
        "children": np.random.choice([0, 1, 2, 3, 4, 5, -1, 10], n, p=[0.2, 0.25, 0.25, 0.15, 0.08, 0.03, 0.02, 0.02]),
    })
    
    # Add nulls
    df.loc[df.index[:25], "bmi"] = np.nan
    df.loc[df.index[50:75], "claim_amount"] = np.nan
    
    return df


class TestNotNullRule:
    """Test not_null validation rule."""
    
    def test_no_nulls_passes(self):
        """Dataset with no nulls should pass."""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "income": [50000, 60000, 70000],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "not_null", "columns": ["age", "income"]}],
        )
        
        print(f"\nNO NULLS: {result}")
        assert result["passed"] is True
        assert len(result["violations"]) == 0
    
    def test_nulls_detected(self):
        """Dataset with nulls should fail and suggest impute."""
        df = pd.DataFrame({
            "age": [25, 30, None, 35, None],
            "income": [50000, None, None, 70000, 80000],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "not_null", "columns": ["age", "income"]}],
        )
        
        print(f"\nNULLS DETECTED: {result}")
        assert result["passed"] is False
        assert len(result["violations"]) == 2
        
        # Check suggestions
        for v in result["violations"]:
            assert v["suggested_fix"]["action"] in ["impute", "drop_column"]
    
    def test_threshold_allows_some_nulls(self):
        """Nulls below threshold should pass."""
        df = pd.DataFrame({
            "age": [25, 30, None, 35, 40, 45, 50, 55, 60, 65],  # 10% null
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "not_null", "columns": ["age"]}],
            thresholds={"max_null_pct": 0.15},  # Allow 15%
        )
        
        print(f"\nTHRESHOLD: {result}")
        assert result["passed"] is True


class TestRangeRule:
    """Test range validation rule."""
    
    def test_valid_range_passes(self):
        """Values within range should pass."""
        df = pd.DataFrame({
            "age": [25, 30, 65, 80, 18],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "range", "column": "age", "min": 0, "max": 120}],
        )
        
        print(f"\nVALID RANGE: {result}")
        assert result["passed"] is True
    
    def test_out_of_range_detected(self):
        """Values outside range should fail and suggest clip."""
        df = pd.DataFrame({
            "age": [25, 150, -5, 30, 999],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "range", "column": "age", "min": 0, "max": 120}],
        )
        
        print(f"\nOUT OF RANGE: {result}")
        assert result["passed"] is False
        assert len(result["violations"]) == 1
        
        v = result["violations"][0]
        assert v["rule"] == "range"
        assert v["count"] == 3  # 150, -5, 999
        assert v["suggested_fix"]["action"] == "clip"
        assert v["suggested_fix"]["params"]["min"] == 0
        assert v["suggested_fix"]["params"]["max"] == 120


class TestUniqueRule:
    """Test unique validation rule."""
    
    def test_unique_passes(self):
        """Unique column should pass."""
        df = pd.DataFrame({
            "customer_id": ["A", "B", "C", "D"],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "unique", "column": "customer_id"}],
        )
        
        print(f"\nUNIQUE: {result}")
        assert result["passed"] is True
    
    def test_duplicates_detected(self):
        """Duplicate values should fail and suggest dedupe."""
        df = pd.DataFrame({
            "customer_id": ["A", "B", "A", "C", "B"],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "unique", "column": "customer_id"}],
        )
        
        print(f"\nDUPLICATES: {result}")
        assert result["passed"] is False
        assert result["violations"][0]["suggested_fix"]["action"] == "dedupe"


class TestDtypeRule:
    """Test dtype validation rule."""
    
    def test_correct_dtype_passes(self):
        """Correct dtype should pass."""
        df = pd.DataFrame({
            "age": [25, 30, 35],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "dtype", "column": "age", "expected": "int"}],
        )
        
        print(f"\nCORRECT DTYPE: {result}")
        assert result["passed"] is True
    
    def test_wrong_dtype_detected(self):
        """Wrong dtype should fail and suggest cast."""
        df = pd.DataFrame({
            "age": ["25", "30", "35"],  # String, not int
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "dtype", "column": "age", "expected": "int"}],
        )
        
        print(f"\nWRONG DTYPE: {result}")
        assert result["passed"] is False
        assert result["violations"][0]["suggested_fix"]["action"] == "cast"


class TestAllowedValuesRule:
    """Test allowed_values validation rule."""
    
    def test_valid_values_pass(self):
        """Valid values should pass."""
        df = pd.DataFrame({
            "status": ["active", "closed", "active"],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "allowed_values", "column": "status", "values": ["active", "closed"]}],
        )
        
        print(f"\nVALID VALUES: {result}")
        assert result["passed"] is True
    
    def test_invalid_values_detected(self):
        """Invalid values should fail."""
        df = pd.DataFrame({
            "status": ["active", "closed", "pending", "unknown"],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "allowed_values", "column": "status", "values": ["active", "closed"]}],
        )
        
        print(f"\nINVALID VALUES: {result}")
        assert result["passed"] is False
        assert result["violations"][0]["count"] == 2  # pending, unknown


class TestRegexRule:
    """Test regex validation rule."""
    
    def test_valid_pattern_passes(self):
        """Valid email pattern should pass."""
        df = pd.DataFrame({
            "email": ["a@b.com", "test@example.org"],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "regex", "column": "email", "pattern": r"^.+@.+\..+$"}],
        )
        
        print(f"\nVALID REGEX: {result}")
        assert result["passed"] is True
    
    def test_invalid_pattern_detected(self):
        """Invalid emails should fail."""
        df = pd.DataFrame({
            "email": ["a@b.com", "invalid", "no-at-sign.com", "test@example.org"],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[{"type": "regex", "column": "email", "pattern": r"^.+@.+\..+$"}],
        )
        
        print(f"\nINVALID REGEX: {result}")
        assert result["passed"] is False
        assert result["violations"][0]["count"] == 2  # invalid, no-at-sign.com


class TestMultipleRules:
    """Test multiple rules together."""
    
    def test_multiple_violations(self):
        """Multiple rules with violations."""
        df = pd.DataFrame({
            "age": [25, 150, None, 30],
            "status": ["active", "invalid", "closed", "active"],
            "customer_id": ["A", "B", "A", "C"],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[
                {"type": "not_null", "columns": ["age"]},
                {"type": "range", "column": "age", "min": 0, "max": 120},
                {"type": "allowed_values", "column": "status", "values": ["active", "closed"]},
                {"type": "unique", "column": "customer_id"},
            ],
        )
        
        print(f"\nMULTIPLE VIOLATIONS: {result}")
        assert result["passed"] is False
        assert result["summary"]["failed"] == 4
        assert len(result["violations"]) == 4


class TestToolOutput:
    """Test the LangChain tool output format."""
    
    def test_tool_output_format(self):
        """Tool should produce readable output."""
        df = pd.DataFrame({
            "age": [25, 150, -5, 30],
            "income": [50000, None, 60000, 70000],
        })
        register_dataset("test", df)
        
        result = data_validation_tool.invoke({
            "dataset_ref": "test",
            "rules": [
                {"type": "not_null", "columns": ["income"]},
                {"type": "range", "column": "age", "min": 0, "max": 120},
            ],
        })
        
        print(f"\nTOOL OUTPUT:\n{result}")
        
        assert "VALIDATION FAILED" in result
        assert "range" in result
        assert "clip" in result
        assert "age" in result


class TestSchemaInference:
    """Test schema inference."""
    
    def test_schema_inferred(self):
        """Schema should be correctly inferred."""
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "income": [50000.0, 60000.0, 70000.0],
            "name": ["Alice", "Bob", "Charlie"],
            "active": [True, False, True],
        })
        register_dataset("test", df)
        
        result = validate_dataset(
            dataset_ref="test",
            rules=[],
        )
        
        print(f"\nSCHEMA: {result['schema_inferred']}")
        
        schema = {s["column"]: s["dtype"] for s in result["schema_inferred"]}
        assert "int" in schema["age"]
        assert "float" in schema["income"]
        assert schema["name"] == "object"
        assert schema["active"] == "bool"


# =============================================================================
# COMPREHENSIVE REALISTIC TESTS
# =============================================================================

class TestLoanDatasetValidation:
    """Test validation on realistic loan dataset."""
    
    def test_full_validation_suite(self, loan_dataset):
        """Run comprehensive validation on loan data."""
        register_dataset("loans", loan_dataset)
        
        rules = [
            # Null checks
            {"type": "not_null", "columns": ["loan_id", "customer_id", "age", "income", "credit_score"]},
            
            # Range checks
            {"type": "range", "column": "age", "min": 18, "max": 100},
            {"type": "range", "column": "credit_score", "min": 300, "max": 850},
            {"type": "range", "column": "dti_ratio", "min": 0, "max": 1.0},
            {"type": "range", "column": "income", "min": 0, "max": 10000000},
            
            # Uniqueness
            {"type": "unique", "column": "loan_id"},
            
            # Allowed values
            {"type": "allowed_values", "column": "loan_status", 
             "values": ["active", "closed", "default", "pending"]},
            {"type": "allowed_values", "column": "employment_type",
             "values": ["full_time", "part_time", "self_employed", "retired", "unemployed"]},
            
            # Regex patterns
            {"type": "regex", "column": "email", "pattern": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"},
            {"type": "regex", "column": "phone", "pattern": r"^\d{3}-\d{3}-\d{4}$"},
        ]
        
        result = validate_dataset(
            dataset_ref="loans",
            rules=rules,
            thresholds={"max_null_pct": 0.05, "max_dup_pct": 0.0},
        )
        
        print("\n" + "="*80)
        print("LOAN DATASET VALIDATION RESULTS")
        print("="*80)
        print(f"\nDataset shape: {loan_dataset.shape}")
        print(f"Passed: {result['passed']}")
        print(f"Rules: {result['summary']['passed']}/{result['summary']['total_rules']} passed")
        print(f"\nVIOLATIONS ({len(result['violations'])}):")
        
        for v in result["violations"]:
            print(f"\n  [{v['severity'].upper()}] {v['rule']} on '{v['column']}'")
            print(f"    Count: {v['count']:,} ({v['pct']:.1%})")
            if v["examples"]:
                print(f"    Examples: {v['examples'][:3]}")
            print(f"    Fix: {v['suggested_fix']['action']} {v['suggested_fix']['params']}")
        
        print(f"\nTop actions: {result['summary']['top_action']}")
        
        # Assertions
        assert result["passed"] is False
        assert len(result["violations"]) >= 5  # We injected at least 5 issues
        
        # Check specific violations exist
        violation_rules = {v["rule"] for v in result["violations"]}
        assert "not_null" in violation_rules  # income, credit_score have nulls
        assert "range" in violation_rules  # age, credit_score out of range
        assert "unique" in violation_rules  # loan_id has duplicates
        assert "allowed_values" in violation_rules  # loan_status has invalid values
        assert "regex" in violation_rules  # email, phone have invalid formats
        
        # Check severity ordering (high should come first)
        severities = [v["severity"] for v in result["violations"]]
        severity_order = {"high": 0, "medium": 1, "low": 2}
        assert all(
            severity_order[severities[i]] <= severity_order[severities[i+1]] 
            for i in range(len(severities)-1)
        ), "Violations should be sorted by severity"
    
    def test_fix_suggestions_are_actionable(self, loan_dataset):
        """Verify that fix suggestions map to actual tools."""
        register_dataset("loans", loan_dataset)
        
        result = validate_dataset(
            dataset_ref="loans",
            rules=[
                {"type": "not_null", "columns": ["income"]},
                {"type": "range", "column": "age", "min": 18, "max": 100},
                {"type": "unique", "column": "loan_id"},
                {"type": "allowed_values", "column": "loan_status", "values": ["active", "closed", "default"]},
            ],
        )
        
        valid_actions = {"impute", "drop_column", "clip", "dedupe", "replace_values", "cast", "regex_replace", "fix_pattern"}
        
        print("\n" + "="*80)
        print("FIX SUGGESTION VALIDATION")
        print("="*80)
        
        for v in result["violations"]:
            action = v["suggested_fix"]["action"]
            params = v["suggested_fix"]["params"]
            
            print(f"\n  {v['rule']} on '{v['column']}' → {action}")
            print(f"    Params: {params}")
            
            assert action in valid_actions, f"Unknown action: {action}"
            
            # Verify params are non-empty for actions that need them
            if action == "clip":
                assert "min" in params or "max" in params
            elif action == "impute":
                assert "strategy" in params
            elif action == "cast":
                assert "dtype" in params


class TestInsuranceDatasetValidation:
    """Test validation on insurance dataset."""
    
    def test_insurance_validation(self, insurance_dataset):
        """Validate insurance claims data."""
        register_dataset("insurance", insurance_dataset)
        
        result = validate_dataset(
            dataset_ref="insurance",
            rules=[
                {"type": "not_null", "columns": ["claim_id", "bmi", "claim_amount"]},
                {"type": "range", "column": "bmi", "min": 10, "max": 60},
                {"type": "range", "column": "children", "min": 0, "max": 8},
                {"type": "range", "column": "policy_holder_age", "min": 18, "max": 100},
                {"type": "allowed_values", "column": "smoker", "values": ["yes", "no"]},
                {"type": "allowed_values", "column": "region", 
                 "values": ["northeast", "northwest", "southeast", "southwest"]},
                {"type": "unique", "column": "claim_id"},
            ],
            thresholds={"max_null_pct": 0.02},
        )
        
        print("\n" + "="*80)
        print("INSURANCE DATASET VALIDATION RESULTS")
        print("="*80)
        print(f"\nDataset shape: {insurance_dataset.shape}")
        print(f"Passed: {result['passed']}")
        
        for v in result["violations"]:
            print(f"\n  [{v['severity'].upper()}] {v['rule']} on '{v['column']}'")
            print(f"    {v['count']:,} violations ({v['pct']:.1%})")
            if v["examples"]:
                print(f"    Examples: {v['examples'][:3]}")
        
        assert result["passed"] is False
        assert len(result["violations"]) >= 3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_dataframe(self):
        """Empty dataframe should handle gracefully."""
        df = pd.DataFrame({"a": [], "b": []})
        register_dataset("empty", df)
        
        result = validate_dataset(
            dataset_ref="empty",
            rules=[{"type": "not_null", "columns": ["a"]}],
        )
        
        print(f"\nEMPTY DF: {result}")
        # Should pass (no rows to violate)
        assert result["passed"] is True
        assert result["summary"]["rows_checked"] == 0
    
    def test_all_nulls_column(self):
        """Column with all nulls should be high severity."""
        df = pd.DataFrame({
            "data": [None, None, None, None, None],
        })
        register_dataset("all_nulls", df)
        
        result = validate_dataset(
            dataset_ref="all_nulls",
            rules=[{"type": "not_null", "columns": ["data"]}],
        )
        
        print(f"\nALL NULLS: {result}")
        assert result["passed"] is False
        assert result["violations"][0]["pct"] == 1.0
        assert result["violations"][0]["severity"] == "high"
        assert result["violations"][0]["suggested_fix"]["action"] == "drop_column"
    
    def test_boundary_values_range(self):
        """Boundary values should be handled correctly."""
        df = pd.DataFrame({
            "score": [0, 100, 50, 0, 100],  # Exactly at boundaries
        })
        register_dataset("boundary", df)
        
        result = validate_dataset(
            dataset_ref="boundary",
            rules=[{"type": "range", "column": "score", "min": 0, "max": 100}],
        )
        
        print(f"\nBOUNDARY: {result}")
        assert result["passed"] is True  # Boundaries are inclusive
    
    def test_min_only_range(self):
        """Range with only min should work."""
        df = pd.DataFrame({
            "value": [10, 20, 0, -5, 100],
        })
        register_dataset("min_only", df)
        
        result = validate_dataset(
            dataset_ref="min_only",
            rules=[{"type": "range", "column": "value", "min": 0}],
        )
        
        print(f"\nMIN ONLY: {result}")
        assert result["passed"] is False
        assert result["violations"][0]["count"] == 1  # Only -5 violates
    
    def test_max_only_range(self):
        """Range with only max should work."""
        df = pd.DataFrame({
            "value": [10, 20, 0, -5, 100, 200],
        })
        register_dataset("max_only", df)
        
        result = validate_dataset(
            dataset_ref="max_only",
            rules=[{"type": "range", "column": "value", "max": 100}],
        )
        
        print(f"\nMAX ONLY: {result}")
        assert result["passed"] is False
        assert result["violations"][0]["count"] == 1  # Only 200 violates
    
    def test_missing_column_in_rule(self):
        """Rule for missing column should not crash."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        register_dataset("missing_col", df)
        
        result = validate_dataset(
            dataset_ref="missing_col",
            rules=[{"type": "range", "column": "nonexistent", "min": 0, "max": 100}],
        )
        
        print(f"\nMISSING COL: {result}")
        # Should pass (column doesn't exist, rule is skipped)
        assert result["passed"] is True
    
    def test_invalid_regex_pattern(self):
        """Invalid regex should be handled gracefully."""
        df = pd.DataFrame({"text": ["abc", "def"]})
        register_dataset("bad_regex", df)
        
        result = validate_dataset(
            dataset_ref="bad_regex",
            rules=[{"type": "regex", "column": "text", "pattern": "[invalid(regex"}],
        )
        
        print(f"\nBAD REGEX: {result}")
        assert result["passed"] is False
        assert "Invalid regex" in str(result["violations"][0]["examples"])
    
    def test_case_sensitivity_allowed_values(self):
        """Allowed values check should be case-sensitive."""
        df = pd.DataFrame({
            "status": ["Active", "ACTIVE", "active", "closed"],
        })
        register_dataset("case_test", df)
        
        result = validate_dataset(
            dataset_ref="case_test",
            rules=[{"type": "allowed_values", "column": "status", "values": ["active", "closed"]}],
        )
        
        print(f"\nCASE SENSITIVITY: {result}")
        assert result["passed"] is False
        assert result["violations"][0]["count"] == 2  # "Active" and "ACTIVE"


class TestSampling:
    """Test dataset sampling for large datasets."""
    
    def test_sampling_large_dataset(self):
        """Sampling should work on large datasets."""
        np.random.seed(42)
        n = 100000
        
        df = pd.DataFrame({
            "id": range(n),
            "value": np.random.normal(50, 10, n),
            "category": np.random.choice(["A", "B", "C", "INVALID"], n, p=[0.4, 0.3, 0.2, 0.1]),
        })
        
        # Add ~5% nulls
        null_idx = np.random.choice(n, size=5000, replace=False)
        df.loc[null_idx, "value"] = np.nan
        
        register_dataset("large", df)
        
        # Without sampling
        result_full = validate_dataset(
            dataset_ref="large",
            rules=[
                {"type": "not_null", "columns": ["value"]},
                {"type": "allowed_values", "column": "category", "values": ["A", "B", "C"]},
            ],
        )
        
        # With sampling
        result_sampled = validate_dataset(
            dataset_ref="large",
            rules=[
                {"type": "not_null", "columns": ["value"]},
                {"type": "allowed_values", "column": "category", "values": ["A", "B", "C"]},
            ],
            sample_n=10000,
        )
        
        print("\n" + "="*80)
        print("SAMPLING COMPARISON")
        print("="*80)
        print(f"\nFull dataset: {result_full['summary']['rows_checked']:,} rows")
        print(f"Sampled: {result_sampled['summary']['rows_checked']:,} rows")
        
        for label, result in [("Full", result_full), ("Sampled", result_sampled)]:
            print(f"\n{label}:")
            for v in result["violations"]:
                print(f"  {v['rule']} on '{v['column']}': {v['pct']:.1%}")
        
        assert result_sampled["summary"]["rows_checked"] == 10000
        
        # Percentages should be similar (within reasonable tolerance)
        full_pcts = {v["column"]: v["pct"] for v in result_full["violations"]}
        sampled_pcts = {v["column"]: v["pct"] for v in result_sampled["violations"]}
        
        for col in full_pcts:
            if col in sampled_pcts:
                diff = abs(full_pcts[col] - sampled_pcts[col])
                assert diff < 0.03, f"Sampling gave very different results for {col}"


class TestSeverityRanking:
    """Test that violations are correctly ranked by severity."""
    
    def test_severity_order(self):
        """High severity should come before low severity."""
        df = pd.DataFrame({
            "critical": [None] * 100,  # 100% null -> high
            "moderate": list(range(90)) + [None] * 10,  # 10% null -> low/medium
            "minor": list(range(99)) + [None],  # 1% null -> low
        })
        register_dataset("severity_test", df)
        
        result = validate_dataset(
            dataset_ref="severity_test",
            rules=[
                {"type": "not_null", "columns": ["critical", "moderate", "minor"]},
            ],
        )
        
        print("\n" + "="*80)
        print("SEVERITY RANKING")
        print("="*80)
        
        for v in result["violations"]:
            print(f"  [{v['severity'].upper():6}] {v['column']}: {v['pct']:.1%}")
        
        assert result["violations"][0]["column"] == "critical"
        assert result["violations"][0]["severity"] == "high"


class TestToolOutputForAgent:
    """Test the formatted output that agents will parse."""
    
    def test_comprehensive_tool_output(self, loan_dataset):
        """Test full tool output format."""
        register_dataset("loans", loan_dataset)
        
        output = data_validation_tool.invoke({
            "dataset_ref": "loans",
            "rules": [
                {"type": "not_null", "columns": ["income", "credit_score"]},
                {"type": "range", "column": "age", "min": 18, "max": 100},
                {"type": "range", "column": "credit_score", "min": 300, "max": 850},
                {"type": "unique", "column": "loan_id"},
                {"type": "allowed_values", "column": "loan_status", 
                 "values": ["active", "closed", "default", "pending"]},
            ],
            "thresholds": {"max_null_pct": 0.05, "max_dup_pct": 0.0},
        })
        
        print("\n" + "="*80)
        print("AGENT-READY OUTPUT")
        print("="*80)
        print(output)
        print("="*80)
        
        # Check output structure
        assert "VALIDATION" in output
        assert "Summary" in output
        assert "Violations" in output
        assert "Schema" in output
        
        # Check actionable items are present
        assert "Fix:" in output
        assert "clip" in output or "impute" in output or "dedupe" in output
        
        # Check severity icons
        assert "🔴" in output or "🟡" in output or "🟢" in output
    
    def test_passing_validation_output(self):
        """Test output when validation passes."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score": [85, 90, 78, 92, 88],
        })
        register_dataset("clean", df)
        
        output = data_validation_tool.invoke({
            "dataset_ref": "clean",
            "rules": [
                {"type": "not_null", "columns": ["id", "name", "score"]},
                {"type": "range", "column": "score", "min": 0, "max": 100},
                {"type": "unique", "column": "id"},
            ],
        })
        
        print("\n" + "="*80)
        print("PASSING VALIDATION OUTPUT")
        print("="*80)
        print(output)
        
        assert "✓ VALIDATION PASSED" in output
        assert "Violations" not in output or "0" in output


class TestDtypeValidation:
    """Additional dtype validation tests."""
    
    def test_dtype_group_matching(self):
        """Test that dtype groups match correctly."""
        df = pd.DataFrame({
            "int_col": pd.array([1, 2, 3], dtype="int32"),
            "float_col": pd.array([1.0, 2.0, 3.0], dtype="float32"),
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })
        register_dataset("dtypes", df)
        
        # These should all pass (group matching)
        result = validate_dataset(
            dataset_ref="dtypes",
            rules=[
                {"type": "dtype", "column": "int_col", "expected": "int"},
                {"type": "dtype", "column": "float_col", "expected": "float"},
                {"type": "dtype", "column": "str_col", "expected": "str"},
                {"type": "dtype", "column": "bool_col", "expected": "bool"},
            ],
        )
        
        print(f"\nDTYPE GROUPS: {result}")
        assert result["passed"] is True
    
    def test_specific_dtype_matching(self):
        """Test that specific dtypes match."""
        df = pd.DataFrame({
            "col": pd.array([1, 2, 3], dtype="int64"),
        })
        register_dataset("specific_dtype", df)
        
        result = validate_dataset(
            dataset_ref="specific_dtype",
            rules=[{"type": "dtype", "column": "col", "expected": "int64"}],
        )
        
        print(f"\nSPECIFIC DTYPE: {result}")
        assert result["passed"] is True


# =============================================================================
# REAL DATASET TESTS (from datasets/sql/)
# =============================================================================

class TestRealDatasets:
    """Test validation on real datasets from datasets/sql/."""
    
    @pytest.fixture
    def credit_card_risk_df(self):
        """Load credit_card_risk dataset from SQL file."""
        import re
        
        sql_path = Path(__file__).parent.parent / "datasets" / "sql" / "credit_card_risk.sql"
        if not sql_path.exists():
            pytest.skip(f"Dataset not found: {sql_path}")
        
        # Parse INSERT statements from SQL
        with open(sql_path, 'r') as f:
            content = f.read()
        
        # Extract column names from CREATE TABLE
        columns = ["ID", "AGE", "INCOME", "GENDER", "MARITAL", "NUMKIDS", 
                   "NUMCARDS", "HOWPAID", "MORTGAGE", "STORECAR", "LOANS", "RISK"]
        
        # Parse VALUES from INSERT statements
        pattern = r"VALUES \(([^)]+)\)"
        rows = []
        for match in re.finditer(pattern, content):
            values_str = match.group(1)
            # Parse values (handle strings with quotes)
            values = []
            for val in re.findall(r"'[^']*'|\d+", values_str):
                if val.startswith("'"):
                    values.append(val.strip("'"))
                else:
                    values.append(int(val))
            if len(values) == len(columns):
                rows.append(values)
        
        df = pd.DataFrame(rows, columns=columns)
        return df
    
    @pytest.fixture
    def insurance_df(self):
        """Load insurance dataset from SQL file."""
        import re
        
        sql_path = Path(__file__).parent.parent / "datasets" / "sql" / "insurance.sql"
        if not sql_path.exists():
            pytest.skip(f"Dataset not found: {sql_path}")
        
        # Parse INSERT statements from SQL
        with open(sql_path, 'r') as f:
            content = f.read()
        
        columns = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
        
        # Parse VALUES from INSERT statements
        pattern = r"VALUES \(([^)]+)\)"
        rows = []
        for match in re.finditer(pattern, content):
            values_str = match.group(1)
            values = []
            for val in re.findall(r"'[^']*'|[\d.]+", values_str):
                if val.startswith("'"):
                    values.append(val.strip("'"))
                elif '.' in val:
                    values.append(float(val))
                else:
                    values.append(int(val))
            if len(values) == len(columns):
                rows.append(values)
        
        df = pd.DataFrame(rows, columns=columns)
        return df
    
    def test_credit_card_risk_validation(self, credit_card_risk_df):
        """Validate real credit card risk dataset."""
        df = credit_card_risk_df
        register_dataset("credit_risk", df)
        
        print("\n" + "="*80)
        print("CREDIT CARD RISK DATASET")
        print("="*80)
        print(f"Shape: {df.shape}")
        print(f"\nColumn dtypes:\n{df.dtypes}")
        print(f"\nSample data:\n{df.head(3)}")
        
        # Define validation rules based on expected schema
        rules = [
            # All columns should be non-null
            {"type": "not_null", "columns": ["ID", "AGE", "INCOME", "GENDER", "MARITAL", "RISK"]},
            
            # Age should be reasonable
            {"type": "range", "column": "AGE", "min": 18, "max": 100},
            
            # Income should be positive
            {"type": "range", "column": "INCOME", "min": 0},
            
            # Numeric columns should be non-negative
            {"type": "range", "column": "NUMKIDS", "min": 0, "max": 10},
            {"type": "range", "column": "NUMCARDS", "min": 0, "max": 20},
            {"type": "range", "column": "LOANS", "min": 0, "max": 50},
            
            # ID should be unique
            {"type": "unique", "column": "ID"},
            
            # Categorical columns - check for unexpected values
            {"type": "allowed_values", "column": "GENDER", "values": ["m", "f", "M", "F", "male", "female"]},
            {"type": "allowed_values", "column": "MORTGAGE", "values": ["y", "n", "Y", "N", "yes", "no"]},
            {"type": "allowed_values", "column": "HOWPAID", "values": ["monthly", "weekly", "annually"]},
            
            # RISK should have clean values (checking for trailing spaces)
            {"type": "allowed_values", "column": "RISK", 
             "values": ["good risk", "bad risk", "bad loss", "bad profit"]},
        ]
        
        result = validate_dataset(
            dataset_ref="credit_risk",
            rules=rules,
            thresholds={"max_null_pct": 0.0, "max_dup_pct": 0.0},
        )
        
        print(f"\n{'='*40}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*40}")
        print(f"Passed: {result['passed']}")
        print(f"Rules: {result['summary']['passed']}/{result['summary']['total_rules']} passed")
        
        if result["violations"]:
            print(f"\nVIOLATIONS ({len(result['violations'])}):")
            for v in result["violations"]:
                print(f"\n  [{v['severity'].upper()}] {v['rule']} on '{v['column']}'")
                print(f"    Count: {v['count']:,} ({v['pct']:.1%})")
                if v["examples"]:
                    print(f"    Examples: {v['examples'][:5]}")
                print(f"    Fix: {v['suggested_fix']['action']} {v['suggested_fix']['params']}")
        
        print(f"\nTop actions: {result['summary']['top_action']}")
        
        # The dataset should have trailing space issues in RISK column
        risk_violation = next((v for v in result["violations"] if v["column"] == "RISK"), None)
        assert risk_violation is not None, "Expected RISK column to have trailing space issues"
        print(f"\n✓ Detected RISK column has dirty values (trailing spaces): {risk_violation['examples']}")
    
    def test_insurance_validation(self, insurance_df):
        """Validate real insurance dataset."""
        df = insurance_df
        register_dataset("insurance_real", df)
        
        print("\n" + "="*80)
        print("INSURANCE DATASET")
        print("="*80)
        print(f"Shape: {df.shape}")
        print(f"\nColumn dtypes:\n{df.dtypes}")
        print(f"\nSample data:\n{df.head(3)}")
        print(f"\nValue counts for 'smoker': {df['smoker'].value_counts().to_dict()}")
        print(f"Value counts for 'region': {df['region'].value_counts().to_dict()}")
        
        # Define validation rules
        rules = [
            # All columns should be non-null
            {"type": "not_null", "columns": ["age", "sex", "bmi", "children", "smoker", "region", "charges"]},
            
            # Age should be reasonable (insurance typically 18+)
            {"type": "range", "column": "age", "min": 18, "max": 100},
            
            # BMI should be in reasonable range
            {"type": "range", "column": "bmi", "min": 10, "max": 60},
            
            # Children should be non-negative
            {"type": "range", "column": "children", "min": 0, "max": 10},
            
            # Charges should be positive
            {"type": "range", "column": "charges", "min": 0},
            
            # Categorical columns
            {"type": "allowed_values", "column": "sex", "values": ["male", "female"]},
            {"type": "allowed_values", "column": "smoker", "values": ["yes", "no"]},
            {"type": "allowed_values", "column": "region", 
             "values": ["northeast", "northwest", "southeast", "southwest"]},
            
            # Check dtypes
            {"type": "dtype", "column": "age", "expected": "int"},
            {"type": "dtype", "column": "bmi", "expected": "float"},
            {"type": "dtype", "column": "charges", "expected": "float"},
        ]
        
        result = validate_dataset(
            dataset_ref="insurance_real",
            rules=rules,
            thresholds={"max_null_pct": 0.0},
        )
        
        print(f"\n{'='*40}")
        print(f"VALIDATION RESULTS")
        print(f"{'='*40}")
        print(f"Passed: {result['passed']}")
        print(f"Rules: {result['summary']['passed']}/{result['summary']['total_rules']} passed")
        
        if result["violations"]:
            print(f"\nVIOLATIONS ({len(result['violations'])}):")
            for v in result["violations"]:
                print(f"\n  [{v['severity'].upper()}] {v['rule']} on '{v['column']}'")
                print(f"    Count: {v['count']:,} ({v['pct']:.1%})")
                if v["examples"]:
                    print(f"    Examples: {v['examples'][:5]}")
                print(f"    Fix: {v['suggested_fix']['action']} {v['suggested_fix']['params']}")
        else:
            print("\n✓ No violations found - this is a clean dataset!")
        
        print(f"\nSchema inferred:")
        for col in result["schema_inferred"]:
            print(f"  {col['column']}: {col['dtype']} (null={col['nullable']}, unique={col['unique_count']})")
    
    def test_tool_output_on_real_data(self, credit_card_risk_df):
        """Test agent-ready output format on real data."""
        register_dataset("credit_risk_tool", credit_card_risk_df)
        
        output = data_validation_tool.invoke({
            "dataset_ref": "credit_risk_tool",
            "rules": [
                {"type": "not_null", "columns": ["ID", "AGE", "INCOME", "RISK"]},
                {"type": "range", "column": "AGE", "min": 18, "max": 100},
                {"type": "unique", "column": "ID"},
                {"type": "allowed_values", "column": "RISK", 
                 "values": ["good risk", "bad risk", "bad loss", "bad profit"]},
                {"type": "allowed_values", "column": "MARITAL", 
                 "values": ["married", "single", "divorced"]},
            ],
        })
        
        print("\n" + "="*80)
        print("AGENT OUTPUT FOR CREDIT CARD RISK DATA")
        print("="*80)
        print(output)
        print("="*80)
        
        # Should detect trailing space issues
        assert "RISK" in output or "MARITAL" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

