"""
Comprehensive tests for feature_diagnostics tool.

Tests cover:
- Redundancy detection
- Leakage detection
- Null pattern analysis
- Transform suggestions
- Drop list generation
- Real dataset testing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from analysis.feature_diagnostics import (feature_diagnostics_tool,
                                          run_feature_diagnostics)
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
def dataset_with_redundancy():
    """Dataset with redundant (highly correlated) features."""
    np.random.seed(42)
    n = 1000
    
    income = np.random.lognormal(10, 1, n)
    
    df = pd.DataFrame({
        "income": income,
        "salary": income * np.random.uniform(0.98, 1.02, n),  # r > 0.99
        "annual_pay": income * np.random.uniform(0.95, 1.05, n),  # r > 0.95
        "age": np.random.randint(18, 70, n),
        "debt_ratio": np.random.uniform(0, 1, n),
        "target": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })
    
    return df


@pytest.fixture
def dataset_with_leakage():
    """Dataset with features that leak target information."""
    np.random.seed(42)
    n = 500
    
    target = np.random.choice([0, 1], n, p=[0.7, 0.3])
    
    df = pd.DataFrame({
        "income": np.random.lognormal(10, 1, n),
        "age": np.random.randint(18, 70, n),
        # Leaky features
        "final_balance": np.where(target == 1, 0, np.random.uniform(100, 1000, n)),
        "approval_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "outcome_status": np.where(target == 1, "rejected", "approved"),
        # Feature perfectly correlated with target (leakage)
        "risk_score": target + np.random.normal(0, 0.01, n),
        "target": target,
    })
    
    return df


@pytest.fixture
def dataset_with_nulls():
    """Dataset with various null patterns."""
    np.random.seed(42)
    n = 1000
    
    target = np.random.choice([0, 1], n, p=[0.8, 0.2])
    
    df = pd.DataFrame({
        "clean_feature": np.random.randn(n),
        "moderate_nulls": np.where(np.random.random(n) > 0.7, np.random.randn(n), np.nan),
        "high_nulls": np.where(np.random.random(n) > 0.2, np.nan, np.random.randn(n)),
        # Nulls correlated with target (suspicious)
        "suspicious_nulls": np.where(
            (target == 1) & (np.random.random(n) > 0.3),
            np.nan,
            np.random.randn(n)
        ),
        "target": target,
    })
    
    return df


@pytest.fixture
def dataset_with_skew():
    """Dataset with skewed features."""
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        "high_skew": np.random.lognormal(10, 2, n),  # Very right skewed
        "moderate_skew": np.random.lognormal(5, 0.8, n),  # Moderately skewed
        "normal": np.random.normal(50, 10, n),  # Normal distribution
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    
    return df


# =============================================================================
# REDUNDANCY TESTS
# =============================================================================

class TestRedundancy:
    """Test redundancy detection."""
    
    def test_redundant_features_detected(self, dataset_with_redundancy):
        """Highly correlated features should be detected."""
        register_dataset("redundancy", dataset_with_redundancy)
        
        result = run_feature_diagnostics(
            dataset_ref="redundancy",
            feature_cols=["income", "salary", "annual_pay", "age", "debt_ratio"],
            target_col="target",
            task_type="classification",
        )
        
        print(f"\nRedundancy groups: {result['redundancy_groups']}")
        
        # Should detect income/salary/annual_pay as redundant
        assert len(result["redundancy_groups"]) >= 1
        
        group_cols = []
        for g in result["redundancy_groups"]:
            group_cols.extend(g["columns"])
        
        assert "income" in group_cols or "salary" in group_cols
    
    def test_redundant_features_in_drop_list(self, dataset_with_redundancy):
        """Redundant features (except first) should be in drop list."""
        register_dataset("redundancy", dataset_with_redundancy)
        
        result = run_feature_diagnostics(
            dataset_ref="redundancy",
            feature_cols=["income", "salary", "annual_pay", "age", "debt_ratio"],
            target_col="target",
            task_type="classification",
        )
        
        print(f"\nDrop list: {result['drop_list']}")
        
        drop_cols = [d["column"] for d in result["drop_list"]]
        drop_reasons = [d["reason"] for d in result["drop_list"]]
        
        # At least one redundant column should be dropped
        redundant_drops = [r for r in drop_reasons if "redundant" in r.lower()]
        assert len(redundant_drops) >= 1


# =============================================================================
# LEAKAGE TESTS
# =============================================================================

class TestLeakage:
    """Test leakage detection."""
    
    def test_suspect_cols_flagged(self, dataset_with_leakage):
        """Columns in suspect_cols should be flagged as high leakage."""
        register_dataset("leakage", dataset_with_leakage)
        
        result = run_feature_diagnostics(
            dataset_ref="leakage",
            feature_cols=["income", "age", "final_balance", "risk_score"],
            target_col="target",
            task_type="classification",
            leakage_checks={"suspect_cols": ["final_balance"]},
        )
        
        print(f"\nFeature health: {result['feature_health']}")
        
        # final_balance should be flagged
        final_balance_health = next(
            (f for f in result["feature_health"] if f["column"] == "final_balance"), 
            None
        )
        assert final_balance_health is not None
        assert final_balance_health["leakage_risk"] == "high"
    
    def test_high_correlation_leakage(self, dataset_with_leakage):
        """Features with very high target correlation should be flagged."""
        register_dataset("leakage", dataset_with_leakage)
        
        result = run_feature_diagnostics(
            dataset_ref="leakage",
            feature_cols=["income", "age", "risk_score"],
            target_col="target",
            task_type="classification",
        )
        
        print(f"\nDrop list: {result['drop_list']}")
        
        # risk_score should be flagged (r ≈ 1 with target)
        risk_health = next(
            (f for f in result["feature_health"] if f["column"] == "risk_score"),
            None
        )
        assert risk_health is not None
        assert risk_health["leakage_risk"] == "high"
    
    def test_leaky_keyword_detection(self, dataset_with_leakage):
        """Features with leaky keywords in name should be flagged."""
        register_dataset("leakage", dataset_with_leakage)
        
        result = run_feature_diagnostics(
            dataset_ref="leakage",
            feature_cols=["income", "final_balance"],
            target_col="target",
            task_type="classification",
        )
        
        # final_balance has "final" keyword
        final_health = next(
            (f for f in result["feature_health"] if f["column"] == "final_balance"),
            None
        )
        assert final_health is not None
        # Should at least be medium risk due to keyword
        assert final_health["leakage_risk"] in ["high", "medium"]


# =============================================================================
# NULL PATTERN TESTS
# =============================================================================

class TestNullPatterns:
    """Test null pattern analysis."""
    
    def test_high_nulls_dropped(self, dataset_with_nulls):
        """Columns with >70% nulls should be in drop list."""
        register_dataset("nulls", dataset_with_nulls)
        
        result = run_feature_diagnostics(
            dataset_ref="nulls",
            feature_cols=["clean_feature", "moderate_nulls", "high_nulls", "suspicious_nulls"],
            target_col="target",
            task_type="classification",
        )
        
        print(f"\nFeature health: {result['feature_health']}")
        print(f"Drop list: {result['drop_list']}")
        
        # high_nulls (80% null) should be dropped
        drop_cols = [d["column"] for d in result["drop_list"]]
        assert "high_nulls" in drop_cols
    
    def test_null_target_correlation(self, dataset_with_nulls):
        """Null-target correlation should be computed."""
        register_dataset("nulls", dataset_with_nulls)
        
        result = run_feature_diagnostics(
            dataset_ref="nulls",
            feature_cols=["clean_feature", "moderate_nulls", "suspicious_nulls"],
            target_col="target",
            task_type="classification",
        )
        
        # suspicious_nulls should have non-zero null_target_corr
        suspicious = next(
            (f for f in result["feature_health"] if f["column"] == "suspicious_nulls"),
            None
        )
        assert suspicious is not None
        # Should have some correlation computed
        assert suspicious["null_target_corr"] is not None or suspicious["null_pct"] > 0


# =============================================================================
# TRANSFORM SUGGESTION TESTS
# =============================================================================

class TestTransformSuggestions:
    """Test transform suggestion generation."""
    
    def test_skewed_features_log_suggestion(self, dataset_with_skew):
        """Highly skewed features should get log transform suggestion."""
        register_dataset("skew", dataset_with_skew)
        
        result = run_feature_diagnostics(
            dataset_ref="skew",
            feature_cols=["high_skew", "moderate_skew", "normal"],
            target_col="target",
            task_type="classification",
        )
        
        print(f"\nTransform suggestions: {result['transform_suggestions']}")
        
        transform_cols = [t["column"] for t in result["transform_suggestions"]]
        transform_actions = [t["action"] for t in result["transform_suggestions"]]
        
        # high_skew should get log1p suggestion
        assert "high_skew" in transform_cols
        
        # Should suggest log1p for high skew
        high_skew_transform = next(
            (t for t in result["transform_suggestions"] if t["column"] == "high_skew"),
            None
        )
        assert high_skew_transform is not None
        assert high_skew_transform["action"] == "log1p"
    
    def test_normal_features_no_transform(self, dataset_with_skew):
        """Normal features should not get transform suggestions."""
        register_dataset("skew", dataset_with_skew)
        
        result = run_feature_diagnostics(
            dataset_ref="skew",
            feature_cols=["high_skew", "moderate_skew", "normal"],
            target_col="target",
            task_type="classification",
        )
        
        transform_cols = [t["column"] for t in result["transform_suggestions"]]
        
        # normal should not have transform suggestion
        assert "normal" not in transform_cols


# =============================================================================
# KEEP LIST TESTS
# =============================================================================

class TestKeepList:
    """Test keep list generation."""
    
    def test_clean_features_in_keep_list(self, dataset_with_redundancy):
        """Clean features should be in keep list."""
        register_dataset("redundancy", dataset_with_redundancy)
        
        result = run_feature_diagnostics(
            dataset_ref="redundancy",
            feature_cols=["income", "salary", "age", "debt_ratio"],
            target_col="target",
            task_type="classification",
        )
        
        print(f"\nKeep list: {result['keep_list']}")
        
        # age and debt_ratio should be in keep list (not redundant)
        assert "age" in result["keep_list"]
        assert "debt_ratio" in result["keep_list"]


# =============================================================================
# TOOL OUTPUT TESTS
# =============================================================================

class TestToolOutput:
    """Test LangChain tool output format."""
    
    def test_tool_output_format(self, dataset_with_redundancy):
        """Tool should produce readable output."""
        register_dataset("test", dataset_with_redundancy)
        
        output = feature_diagnostics_tool.invoke({
            "dataset_ref": "test",
            "feature_cols": ["income", "salary", "annual_pay", "age", "debt_ratio"],
            "target_col": "target",
            "task_type": "classification",
        })
        
        print("\n" + "="*80)
        print("TOOL OUTPUT")
        print("="*80)
        print(output)
        
        assert "# Feature Diagnostics Report" in output
        assert "Features checked" in output
        assert "Drop" in output or "Keep" in output
    
    def test_comprehensive_output(self, dataset_with_leakage):
        """Test output with leakage and other issues."""
        register_dataset("test", dataset_with_leakage)
        
        output = feature_diagnostics_tool.invoke({
            "dataset_ref": "test",
            "feature_cols": ["income", "age", "final_balance", "risk_score"],
            "target_col": "target",
            "task_type": "classification",
            "leakage_checks": {"suspect_cols": ["final_balance"]},
        })
        
        print("\n" + "="*80)
        print("COMPREHENSIVE OUTPUT")
        print("="*80)
        print(output)
        
        # Should mention leakage
        assert "leakage" in output.lower() or "drop" in output.lower()


# =============================================================================
# REAL DATASET TESTS
# =============================================================================

class TestRealDatasets:
    """Test on real datasets."""
    
    @pytest.fixture
    def credit_card_risk_df(self):
        """Load credit_card_risk dataset."""
        import re
        
        sql_path = Path(__file__).parent.parent / "datasets" / "sql" / "credit_card_risk.sql"
        if not sql_path.exists():
            pytest.skip(f"Dataset not found: {sql_path}")
        
        with open(sql_path, 'r') as f:
            content = f.read()
        
        columns = ["ID", "AGE", "INCOME", "GENDER", "MARITAL", "NUMKIDS", 
                   "NUMCARDS", "HOWPAID", "MORTGAGE", "STORECAR", "LOANS", "RISK"]
        
        pattern = r"VALUES \(([^)]+)\)"
        rows = []
        for match in re.finditer(pattern, content):
            values_str = match.group(1)
            values = []
            for val in re.findall(r"'[^']*'|\d+", values_str):
                if val.startswith("'"):
                    values.append(val.strip("'"))
                else:
                    values.append(int(val))
            if len(values) == len(columns):
                rows.append(values)
        
        df = pd.DataFrame(rows, columns=columns)
        df["is_bad"] = df["RISK"].str.strip().apply(lambda x: 1 if "bad" in x.lower() else 0)
        return df
    
    @pytest.fixture
    def insurance_df(self):
        """Load insurance dataset."""
        csv_path = Path(__file__).parent.parent / "datasets" / "csv" / "insurance.csv"
        if not csv_path.exists():
            pytest.skip(f"Dataset not found: {csv_path}")
        return pd.read_csv(csv_path)
    
    @pytest.fixture
    def loan_default_df(self):
        """Load loan default dataset."""
        csv_path = Path(__file__).parent.parent / "datasets" / "csv" / "Loan_default.csv"
        if not csv_path.exists():
            pytest.skip(f"Dataset not found: {csv_path}")
        return pd.read_csv(csv_path)
    
    @pytest.fixture
    def financial_distress_df(self):
        """Load financial distress dataset."""
        csv_path = Path(__file__).parent.parent / "datasets" / "csv" / "Financial Distress.csv"
        if not csv_path.exists():
            pytest.skip(f"Dataset not found: {csv_path}")
        df = pd.read_csv(csv_path)
        # Create binary target
        df["is_distressed"] = (df["Financial Distress"] < -0.5).astype(int)
        return df
    
    def test_credit_card_risk_diagnostics(self, credit_card_risk_df):
        """Test diagnostics on credit card risk dataset."""
        register_dataset("credit_risk", credit_card_risk_df)
        
        result = run_feature_diagnostics(
            dataset_ref="credit_risk",
            feature_cols=["ID", "AGE", "INCOME", "NUMKIDS", "NUMCARDS", "STORECAR", "LOANS"],
            target_col="is_bad",
            task_type="classification",
            leakage_checks={"suspect_cols": ["ID"]},  # ID shouldn't predict
        )
        
        print("\n" + "="*80)
        print("CREDIT CARD RISK DIAGNOSTICS")
        print("="*80)
        print(f"Feature health: {len(result['feature_health'])} features")
        print(f"Drop list: {result['drop_list']}")
        print(f"Transform suggestions: {result['transform_suggestions']}")
        print(f"Keep list: {result['keep_list']}")
        print(f"Summary: {result['summary']}")
        
        # ID should be flagged
        id_health = next(
            (f for f in result["feature_health"] if f["column"] == "ID"),
            None
        )
        assert id_health is not None
        assert id_health["leakage_risk"] == "high"
    
    def test_credit_card_risk_tool_output(self, credit_card_risk_df):
        """Test tool output on credit card risk."""
        register_dataset("credit_risk", credit_card_risk_df)
        
        output = feature_diagnostics_tool.invoke({
            "dataset_ref": "credit_risk",
            "feature_cols": ["ID", "AGE", "INCOME", "NUMKIDS", "NUMCARDS", "STORECAR", "LOANS"],
            "target_col": "is_bad",
            "task_type": "classification",
            "leakage_checks": {"suspect_cols": ["ID"]},
        })
        
        print("\n" + "="*80)
        print("CREDIT CARD RISK - AGENT OUTPUT")
        print("="*80)
        print(output)
        
        assert "Feature Diagnostics" in output
        assert "ID" in output  # Should mention ID somewhere
    
    def test_insurance_diagnostics(self, insurance_df):
        """Test diagnostics on insurance dataset (regression)."""
        register_dataset("insurance", insurance_df)
        
        feature_cols = [c for c in insurance_df.columns if c != "charges"]
        
        print("\n" + "="*80)
        print("INSURANCE DATASET - INPUT")
        print("="*80)
        print(f"Shape: {insurance_df.shape}")
        print(f"Columns: {list(insurance_df.columns)}")
        print(f"Feature cols: {feature_cols}")
        print(f"Target: charges (regression)")
        print(f"\nSample data:\n{insurance_df.head()}")
        
        result = run_feature_diagnostics(
            dataset_ref="insurance",
            feature_cols=feature_cols,
            target_col="charges",
            task_type="regression",
        )
        
        print("\n" + "="*80)
        print("INSURANCE DATASET - OUTPUT")
        print("="*80)
        print(f"\n--- Feature Health ---")
        for f in result["feature_health"]:
            print(f"  {f['column']}: null={f['null_pct']}, skew={f['skew']}, "
                  f"leakage={f['leakage_risk']}, redundancy_group={f['redundancy_group']}")
        print(f"\n--- Redundancy Groups ---")
        print(f"  {result['redundancy_groups']}")
        print(f"\n--- Drop List ---")
        for d in result["drop_list"]:
            print(f"  {d['column']}: {d['reason']}")
        print(f"\n--- Transform Suggestions ---")
        for t in result["transform_suggestions"]:
            print(f"  {t['column']} → {t['action']}: {t['reason']}")
        print(f"\n--- Keep List ---")
        print(f"  {result['keep_list']}")
        print(f"\n--- Summary ---")
        print(f"  {result['summary']}")
        
        # Assertions
        assert len(result["feature_health"]) == len(feature_cols)
        assert "keep_list" in result
    
    def test_loan_default_diagnostics(self, loan_default_df):
        """Test diagnostics on loan default dataset (many features, classification)."""
        register_dataset("loan_default", loan_default_df)
        
        # Use actual column names from this dataset
        feature_cols = ["LoanID", "Age", "Income", "LoanAmount", "CreditScore", 
                       "MonthsEmployed", "NumCreditLines", "InterestRate", 
                       "LoanTerm", "DTIRatio"]
        
        print("\n" + "="*80)
        print("LOAN DEFAULT DATASET - INPUT")
        print("="*80)
        print(f"Shape: {loan_default_df.shape}")
        print(f"Feature cols: {feature_cols}")
        print(f"Target: Default (1=default)")
        print(f"\nColumn dtypes:")
        for c in feature_cols:
            print(f"  {c}: {loan_default_df[c].dtype}, nulls={loan_default_df[c].isna().sum()}")
        
        result = run_feature_diagnostics(
            dataset_ref="loan_default",
            feature_cols=feature_cols,
            target_col="Default",
            task_type="classification",
            leakage_checks={"suspect_cols": ["LoanID"]},  # ID shouldn't predict
        )
        
        print("\n" + "="*80)
        print("LOAN DEFAULT DATASET - OUTPUT")
        print("="*80)
        print(f"\n--- Feature Health (sorted by issues) ---")
        # Sort by issues
        for f in sorted(result["feature_health"], 
                       key=lambda x: (x["leakage_risk"] != "none", x["null_pct"]), 
                       reverse=True):
            issues = []
            if f["null_pct"] > 0.1:
                issues.append(f"high_null({f['null_pct']:.1%})")
            if f["skew"] and abs(f["skew"]) > 1:
                issues.append(f"skewed({f['skew']})")
            if f["leakage_risk"] != "none":
                issues.append(f"leakage({f['leakage_risk']})")
            issues_str = ", ".join(issues) if issues else "clean"
            print(f"  {f['column']}: {issues_str}")
        
        print(f"\n--- Redundancy Groups ---")
        for g in result["redundancy_groups"]:
            print(f"  {g['group']}: {g['columns']} (r={g['max_corr']})")
        
        print(f"\n--- Drop List ({len(result['drop_list'])}) ---")
        for d in result["drop_list"]:
            print(f"  DROP {d['column']}: {d['reason']}")
        
        print(f"\n--- Transform Suggestions ({len(result['transform_suggestions'])}) ---")
        for t in result["transform_suggestions"]:
            print(f"  {t['column']} → {t['action']}: {t['reason']}")
        
        print(f"\n--- Summary ---")
        print(f"  Checked: {result['summary']['features_checked']}")
        print(f"  Drop: {result['summary']['drop']}")
        print(f"  Transform: {result['summary']['transform']}")
        print(f"  Keep as-is: {result['summary']['keep_as_is']}")
        print(f"  Top actions: {result['summary']['top_actions']}")
        
        # Should handle nulls correctly
        assert len(result["feature_health"]) > 0
    
    def test_financial_distress_diagnostics(self, financial_distress_df):
        """Test diagnostics on financial distress dataset (many numeric features)."""
        register_dataset("fin_distress", financial_distress_df)
        
        # Get numeric feature columns (x1, x2, ... x83)
        feature_cols = [c for c in financial_distress_df.columns 
                       if c.startswith("x") and c[1:].isdigit()][:15]  # First 15
        
        print("\n" + "="*80)
        print("FINANCIAL DISTRESS DATASET - INPUT")
        print("="*80)
        print(f"Shape: {financial_distress_df.shape}")
        print(f"Feature cols (first 15): {feature_cols}")
        print(f"Target: is_distressed (binary)")
        print(f"Target distribution: {financial_distress_df['is_distressed'].value_counts().to_dict()}")
        
        result = run_feature_diagnostics(
            dataset_ref="fin_distress",
            feature_cols=feature_cols,
            target_col="is_distressed",
            task_type="classification",
        )
        
        print("\n" + "="*80)
        print("FINANCIAL DISTRESS DATASET - OUTPUT")
        print("="*80)
        
        # Check for redundancy (many financial ratios are correlated)
        print(f"\n--- Redundancy Groups Found ---")
        for g in result["redundancy_groups"]:
            print(f"  {g['group']}: {g['columns']} (max_r={g['max_corr']})")
        
        print(f"\n--- Features with Skew > 2 ---")
        skewed = [f for f in result["feature_health"] if f["skew"] and abs(f["skew"]) > 2]
        for f in skewed[:5]:
            print(f"  {f['column']}: skew={f['skew']}")
        
        print(f"\n--- Drop List ---")
        for d in result["drop_list"]:
            print(f"  {d['column']}: {d['reason']}")
        
        print(f"\n--- Transform Suggestions ---")
        for t in result["transform_suggestions"][:5]:
            print(f"  {t['column']} → {t['action']}")
        
        print(f"\n--- Summary ---")
        print(f"  {result['summary']}")
        
        # Financial data often has redundant features
        # Should detect some
        assert len(result["feature_health"]) == len(feature_cols)
    
    def test_comprehensive_evaluation(self, loan_default_df):
        """Comprehensive test showing full agent workflow with input/output."""
        register_dataset("loan", loan_default_df)
        
        # Use actual column names
        feature_cols = ["LoanID", "Age", "Income", "LoanAmount", "CreditScore",
                       "MonthsEmployed", "NumCreditLines", "InterestRate",
                       "LoanTerm", "DTIRatio"]
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION - AGENT INPUT")
        print("="*80)
        print(f"""
feature_diagnostics_tool.invoke({{
    "dataset_ref": "loan",
    "feature_cols": {feature_cols},
    "target_col": "Default",
    "task_type": "classification",
    "leakage_checks": {{"suspect_cols": ["LoanID"]}},
    "sample_n": 50000
}})
""")
        
        output = feature_diagnostics_tool.invoke({
            "dataset_ref": "loan",
            "feature_cols": feature_cols,
            "target_col": "Default",
            "task_type": "classification",
            "leakage_checks": {"suspect_cols": ["LoanID"]},
            "sample_n": 50000,
        })
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION - AGENT OUTPUT")
        print("="*80)
        print(output)
        
        # Verify it's useful
        assert "Feature Diagnostics Report" in output
        assert "Drop" in output or "Keep" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

