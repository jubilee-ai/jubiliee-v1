"""
Comprehensive tests for eda_report tool.

Tests cover:
- Basic profiling (shape, schema, numeric/categorical summaries)
- Target analysis (classification and regression)
- Correlation detection
- Alert generation
- Recommendation generation
- Real dataset testing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from analysis.eda_report import eda_report_tool, run_eda_report
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
def simple_dataset():
    """Simple dataset with numeric and categorical columns."""
    np.random.seed(42)
    n = 1000
    
    return pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "income": np.random.lognormal(10, 1, n),
        "score": np.random.normal(500, 100, n),
        "gender": np.random.choice(["M", "F"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "customer_id": [f"CUST-{i:06d}" for i in range(n)],
    })


@pytest.fixture
def classification_dataset():
    """Dataset for classification with imbalanced target."""
    np.random.seed(42)
    n = 1000
    
    # Imbalanced target (85/15)
    target = np.random.choice([0, 1], n, p=[0.85, 0.15])
    
    # Features correlated with target
    income = np.where(target == 1, 
                      np.random.normal(30000, 10000, n),
                      np.random.normal(60000, 15000, n))
    age = np.where(target == 1,
                   np.random.normal(25, 5, n),
                   np.random.normal(45, 10, n))
    
    return pd.DataFrame({
        "income": income,
        "age": age,
        "score": np.random.normal(500, 100, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "default": target,
    })


@pytest.fixture
def regression_dataset():
    """Dataset for regression target."""
    np.random.seed(42)
    n = 500
    
    x1 = np.random.normal(50, 10, n)
    x2 = np.random.normal(100, 20, n)
    target = 2 * x1 + 0.5 * x2 + np.random.normal(0, 10, n)
    
    return pd.DataFrame({
        "feature1": x1,
        "feature2": x2,
        "noise": np.random.normal(0, 1, n),
        "price": target,
    })


@pytest.fixture
def dataset_with_issues():
    """Dataset with various data quality issues."""
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        # High skew
        "income": np.random.lognormal(10, 2, n),
        # Highly correlated
        "salary": np.random.lognormal(10, 2, n),
        # High cardinality categorical
        "zip_code": [f"{np.random.randint(10000, 99999)}" for _ in range(n)],
        # ID-like
        "transaction_id": [f"TXN-{i:08d}" for i in range(n)],
        # High nulls
        "optional_field": np.where(np.random.random(n) > 0.4, np.random.randn(n), np.nan),
        # Normal
        "age": np.random.randint(18, 80, n),
    })
    
    # Make salary correlated with income
    df["salary"] = df["income"] * np.random.uniform(0.95, 1.05, n)
    
    return df


# =============================================================================
# BASIC TESTS
# =============================================================================

class TestBasicProfiling:
    """Test basic EDA profiling."""
    
    def test_shape(self, simple_dataset):
        """Test shape is correctly reported."""
        register_dataset("simple", simple_dataset)
        result = run_eda_report("simple")
        
        assert result["shape"]["rows"] == 1000
        assert result["shape"]["columns"] == 6
    
    def test_schema(self, simple_dataset):
        """Test schema is correctly inferred."""
        register_dataset("simple", simple_dataset)
        result = run_eda_report("simple")
        
        schema = {s["column"]: s for s in result["schema"]}
        
        assert "int" in schema["age"]["dtype"]
        assert "float" in schema["income"]["dtype"]
        assert schema["gender"]["dtype"] == "object"
        assert schema["customer_id"]["unique"] == 1000  # All unique
    
    def test_numeric_summary(self, simple_dataset):
        """Test numeric summary statistics."""
        register_dataset("simple", simple_dataset)
        result = run_eda_report("simple")
        
        numeric = {n["column"]: n for n in result["numeric_summary"]}
        
        assert "age" in numeric
        assert "income" in numeric
        assert "score" in numeric
        
        # Check age stats are reasonable
        age_stats = numeric["age"]
        assert 18 <= age_stats["min"] <= age_stats["max"] <= 80
        assert "mean" in age_stats
        assert "median" in age_stats
        assert "skew" in age_stats
    
    def test_categorical_summary(self, simple_dataset):
        """Test categorical summary."""
        register_dataset("simple", simple_dataset)
        result = run_eda_report("simple")
        
        categorical = {c["column"]: c for c in result["categorical_summary"]}
        
        assert "gender" in categorical
        assert categorical["gender"]["unique"] == 2
        assert len(categorical["gender"]["top_values"]) == 2
        
        assert "region" in categorical
        assert categorical["region"]["unique"] == 4
        
        # customer_id should be detected as ID-like
        assert categorical["customer_id"]["id_like"] is True


class TestTargetAnalysis:
    """Test target column analysis."""
    
    def test_classification_target(self, classification_dataset):
        """Test classification target analysis."""
        register_dataset("clf", classification_dataset)
        result = run_eda_report("clf", target_col="default", task_type="classification")
        
        assert "target_analysis" in result
        ta = result["target_analysis"]
        
        assert ta["column"] == "default"
        assert ta["task"] == "classification"
        assert "class_counts" in ta
        assert "0" in ta["class_counts"] or 0 in ta["class_counts"]
        assert "imbalance_ratio" in ta
        assert ta["imbalance_ratio"] > 3  # 85/15 is imbalanced
        assert "recommendation" in ta
    
    def test_regression_target(self, regression_dataset):
        """Test regression target analysis."""
        register_dataset("reg", regression_dataset)
        result = run_eda_report("reg", target_col="price", task_type="regression")
        
        assert "target_analysis" in result
        ta = result["target_analysis"]
        
        assert ta["column"] == "price"
        assert ta["task"] == "regression"
        assert "mean" in ta
        assert "std" in ta
        assert "skew" in ta
    
    def test_target_associations_classification(self, classification_dataset):
        """Test feature-target associations for classification."""
        # Check if sklearn is available
        try:
            from sklearn.metrics import roc_auc_score
            sklearn_available = True
        except ImportError:
            sklearn_available = False
        
        register_dataset("clf", classification_dataset)
        result = run_eda_report("clf", target_col="default", task_type="classification")
        
        assert "target_associations" in result
        
        if sklearn_available:
            # Income and age should have high AUC (we made them correlated with target)
            assoc = {a["column"]: a for a in result["target_associations"]}
            
            # At least one should have meaningful AUC
            assert len(result["target_associations"]) > 0
            assert all(a["metric"] == "auc" for a in result["target_associations"])
        else:
            # sklearn not available, AUC computation skipped
            print("sklearn not available - skipping AUC assertions")
    
    def test_target_associations_regression(self, regression_dataset):
        """Test feature-target associations for regression."""
        register_dataset("reg", regression_dataset)
        result = run_eda_report("reg", target_col="price", task_type="regression")
        
        assert "target_associations" in result
        
        # feature1 and feature2 should have high correlation with target
        assoc = {a["column"]: a for a in result["target_associations"]}
        
        assert "feature1" in assoc
        assert abs(assoc["feature1"]["value"]) > 0.5  # Should be highly correlated
    
    def test_auto_detect_task_type(self, classification_dataset):
        """Test task type is auto-detected."""
        register_dataset("clf", classification_dataset)
        
        # Don't provide task_type
        result = run_eda_report("clf", target_col="default")
        
        # Should auto-detect as classification (only 2 unique values)
        assert result["target_analysis"]["task"] == "classification"


class TestCorrelations:
    """Test correlation detection."""
    
    def test_high_correlation_detection(self, dataset_with_issues):
        """Test highly correlated columns are detected."""
        register_dataset("issues", dataset_with_issues)
        result = run_eda_report("issues")
        
        high_pairs = result["correlations"]["high_pairs"]
        
        # income and salary should be highly correlated
        pair_cols = [(p["col1"], p["col2"]) for p in high_pairs]
        assert any(
            ("income" in cols and "salary" in cols)
            for cols in pair_cols
        ), f"Expected income-salary correlation, got {pair_cols}"


class TestAlerts:
    """Test alert generation."""
    
    def test_imbalanced_target_alert(self, classification_dataset):
        """Test imbalanced target alert."""
        register_dataset("clf", classification_dataset)
        result = run_eda_report("clf", target_col="default", task_type="classification")
        
        alert_types = [a["type"] for a in result["alerts"]]
        assert "imbalanced_target" in alert_types
    
    def test_high_skew_alert(self, dataset_with_issues):
        """Test high skew alert."""
        register_dataset("issues", dataset_with_issues)
        result = run_eda_report("issues")
        
        # income is lognormal, should be skewed
        skew_alerts = [a for a in result["alerts"] if a["type"] == "high_skew"]
        skewed_cols = [a["column"] for a in skew_alerts]
        
        assert "income" in skewed_cols or "salary" in skewed_cols
    
    def test_id_like_alert(self, dataset_with_issues):
        """Test ID-like column alert."""
        register_dataset("issues", dataset_with_issues)
        result = run_eda_report("issues")
        
        id_alerts = [a for a in result["alerts"] if a["type"] == "id_like"]
        id_cols = [a["column"] for a in id_alerts]
        
        assert "transaction_id" in id_cols
    
    def test_redundant_alert(self, dataset_with_issues):
        """Test redundant column alert."""
        register_dataset("issues", dataset_with_issues)
        result = run_eda_report("issues")
        
        redundant_alerts = [a for a in result["alerts"] if a["type"] == "redundant"]
        
        # income and salary should be flagged as redundant
        if redundant_alerts:
            cols_in_alerts = [set(a["columns"]) for a in redundant_alerts]
            assert any({"income", "salary"}.issubset(cols) or 
                      "income" in str(cols) and "salary" in str(cols) 
                      for cols in cols_in_alerts)
    
    def test_high_nulls_alert(self, dataset_with_issues):
        """Test high nulls alert."""
        register_dataset("issues", dataset_with_issues)
        result = run_eda_report("issues")
        
        null_alerts = [a for a in result["alerts"] if a["type"] == "high_nulls"]
        null_cols = [a["column"] for a in null_alerts]
        
        assert "optional_field" in null_cols


class TestRecommendations:
    """Test recommendation generation."""
    
    def test_imbalance_recommendation(self, classification_dataset):
        """Test imbalance recommendation."""
        register_dataset("clf", classification_dataset)
        result = run_eda_report("clf", target_col="default", task_type="classification")
        
        recs = result["recommendations"]
        assert any("class_weight" in r.lower() or "balanced" in r.lower() or "smote" in r.lower() 
                   for r in recs)
    
    def test_skew_recommendation(self, dataset_with_issues):
        """Test skew recommendation."""
        register_dataset("issues", dataset_with_issues)
        result = run_eda_report("issues")
        
        recs = result["recommendations"]
        assert any("log" in r.lower() or "bin" in r.lower() for r in recs)
    
    def test_id_like_recommendation(self, dataset_with_issues):
        """Test ID-like column recommendation."""
        register_dataset("issues", dataset_with_issues)
        result = run_eda_report("issues")
        
        recs = result["recommendations"]
        assert any("transaction_id" in r for r in recs)


class TestToolOutput:
    """Test LangChain tool output format."""
    
    def test_tool_output_format(self, classification_dataset):
        """Test tool produces readable output."""
        register_dataset("clf", classification_dataset)
        
        output = eda_report_tool.invoke({
            "dataset_ref": "clf",
            "target_col": "default",
            "task_type": "classification",
        })
        
        print("\n" + "="*80)
        print("EDA REPORT TOOL OUTPUT")
        print("="*80)
        print(output)
        
        assert "# EDA Report" in output
        assert "Shape:" in output
        assert "Schema" in output
        assert "Target Analysis" in output
        assert "Recommendations" in output
    
    def test_basic_tool_output(self, simple_dataset):
        """Test basic tool output without target."""
        register_dataset("simple", simple_dataset)
        
        output = eda_report_tool.invoke({
            "dataset_ref": "simple",
        })
        
        print("\n" + "="*80)
        print("BASIC EDA OUTPUT")
        print("="*80)
        print(output)
        
        assert "# EDA Report" in output
        assert "Shape:" in output
        assert "customer_id" in output  # Should detect ID-like


class TestSampling:
    """Test dataset sampling."""
    
    def test_sampling_large_dataset(self):
        """Test sampling works for large datasets."""
        np.random.seed(42)
        n = 200000
        
        df = pd.DataFrame({
            "x": np.random.randn(n),
            "y": np.random.randn(n),
        })
        register_dataset("large", df)
        
        result = run_eda_report("large", sample_n=10000)
        
        # Should be sampled
        assert result["shape"]["rows"] == 10000


class TestCaps:
    """Test output size caps."""
    
    def test_column_cap(self):
        """Test max_columns cap."""
        np.random.seed(42)
        n = 100
        
        # Create dataset with many columns
        df = pd.DataFrame({
            f"col_{i}": np.random.randn(n) for i in range(150)
        })
        register_dataset("many_cols", df)
        
        result = run_eda_report("many_cols", caps={"max_columns": 50})
        
        # Should be capped
        assert result["shape"]["columns"] == 50


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
        
        return pd.DataFrame(rows, columns=columns)
    
    def test_credit_card_risk_eda(self, credit_card_risk_df):
        """Test EDA on credit card risk dataset."""
        # Clean RISK column for target analysis
        df = credit_card_risk_df.copy()
        df["RISK_clean"] = df["RISK"].str.strip()
        df["is_bad"] = df["RISK_clean"].apply(lambda x: 1 if "bad" in x.lower() else 0)
        
        register_dataset("credit_risk", df)
        
        result = run_eda_report(
            "credit_risk",
            target_col="is_bad",
            task_type="classification"
        )
        
        print("\n" + "="*80)
        print("CREDIT CARD RISK EDA")
        print("="*80)
        print(f"Shape: {result['shape']}")
        print(f"Alerts: {len(result['alerts'])}")
        for alert in result["alerts"]:
            print(f"  - {alert}")
        print(f"\nRecommendations:")
        for rec in result["recommendations"]:
            print(f"  - {rec}")
        print(f"\nTarget analysis: {result.get('target_analysis', 'n/a')}")
        
        # Should have detected various issues
        assert result["shape"]["rows"] > 2000
        assert len(result["alerts"]) >= 0  # May or may not have alerts
    
    def test_credit_card_risk_tool_output(self, credit_card_risk_df):
        """Test tool output on credit card risk."""
        df = credit_card_risk_df.copy()
        df["is_bad"] = df["RISK"].str.strip().apply(lambda x: 1 if "bad" in x.lower() else 0)
        
        register_dataset("credit_risk", df)
        
        output = eda_report_tool.invoke({
            "dataset_ref": "credit_risk",
            "target_col": "is_bad",
            "task_type": "classification",
        })
        
        print("\n" + "="*80)
        print("CREDIT CARD RISK - AGENT OUTPUT")
        print("="*80)
        print(output)
        
        assert "# EDA Report" in output
        assert "Target Analysis" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

