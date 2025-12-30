"""
Visual tests for transformations - shows actual data before/after.

Run with: pytest tests/test_transformations_visual.py -v -s
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from transformations import (
    select, drop, rename, cast, parse_datetime, add_column,
    transform_apply, transform_dataset,
)
from transformations.recipes import clean_for_ml, clean_basic, normalize_column_names
from utils import register_dataset, clear_registry, get_registered_dataset


def print_table(df: pd.DataFrame, title: str = "", max_rows: int = 10):
    """Print DataFrame as a formatted table."""
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes: {dict(df.dtypes)}")
    print(f"{'-'*60}")
    print(df.head(max_rows).to_string(index=True))
    print(f"{'='*60}\n")


@pytest.fixture(autouse=True)
def setup():
    clear_registry()
    yield
    clear_registry()


# =============================================================================
# LOAN DATA SCENARIO
# =============================================================================

class TestLoanDataTransformations:
    """Test transformations on realistic loan application data."""
    
    @pytest.fixture
    def loan_data(self):
        """Realistic loan application data with common issues."""
        return pd.DataFrame({
            "application_id": ["APP001", "APP002", "APP003", "APP004", "APP005", "APP006"],
            "applicant_name": ["John Smith", "Jane Doe", "Bob Wilson", "Alice Brown", "Charlie Davis", "Eva Martinez"],
            "age": ["35", "28", "45", "32", "55", "41"],  # Strings!
            "annual_income": [75000.0, 52000.0, 120000.0, 68000.0, 95000.0, 82000.0],
            "loan_amount": [25000, 15000, 50000, 20000, 35000, 28000],
            "credit_score": [720, 680, 750, 695, 710, 730],
            "employment_status": ["employed", "employed", "self-employed", "employed", "retired", "employed"],
            "application_date": ["2024-01-15", "2024-01-18", "2024-02-01", "2024-02-10", "2024-02-15", "2024-02-20"],
            "notes": [None, "First-time buyer", None, None, "Existing customer", None],  # 67% null
        })
    
    def test_select_columns_for_risk_model(self, loan_data):
        """Select only columns needed for risk assessment."""
        print_table(loan_data, "INPUT: Full Loan Application Data")
        
        result = select(["application_id", "annual_income", "loan_amount", "credit_score"]).execute(loan_data)
        
        print_table(result.df, "OUTPUT: Selected Columns for Risk Model")
        
        assert list(result.df.columns) == ["application_id", "annual_income", "loan_amount", "credit_score"]
        assert len(result.df) == 6
    
    def test_cast_age_and_add_risk_features(self, loan_data):
        """Cast age to int and add computed risk features."""
        print_table(loan_data[["application_id", "age", "annual_income", "loan_amount"]], 
                   "INPUT: Loan Data (age is string)")
        
        # Cast age to integer
        df1 = cast("age", "integer").execute(loan_data).df
        
        # Add debt-to-income ratio
        df2 = add_column("dti_ratio", "loan_amount / annual_income").execute(df1).df
        
        # Add risk tier
        result = add_column("risk_tier", "'low' if credit_score >= 720 else ('medium' if credit_score >= 680 else 'high')").execute(df2)
        
        print_table(result.df[["application_id", "age", "dti_ratio", "credit_score", "risk_tier"]], 
                   "OUTPUT: With age cast + computed features")
        
        assert result.df["age"].dtype.name == "Int64"
        assert all(result.df["dti_ratio"] > 0)
        assert set(result.df["risk_tier"].unique()) == {"low", "medium"}
    
    def test_full_pipeline_for_ml(self, loan_data):
        """Full transformation pipeline preparing data for ML."""
        print_table(loan_data, "INPUT: Raw Loan Application Data")
        
        result = transform_apply(loan_data, [
            # Drop high-null column
            {"op": "drop", "columns": ["notes"]},
            # Parse dates
            {"op": "parse_datetime", "column": "application_date"},
            # Cast types
            {"op": "cast", "column": "age", "dtype": "integer"},
            {"op": "cast", "column": "employment_status", "dtype": "category"},
            # Add features
            {"op": "add_column", "name": "dti_ratio", "expression": "round(loan_amount / annual_income, 3)"},
            {"op": "add_column", "name": "income_per_year_age", "expression": "annual_income / age"},
            {"op": "add_column", "name": "is_high_income", "expression": "annual_income >= 80000"},
            # Rename for clarity
            {"op": "rename", "mapping": {"annual_income": "income", "loan_amount": "loan"}},
        ])
        
        final_df = get_registered_dataset(result.dataset_ref)
        print_table(final_df, "OUTPUT: Transformed for ML")
        
        print(f"\nPipeline Summary:")
        print(f"  Operations: {result.ops_applied}")
        print(f"  Execution time: {result.execution_time_ms:.2f}ms")
        print(f"  Final columns: {[c['name'] for c in result.final_schema]}")
        
        assert result.success
        assert "dti_ratio" in final_df.columns
        assert "is_high_income" in final_df.columns
        assert pd.api.types.is_datetime64_any_dtype(final_df["application_date"])


# =============================================================================
# CUSTOMER DATA SCENARIO
# =============================================================================

class TestCustomerDataTransformations:
    """Test transformations on customer/CRM data."""
    
    @pytest.fixture
    def customer_data(self):
        """Customer data with messy formatting."""
        return pd.DataFrame({
            "CustomerID": ["  C001  ", "C002", " C003", "C004  ", "C005"],
            "FirstName": ["john", "JANE", "Bob", "alice", "CHARLIE"],
            "LastName": ["SMITH", "doe", "Wilson", "BROWN", "davis"],
            "Email": ["john@email.com", "jane@email.com", "bob@email.com", "alice@email.com", "charlie@email.com"],
            "SignupDate": ["01/15/2023", "02/20/2023", "03/10/2023", "04/05/2023", "05/12/2023"],
            "TotalPurchases": [1500.50, 2300.75, 890.00, 3400.25, 1200.00],
            "LastPurchaseDate": ["12/01/2024", "11/15/2024", "10/20/2024", "12/10/2024", "09/30/2024"],
            "Status": ["active", "active", "inactive", "active", "inactive"],
        })
    
    def test_normalize_column_names(self, customer_data):
        """Convert CamelCase columns to snake_case."""
        print_table(customer_data, "INPUT: Customer Data (CamelCase columns)")
        
        ops = normalize_column_names(customer_data)
        print(f"\nGenerated rename mapping: {ops[0]['mapping']}")
        
        result = transform_apply(customer_data, ops)
        final_df = get_registered_dataset(result.dataset_ref)
        
        print_table(final_df, "OUTPUT: snake_case column names")
        
        # Acronyms handled properly: CustomerID -> customer_id
        assert "customer_id" in final_df.columns
        assert "first_name" in final_df.columns
        assert "last_name" in final_df.columns
        assert "signup_date" in final_df.columns
        assert "CustomerID" not in final_df.columns  # Original name should be gone
    
    def test_clean_customer_ids(self, customer_data):
        """Strip whitespace from customer IDs."""
        print_table(customer_data[["CustomerID", "FirstName"]], "INPUT: CustomerID has whitespace")
        print(f"Raw CustomerID values: {customer_data['CustomerID'].tolist()}")
        
        result = add_column("CustomerID", "strip(CustomerID)", overwrite=True).execute(customer_data)
        
        print_table(result.df[["CustomerID", "FirstName"]], "OUTPUT: Cleaned CustomerID")
        print(f"Cleaned CustomerID values: {result.df['CustomerID'].tolist()}")
        
        assert result.df["CustomerID"].tolist() == ["C001", "C002", "C003", "C004", "C005"]
    
    def test_standardize_names(self, customer_data):
        """Capitalize names properly."""
        print_table(customer_data[["FirstName", "LastName"]], "INPUT: Inconsistent name casing")
        
        df = customer_data.copy()
        # Note: We'd need to add a 'title' or 'capitalize' function to do this properly
        # For now, let's just use upper/lower to demonstrate
        
        result = transform_apply(df, [
            {"op": "add_column", "name": "FullName", "expression": "FirstName + ' ' + LastName"},
            {"op": "add_column", "name": "NameUpper", "expression": "upper(FirstName)"},
        ])
        
        final_df = get_registered_dataset(result.dataset_ref)
        print_table(final_df[["FirstName", "LastName", "FullName", "NameUpper"]], 
                   "OUTPUT: With computed name columns")
    
    def test_customer_segmentation(self, customer_data):
        """Add customer segments based on purchase behavior."""
        print_table(customer_data[["CustomerID", "TotalPurchases", "Status"]], 
                   "INPUT: Customer purchase data")
        
        result = transform_apply(customer_data, [
            {"op": "add_column", "name": "ValueTier", 
             "expression": "'platinum' if TotalPurchases >= 3000 else ('gold' if TotalPurchases >= 1500 else 'silver')"},
            {"op": "add_column", "name": "IsHighValue", "expression": "TotalPurchases >= 2000"},
            {"op": "add_column", "name": "NeedsReactivation", 
             "expression": "Status == 'inactive'"},
        ])
        
        final_df = get_registered_dataset(result.dataset_ref)
        print_table(final_df[["CustomerID", "TotalPurchases", "ValueTier", "IsHighValue", "NeedsReactivation"]], 
                   "OUTPUT: With customer segments")
        
        # Verify segmentation
        assert final_df[final_df["TotalPurchases"] >= 3000]["ValueTier"].iloc[0] == "platinum"
        assert final_df[final_df["Status"] == "inactive"]["NeedsReactivation"].all()


# =============================================================================
# INSURANCE CLAIMS SCENARIO
# =============================================================================

class TestInsuranceClaimsTransformations:
    """Test transformations on insurance claims data."""
    
    @pytest.fixture
    def claims_data(self):
        """Insurance claims with various data quality issues."""
        return pd.DataFrame({
            "claim_id": ["CLM-2024-001", "CLM-2024-002", "CLM-2024-003", "CLM-2024-004", "CLM-2024-005"],
            "policy_holder": ["John Smith", "Jane Doe", "Bob Wilson", "Alice Brown", "Charlie Davis"],
            "claim_type": ["auto", "home", "auto", "health", "auto"],
            "claim_amount": ["5000.50", "12000.00", "3500.75", "8500.00", "15000.25"],  # Strings!
            "deductible": [500, 1000, 500, 250, 500],
            "incident_date": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "2024-05-12"],
            "filed_date": ["2024-01-18", "2024-02-25", "2024-03-12", "2024-04-10", "2024-05-15"],
            "status": ["approved", "pending", "approved", "denied", "pending"],
            "adjuster_notes": [None, "Under review", None, "Pre-existing condition", None],  # 60% null
        })
    
    def test_clean_claims_for_analysis(self, claims_data):
        """Apply clean_for_ml recipe to claims data."""
        print_table(claims_data, "INPUT: Raw Insurance Claims")
        
        ops = clean_for_ml(claims_data)
        print(f"\nRecipe generated {len(ops)} operations:")
        for op in ops:
            print(f"  - {op}")
        
        result = transform_apply(claims_data, ops)
        final_df = get_registered_dataset(result.dataset_ref)
        
        print_table(final_df, "OUTPUT: Cleaned for ML")
        
        # adjuster_notes should be dropped (60% null)
        assert "adjuster_notes" not in final_df.columns
        # Dates should be parsed
        assert pd.api.types.is_datetime64_any_dtype(final_df["incident_date"])
        # claim_amount (numeric string) should be float, not category
        assert final_df["claim_amount"].dtype == "float64"
    
    def test_claims_processing_pipeline(self, claims_data):
        """Full claims processing transformation."""
        print_table(claims_data, "INPUT: Raw Claims Data")
        
        result = transform_apply(claims_data, [
            # Drop notes column
            {"op": "drop", "columns": ["adjuster_notes"]},
            # Cast claim_amount to float
            {"op": "cast", "column": "claim_amount", "dtype": "float"},
            # Parse dates
            {"op": "parse_datetime", "column": "incident_date"},
            {"op": "parse_datetime", "column": "filed_date"},
            # Cast claim_type to category
            {"op": "cast", "column": "claim_type", "dtype": "category"},
            {"op": "cast", "column": "status", "dtype": "category"},
            # Add computed columns
            {"op": "add_column", "name": "net_claim", "expression": "claim_amount - deductible"},
            {"op": "add_column", "name": "is_large_claim", "expression": "claim_amount >= 10000"},
            {"op": "add_column", "name": "is_approved", "expression": "status == 'approved'"},
        ])
        
        final_df = get_registered_dataset(result.dataset_ref)
        print_table(final_df, "OUTPUT: Processed Claims")
        
        # Verify transformations
        assert final_df["claim_amount"].dtype == "float64"
        assert final_df["net_claim"].iloc[0] == 5000.50 - 500
        assert final_df["is_large_claim"].sum() == 2  # Two claims >= 10000


# =============================================================================
# AGENT TOOL VISUAL TEST
# =============================================================================

class TestAgentToolVisual:
    """Visual test of the agent-facing tool."""
    
    @pytest.fixture
    def sales_data(self):
        return pd.DataFrame({
            "OrderID": ["ORD-001", "ORD-002", "ORD-003", "ORD-004", "ORD-005"],
            "Product": ["Widget A", "Widget B", "Gadget X", "Widget A", "Gadget Y"],
            "Quantity": [10, 5, 3, 8, 12],
            "UnitPrice": [25.00, 45.00, 120.00, 25.00, 85.00],
            "OrderDate": ["2024-11-01", "2024-11-05", "2024-11-10", "2024-11-15", "2024-11-20"],
            "Region": ["North", "South", "North", "East", "West"],
        })
    
    def test_agent_tool_full_workflow(self, sales_data):
        """Simulate how an AI agent would use the tool."""
        register_dataset("sales_data", sales_data)
        
        print_table(sales_data, "INPUT: Sales Data (registered as 'sales_data')")
        
        # Agent invokes the tool
        result = transform_dataset.invoke({
            "dataset_ref": "sales_data",
            "add_features": {
                "TotalValue": "Quantity * UnitPrice",
                "IsHighValue": "Quantity * UnitPrice >= 500",
            },
            "parse_dates": ["OrderDate"],
            "cast_types": {"Region": "category"},
        })
        
        print("\n" + "="*60)
        print("  AGENT TOOL RESPONSE")
        print("="*60)
        print(result)
        
        # Get the transformed data
        # Extract dataset_ref from the result
        import re
        match = re.search(r'`(ds_[a-f0-9_]+)`', result)
        if match:
            ref = match.group(1)
            final_df = get_registered_dataset(ref)
            print_table(final_df, "TRANSFORMED DATA")
            
            assert "TotalValue" in final_df.columns
            assert final_df["TotalValue"].iloc[0] == 10 * 25.00


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

