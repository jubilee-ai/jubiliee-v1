"""
Tests for data transformation tools.

Tests:
- Individual transforms (select, drop, rename, cast, parse_datetime, add_column)
- Recipes (clean_for_ml, clean_basic, prepare_for_join, normalize_names)
- Agent tool (transform_dataset)
- Pipeline orchestration (transform_apply)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

from transformations import (
    select, drop, rename, cast, parse_datetime, add_column,
    transform_apply, transform_dataset, list_recipes,
)
from transformations.recipes import (
    clean_for_ml, clean_basic, prepare_for_join, 
    normalize_column_names, select_features,
)
from utils import register_dataset, clear_registry


# =============================================================================
# MOCK DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_df():
    """Basic sample DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, 35, 28, 42],
        "salary": [50000, 60000, 75000, 55000, 90000],
        "department": ["Sales", "Engineering", "Engineering", "Sales", "Marketing"],
    })


@pytest.fixture
def messy_df():
    """Messy DataFrame with type issues, nulls, and date strings."""
    return pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5],
        "UserName": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age_str": ["25", "30", "35", "28", "42"],  # Strings that should be int
        "Salary": [50000.0, 60000.0, None, 55000.0, 90000.0],  # Has null
        "created_at": ["2020-01-15", "2019-06-20", "2018-03-10", "2021-09-01", "2017-11-30"],
        "notes": [None, None, "VIP", None, "VIP"],  # High null rate (60%)
        "status": ["active", "inactive", "active", "active", "inactive"],  # Low cardinality
    })


@pytest.fixture
def join_df():
    """DataFrame for join preparation testing."""
    return pd.DataFrame({
        "customer_id": ["  ABC123  ", "DEF456", " GHI789 ", "JKL012"],
        "order_id": [1, 2, 3, 4],
        "amount": [100.50, 200.75, 150.25, 300.00],
    })


@pytest.fixture
def camel_case_df():
    """DataFrame with camelCase and mixed column names."""
    return pd.DataFrame({
        "userId": [1, 2, 3],
        "firstName": ["John", "Jane", "Bob"],
        "lastName": ["Doe", "Smith", "Jones"],
        "emailAddress": ["john@example.com", "jane@example.com", "bob@example.com"],
        "createdAt": ["2020-01-01", "2020-02-01", "2020-03-01"],
    })


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the dataset registry before each test."""
    clear_registry()
    yield
    clear_registry()


# =============================================================================
# INDIVIDUAL TRANSFORM TESTS
# =============================================================================

class TestSelectTransform:
    """Tests for select transform."""
    
    def test_select_columns(self, sample_df):
        """Select specific columns."""
        # Input
        print(f"\nInput columns: {list(sample_df.columns)}")
        
        # Transform
        result = select(["id", "name"]).execute(sample_df)
        
        # Output
        print(f"Output columns: {list(result.df.columns)}")
        print(f"Audit: {result.audit.to_dict()}")
        
        assert list(result.df.columns) == ["id", "name"]
        assert result.audit.success
        assert len(result.audit.schema_changes) == 3  # Removed 3 columns
    
    def test_select_missing_column_warns(self, sample_df):
        """Missing columns generate warnings, not errors."""
        result = select(["id", "nonexistent"]).execute(sample_df)
        
        print(f"\nOutput columns: {list(result.df.columns)}")
        print(f"Warnings: {[w.message for w in result.audit.warnings]}")
        
        assert list(result.df.columns) == ["id"]
        assert len(result.audit.warnings) == 1
        assert "nonexistent" in result.audit.warnings[0].message


class TestDropTransform:
    """Tests for drop transform."""
    
    def test_drop_columns(self, sample_df):
        """Drop specific columns."""
        print(f"\nInput columns: {list(sample_df.columns)}")
        
        result = drop(["age", "salary"]).execute(sample_df)
        
        print(f"Output columns: {list(result.df.columns)}")
        
        assert "age" not in result.df.columns
        assert "salary" not in result.df.columns
        assert "id" in result.df.columns
        assert result.audit.success


class TestRenameTransform:
    """Tests for rename transform."""
    
    def test_rename_columns(self, sample_df):
        """Rename columns according to mapping."""
        print(f"\nInput columns: {list(sample_df.columns)}")
        
        result = rename({"name": "full_name", "salary": "annual_pay"}).execute(sample_df)
        
        print(f"Output columns: {list(result.df.columns)}")
        print(f"Schema changes: {[sc.to_dict() for sc in result.audit.schema_changes]}")
        
        assert "full_name" in result.df.columns
        assert "annual_pay" in result.df.columns
        assert "name" not in result.df.columns
        assert len(result.audit.schema_changes) == 2


class TestCastTransform:
    """Tests for cast transform."""
    
    def test_cast_string_to_integer(self, messy_df):
        """Cast string column to integer."""
        print(f"\nInput dtype: {messy_df['age_str'].dtype}")
        print(f"Input values: {messy_df['age_str'].tolist()}")
        
        result = cast("age_str", "integer").execute(messy_df)
        
        print(f"Output dtype: {result.df['age_str'].dtype}")
        print(f"Output values: {result.df['age_str'].tolist()}")
        
        assert "int" in str(result.df["age_str"].dtype).lower()
        assert result.df["age_str"].tolist() == [25, 30, 35, 28, 42]
    
    def test_cast_to_category(self, sample_df):
        """Cast string column to category."""
        result = cast("department", "category").execute(sample_df)
        
        print(f"\nOutput dtype: {result.df['department'].dtype}")
        
        assert result.df["department"].dtype.name == "category"


class TestParseDatetimeTransform:
    """Tests for parse_datetime transform."""
    
    def test_parse_date_strings(self, messy_df):
        """Parse date strings to datetime."""
        print(f"\nInput dtype: {messy_df['created_at'].dtype}")
        print(f"Input values: {messy_df['created_at'].tolist()[:3]}")
        
        result = parse_datetime("created_at").execute(messy_df)
        
        print(f"Output dtype: {result.df['created_at'].dtype}")
        print(f"Output values: {result.df['created_at'].tolist()[:3]}")
        
        assert pd.api.types.is_datetime64_any_dtype(result.df["created_at"])
        assert result.df["created_at"].dt.year.tolist() == [2020, 2019, 2018, 2021, 2017]
    
    def test_parse_with_format(self):
        """Parse with explicit format."""
        df = pd.DataFrame({"date": ["15-01-2020", "20-06-2019", "10-03-2018"]})
        
        result = parse_datetime("date", format="%d-%m-%Y").execute(df)
        
        print(f"\nInput: {df['date'].tolist()}")
        print(f"Output years: {result.df['date'].dt.year.tolist()}")
        
        assert result.df["date"].dt.year.tolist() == [2020, 2019, 2018]


class TestAddColumnTransform:
    """Tests for add_column transform."""
    
    def test_arithmetic_expression(self, sample_df):
        """Add column with arithmetic expression."""
        result = add_column("double_salary", "salary * 2").execute(sample_df)
        
        print(f"\nInput salary: {sample_df['salary'].tolist()}")
        print(f"Output double_salary: {result.df['double_salary'].tolist()}")
        
        assert result.df["double_salary"].tolist() == [100000, 120000, 150000, 110000, 180000]
    
    def test_ternary_expression(self, sample_df):
        """Add column with ternary/conditional expression."""
        result = add_column("salary_tier", "'high' if salary >= 60000 else 'normal'").execute(sample_df)
        
        print(f"\nInput salary: {sample_df['salary'].tolist()}")
        print(f"Output salary_tier: {result.df['salary_tier'].tolist()}")
        
        expected = ["normal", "high", "high", "normal", "high"]
        assert result.df["salary_tier"].tolist() == expected
    
    def test_comparison_expression(self, sample_df):
        """Add column with comparison expression."""
        result = add_column("is_senior", "age >= 35").execute(sample_df)
        
        print(f"\nInput age: {sample_df['age'].tolist()}")
        print(f"Output is_senior: {result.df['is_senior'].tolist()}")
        
        expected = [False, False, True, False, True]
        assert result.df["is_senior"].tolist() == expected
    
    def test_function_expression(self, sample_df):
        """Add column using built-in functions."""
        result = add_column("log_salary", "log(salary)").execute(sample_df)
        
        print(f"\nInput salary: {sample_df['salary'].tolist()}")
        print(f"Output log_salary: {[round(x, 2) for x in result.df['log_salary'].tolist()]}")
        
        assert all(result.df["log_salary"] > 0)
        assert result.df["log_salary"].iloc[0] == pytest.approx(np.log(50000), rel=0.01)


# =============================================================================
# RECIPE TESTS
# =============================================================================

class TestCleanForMLRecipe:
    """Tests for clean_for_ml recipe."""
    
    def test_drops_high_null_columns(self, messy_df):
        """Recipe should drop columns with >50% nulls."""
        print(f"\nInput columns: {list(messy_df.columns)}")
        print(f"Null rates: {messy_df.isnull().mean().to_dict()}")
        
        ops = clean_for_ml(messy_df)
        print(f"Generated ops: {ops}")
        
        # Should have a drop op for 'notes' column (60% null)
        drop_ops = [op for op in ops if op["op"] == "drop"]
        assert len(drop_ops) == 1
        assert "notes" in drop_ops[0]["columns"]
    
    def test_parses_date_columns(self, messy_df):
        """Recipe should parse obvious date columns."""
        ops = clean_for_ml(messy_df)
        
        datetime_ops = [op for op in ops if op["op"] == "parse_datetime"]
        print(f"\nDatetime ops: {datetime_ops}")
        
        assert any(op["column"] == "created_at" for op in datetime_ops)
    
    def test_casts_low_cardinality_to_category(self, messy_df):
        """Recipe should cast low-cardinality strings to category."""
        ops = clean_for_ml(messy_df)
        
        cast_ops = [op for op in ops if op["op"] == "cast" and op.get("dtype") == "category"]
        print(f"\nCategory cast ops: {cast_ops}")
        
        # 'status' has only 2 unique values
        assert any(op["column"] == "status" for op in cast_ops)


class TestCleanBasicRecipe:
    """Tests for clean_basic recipe."""
    
    def test_parses_dates_only(self, messy_df):
        """Recipe should only parse date columns."""
        ops = clean_basic(messy_df)
        
        print(f"\nGenerated ops: {ops}")
        
        # Should only have parse_datetime ops
        assert all(op["op"] == "parse_datetime" for op in ops)
        assert any(op["column"] == "created_at" for op in ops)


class TestPrepareForJoinRecipe:
    """Tests for prepare_for_join recipe."""
    
    def test_strips_whitespace_from_keys(self, join_df):
        """Recipe should strip whitespace from join key columns."""
        print(f"\nInput customer_id: {join_df['customer_id'].tolist()}")
        
        ops = prepare_for_join(join_df, {"key_columns": ["customer_id"]})
        print(f"Generated ops: {ops}")
        
        # Apply the ops
        result = transform_apply(join_df, ops)
        print(f"Output customer_id: {result.audits}")
        
        # The add_column with strip should be generated
        add_ops = [op for op in ops if op["op"] == "add_column"]
        assert len(add_ops) == 1
        assert "strip" in add_ops[0]["expression"]


class TestNormalizeNamesRecipe:
    """Tests for normalize_column_names recipe."""
    
    def test_converts_to_snake_case(self, camel_case_df):
        """Recipe should convert camelCase to snake_case."""
        print(f"\nInput columns: {list(camel_case_df.columns)}")
        
        ops = normalize_column_names(camel_case_df)
        print(f"Generated ops: {ops}")
        
        # Apply the ops
        result = transform_apply(camel_case_df, ops)
        output_cols = list(result.final_schema)
        
        print(f"Output columns: {[c['name'] for c in output_cols]}")
        
        # Should have rename op with snake_case mapping
        assert len(ops) == 1
        assert ops[0]["op"] == "rename"
        mapping = ops[0]["mapping"]
        assert "userId" in mapping
        assert mapping["userId"] == "user_id"


class TestSelectFeaturesRecipe:
    """Tests for select_features recipe."""
    
    def test_selects_specified_columns(self, sample_df):
        """Recipe should select only specified columns."""
        ops = select_features(sample_df, {"columns": ["id", "name", "salary"]})
        
        print(f"\nGenerated ops: {ops}")
        
        assert len(ops) == 1
        assert ops[0]["op"] == "select"
        assert ops[0]["columns"] == ["id", "name", "salary"]


# =============================================================================
# TRANSFORM_APPLY TESTS
# =============================================================================

class TestTransformApply:
    """Tests for transform_apply orchestrator."""
    
    def test_pipeline_execution(self, sample_df):
        """Execute a multi-step pipeline."""
        print(f"\nInput: {len(sample_df)} rows, {len(sample_df.columns)} columns")
        
        result = transform_apply(sample_df, [
            {"op": "select", "columns": ["id", "name", "salary"]},
            {"op": "add_column", "name": "bonus", "expression": "salary * 0.1"},
            {"op": "rename", "mapping": {"salary": "base_pay"}},
        ])
        
        print(f"Output: {result.final_rows} rows, {len(result.final_schema)} columns")
        print(f"Final schema: {[c['name'] for c in result.final_schema]}")
        print(f"Ops applied: {result.ops_applied}/{result.ops_total}")
        print(f"Execution time: {result.execution_time_ms:.2f}ms")
        
        assert result.success
        assert result.ops_applied == 3
        assert [c["name"] for c in result.final_schema] == ["id", "name", "base_pay", "bonus"]
    
    def test_dry_run_validates_without_executing(self, sample_df):
        """Dry run should validate but not modify data."""
        result = transform_apply(sample_df, [
            {"op": "select", "columns": ["id", "nonexistent"]},
            {"op": "cast", "column": "id", "dtype": "float"},
        ], dry_run=True)
        
        print(f"\nDry run success: {result.success}")
        print(f"Warnings: {result.warnings_total}")
        
        assert result.success
        assert result.dataset_ref == "[dry_run]"
        assert result.warnings_total == 1  # nonexistent column warning
    
    def test_lineage_tracking(self, sample_df):
        """Lineage should track all steps."""
        result = transform_apply(sample_df, [
            {"op": "select", "columns": ["id", "name"]},
            {"op": "add_column", "name": "upper_name", "expression": "upper(name)"},
        ], return_lineage=True)
        
        print(f"\nLineage: {result.lineage.to_dict()}")
        
        assert result.lineage is not None
        assert len(result.lineage.steps) == 2
        assert result.lineage.steps[0].op_name == "select"
        assert result.lineage.steps[1].op_name == "add_column"


# =============================================================================
# AGENT TOOL TESTS
# =============================================================================

class TestTransformDatasetTool:
    """Tests for the agent-facing transform_dataset tool."""
    
    def test_recipe_invocation(self, messy_df):
        """Agent can invoke with a recipe."""
        register_dataset("messy_data", messy_df)
        
        result = transform_dataset.invoke({
            "dataset_ref": "messy_data",
            "recipe": "clean_basic",
        })
        
        print(f"\nTool output:\n{result}")
        
        assert "SUCCESS" in result
        assert "parse_datetime" in result
    
    def test_declarative_options(self, sample_df):
        """Agent can use declarative options."""
        register_dataset("sample_data", sample_df)
        
        result = transform_dataset.invoke({
            "dataset_ref": "sample_data",
            "keep_columns": ["id", "name", "salary"],
            "add_features": {"is_high_earner": "salary >= 60000"},
        })
        
        print(f"\nTool output:\n{result}")
        
        assert "SUCCESS" in result
        assert "is_high_earner" in result
        assert "Columns: 4" in result  # id, name, salary, is_high_earner
    
    def test_combined_recipe_and_options(self, camel_case_df):
        """Agent can combine recipe with additional options."""
        register_dataset("camel_data", camel_case_df)
        
        # Note: parse_dates runs AFTER recipe, so use the renamed column name
        result = transform_dataset.invoke({
            "dataset_ref": "camel_data",
            "recipe": "normalize_names",
            "parse_dates": ["created_at"],  # Uses new name after normalize_names renames it
        })
        
        print(f"\nTool output:\n{result}")
        
        assert "SUCCESS" in result
        assert "datetime" in result
    
    def test_dry_run_with_warnings(self, sample_df):
        """Agent can do dry run to check for issues."""
        register_dataset("sample_data", sample_df)
        
        result = transform_dataset.invoke({
            "dataset_ref": "sample_data",
            "keep_columns": ["id", "nonexistent_column"],
            "dry_run": True,
        })
        
        print(f"\nTool output:\n{result}")
        
        assert "DRY RUN" in result
        assert "Warnings: 1" in result
        assert "not found" in result
    
    def test_cast_types(self, sample_df):
        """Agent can cast column types."""
        register_dataset("sample_data", sample_df)
        
        result = transform_dataset.invoke({
            "dataset_ref": "sample_data",
            "cast_types": {"age": "float", "salary": "float"},
        })
        
        print(f"\nTool output:\n{result}")
        
        assert "SUCCESS" in result
        assert "float" in result


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Handle empty DataFrame."""
        df = pd.DataFrame({"a": [], "b": []})
        
        result = transform_apply(df, [
            {"op": "add_column", "name": "c", "expression": "a + b"},
        ])
        
        print(f"\nEmpty df result: {result.success}, rows: {result.final_rows}")
        
        assert result.success
        assert result.final_rows == 0
    
    def test_all_null_column(self):
        """Handle column with all nulls."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
        
        result = cast("b", "float").execute(df)
        
        print(f"\nAll-null column cast: dtype={result.df['b'].dtype}")
        
        assert result.audit.success
    
    def test_invalid_expression(self, sample_df):
        """Invalid expression should fail gracefully."""
        with pytest.raises(ValueError) as exc_info:
            add_column("bad", "undefined_var + 1").execute(sample_df)
        
        print(f"\nError message: {exc_info.value}")
        
        assert "Unknown" in str(exc_info.value)
    
    def test_unknown_recipe(self, sample_df):
        """Unknown recipe should fail with helpful message."""
        register_dataset("sample_data", sample_df)
        
        result = transform_dataset.invoke({
            "dataset_ref": "sample_data",
            "recipe": "nonexistent_recipe",
        })
        
        print(f"\nUnknown recipe result:\n{result}")
        
        assert "Error" in result
        assert "nonexistent_recipe" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

