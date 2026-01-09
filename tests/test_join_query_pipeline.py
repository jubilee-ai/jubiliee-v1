"""
Test the JOIN → SQL Query pipeline.
Verifies that after a join, the resulting table can be queried with SQL.
"""

import importlib.util
import re
import sys
from pathlib import Path


def load_module(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Setup module loading
base_path = Path(__file__).parent.parent / "tools" / "data-tools"


class FakePackage:
    pass


sys.modules["data_tools"] = FakePackage()

# Load modules
utils = load_module("data_tools.utils", base_path / "utils.py")
embedder = load_module("data_tools.embedder", base_path / "embedder.py")
sql_query = load_module("data_tools.sql_query", base_path / "sql_query.py")
data_loader = load_module("data_tools.data_loader", base_path / "data_loader.py")
retrieval = load_module("data_tools.retrieval", base_path / "retrieval.py")
join_merge = load_module("data_tools.join_merge", base_path / "join_merge.py")


def test_join_then_query():
    """Test that we can join two tables and then query the result with SQL."""
    print("=" * 60)
    print("TEST: JOIN → SQL QUERY PIPELINE")
    print("=" * 60)

    # Step 1: Check available tables
    print("\n1. Available SQL tables:")
    warehouse = sql_query.get_warehouse()
    print(f"   {list(warehouse.tables.keys())}")

    # Step 2: Check schemas
    print("\n2. Checking schemas for join compatibility...")
    loan_cols = [c["name"] for c in warehouse.tables["loan_default"].columns]
    insurance_cols = [c["name"] for c in warehouse.tables["insurance"].columns]
    print(f"   loan_default has 'Age': {'Age' in loan_cols}")
    print(f"   insurance has 'age': {'age' in insurance_cols}")

    # Step 3: Perform a join
    print("\n3. Performing join: loan_default + insurance on Age/age...")
    result = join_merge.join_merge_tool.invoke(
        {
            "left_ref": "loan_default",
            "right_ref": "insurance",
            "keys": {"Age": "age"},
            "join_type": "inner",
        }
    )

    # Extract the dataset_ref from the result
    # Try multiple patterns
    ref_match = re.search(r"\*\*Reference:\*\*\s*`([^`]+)`", result)
    if not ref_match:
        ref_match = re.search(r"Reference:\s*`([^`]+)`", result)
    if not ref_match:
        ref_match = re.search(r"dataset_ref.*?['\"]([^'\"]+)['\"]", result)
    join_ref = ref_match.group(1) if ref_match else None
    print(f"   Join created with ref: {join_ref}")

    if not join_ref:
        print("   ERROR: Could not extract join ref from result")
        print(f"   Result: {result[:500]}")
        return False

    # Step 4: Check if joined table is in SQL warehouse
    print("\n4. Checking SQL warehouse for joined table...")
    warehouse = sql_query.get_warehouse()
    all_tables = list(warehouse.tables.keys())
    print(f"   All tables now: {all_tables}")

    # The table name is sanitized (lowercase)
    sql_table_name = join_ref.lower().replace("-", "_")
    table_exists = sql_table_name in all_tables
    print(f"   Looking for: {sql_table_name}")
    print(f"   Table exists: {table_exists}")

    if not table_exists:
        print("   ERROR: Joined table not found in SQL warehouse!")
        return False

    # Step 5: Query the joined table
    print("\n5. Querying the joined table with SQL...")
    query = f"SELECT Age, Income, CreditScore, bmi, smoker FROM {sql_table_name} LIMIT 5"
    print(f"   Query: {query}")

    query_result = sql_query.sql_query_tool.invoke({"query": query})
    print(f"   Result: {query_result[:400]}...")

    # Step 6: Get dataset via registry
    print("\n6. Getting dataset via get_registered_dataset...")
    df = utils.get_registered_dataset(join_ref)
    if df is not None:
        print(f"   Got DataFrame: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Sample columns: {list(df.columns)[:8]}")
    else:
        print("   Not found in registry (checking disk...)")
        # Try loading from SQL
        df, _ = warehouse.execute_query(f"SELECT * FROM {sql_table_name} LIMIT 10")
        print(f"   Loaded from SQL: {len(df)} rows")

    # Cleanup
    print("\n7. Cleaning up...")
    utils.clear_registry(clear_disk=True, clear_sql=True)
    print("   Done!")

    print("\n" + "=" * 60)
    print("✓ PIPELINE TEST PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_join_then_query()
    exit(0 if success else 1)
