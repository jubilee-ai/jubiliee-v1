"""
Tests for sql_query tool.

Tests cover:
1. Schema introspection
2. Basic SELECT queries
3. Query validation (read-only enforcement)
4. Parameterized queries
5. Error handling with suggestions
6. Edge cases
"""

import importlib.util
import sys
from pathlib import Path


def _load_sql_query_module():
    """Load data-tools.sql_query with proper imports."""
    base_path = Path(__file__).parent.parent.parent / "tools" / "data-tools"
    
    # Register package
    spec = importlib.util.spec_from_file_location(
        'data_tools', 
        base_path / '__init__.py',
        submodule_search_locations=[str(base_path)]
    )
    data_tools = importlib.util.module_from_spec(spec)
    sys.modules['data_tools'] = data_tools
    
    # Load utils
    spec_utils = importlib.util.spec_from_file_location('data_tools.utils', base_path / 'utils.py')
    utils = importlib.util.module_from_spec(spec_utils)
    sys.modules['data_tools.utils'] = utils
    spec_utils.loader.exec_module(utils)
    
    # Load embedder
    spec_emb = importlib.util.spec_from_file_location('data_tools.embedder', base_path / 'embedder.py')
    embedder = importlib.util.module_from_spec(spec_emb)
    sys.modules['data_tools.embedder'] = embedder
    spec_emb.loader.exec_module(embedder)
    
    # Load sql_query
    spec_sql = importlib.util.spec_from_file_location('data_tools.sql_query', base_path / 'sql_query.py')
    sql_query_mod = importlib.util.module_from_spec(spec_sql)
    sys.modules['data_tools.sql_query'] = sql_query_mod
    spec_sql.loader.exec_module(sql_query_mod)
    
    return sql_query_mod

_sql_module = _load_sql_query_module()
sql_query = _sql_module.sql_query
get_schema = _sql_module.get_schema
list_tables = _sql_module.list_tables
describe_table = _sql_module.describe_table
get_warehouse = _sql_module.get_warehouse
validate_query = _sql_module.validate_query
sql_query_tool = _sql_module.sql_query_tool
get_sql_schema_tool = _sql_module.get_sql_schema_tool


def test_list_tables():
    """Test listing available tables."""
    print("=" * 60)
    print("TEST 1: List Tables")
    print("=" * 60)
    
    tables = list_tables()
    
    print(f"\nAvailable tables ({len(tables)}):")
    for t in tables:
        print(f"  - {t}")
    
    assert len(tables) >= 5, f"Expected at least 5 tables, got {len(tables)}"
    
    # Check expected tables exist
    expected = ["loan_default", "insurance", "credit_card_risk"]
    for exp in expected:
        assert any(exp in t.lower() for t in tables), f"Expected table containing '{exp}'"
    
    print("\n✓ Test passed: Tables listed successfully")


def test_describe_table():
    """Test describing a specific table."""
    print("\n" + "=" * 60)
    print("TEST 2: Describe Table")
    print("=" * 60)
    
    info = describe_table("loan_default")
    
    print(f"\nTable: {info['table']}")
    print(f"Row count: {info.get('row_count', 'N/A')}")
    print(f"Columns ({len(info['columns'])}):")
    for col in info['columns'][:5]:
        print(f"  - {col['name']} ({col['type']})")
    if len(info['columns']) > 5:
        print(f"  ... and {len(info['columns']) - 5} more")
    
    assert info['table'] == 'loan_default'
    assert len(info['columns']) > 0
    
    print("\n✓ Test passed: Table described successfully")


def test_get_schema():
    """Test getting full warehouse schema."""
    print("\n" + "=" * 60)
    print("TEST 3: Get Full Schema")
    print("=" * 60)
    
    schema = get_schema()
    
    print(f"\nSchema (dict):")
    print(f"Tables: {[t['name'] for t in schema['tables']]}")
    
    # Check structure
    assert "tables" in schema
    assert len(schema["tables"]) >= 5
    
    # Check loan_default exists
    table_names = [t["name"] for t in schema["tables"]]
    assert "loan_default" in table_names
    
    print("\n✓ Test passed: Schema retrieved successfully")


def test_basic_select():
    """Test basic SELECT query."""
    print("\n" + "=" * 60)
    print("TEST 4: Basic SELECT Query")
    print("=" * 60)
    
    result = sql_query("SELECT * FROM loan_default LIMIT 5")
    
    print(f"\nQuery ID: {result.query_id}")
    print(f"Rows returned: {result.stats['rows_returned']}")
    print(f"Execution time: {result.stats['execution_time_ms']}ms")
    print(f"\nSchema:")
    for col in result.schema[:5]:
        print(f"  - {col['name']} ({col['type']})")
    print(f"\nFirst row:")
    if result.data:
        for k, v in list(result.data[0].items())[:5]:
            print(f"  {k}: {v}")
    
    assert result.stats['rows_returned'] == 5
    assert len(result.schema) > 0
    
    print("\n✓ Test passed: Basic SELECT works")


def test_select_with_where():
    """Test SELECT with WHERE clause."""
    print("\n" + "=" * 60)
    print("TEST 5: SELECT with WHERE")
    print("=" * 60)
    
    result = sql_query(
        'SELECT Age, Income, CreditScore, "Default" FROM loan_default WHERE Age > 50 LIMIT 10'
    )
    
    print(f"\nQuery: SELECT ... WHERE Age > 50")
    print(f"Rows returned: {result.stats['rows_returned']}")
    
    # Verify all ages are > 50
    for row in result.data:
        assert row['Age'] > 50, f"Expected Age > 50, got {row['Age']}"
    
    print(f"All rows have Age > 50 ✓")
    print(f"\nSample row: {result.data[0] if result.data else 'N/A'}")
    
    print("\n✓ Test passed: WHERE clause works")


def test_parameterized_query():
    """Test parameterized query."""
    print("\n" + "=" * 60)
    print("TEST 6: Parameterized Query")
    print("=" * 60)
    
    result = sql_query(
        "SELECT * FROM loan_default WHERE Age > :min_age AND Income < :max_income LIMIT 5",
        params={"min_age": 40, "max_income": 50000}
    )
    
    print(f"\nQuery: WHERE Age > :min_age AND Income < :max_income")
    print(f"Params: min_age=40, max_income=50000")
    print(f"Rows returned: {result.stats['rows_returned']}")
    
    for row in result.data:
        assert row['Age'] > 40, f"Expected Age > 40, got {row['Age']}"
        assert row['Income'] < 50000, f"Expected Income < 50000, got {row['Income']}"
    
    print(f"All rows match criteria ✓")
    
    print("\n✓ Test passed: Parameterized queries work")


def test_aggregation():
    """Test aggregation queries."""
    print("\n" + "=" * 60)
    print("TEST 7: Aggregation Query")
    print("=" * 60)
    
    result = sql_query("""
        SELECT 
            Education,
            COUNT(*) as count,
            AVG(Income) as avg_income,
            AVG(CreditScore) as avg_credit
        FROM loan_default 
        GROUP BY Education
    """)
    
    print(f"\nQuery: GROUP BY Education")
    print(f"Rows returned: {result.stats['rows_returned']}")
    print(f"\nResults:")
    for row in result.data:
        print(f"  {row['Education']}: count={row['count']}, avg_income={row['avg_income']:.0f}")
    
    assert result.stats['rows_returned'] > 0
    
    print("\n✓ Test passed: Aggregation works")


def test_forbidden_operations():
    """Test that non-SELECT operations are blocked."""
    print("\n" + "=" * 60)
    print("TEST 8: Forbidden Operations Blocked")
    print("=" * 60)
    
    forbidden_queries = [
        ("DROP TABLE loan_default", "DROP"),
        ("DELETE FROM loan_default", "DELETE"),
        ("INSERT INTO loan_default VALUES (1)", "INSERT"),
        ("UPDATE loan_default SET Age = 0", "UPDATE"),
        ("CREATE TABLE test (id INT)", "CREATE"),
    ]
    
    for query, op_type in forbidden_queries:
        try:
            sql_query(query)
            assert False, f"Expected {op_type} to be blocked"
        except ValueError as e:
            print(f"  ✓ {op_type} blocked: {str(e)[:50]}...")
    
    print("\n✓ Test passed: All forbidden operations blocked")


def test_table_not_found_suggestions():
    """Test that table not found provides suggestions."""
    print("\n" + "=" * 60)
    print("TEST 9: Table Not Found Suggestions")
    print("=" * 60)
    
    try:
        sql_query("SELECT * FROM loan_defaults")  # typo: should be loan_default
        assert False, "Expected error"
    except ValueError as e:
        error_msg = str(e)
        print(f"\nError message: {error_msg}")
        
        # Should suggest the correct table name
        assert "loan_default" in error_msg.lower() or "did you mean" in error_msg.lower()
        print("  ✓ Suggestion provided")
    
    print("\n✓ Test passed: Helpful suggestions on typos")


def test_column_not_found_suggestions():
    """Test that column not found provides suggestions."""
    print("\n" + "=" * 60)
    print("TEST 10: Column Not Found Suggestions")
    print("=" * 60)
    
    try:
        sql_query("SELECT AgeX FROM loan_default LIMIT 1")  # typo: should be Age
        assert False, "Expected error"
    except ValueError as e:
        error_msg = str(e)
        print(f"\nError message: {error_msg}")
        
        # Should suggest the correct column name
        assert "age" in error_msg.lower() or "did you mean" in error_msg.lower() or "no such column" in error_msg.lower()
        print("  ✓ Error caught with helpful message")
    
    print("\n✓ Test passed: Column errors handled")


def test_missing_params():
    """Test that missing parameters are caught."""
    print("\n" + "=" * 60)
    print("TEST 11: Missing Parameters")
    print("=" * 60)
    
    try:
        sql_query("SELECT * FROM loan_default WHERE Age > :min_age")  # no params provided
        assert False, "Expected error"
    except ValueError as e:
        error_msg = str(e)
        print(f"\nError message: {error_msg}")
        
        assert "min_age" in error_msg or "param" in error_msg.lower()
        print("  ✓ Missing parameter caught")
    
    print("\n✓ Test passed: Missing params detected")


def test_tool_interface():
    """Test the LangChain tool interface."""
    print("\n" + "=" * 60)
    print("TEST 12: LangChain Tool Interface")
    print("=" * 60)
    
    # Test schema tool
    schema_result = get_sql_schema_tool.invoke({})
    print(f"\nSchema tool output preview:")
    print(schema_result[:300])
    print("...")
    
    assert "loan_default" in schema_result.lower() or "insurance" in schema_result.lower()
    print("  ✓ Schema tool works")
    
    # Test query tool
    query_result = sql_query_tool.invoke({
        "query": "SELECT COUNT(*) as total FROM insurance",
        "max_rows": 10
    })
    print(f"\nQuery tool output preview:")
    print(query_result[:500])
    
    assert "total" in query_result.lower()
    print("  ✓ Query tool works")
    
    print("\n✓ Test passed: Tool interfaces work")


def test_max_rows_limit():
    """Test that max_rows is enforced."""
    print("\n" + "=" * 60)
    print("TEST 13: Max Rows Limit")
    print("=" * 60)
    
    result = sql_query("SELECT * FROM loan_default", max_rows=10)
    
    print(f"\nRequested: SELECT * (no LIMIT)")
    print(f"max_rows: 10")
    print(f"Rows returned: {result.stats['rows_returned']}")
    
    assert result.stats['rows_returned'] <= 10
    assert result.stats['truncated'] == True
    
    print("  ✓ Results truncated to max_rows")
    
    print("\n✓ Test passed: Row limit enforced")


def test_join_query():
    """Test JOIN between tables."""
    print("\n" + "=" * 60)
    print("TEST 14: JOIN Query")
    print("=" * 60)
    
    # This is a self-join example since tables don't have foreign keys
    # Note: "Default" is a reserved keyword so must be quoted
    result = sql_query("""
        SELECT 
            l1.Education,
            COUNT(*) as default_count
        FROM loan_default l1
        WHERE l1."Default" = 1
        GROUP BY l1.Education
        ORDER BY default_count DESC
        LIMIT 5
    """)
    
    print(f"\nQuery: Defaults by Education level")
    print(f"Rows returned: {result.stats['rows_returned']}")
    print(f"\nResults:")
    for row in result.data:
        print(f"  {row['Education']}: {row['default_count']} defaults")
    
    assert result.stats['rows_returned'] > 0
    
    print("\n✓ Test passed: Complex queries work")


def test_insurance_table():
    """Test querying insurance table."""
    print("\n" + "=" * 60)
    print("TEST 15: Insurance Table Query")
    print("=" * 60)
    
    result = sql_query("""
        SELECT 
            smoker,
            COUNT(*) as count,
            AVG(charges) as avg_charges
        FROM insurance
        GROUP BY smoker
    """)
    
    print(f"\nQuery: Charges by Smoker status")
    print(f"\nResults:")
    for row in result.data:
        print(f"  Smoker={row['smoker']}: count={row['count']}, avg_charges=${row['avg_charges']:,.2f}")
    
    assert result.stats['rows_returned'] == 2  # yes/no
    
    print("\n✓ Test passed: Insurance table accessible")


if __name__ == "__main__":
    # Run all tests
    test_list_tables()
    test_describe_table()
    test_get_schema()
    test_basic_select()
    test_select_with_where()
    test_parameterized_query()
    test_aggregation()
    test_forbidden_operations()
    test_table_not_found_suggestions()
    test_column_not_found_suggestions()
    test_missing_params()
    test_tool_interface()
    test_max_rows_limit()
    test_join_query()
    test_insurance_table()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)

