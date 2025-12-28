"""
Tests for the dataset_get data loader tool.

Includes:
- Basic data loading tests
- Column selection and filtering tests
- End-to-end workflow: catalog_search -> dataset_get
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Add project root to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import data-tools as a package
import importlib.util


def _load_data_tools():
    """Load data-tools as a package with proper relative imports."""
    base_path = Path(__file__).parent.parent.parent / "tools" / "data-tools"
    
    # Register package
    spec = importlib.util.spec_from_file_location(
        'data_tools', 
        base_path / '__init__.py',
        submodule_search_locations=[str(base_path)]
    )
    data_tools = importlib.util.module_from_spec(spec)
    sys.modules['data_tools'] = data_tools
    
    # Load modules in order
    for mod_name in ['utils', 'embedder', 'retrieval', 'data_loader']:
        spec_mod = importlib.util.spec_from_file_location(
            f'data_tools.{mod_name}', 
            base_path / f'{mod_name}.py'
        )
        mod = importlib.util.module_from_spec(spec_mod)
        sys.modules[f'data_tools.{mod_name}'] = mod
        spec_mod.loader.exec_module(mod)
    
    return sys.modules['data_tools.data_loader'], sys.modules['data_tools.retrieval']

_data_loader, _retrieval = _load_data_tools()
dataset_get = _data_loader.dataset_get
dataset_get_tool = _data_loader.dataset_get_tool
catalog_search = _retrieval.catalog_search


def test_basic_data_loading():
    """Test basic data loading with default parameters."""
    print("=" * 60)
    print("TEST 1: Basic Data Loading - insurance.csv")
    print("=" * 60)
    
    print("\nInput:")
    print("  asset_id: 'insurance.csv'")
    print("  limit: 10 (default)")
    
    result = dataset_get('insurance.csv', limit=10)
    
    print(f"\nOutput:")
    print(f"  Dataset ref: {result.dataset_ref}")
    print(f"  Total rows in dataset: {result.stats['total_rows_in_dataset']}")
    print(f"  Rows returned: {result.stats['rows_returned']}")
    print(f"  Columns: {[c.name for c in result.schema]}")
    print(f"\n  Schema details:")
    for col in result.schema:
        dtype_info = f"{col.dtype}"
        if col.sample_values:
            dtype_info += f", samples: {col.sample_values[:3]}"
        elif col.min_value is not None:
            dtype_info += f", range: [{col.min_value}, {col.max_value}]"
        print(f"    - {col.name}: {dtype_info}")
    print(f"\n  First row: {result.data[0]}")
    
    # Assertions
    assert result.dataset_ref == 'insurance.csv'
    assert result.stats['rows_returned'] == 10
    assert len(result.schema) == 7  # insurance.csv has 7 columns
    assert 'age' in [c.name for c in result.schema]
    
    print("\n✓ Test passed: Basic data loading works correctly")
    return result


def test_column_selection_and_filters():
    """Test data loading with column selection and filters."""
    print("\n" + "=" * 60)
    print("TEST 2: Column Selection + Filters - Loan_default.csv")
    print("=" * 60)
    
    columns = ['Age', 'Income', 'CreditScore', 'Default', 'LoanPurpose']
    filters = {
        'Age': {'$gte': 30, '$lte': 50},
        'CreditScore': {'$gte': 700},
        'Default': 1
    }
    
    print("\nInput:")
    print(f"  asset_id: 'Loan_default.csv'")
    print(f"  columns: {columns}")
    print(f"  filters: {filters}")
    print(f"  limit: 10")
    print(f"  sample_strategy: 'random'")
    
    result = dataset_get(
        'Loan_default.csv',
        columns=columns,
        filters=filters,
        limit=10,
        sample_strategy='random'
    )
    
    print(f"\nOutput:")
    print(f"  Total rows in dataset: {result.stats['total_rows_in_dataset']:,}")
    print(f"  Rows after filtering: {result.stats['rows_after_filter']:,}")
    print(f"  Rows returned: {result.stats['rows_returned']}")
    print(f"  Columns returned: {[c.name for c in result.schema]}")
    print(f"\n  Data sample:")
    for i, row in enumerate(result.data[:5], 1):
        print(f"    {i}. Age={row['Age']}, Income={row['Income']:,}, "
              f"CreditScore={row['CreditScore']}, Default={row['Default']}, "
              f"Purpose={row['LoanPurpose']}")
    
    # Assertions
    assert result.stats['rows_returned'] == 10
    assert len(result.schema) == 5  # Only requested columns
    
    # Verify filters were applied
    for row in result.data:
        assert 30 <= row['Age'] <= 50, f"Age filter failed: {row['Age']}"
        assert row['CreditScore'] >= 700, f"CreditScore filter failed: {row['CreditScore']}"
        assert row['Default'] == 1, f"Default filter failed: {row['Default']}"
    
    print("\n✓ Test passed: Column selection and filters work correctly")
    return result


def test_catalog_search_then_dataset_get():
    """
    End-to-end workflow: Use catalog_search to find a dataset,
    then use dataset_get to load data from it.
    """
    print("\n" + "=" * 60)
    print("TEST 3: End-to-End Workflow - Search → Get Data")
    print("=" * 60)
    
    # Step 1: Search for datasets
    search_query = "predict insurance costs health"
    print(f"\n--- Step 1: Catalog Search ---")
    print(f"Input: query='{search_query}'")
    
    search_results = catalog_search(search_query, top_k=3)
    
    print(f"\nSearch Output ({len(search_results)} results):")
    for i, r in enumerate(search_results, 1):
        print(f"  {i}. {r['name']}")
        print(f"     Asset ID: {r['asset_id']}")
        print(f"     Score: {r['score']}")
        print(f"     Use Case: {r['use_case'][:60]}...")
    
    # Get the top result's asset_id
    top_result = search_results[0]
    asset_id = top_result['asset_id']
    
    print(f"\n→ Selected top result: '{top_result['name']}' (asset_id: {asset_id})")
    
    # Step 2: Load data from the discovered dataset
    print(f"\n--- Step 2: Dataset Get ---")
    print(f"Input:")
    print(f"  asset_id: '{asset_id}'")
    print(f"  columns: ['age', 'bmi', 'smoker', 'charges']")
    print(f"  filters: {{'smoker': 'yes'}}")
    print(f"  limit: 10")
    
    data_result = dataset_get(
        asset_id,
        columns=['age', 'bmi', 'smoker', 'charges'],
        filters={'smoker': 'yes'},
        limit=10
    )
    
    print(f"\nData Output:")
    print(f"  Provenance:")
    print(f"    - Source: {data_result.provenance['source']}")
    print(f"    - Retrieved at: {data_result.provenance['retrieved_at']}")
    print(f"    - Filters applied: {data_result.provenance['filters_applied']}")
    
    print(f"\n  Stats:")
    print(f"    - Total rows: {data_result.stats['total_rows_in_dataset']:,}")
    print(f"    - After filter: {data_result.stats['rows_after_filter']}")
    print(f"    - Returned: {data_result.stats['rows_returned']}")
    
    print(f"\n  Schema:")
    for col in data_result.schema:
        info = f"{col.dtype}"
        if col.mean_value is not None:
            info += f", mean={col.mean_value:.2f}"
        print(f"    - {col.name}: {info}")
    
    print(f"\n  Data (smokers only):")
    for i, row in enumerate(data_result.data[:5], 1):
        print(f"    {i}. age={row['age']}, bmi={row['bmi']:.1f}, "
              f"smoker={row['smoker']}, charges=${row['charges']:,.2f}")
    
    # Assertions (catalog uses full paths like 'csv/insurance.csv')
    assert 'insurance.csv' in top_result['asset_id'], "Expected insurance dataset as top result"
    assert 'insurance.csv' in data_result.dataset_ref
    assert all(row['smoker'] == 'yes' for row in data_result.data), "Filter not applied correctly"
    
    # Verify the workflow produces usable data for ML
    print(f"\n  Numeric summary for ML:")
    if data_result.stats.get('numeric_summary'):
        for col_name, stats in data_result.stats['numeric_summary'].items():
            print(f"    {col_name}: mean={stats['mean']}, std={stats['std']}, "
                  f"range=[{stats['min']}, {stats['max']}]")
    
    print("\n✓ Test passed: End-to-end workflow (search → get) works correctly")
    return search_results, data_result


def run_all_tests():
    """Run all data loader tests."""
    print("\n" + "=" * 60)
    print("DATA LOADER TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_basic_data_loading,
        test_column_selection_and_filters,
        test_catalog_search_then_dataset_get,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ Test FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ Test ERROR: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

