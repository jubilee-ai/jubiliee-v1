"""
Test the full Data Retrieval Agent flow.

These tests run the agent with natural language prompts and verify
it correctly retrieves and transforms data from the available datasets.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.data_retrieval import (
    DataRetrievalResult,
    get_dataset,
    retrieve_data,
)

# Delay between tests to avoid rate limits
TEST_DELAY_SECONDS = 5


def print_result(name: str, request: str, result):
    """Pretty print test results."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"\n📥 INPUT:")
    print(f"   {request}")
    print(f"\n📤 OUTPUT:")
    
    if isinstance(result, DataRetrievalResult):
        print(f"   ✓ Success - DataRetrievalResult")
        print(f"   dataset_ref: {result.dataset_ref}")
        print(f"   description: {result.description}")
        print(f"   rows: {result.rows}")
        print(f"   columns: {result.columns[:8]}{'...' if len(result.columns) > 8 else ''}")
        print(f"   source: {result.source}")
        
        # Try to get the actual data
        df = get_dataset(result.dataset_ref)
        if df is not None:
            print(f"\n   📊 Data preview (first 3 rows):")
            print(f"   {df.head(3).to_string().replace(chr(10), chr(10) + '   ')}")
        return True
    else:
        print(f"   ⚠️ Fallback response (not structured)")
        if "error" in result:
            print(f"   Error: {result.get('error')}")
        else:
            response = result.get("response", "")
            print(f"   Response: {response[:500]}...")
        return False


def test_1_simple_retrieval():
    """Test 1: Simple dataset retrieval - get insurance data."""
    request = "Get me the health insurance dataset"
    result = retrieve_data(request)
    return print_result("Simple Retrieval - Insurance Data", request, result)


def test_2_filtered_query():
    """Test 2: Filtered SQL query - high credit score loans."""
    request = "Find loan default data where credit score is above 750, limit to 100 rows"
    result = retrieve_data(request)
    return print_result("Filtered Query - High Credit Score Loans", request, result)


def test_3_aggregation():
    """Test 3: SQL aggregation - average charges by smoker status."""
    request = "Query the insurance SQL table to get average charges grouped by smoker column"
    result = retrieve_data(request)
    return print_result("Aggregation - Avg Charges by Smoker Status", request, result)


def test_4_credit_risk_search():
    """Test 4: Search for credit risk data and retrieve it."""
    request = "Find data about credit card customer risk classification"
    result = retrieve_data(request)
    return print_result("Search + Retrieve - Credit Card Risk", request, result)


def test_5_join_tables():
    """Test 5: Join two tables - loan and insurance by age."""
    request = "Use join_merge_tool to join the loan_default SQL table with the insurance SQL table on Age/age columns. Use dedupe_strategy='first' to avoid row explosion."
    result = retrieve_data(request)
    return print_result("Join Tables - Loan + Insurance by Age", request, result)


def run_all_tests():
    """Run all agent tests."""
    print("\n" + "=" * 70)
    print(" DATA RETRIEVAL AGENT - FULL FLOW TESTS")
    print("=" * 70)
    
    tests = [
        ("Simple Retrieval", test_1_simple_retrieval),
        ("Filtered Query", test_2_filtered_query),
        ("Aggregation", test_3_aggregation),
        ("Credit Risk Search", test_4_credit_risk_search),
        ("Join Tables", test_5_join_tables),
    ]
    
    results = []
    for i, (name, test_fn) in enumerate(tests):
        if i > 0:
            print(f"\n⏳ Waiting {TEST_DELAY_SECONDS}s before next test (rate limit)...")
            time.sleep(TEST_DELAY_SECONDS)
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, s in results if s)
    print(f"\n   Passed: {passed}/{len(results)}")
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"   {status} {name}")
    print()
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
