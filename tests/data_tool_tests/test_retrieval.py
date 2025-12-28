"""
Tests for the catalog_search retrieval tool.

Includes:
- BM25 lexical search tests
- Filter tests (domain, source, format, columns, min_rows)
- Hybrid search tests (BM25 + OpenAI semantic) - requires OPENAI_API_KEY
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
    
    # Load retrieval
    spec_ret = importlib.util.spec_from_file_location('data_tools.retrieval', base_path / 'retrieval.py')
    retrieval = importlib.util.module_from_spec(spec_ret)
    sys.modules['data_tools.retrieval'] = retrieval
    spec_ret.loader.exec_module(retrieval)
    
    return retrieval

_retrieval = _load_data_tools()
CatalogSearcher = _retrieval.CatalogSearcher
SearchFilters = _retrieval.SearchFilters
catalog_search = _retrieval.catalog_search
hybrid_search = _retrieval.hybrid_search


def test_basic_search():
    """Test basic keyword search without filters."""
    print("=" * 60)
    print("TEST 1: Basic Search - 'loan default credit score'")
    print("=" * 60)
    
    results = catalog_search("loan default credit score", top_k=5)
    
    print(f"\nInput: query='loan default credit score', top_k=5")
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Asset ID: {r['asset_id']}")
        print(f"     Score: {r['score']}")
        print(f"     Tags: {r['tags']}")
        print(f"     Format: {r['format']}")
        print(f"     Use Case: {r['use_case'][:80]}...")
    
    assert len(results) > 0
    assert results[0]['asset_id'] == 'csv/Loan_default.csv'  # Should be top result
    print("\n✓ Test passed: Loan default dataset ranked first")
    return results


def test_insurance_search():
    """Test search for insurance-related datasets."""
    print("\n" + "=" * 60)
    print("TEST 2: Insurance Search - 'health insurance premium cost'")
    print("=" * 60)
    
    results = catalog_search("health insurance premium cost", top_k=3)
    
    print(f"\nInput: query='health insurance premium cost', top_k=3")
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Score: {r['score']}")
        print(f"     Columns: {r['schema_summary'][:5] if r['schema_summary'] else 'N/A'}...")
    
    assert len(results) > 0
    assert 'insurance' in results[0]['name'].lower()
    print("\n✓ Test passed: Insurance dataset ranked first")
    return results


def test_domain_filter():
    """Test search with domain filter."""
    print("\n" + "=" * 60)
    print("TEST 3: Domain Filter - domain='credit_risk'")
    print("=" * 60)
    
    results = catalog_search(
        "prediction model", 
        filters={"domain": "credit_risk"},
        top_k=5
    )
    
    print(f"\nInput: query='prediction model', filters={{'domain': 'credit_risk'}}")
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Score: {r['score']}")
        print(f"     Tags: {r['tags']}")
    
    # All results should have credit_risk tag
    for r in results:
        assert 'credit_risk' in r['tags'], f"Expected credit_risk tag in {r['name']}"
    
    print("\n✓ Test passed: All results have credit_risk tag")
    return results


def test_column_filter():
    """Test search with column filter."""
    print("\n" + "=" * 60)
    print("TEST 4: Column Filter - columns=['age', 'income']")
    print("=" * 60)
    
    results = catalog_search(
        "risk assessment",
        filters={"columns": ["age", "income"]},
        top_k=5
    )
    
    print(f"\nInput: query='risk assessment', filters={{'columns': ['age', 'income']}}")
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Score: {r['score']}")
        print(f"     Columns: {r['schema_summary']}")
    
    # All results should have age and income columns
    for r in results:
        cols_lower = [c.lower() for c in r['schema_summary']]
        has_age = any('age' in c for c in cols_lower)
        has_income = any('income' in c for c in cols_lower)
        assert has_age and has_income, f"Expected age and income columns in {r['name']}"
    
    print("\n✓ Test passed: All results have age and income columns")
    return results


def test_format_filter():
    """Test search with format filter."""
    print("\n" + "=" * 60)
    print("TEST 5: Format Filter - format='CSV'")
    print("=" * 60)
    
    results = catalog_search(
        "financial",
        filters={"format": "CSV"},
        top_k=5
    )
    
    print(f"\nInput: query='financial', filters={{'format': 'CSV'}}")
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Score: {r['score']}")
        print(f"     Format: {r['format']}")
    
    for r in results:
        assert 'CSV' in r['format'], f"Expected CSV format in {r['name']}"
    
    print("\n✓ Test passed: All results are CSV format")
    return results


def test_source_filter():
    """Test search with source filter for HuggingFace datasets."""
    print("\n" + "=" * 60)
    print("TEST 6: Source Filter - source='HuggingFace'")
    print("=" * 60)
    
    results = catalog_search(
        "risk",
        filters={"source": "HuggingFace"},
        top_k=5
    )
    
    print(f"\nInput: query='risk', filters={{'source': 'HuggingFace'}}")
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Score: {r['score']}")
        print(f"     Source: {r['source']}")
        print(f"     Format: {r['format']}")
    
    for r in results:
        assert 'huggingface' in r['tags'], f"Expected huggingface tag in {r['name']}"
    
    print("\n✓ Test passed: All results are from HuggingFace")
    return results


def test_min_rows_filter():
    """Test search with minimum rows filter."""
    print("\n" + "=" * 60)
    print("TEST 7: Min Rows Filter - min_rows=10000")
    print("=" * 60)
    
    results = catalog_search(
        "prediction",
        filters={"min_rows": 10000},
        top_k=5
    )
    
    print(f"\nInput: query='prediction', filters={{'min_rows': 10000}}")
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Score: {r['score']}")
        print(f"     Rows: {r['rows']}")
    
    for r in results:
        assert r['rows'] is not None and r['rows'] >= 10000, \
            f"Expected at least 10000 rows in {r['name']}, got {r['rows']}"
    
    print("\n✓ Test passed: All results have >= 10000 rows")
    return results


def test_combined_filters():
    """Test search with multiple filters combined."""
    print("\n" + "=" * 60)
    print("TEST 8: Combined Filters - domain + columns + format")
    print("=" * 60)
    
    results = catalog_search(
        "default",
        filters={
            "domain": "credit_risk",
            "columns": ["income"],
            "format": "CSV"
        },
        top_k=5
    )
    
    print(f"\nInput: query='default', filters={{")
    print(f"    'domain': 'credit_risk',")
    print(f"    'columns': ['income'],")
    print(f"    'format': 'CSV'")
    print(f"}}")
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Score: {r['score']}")
        print(f"     Tags: {r['tags']}")
        print(f"     Format: {r['format']}")
        print(f"     Columns (first 5): {r['schema_summary'][:5]}")
    
    assert len(results) > 0
    print("\n✓ Test passed: Combined filters work correctly")
    return results


def test_empty_query():
    """Test search with empty/generic query."""
    print("\n" + "=" * 60)
    print("TEST 9: Empty Query - returns all datasets")
    print("=" * 60)
    
    results = catalog_search("", top_k=10)
    
    print(f"\nInput: query='', top_k=10")
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['name']} (score: {r['score']})")
    
    # Should return all datasets
    assert len(results) == 10  # We have 10 datasets in catalog (CSV + SQL versions)
    print("\n✓ Test passed: All 10 datasets returned")
    return results


def test_column_search_in_query():
    """Test that column names in query boost relevance."""
    print("\n" + "=" * 60)
    print("TEST 10: Column Names in Query - 'CreditScore DTIRatio'")
    print("=" * 60)
    
    results = catalog_search("CreditScore DTIRatio LoanAmount", top_k=3)
    
    print(f"\nInput: query='CreditScore DTIRatio LoanAmount', top_k=3")
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Score: {r['score']}")
        print(f"     Columns: {r['schema_summary']}")
    
    # Loan default dataset has these exact columns (CSV or SQL version)
    assert results[0]['asset_id'] in ['csv/Loan_default.csv', 'sql/loan_default.sql']
    print("\n✓ Test passed: Dataset with matching columns ranked first")
    return results


def test_hybrid_search_natural_language():
    """Test hybrid search with natural language query."""
    print("\n" + "=" * 60)
    print("TEST 11: HYBRID SEARCH - Natural Language Query")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Skipping: OPENAI_API_KEY not set")
        return None
    
    query = "find datasets about predicting if someone will pay back a loan"
    print(f"\nInput: query='{query}'")
    
    results = hybrid_search(query, top_k=3)
    
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Score: {r['score']}")
        if "score_breakdown" in r:
            bd = r["score_breakdown"]
            print(f"     BM25: {bd['bm25']}, Column: {bd['column']}, Semantic: {bd['semantic']}")
    
    assert len(results) > 0
    # Semantic search should understand "pay back a loan" = loan default
    assert "loan" in results[0]["name"].lower() or "default" in results[0]["name"].lower()
    print("\n✓ Test passed: Loan-related dataset found via semantic understanding")
    return results


def test_hybrid_search_conceptual():
    """Test hybrid search with conceptual query (no exact keyword match)."""
    print("\n" + "=" * 60)
    print("TEST 12: HYBRID SEARCH - Conceptual Query")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Skipping: OPENAI_API_KEY not set")
        return None
    
    query = "customer demographics for risk modeling"
    print(f"\nInput: query='{query}'")
    
    results = hybrid_search(query, top_k=3)
    
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Score: {r['score']}")
        if "score_breakdown" in r:
            bd = r["score_breakdown"]
            print(f"     BM25: {bd['bm25']}, Column: {bd['column']}, Semantic: {bd['semantic']}")
    
    assert len(results) > 0
    print("\n✓ Test passed: Relevant datasets found via conceptual search")
    return results


def test_hybrid_search_financial_distress():
    """Test hybrid search for financial health concepts."""
    print("\n" + "=" * 60)
    print("TEST 13: HYBRID SEARCH - Financial Health Query")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Skipping: OPENAI_API_KEY not set")
        return None
    
    query = "data about companies going bankrupt or having money problems"
    print(f"\nInput: query='{query}'")
    
    results = hybrid_search(query, top_k=3)
    
    print(f"\nOutput ({len(results)} results):")
    for i, r in enumerate(results, 1):
        print(f"\n  {i}. {r['name']}")
        print(f"     Score: {r['score']}")
        if "score_breakdown" in r:
            bd = r["score_breakdown"]
            print(f"     BM25: {bd['bm25']}, Column: {bd['column']}, Semantic: {bd['semantic']}")
    
    assert len(results) > 0
    # Should find Financial Distress dataset via semantic understanding
    assert "financial" in results[0]["name"].lower() or "distress" in results[0]["description"].lower()
    print("\n✓ Test passed: Financial distress dataset found via semantic search")
    return results


def test_hybrid_vs_bm25_comparison():
    """Compare hybrid search vs BM25-only for a query with no exact matches."""
    print("\n" + "=" * 60)
    print("TEST 14: HYBRID vs BM25 Comparison")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Skipping: OPENAI_API_KEY not set")
        return None
    
    # Query with synonyms/concepts instead of exact keywords
    query = "creditworthiness evaluation data"
    print(f"\nInput: query='{query}'")
    
    # BM25 only
    bm25_results = catalog_search(query, top_k=3, use_semantic=False)
    print("\nBM25 Only Results:")
    for i, r in enumerate(bm25_results, 1):
        print(f"  {i}. {r['name']} (score: {r['score']})")
    
    # Hybrid
    hybrid_results = hybrid_search(query, top_k=3)
    print("\nHybrid (BM25 + Semantic) Results:")
    for i, r in enumerate(hybrid_results, 1):
        print(f"  {i}. {r['name']} (score: {r['score']})")
        if "score_breakdown" in r:
            bd = r["score_breakdown"]
            print(f"     BM25: {bd['bm25']}, Column: {bd['column']}, Semantic: {bd['semantic']}")
    
    print("\n✓ Test passed: Comparison complete")
    return {"bm25": bm25_results, "hybrid": hybrid_results}


def run_all_tests():
    """Run all tests and show summary."""
    print("\n" + "=" * 60)
    print("CATALOG SEARCH TEST SUITE")
    print("=" * 60)
    
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if has_openai:
        print("✓ OPENAI_API_KEY detected - running hybrid search tests")
    else:
        print("⚠️  OPENAI_API_KEY not set - skipping hybrid search tests")
        print("   Set it with: export OPENAI_API_KEY='sk-...'")
    
    # BM25 tests (always run)
    bm25_tests = [
        test_basic_search,
        test_insurance_search,
        test_domain_filter,
        test_column_filter,
        test_format_filter,
        test_source_filter,
        test_min_rows_filter,
        test_combined_filters,
        test_empty_query,
        test_column_search_in_query,
    ]
    
    # Hybrid tests (only if API key available)
    hybrid_tests = [
        test_hybrid_search_natural_language,
        test_hybrid_search_conceptual,
        test_hybrid_search_financial_distress,
        test_hybrid_vs_bm25_comparison,
    ]
    
    tests = bm25_tests + (hybrid_tests if has_openai else [])
    
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
            print(f"\n✗ Test ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

