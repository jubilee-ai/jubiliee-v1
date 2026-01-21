"""
Tests for cleaning_standardization subagent.

Tests cover:
- Basic cleaning flow
- EDA + validation integration
- Tool execution
- Iteration logic
- Edge cases
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agents" / "training"))

from utils import (clear_dataset_registry, get_registered_dataset,
                   register_dataset)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def clear_registry():
    """Clear dataset registry before and after each test."""
    clear_dataset_registry()
    yield
    clear_dataset_registry()


@pytest.fixture
def clean_dataset():
    """A clean dataset with no issues."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "income": np.random.normal(50000, 15000, n),
        "score": np.random.normal(500, 100, n),
        "category": np.random.choice(["A", "B", "C"], n),
    })


@pytest.fixture
def dirty_dataset():
    """A dataset with issues that need cleaning."""
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        # High nulls (40%)
        "income": np.where(np.random.random(n) > 0.4, np.random.normal(50000, 15000, n), np.nan),
        # High skew (lognormal)
        "claim_amount": np.random.lognormal(8, 2, n),
        # Some nulls (10%)
        "age": np.where(np.random.random(n) > 0.1, np.random.randint(18, 80, n), np.nan),
        # Clean categorical
        "region": np.random.choice(["North", "South", "East", "West"], n),
        # ID-like column
        "customer_id": [f"CUST-{i:06d}" for i in range(n)],
    })
    return df


@pytest.fixture
def base_state():
    """Base state for testing."""
    return {
        "goal": "Build a model to predict customer churn",
        "collected_dataset_ref": None,
        "cleaned_dataset_ref": None,
        "cleaning_transformations": [],
        "cleaning_iteration": 0,
        "cleaning_status": None,
        "explanations": [],
        "audit_trace": [],
        "error": None,
    }


# =============================================================================
# UNIT TESTS (WITH MOCKS)
# =============================================================================

class TestCleaningWithMocks:
    """Test cleaning logic with mocked LLM."""
    
    def test_no_dataset_returns_error(self, base_state):
        """Test error is returned when no dataset is available."""
        from cleaning_standardization import cleaning_standardization
        
        result = cleaning_standardization(base_state)
        
        assert result.get("error") == "No dataset available for cleaning"
    
    def test_max_iterations_exits(self, base_state, clean_dataset):
        """Test that max iterations causes exit."""
        from cleaning_standardization import (MAX_CLEANING_ITERATIONS,
                                              cleaning_standardization)
        
        register_dataset("test_data", clean_dataset)
        state = {
            **base_state,
            "collected_dataset_ref": "test_data",
            "cleaning_iteration": MAX_CLEANING_ITERATIONS,  # At max
        }
        
        result = cleaning_standardization(state)
        
        assert result["cleaning_status"] == "done"
        assert "Max iterations" in result["explanations"][-1]
    
    def test_eda_runs_successfully(self, base_state, dirty_dataset):
        """Test EDA runs and produces results."""
        from analysis.eda_report import run_eda_report
        from cleaning_standardization import cleaning_standardization
        
        register_dataset("dirty", dirty_dataset)
        
        # Just test EDA runs - we'll mock the LLM
        eda_result = run_eda_report("dirty")
        
        assert eda_result["shape"]["rows"] == 500
        assert len(eda_result["alerts"]) > 0  # Should have alerts for nulls, skew
        
        # Check for high null alert
        alert_types = [a["type"] for a in eda_result["alerts"]]
        assert "high_nulls" in alert_types
    
    @patch("cleaning_standardization._get_llm")
    def test_llm_calls_mark_complete(self, mock_get_llm, base_state, clean_dataset):
        """Test flow when LLM calls mark_cleaning_complete."""
        from cleaning_standardization import cleaning_standardization
        
        register_dataset("clean_data", clean_dataset)
        state = {
            **base_state,
            "collected_dataset_ref": "clean_data",
        }
        
        # Mock LLM response with tool call
        mock_response = MagicMock()
        mock_response.tool_calls = [{
            "name": "mark_cleaning_complete",
            "args": {
                "dataset_ref": "clean_data",
                "reasoning": "Data is clean, no issues found"
            }
        }]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        result = cleaning_standardization(state)
        
        assert result["cleaning_status"] == "done"
        assert result["cleaned_dataset_ref"].startswith("cleaned_")
        assert "Cleaning complete" in result["explanations"][-1]
    
    @patch("cleaning_standardization._get_llm")
    def test_llm_calls_transformation(self, mock_get_llm, base_state, dirty_dataset):
        """Test flow when LLM calls a transformation tool."""
        from cleaning_standardization import cleaning_standardization
        
        register_dataset("dirty_data", dirty_dataset)
        state = {
            **base_state,
            "collected_dataset_ref": "dirty_data",
        }
        
        # Mock LLM response with transformation tool call
        mock_response = MagicMock()
        mock_response.tool_calls = [{
            "name": "drop_nulls_tool",
            "args": {
                "dataset_ref": "dirty_data",
                "columns": ["income"],
            }
        }]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        result = cleaning_standardization(state)
        
        assert result["cleaning_status"] == "iterate"
        assert result["cleaning_iteration"] == 1
        assert len(result["cleaning_transformations"]) == 1
        assert result["cleaning_transformations"][0]["tool"] == "drop_nulls_tool"
    
    @patch("cleaning_standardization._get_llm")
    def test_no_tool_calls_retries(self, mock_get_llm, base_state, clean_dataset):
        """Test that no tool calls triggers retry."""
        from cleaning_standardization import cleaning_standardization
        
        register_dataset("data", clean_dataset)
        state = {
            **base_state,
            "collected_dataset_ref": "data",
        }
        
        # Mock LLM response with no tool calls
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        
        result = cleaning_standardization(state)
        
        assert result["cleaning_status"] == "iterate"
        assert result["cleaning_iteration"] == 1
        assert "LLM returned no actions" in result["explanations"][-1]


class TestConditionalEdge:
    """Test the should_continue_cleaning conditional edge."""
    
    def test_done_status_returns_done(self):
        """Test done status returns done."""
        from cleaning_standardization import should_continue_cleaning
        
        state = {"cleaning_status": "done"}
        assert should_continue_cleaning(state) == "done"
    
    def test_iterate_status_returns_iterate(self):
        """Test iterate status returns iterate."""
        from cleaning_standardization import should_continue_cleaning
        
        state = {"cleaning_status": "iterate"}
        assert should_continue_cleaning(state) == "iterate"
    
    def test_no_status_returns_iterate(self):
        """Test missing status returns iterate (to start the loop)."""
        from cleaning_standardization import should_continue_cleaning
        
        state = {}
        assert should_continue_cleaning(state) == "iterate"


class TestMarkCleaningComplete:
    """Test the mark_cleaning_complete tool."""
    
    def test_registers_new_dataset(self, clean_dataset):
        """Test that mark_cleaning_complete registers a new dataset."""
        from cleaning_standardization import mark_cleaning_complete
        
        register_dataset("original", clean_dataset)
        
        result = mark_cleaning_complete.invoke({
            "dataset_ref": "original",
            "reasoning": "All clean"
        })
        
        assert result.startswith("CLEANING_COMPLETE|")
        parts = result.split("|")
        assert len(parts) == 3
        assert parts[1].startswith("cleaned_")
        assert parts[2] == "All clean"
        
        # Verify new dataset was registered
        new_df = get_registered_dataset(parts[1])
        assert new_df is not None
        assert len(new_df) == len(clean_dataset)
    
    def test_missing_dataset_returns_error(self):
        """Test error when dataset doesn't exist."""
        from cleaning_standardization import mark_cleaning_complete
        
        result = mark_cleaning_complete.invoke({
            "dataset_ref": "nonexistent",
            "reasoning": "test"
        })
        
        assert "not found" in result


# =============================================================================
# INTEGRATION TESTS (WITH REAL LLM)
# =============================================================================

@pytest.mark.integration
class TestCleaningIntegration:
    """Integration tests that use real LLM. Skip if no API key."""
    
    @pytest.fixture(autouse=True)
    def check_api_key(self):
        """Skip if no OpenAI API key."""
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
    
    def test_full_cleaning_flow_clean_data(self, base_state, clean_dataset):
        """Test full flow with clean data - should complete quickly."""
        from cleaning_standardization import cleaning_standardization
        
        register_dataset("clean_test", clean_dataset)
        state = {
            **base_state,
            "collected_dataset_ref": "clean_test",
        }
        
        result = cleaning_standardization(state)
        
        print("\n" + "="*60)
        print("CLEAN DATA TEST RESULT")
        print("="*60)
        print(f"Status: {result.get('cleaning_status')}")
        print(f"Iterations: {result.get('cleaning_iteration')}")
        print(f"Transformations: {result.get('cleaning_transformations')}")
        print(f"Explanations: {result.get('explanations')}")
        
        # Should either complete or iterate (both are valid)
        assert result["cleaning_status"] in ["done", "iterate"]
    
    def test_full_cleaning_flow_dirty_data(self, base_state, dirty_dataset):
        """Test full flow with dirty data - should apply transformations."""
        from cleaning_standardization import cleaning_standardization
        
        register_dataset("dirty_test", dirty_dataset)
        state = {
            **base_state,
            "collected_dataset_ref": "dirty_test",
        }
        
        result = cleaning_standardization(state)
        
        print("\n" + "="*60)
        print("DIRTY DATA TEST RESULT")
        print("="*60)
        print(f"Status: {result.get('cleaning_status')}")
        print(f"Iterations: {result.get('cleaning_iteration')}")
        print(f"Transformations: {len(result.get('cleaning_transformations', []))}")
        for t in result.get("cleaning_transformations", []):
            print(f"  - {t['tool']}: {t.get('result', t.get('error', 'n/a'))[:100]}")
        print(f"Explanations: {result.get('explanations')}")
        
        # Should have done something
        assert result["cleaning_status"] in ["done", "iterate"]
        assert result["cleaning_iteration"] == 1
    
    def test_multiple_iterations(self, base_state, dirty_dataset):
        """Test running multiple iterations."""
        from cleaning_standardization import (cleaning_standardization,
                                              should_continue_cleaning)
        
        register_dataset("dirty_multi", dirty_dataset)
        state = {
            **base_state,
            "collected_dataset_ref": "dirty_multi",
        }
        
        max_test_iterations = 3
        iteration = 0
        
        print("\n" + "="*60)
        print("MULTI-ITERATION TEST")
        print("="*60)
        
        while iteration < max_test_iterations:
            result = cleaning_standardization(state)
            iteration += 1
            
            print(f"\nIteration {iteration}:")
            print(f"  Status: {result.get('cleaning_status')}")
            print(f"  Dataset: {result.get('cleaned_dataset_ref')}")
            print(f"  Transformations this round: {len(result.get('cleaning_transformations', [])) - len(state.get('cleaning_transformations', []))}")
            
            if should_continue_cleaning(result) == "done":
                print("  -> DONE, exiting loop")
                break
            
            state = result
        
        print(f"\nTotal iterations: {iteration}")
        print(f"Final status: {result.get('cleaning_status')}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run unit tests by default, integration tests with -m integration
    pytest.main([__file__, "-v", "-s", "-m", "not integration"])
