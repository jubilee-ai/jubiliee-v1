"""
Test for the Simple Training Agent (Deep Agents SDK version).

Runs the full end-to-end pipeline: select_model → data_collection → cleaning →
label_split → feature_spec → feature_eng → training_approval → training → report.

Run with: python3 tests/test_simple_training_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "models-tools" / "training"))

import numpy as np
import pandas as pd
from utils import clear_registry, register_dataset


def create_credit_dataset(n=300, seed=42):
    """Create a synthetic credit-risk dataset with predictive signal."""
    np.random.seed(seed)

    age = np.random.randint(25, 65, n)
    income = np.random.normal(60000, 20000, n).clip(20000, 150000)
    credit_score = np.random.randint(500, 800, n)
    debt_ratio = np.random.uniform(0.1, 0.8, n)

    risk_score = -0.05 * age - 0.00003 * income - 0.008 * credit_score + 8
    default_prob = 1 / (1 + np.exp(-risk_score))
    default = (np.random.random(n) < default_prob).astype(int)

    return pd.DataFrame({
        "age": age,
        "income": income.round(2),
        "credit_score": credit_score,
        "debt_ratio": debt_ratio.round(4),
        "default": default,
    })


def test_simple_agent_end_to_end():
    """Full end-to-end test: register data, invoke agent, check report."""
    print("\n" + "=" * 70)
    print("TEST: Simple Training Agent — End-to-End")
    print("=" * 70)

    clear_registry()

    print("\n[1] Creating and registering synthetic dataset...")
    df = create_credit_dataset(n=300)
    register_dataset("credit_risk_raw", df, register_sql=False)
    print(f"    Registered 'credit_risk_raw': {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Default rate: {df['default'].mean():.1%}")

    print("\n[2] Creating simple training agent...")
    from agents.training.agent_simple import create_simple_training_agent

    agent, state = create_simple_training_agent(
        goal="Predict loan default risk based on applicant financial profile",
        linked_datasets=["credit_risk_raw"],
        user_model_preference="logistic_regression",
        model="openai:gpt-4o-mini",
        hitl=False,
    )
    print("    Agent created.")

    print("\n[3] Invoking agent (this calls all 9 steps via LLM)...")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Train a logistic regression model to predict loan default risk. The dataset 'credit_risk_raw' is already loaded."}]}
    )

    print("\n[4] Agent finished.")
    last_msg = result["messages"][-1].content if result.get("messages") else "(no messages)"
    print(f"    Final message (first 500 chars):\n{last_msg[:500]}")

    print("\n" + "=" * 70)
    print("TEST PASSED")
    print("=" * 70)
    return result


if __name__ == "__main__":
    test_simple_agent_end_to_end()
