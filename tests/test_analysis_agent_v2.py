"""
Tests for Analysis Agent v2 - Full agent tests with data retrieval, 
statistical analysis, and model execution.

Tests vary in difficulty:
1. Simple: Just model execution with provided data
2. Medium: Data retrieval + basic analysis
3. Complex: Multi-step flows with retrieval + analysis + insights

Run with: python tests/test_analysis_agent_v2.py
"""

import os
import sys
import time
from pathlib import Path

# Determine project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add ALL required paths BEFORE any agent imports
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "tools"))
sys.path.insert(0, str(_PROJECT_ROOT / "tools" / "data-tools"))
sys.path.insert(0, str(_PROJECT_ROOT / "tools" / "models-tools" / "pretrained"))
sys.path.insert(0, str(_PROJECT_ROOT / "agents" / "data-retrieval"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

# Enable LangSmith tracing
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "jubiliee-analysis-agent-tests")

from langchain_openai import ChatOpenAI

# Now import the agent (paths are set up)
from agents.analysis_agent_v2 import agent

llm = ChatOpenAI(model="gpt-5.1", temperature=0)


# Delay between tests to avoid rate limits
TEST_DELAY_SECONDS = 3


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_result(result: str, max_len: int = 3000):
    print("-" * 60)
    if len(result) > max_len:
        print(result[:max_len] + "\n\n... [truncated]")
    else:
        print(result)
    print("-" * 60)


def run_agent(query: str, run_name: str = None) -> str:
    """Run the agent and extract the final response."""
    config = {"run_name": run_name} if run_name else {}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config
    )
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    return "No response"


# =============================================================================
# TEST 1: Simple - Sentiment Analysis (Model Only)
# =============================================================================

def test_1_sentiment_analysis():
    """TEST 1 (Simple): Analyze sentiment of financial text."""
    print_section("TEST 1: Sentiment Analysis (Simple - Model Only)")
    
    query = """Analyze the sentiment of this quarterly earnings excerpt:

"Despite challenging macroeconomic conditions, we delivered exceptional results this quarter. 
Revenue grew 23% year-over-year, driven by strong customer acquisition and improved retention rates. 
Our operating margin expanded by 400 basis points, reflecting disciplined cost management 
and economies of scale. We're raising our full-year guidance."
"""
    
    print(f"\n📥 QUERY:")
    print(query)
    
    result = run_agent(query, "Test1_SentimentAnalysis")
    
    print(f"\n📤 AGENT RESPONSE:")
    print_result(result)
    
    success = len(result) > 100 and ("sentiment" in result.lower() or "positive" in result.lower())
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}")
    return success


# =============================================================================
# TEST 2: Simple - Credit Risk Prediction (Model Only)
# =============================================================================

def test_2_credit_risk():
    """TEST 2 (Simple): Predict credit risk for a loan applicant."""
    print_section("TEST 2: Credit Risk Prediction (Simple - Model Only)")
    
    query = """I need to assess credit risk for this loan applicant:

- 52 year old married male
- Annual income: $120,000, paid monthly
- Has a mortgage
- 3 dependent children
- Currently holds 1 credit card and 1 store card
- Has 2 active loans

What is the risk classification for this applicant?"""
    
    print(f"\n📥 QUERY:")
    print(query)
    
    result = run_agent(query, "Test2_CreditRisk")
    
    print(f"\n📤 AGENT RESPONSE:")
    print_result(result)
    
    success = len(result) > 100 and ("risk" in result.lower() or "good" in result.lower() or "bad" in result.lower())
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}")
    return success


# =============================================================================
# TEST 3: Medium - Negative Sentiment with Reasoning
# =============================================================================

def test_3_negative_sentiment():
    """TEST 3 (Medium): Analyze negative sentiment and explain implications."""
    print_section("TEST 3: Negative Sentiment Analysis (Medium)")
    
    query = """Analyze this CEO statement from a quarterly call and explain the implications:

"We are disappointed to report that our Q3 results fell significantly short of expectations. 
Revenue declined 15% as we lost several major enterprise contracts. We are implementing 
an immediate cost reduction plan including workforce reductions of approximately 20%. 
The path to profitability is now uncertain, and we are withdrawing our full-year guidance."

What is the sentiment? What are the key concerns an investor should have?"""
    
    print(f"\n📥 QUERY:")
    print(query)
    
    result = run_agent(query, "Test3_NegativeSentiment")
    
    print(f"\n📤 AGENT RESPONSE:")
    print_result(result)
    
    success = len(result) > 150 and ("negative" in result.lower() or "concern" in result.lower())
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}")
    return success


# =============================================================================
# TEST 4: Medium - High Risk Credit Profile
# =============================================================================

def test_4_high_risk_credit():
    """TEST 4 (Medium): Evaluate a high-risk credit applicant."""
    print_section("TEST 4: High Risk Credit Evaluation (Medium)")
    
    query = """Evaluate this credit card application and provide your recommendation:

Applicant Profile:
- 24 year old single female
- Income: $28,000/year, paid weekly
- No mortgage (renting)
- No dependents
- Already has 6 credit cards and 4 store cards
- Currently has 5 active personal loans

Given this profile, should we approve or decline? 
Explain your reasoning based on the risk assessment."""
    
    print(f"\n📥 QUERY:")
    print(query)
    
    result = run_agent(query, "Test4_HighRiskCredit")
    
    print(f"\n📤 AGENT RESPONSE:")
    print_result(result)
    
    success = len(result) > 200 and ("risk" in result.lower() or "decline" in result.lower() or "bad" in result.lower())
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}")
    return success


# =============================================================================
# TEST 5: Complex - Multi-Applicant Comparison
# =============================================================================

def test_5_multi_applicant_comparison():
    """TEST 5 (Complex): Compare two applicants and recommend which to approve."""
    print_section("TEST 5: Multi-Applicant Comparison (Complex)")
    
    query = """We have two credit card applicants and can only approve one. 
Please analyze both and recommend which one we should approve:

APPLICANT A:
- 35 year old married male  
- Income: $65,000/year, monthly pay
- Has a mortgage
- 2 children
- 2 credit cards, 0 store cards
- 1 active loan

APPLICANT B:
- 29 year old single female
- Income: $85,000/year, monthly pay  
- No mortgage (owns home outright)
- No children
- 3 credit cards, 2 store cards
- 3 active loans

For each applicant:
1. Assess their credit risk
2. Identify key risk factors
3. Make a final recommendation on which one to approve and why"""
    
    print(f"\n📥 QUERY:")
    print(query)
    
    result = run_agent(query, "Test5_MultiApplicantComparison")
    
    print(f"\n📤 AGENT RESPONSE:")
    print_result(result)
    
    success = len(result) > 300 and ("applicant" in result.lower() or "recommend" in result.lower())
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}")
    return success


# =============================================================================
# Main
# =============================================================================

def test_6_insurance_smoker_analysis():
    """
    TEST 6 (Data Retrieval + Analysis): Analyze smoking impact on insurance costs.
    
    Expected flow:
    1. Agent uses data_retrieval_tool to find insurance data
    2. Agent uses statistical_analysis_tool to analyze the data
    3. Agent provides insights grounded in the actual data
    """
    print_section("TEST 6: Insurance Smoker Analysis (Data Retrieval + Analysis)")
    
    query = """How much more do smokers pay for health insurance compared to non-smokers? 
    I need actual numbers from the data, not general estimates."""
    
    print(f"\n📥 QUERY:")
    print(query)
    
    result = run_agent(query, "Test6_InsuranceSmokerAnalysis")
    
    print(f"\n📤 AGENT RESPONSE:")
    print_result(result)
    
    # Check for evidence of data-driven analysis
    success = (
        len(result) > 200 and
        ("smoker" in result.lower() or "smoking" in result.lower()) and
        any(char.isdigit() for char in result)  # Should have actual numbers
    )
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}")
    return success


def test_7_insurance_regional_analysis():
    """
    TEST 7 (Data Retrieval + Analysis): Compare insurance costs across regions.
    
    Expected flow:
    1. Agent retrieves insurance data
    2. Agent performs statistical analysis by region
    3. Agent identifies patterns and explains findings
    """
    print_section("TEST 7: Insurance Regional Comparison (Medium)")
    
    query = """Which US region has the highest average health insurance costs? 
    Are there any interesting patterns when you factor in BMI and smoking status?
    Show me the data."""
    
    print(f"\n📥 QUERY:")
    print(query)
    
    result = run_agent(query, "Test7_InsuranceRegionalAnalysis")
    
    print(f"\n📤 AGENT RESPONSE:")
    print_result(result)
    
    # Check for regional analysis with data
    success = (
        len(result) > 300 and
        any(region in result.lower() for region in ["northeast", "northwest", "southeast", "southwest"]) and
        any(char.isdigit() for char in result)
    )
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}")
    return success


def test_8_insurance_risk_factors():
    """
    TEST 8 (Complex Data Analysis): Identify key risk factors for high insurance costs.
    
    Expected flow:
    1. Agent retrieves insurance data
    2. Agent performs comprehensive statistical analysis
    3. Agent identifies correlations and key drivers
    4. Agent provides actionable insights
    """
    print_section("TEST 8: Insurance Risk Factor Analysis (Complex)")
    
    query = """I'm an insurance underwriter. Help me understand what factors most 
    strongly predict high medical insurance costs. I need to see the correlations 
    and understand which customer profiles are most expensive to insure."""
    
    print(f"\n📥 QUERY:")
    print(query)
    
    result = run_agent(query, "Test8_InsuranceRiskFactors")
    
    print(f"\n📤 AGENT RESPONSE:")
    print_result(result)
    
    # Check for comprehensive analysis
    success = (
        len(result) > 400 and
        ("correlation" in result.lower() or "factor" in result.lower()) and
        any(char.isdigit() for char in result)
    )
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}")
    return success


def run_all_tests():
    """Run all tests sequentially."""
    print("\n" + "=" * 80)
    print(" ANALYSIS AGENT V2 - FULL AGENT INTEGRATION TESTS")
    print("=" * 80)
    print(f"\nRunning 8 tests with {TEST_DELAY_SECONDS}s delay between each...")
    print("Tests range from simple model calls to complex multi-step reasoning.")
    print("\nNote: These tests call real LLM APIs and may take 20-40 seconds each.\n")
    
    tests = [
        ("1. Sentiment Analysis (Simple)", test_1_sentiment_analysis),
        ("2. Credit Risk Prediction (Simple)", test_2_credit_risk),
        ("3. Negative Sentiment Analysis (Medium)", test_3_negative_sentiment),
        ("4. High Risk Credit Evaluation (Medium)", test_4_high_risk_credit),
        ("5. Multi-Applicant Comparison (Complex)", test_5_multi_applicant_comparison),
        ("6. Insurance Smoker Analysis (Data Retrieval)", test_6_insurance_smoker_analysis),
        ("7. Insurance Regional Comparison (Medium)", test_7_insurance_regional_analysis),
        ("8. Insurance Risk Factor Analysis (Complex)", test_8_insurance_risk_factors),
    ]
    
    results = []
    for i, (name, test_fn) in enumerate(tests):
        if i > 0:
            print(f"\n⏳ Waiting {TEST_DELAY_SECONDS}s before next test...")
            time.sleep(TEST_DELAY_SECONDS)
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            print(f"\n❌ ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    passed = sum(1 for _, s, _ in results if s)
    print(f"\n   Passed: {passed}/{len(results)}")
    print()
    for name, success, error in results:
        status = "✓" if success else "✗"
        suffix = f"\n      Error: {error[:80]}..." if error else ""
        print(f"   {status} {name}{suffix}")
    print()
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
