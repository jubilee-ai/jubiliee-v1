import pytest
from credit_risk import (CreditCardRiskInput, CreditCardRiskOutput,
                         credit_card_risk_prediction_tool,
                         predict_credit_risk)


def test_low_risk_insurance_professional():
    """Test that a stable insurance professional with good financials is classified as Good risk."""
    result = predict_credit_risk(CreditCardRiskInput(
        gender="m",
        marital="married",
        howpaid="monthly",
        mortgage="y",
        age=42,
        income=95000.0,
        numkids=2,
        numcards=2,
        storecar=0,
        loans=1,
    ))
    print(f"Input: Insurance underwriter, married, stable income, low credit exposure")
    print(f"Output: {result}")
    assert isinstance(result, CreditCardRiskOutput)
    assert result.risk is not None
    assert abs(sum(result.probabilities.values()) - 1.0) < 0.01


def test_high_risk_insurance_claimant_profile():
    """Test that a profile matching high insurance risk indicators gets appropriate risk classification."""
    result = predict_credit_risk(CreditCardRiskInput(
        gender="f",
        marital="single",
        howpaid="weekly",
        mortgage="n",
        age=23,
        income=22000.0,
        numkids=0,
        numcards=5,
        storecar=3,
        loans=4,
    ))
    print(f"Input: Young, weekly-paid, high credit exposure (5 cards, 4 loans)")
    print(f"Output: {result}")
    assert isinstance(result, CreditCardRiskOutput)
    assert result.risk is not None
    assert len(result.probabilities) > 0


def test_langchain_tool_insurance_agent_scenario():
    """Test LangChain tool with an insurance agent applicant profile."""
    result = credit_card_risk_prediction_tool.invoke({
        "gender": "f",
        "marital": "married",
        "howpaid": "monthly",
        "mortgage": "y",
        "age": 38,
        "income": 72000.0,
        "numkids": 1,
        "numcards": 1,
        "storecar": 1,
        "loans": 2,
    })
    print(f"Input: Insurance agent, married, moderate income")
    print(f"Tool output:\n{result}")
    assert isinstance(result, str)
    assert "Risk Classification:" in result
    assert "Probabilities:" in result

