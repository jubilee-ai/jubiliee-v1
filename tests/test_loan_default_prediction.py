import pytest
from loan_default_prediction import (LoanApplicantInput, LoanDefaultOutput,
                                     loan_default_prediction_tool,
                                     predict_loan_default)


def test_low_risk_applicant():
    """Test that a stable, high-income applicant is predicted as No Default."""
    result = predict_loan_default(LoanApplicantInput(
        income=500000,
        age=35,
        experience=12,
        married="married",
        house_ownership="owned",
        car_ownership="yes",
        profession="chartered_accountant",
        city="mumbai",
        state="maharashtra",
        current_job_yrs=8,
        current_house_yrs=12,
    ))
    print(f"Input: High-income chartered accountant in Mumbai")
    print(f"Output: {result}")
    assert isinstance(result, LoanDefaultOutput)
    assert result.prediction == "No Default"
    assert result.default_probability < 0.29
    assert abs(sum(result.probabilities.values()) - 1.0) < 0.01


def test_high_risk_applicant():
    """Test that an unstable, low-income applicant has higher default risk."""
    result = predict_loan_default(LoanApplicantInput(
        income=20000,
        age=25,
        experience=2,
        married="single",
        house_ownership="rented",
        car_ownership="no",
        profession="artist",
        city="sikar",
        state="rajasthan",
        current_job_yrs=1,
        current_house_yrs=10,
    ))
    print(f"Input: Low-income artist in Sikar")
    print(f"Output: {result}")
    assert isinstance(result, LoanDefaultOutput)
    assert result.default_probability > 0.1  # Higher risk than typical
    assert "Default" in result.probabilities and "No Default" in result.probabilities


def test_langchain_tool_returns_formatted_string():
    """Test that the LangChain tool returns properly formatted output."""
    result = loan_default_prediction_tool.invoke({
        "income": 300000,
        "age": 30,
        "experience": 5,
        "married": "single",
        "house_ownership": "rented",
        "car_ownership": "no",
        "profession": "analyst",
        "city": "delhi_city",
        "state": "delhi",
        "current_job_yrs": 3,
        "current_house_yrs": 11,
    })
    print(f"Input: Analyst in Delhi")
    print(f"Tool output:\n{result}")
    assert isinstance(result, str)
    assert "Prediction:" in result
    assert "Default Probability:" in result
    assert "Risk Level:" in result
