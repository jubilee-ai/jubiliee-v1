from typing import Literal

import requests
from langchain.tools import tool
from pydantic import BaseModel, Field


class LoanApplicantInput(BaseModel):
    income: int
    age: int
    experience: int
    married: Literal["single", "married"]
    house_ownership: Literal["rented", "owned", "norent_noown"]
    car_ownership: Literal["yes", "no"]
    profession: str
    city: str
    state: str
    current_job_yrs: int
    current_house_yrs: int


class LoanDefaultOutput(BaseModel):
    prediction: Literal["Default", "No Default"]
    default_probability: float
    probabilities: dict[str, float]


API_URL = "https://jensbender-loan-default-prediction-app.hf.space/api/predict"
THRESHOLD = 0.29


def predict_loan_default(input_data: LoanApplicantInput) -> LoanDefaultOutput:
    """Predict loan default probability by calling the hosted HF Space API."""
    response = requests.post(API_URL, json=input_data.model_dump(), timeout=30)
    response.raise_for_status()

    result = response.json()["results"][0]
    default_prob = result["probabilities"]["Default"]
    no_default_prob = result["probabilities"]["No Default"]

    return LoanDefaultOutput(
        prediction=result["prediction"],
        default_probability=default_prob,
        probabilities={"Default": default_prob, "No Default": no_default_prob}
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class LoanDefaultToolInput(BaseModel):
    """Input for loan default prediction. Categorical values must match exactly (case-sensitive)."""

    income: int = Field(description="Annual income in local currency (e.g., 300000)")
    age: int = Field(description="Applicant age in years (e.g., 30)")
    experience: int = Field(description="Total professional experience in years (e.g., 5)")
    married: Literal["single", "married"] = Field(description="Marital status. Must be exactly: 'single' or 'married'")
    house_ownership: Literal["rented", "owned", "norent_noown"] = Field(
        description="Housing status. Must be exactly: 'rented', 'owned', or 'norent_noown'"
    )
    car_ownership: Literal["yes", "no"] = Field(description="Owns a car? Must be exactly: 'yes' or 'no'")
    profession: str = Field(description="Job title (e.g., 'Engineer', 'Artist', 'Doctor')")
    city: str = Field(description="City of residence (e.g., 'Mumbai', 'Delhi')")
    state: str = Field(description="State of residence (e.g., 'Maharashtra', 'Rajasthan')")
    current_job_yrs: int = Field(description="Years at current employer (e.g., 3)")
    current_house_yrs: int = Field(description="Years at current address (e.g., 5)")


@tool("loan_default_prediction", args_schema=LoanDefaultToolInput)
def loan_default_prediction_tool(
    income: int,
    age: int,
    experience: int,
    married: Literal["single", "married"],
    house_ownership: Literal["rented", "owned", "norent_noown"],
    car_ownership: Literal["yes", "no"],
    profession: str,
    city: str,
    state: str,
    current_job_yrs: int,
    current_house_yrs: int,
) -> str:
    """Predict loan default probability via hosted HF Space API (Random Forest trained on 252K Indian loans).

    WHEN TO USE:
    - Credit risk triage for loan applications
    - Quick default probability scoring from applicant demographics
    - POC for agent-driven tabular risk models with threshold-based actions

    WHEN NOT TO USE:
    - Non-Indian applicant data (model trained on Indian geography)
    - When network access is unavailable (requires external API call)
    - High-stakes decisioning without human review

    CONSTRAINTS:
    - Requires network: calls HF Space endpoint (not local inference)
    - Fixed schema: all fields required, categorical values must match exactly
    - Geography bias: trained on Indian loan data, may not generalize

    OUTPUT:
    - prediction: "Default" or "No Default"
    - default_probability: 0.0-1.0 (threshold at 0.29)
    - risk_level: LOW (<29%), MEDIUM (29-50%), HIGH (>50%)

    Args:
        income: Annual income in local currency (e.g., 300000)
        age: Applicant age in years (e.g., 30)
        experience: Total professional experience in years (e.g., 5)
        married: 'single' or 'married'
        house_ownership: 'rented', 'owned', or 'norent_noown'
        car_ownership: 'yes' or 'no'
        profession: Job title (e.g., 'Engineer')
        city: City name (e.g., 'Mumbai')
        state: State name (e.g., 'Maharashtra')
        current_job_yrs: Years at current employer
        current_house_yrs: Years at current address
    """
    result = predict_loan_default(LoanApplicantInput(
        income=income,
        age=age,
        experience=experience,
        married=married,
        house_ownership=house_ownership,
        car_ownership=car_ownership,
        profession=profession,
        city=city,
        state=state,
        current_job_yrs=current_job_yrs,
        current_house_yrs=current_house_yrs,
    ))

    return (
        f"Prediction: {result.prediction}\n"
        f"Default Probability: {result.default_probability:.1%}\n"
        f"Risk Level: {'HIGH' if result.default_probability >= 0.5 else 'MEDIUM' if result.default_probability >= THRESHOLD else 'LOW'}"
    )
