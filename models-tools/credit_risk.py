from typing import Literal

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from langchain.tools import tool
from pydantic import BaseModel, Field


class CreditCardRiskInput(BaseModel):
    """Input schema for credit card risk assessment. All fields required."""
    gender: Literal["m", "f"] = Field(description="Biological sex. Exact values: 'm' (male) or 'f' (female)")
    marital: Literal["married", "single", "divsepwid"] = Field(description="Marital status. Exact values: 'married', 'single', or 'divsepwid' (divorced/separated/widowed)")
    howpaid: Literal["monthly", "weekly"] = Field(description="Salary payment frequency. Exact values: 'monthly' or 'weekly'")
    mortgage: Literal["y", "n"] = Field(description="Whether applicant has a mortgage. Exact values: 'y' or 'n'")
    age: int = Field(description="Age in years. Typical range: 18-80", ge=18)
    income: float = Field(description="Annual income in currency units. Typical range: 10000-500000", gt=0)
    numkids: int = Field(description="Number of dependent children. Typical range: 0-10", ge=0)
    numcards: int = Field(description="Number of existing credit cards held. Typical range: 0-10", ge=0)
    storecar: int = Field(description="Number of retail/store credit cards. Typical range: 0-5", ge=0)
    loans: int = Field(description="Number of active loans (auto, personal, etc.). Typical range: 0-10", ge=0)


class CreditCardRiskOutput(BaseModel):
    """Output schema for credit card risk assessment."""
    risk: str = Field(description="Predicted risk class: 'Good risk' (approve), 'Bad loss' (reject - will default), or 'Bad profit' (reject - unprofitable)")
    probabilities: dict[str, float] = Field(description="Probability distribution across all risk classes. Keys match risk class names, values are floats 0.0-1.0 summing to 1.0")


# Lazy-load the model
_model = None


def _get_model():
    global _model
    if _model is None:
        model_path = hf_hub_download(
            repo_id="saifhmb/Credit-Card-Risk-Model",
            filename="skops-akmmropo.pkl"
        )
        _model = joblib.load(model_path)
    return _model


def predict_credit_risk(input_data: CreditCardRiskInput) -> CreditCardRiskOutput:
    """Classify credit card applicant into risk category using sklearn logistic regression pipeline.
    
    Returns multiclass prediction: 'Good risk', 'Bad loss', or 'Bad profit' with probabilities.
    Model auto-downloads from HuggingFace on first call and is cached for subsequent calls.
    """
    model = _get_model()
    
    df = pd.DataFrame([{
        "GENDER": input_data.gender,
        "MARITAL": input_data.marital,
        "HOWPAID": input_data.howpaid,
        "MORTGAGE": input_data.mortgage,
        "AGE": input_data.age,
        "INCOME": input_data.income,
        "NUMKIDS": input_data.numkids,
        "NUMCARDS": input_data.numcards,
        "STORECAR": input_data.storecar,
        "LOANS": input_data.loans,
    }])
    
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    classes = model.classes_
    
    # Map numeric class labels to descriptive names
    risk_labels = {0: "Bad loss", 1: "Bad profit", 2: "Good risk"}
    risk_name = risk_labels.get(int(prediction), str(prediction))
    
    return CreditCardRiskOutput(
        risk=risk_name,
        probabilities={risk_labels.get(int(c), str(c)): float(p) for c, p in zip(classes, proba)}
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class CreditCardRiskToolInput(BaseModel):
    """Input for credit card risk classification. Categorical fields are case-sensitive lowercase."""
    gender: Literal["m", "f"] = Field(description="Biological sex. EXACT: 'm' (male) | 'f' (female)")
    marital: Literal["married", "single", "divsepwid"] = Field(description="Marital status. EXACT: 'married' | 'single' | 'divsepwid' (divorced/separated/widowed)")
    howpaid: Literal["monthly", "weekly"] = Field(description="Salary payment frequency. EXACT: 'monthly' | 'weekly'")
    mortgage: Literal["y", "n"] = Field(description="Has active mortgage. EXACT: 'y' | 'n'")
    age: int = Field(description="Age in years (e.g., 35). Valid: 18+")
    income: float = Field(description="Annual income in currency units (e.g., 45000.0). Valid: >0")
    numkids: int = Field(description="Dependent children count (e.g., 2). Valid: >=0")
    numcards: int = Field(description="Existing credit cards held (e.g., 3). Valid: >=0")
    storecar: int = Field(description="Retail/store cards held (e.g., 1). Valid: >=0")
    loans: int = Field(description="Active loans count (e.g., 2). Valid: >=0")


@tool("credit_card_risk_prediction", args_schema=CreditCardRiskToolInput)
def credit_card_risk_prediction_tool(
    gender: Literal["m", "f"],
    marital: Literal["married", "single", "divsepwid"],
    howpaid: Literal["monthly", "weekly"],
    mortgage: Literal["y", "n"],
    age: int,
    income: float,
    numkids: int,
    numcards: int,
    storecar: int,
    loans: int,
) -> str:
    """Classify credit card applicant risk using logistic regression (sklearn pipeline, ~70% accuracy).

    MODEL: saifhmb/Credit-Card-Risk-Model - Logistic regression with OneHotEncoder (categorical) + StandardScaler (numerical).
    TRAINING: Bank customer credit card data. Multiclass classification.

    USE CASES:
    - Credit card application screening: determine approve/reject recommendation
    - Risk stratification: segment applicants by predicted loss probability
    - Feature analysis: income, existing credit exposure (numcards, loans) are key predictors

    LIMITATIONS:
    - Training data demographics unknown - may not generalize across geographies
    - Binary gender/marital encoding - limited demographic representation
    - ~70% accuracy - use as triage signal, not final decision
    - First call downloads model (~1MB) - slight latency on cold start

    OUTPUT INTERPRETATION:
    - 'Good risk': Recommend approval - low default/loss probability
    - 'Bad loss': Recommend rejection - high likelihood of default and loss
    - 'Bad profit': Recommend rejection - unlikely to be profitable even if no default
    - Probabilities: Use highest probability class; consider 'Good risk' prob as approval confidence
    """
    result = predict_credit_risk(CreditCardRiskInput(
        gender=gender,
        marital=marital,
        howpaid=howpaid,
        mortgage=mortgage,
        age=age,
        income=income,
        numkids=numkids,
        numcards=numcards,
        storecar=storecar,
        loans=loans,
    ))
    
    proba_str = ", ".join(f"{k}: {v:.1%}" for k, v in result.probabilities.items())
    return f"Risk Classification: {result.risk}\nProbabilities: {proba_str}"

