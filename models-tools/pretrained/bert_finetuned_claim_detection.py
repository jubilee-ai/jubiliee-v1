import torch
from langchain.tools import tool
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ClaimDetectionInput(BaseModel):
    text: str


class ClaimDetectionOutput(BaseModel):
    label: str  # "check-worthy" or "not-check-worthy"
    confidence: float
    scores: dict[str, float]  # all class scores


# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("XiaojingEllen/bert-finetuned-claim-detection")
model = AutoModelForSequenceClassification.from_pretrained("XiaojingEllen/bert-finetuned-claim-detection")

# Label mapping: LABEL_0 = not check-worthy, LABEL_1 = check-worthy
LABELS = ["not-check-worthy", "check-worthy"]


def detect_claim(input_data: ClaimDetectionInput) -> ClaimDetectionOutput:
    """Detect if text contains a check-worthy claim for insurance screening."""
    inputs = tokenizer(
        input_data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

    scores = {label: prob.item() for label, prob in zip(LABELS, probabilities)}
    predicted_idx = probabilities.argmax().item()

    return ClaimDetectionOutput(
        label=LABELS[predicted_idx],
        confidence=probabilities[predicted_idx].item(),
        scores=scores
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class ClaimDetectionToolInput(BaseModel):
    """Input schema for the claim detection tool."""

    text: str = Field(
        description="Insurance claim text or statement to screen. Best with 1-3 sentences, max 512 tokens."
    )


@tool("claim_detection", args_schema=ClaimDetectionToolInput)
def claim_detection_tool(text: str) -> str:
    """Identify check-worthy statements in insurance claims for screening and fraud detection.

    USE FOR:
    - Insurance claim screening and triage
    - Fraud detection in claim descriptions
    - Compliance document filtering
    - Flagging statements that require fact-checking or verification

    DON'T USE FOR:
    - Determining if a claim is TRUE or FALSE (only detects check-worthiness)
    - Sentiment analysis (use finbert_sentiment)
    - Non-English text

    OUTPUT:
    - label: "check-worthy" (needs verification) or "not-check-worthy" (routine/factual)
    - confidence: 0-100%, use >80% for reliable automated routing
    - scores: probabilities for both classes

    TIPS:
    - Route high-confidence check-worthy claims (>85%) to human reviewers
    - Combine with entity extraction to identify specific claim details
    - Confidence 40-60% = borderline, flag for manual review

    EXAMPLES:
    Check-worthy: "The accident caused $50,000 in damages", "My car was stolen last night"
    Not-check-worthy: "Please find attached my policy documents", "Thank you for your assistance"

    Args:
        text: Insurance claim text to analyze (1-3 sentences, max 512 tokens)
    """
    result = detect_claim(ClaimDetectionInput(text=text))

    return (
        f"Classification: {result.label}\n"
        f"Confidence: {result.confidence:.2%}\n"
        f"Scores: {', '.join(f'{k}={v:.2%}' for k, v in result.scores.items())}"
    )