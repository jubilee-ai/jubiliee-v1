import pytest
from bert_finetuned_claim_detection import (ClaimDetectionInput,
                                            ClaimDetectionOutput,
                                            claim_detection_tool, detect_claim)


def test_check_worthy_claim():
    """Test that an insurance claim statement is classified as check-worthy."""
    result = detect_claim(ClaimDetectionInput(
        text="My vehicle was rear-ended and sustained $8,000 in damages."
    ))
    print(f"Input: 'My vehicle was rear-ended and sustained $8,000 in damages.'")
    print(f"Output: {result}")
    assert isinstance(result, ClaimDetectionOutput)
    assert result.confidence > 0.0
    assert abs(sum(result.scores.values()) - 1.0) < 0.01  # scores sum to 1


def test_not_check_worthy():
    """Test that routine correspondence is classified as not-check-worthy."""
    result = detect_claim(ClaimDetectionInput(
        text="Please find my policy documents attached for your reference."
    ))
    print(f"Input: 'Please find my policy documents attached for your reference.'")
    print(f"Output: {result}")
    assert isinstance(result, ClaimDetectionOutput)
    assert result.confidence > 0.0
    assert len(result.scores) == 2  # should have 2 classes


def test_langchain_tool_returns_formatted_string():
    """Test that the LangChain tool returns properly formatted output."""
    result = claim_detection_tool.invoke({
        "text": "The fire destroyed my entire kitchen and caused $25,000 in damages."
    })
    print(f"Input: 'The fire destroyed my entire kitchen and caused $25,000 in damages.'")
    print(f"Output:\n{result}")
    assert isinstance(result, str)
    assert "Classification:" in result
    assert "Confidence:" in result
    assert "Scores:" in result
