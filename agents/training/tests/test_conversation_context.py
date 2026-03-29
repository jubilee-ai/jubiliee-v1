"""Tests for multi-turn chat normalization used by the planner."""

from agents.training.core.conversation_context import (
    normalize_conversation_turns,
    transcript_for_planner_prompt,
)


def test_normalize_appends_trigger_when_last_turn_not_user_message():
    turns = normalize_conversation_turns(
        [{"role": "user", "content": "Hi"}, {"role": "agent", "content": "Hello"}],
        triggering_message="Train a churn model",
    )
    assert turns[-1] == {"role": "user", "content": "Train a churn model"}
    assert len(turns) == 3


def test_normalize_no_duplicate_when_trigger_matches_last_user():
    turns = normalize_conversation_turns(
        [{"role": "user", "content": "Same"}],
        triggering_message="Same",
    )
    assert turns == [{"role": "user", "content": "Same"}]


def test_normalize_empty_conversation_uses_trigger():
    turns = normalize_conversation_turns(None, triggering_message="  Go  ")
    assert turns == [{"role": "user", "content": "Go"}]


def test_transcript_tail_truncation():
    long_turns = [{"role": "user", "content": "x" * 5000} for _ in range(5)]
    text = transcript_for_planner_prompt(long_turns, max_chars=400)
    assert "omitted" in text
    assert len(text) <= 400
