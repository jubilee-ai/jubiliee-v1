from types import SimpleNamespace
from unittest.mock import patch

from backend.chat.intent_router import should_route_to_training_graph
from backend.chat.schemas import ChatRequest


def test_vague_training_request_stays_in_chat_for_clarification():
    req = ChatRequest(message="Help me build a churn model")

    with patch(
        "backend.chat.intent_router.get_settings",
        return_value=SimpleNamespace(openai_api_key="test-key"),
    ), patch("backend.chat.intent_router._classify_via_llm") as classify:
        assert should_route_to_training_graph(req) is False
        classify.assert_not_called()


def test_explicit_training_request_can_still_route_to_graph():
    req = ChatRequest(message="Train on customer_churn.csv and predict churn")

    with patch(
        "backend.chat.intent_router.get_settings",
        return_value=SimpleNamespace(openai_api_key="test-key"),
    ), patch(
        "backend.chat.intent_router._classify_via_llm",
        return_value=True,
    ) as classify:
        assert should_route_to_training_graph(req) is True
        classify.assert_called_once_with(req.message.strip(), req)
