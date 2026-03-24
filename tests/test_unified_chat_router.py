"""TDD-style router tests for unified POST /api/chat (no live LLM)."""

from unittest.mock import patch

import pytest

from backend.chat.schemas import ChatRequest, ResumeTrainingPayload
from backend.chat.service import chat


def test_router_resume_delegates_to_graph_resume():
    with patch(
        "backend.training.service.generate_graph_resume_sse_events",
        return_value=iter(['data: {"type":"x"}\n\n']),
    ) as fn:
        req = ChatRequest(
            message="",
            resume_training=ResumeTrainingPayload(
                thread_id="graph-deadbeef", approved=True, feedback=None
            ),
        )
        tid, gen = chat(req)
        assert tid == "graph-deadbeef"
        fn.assert_called_once_with("graph-deadbeef", True, None)
        assert next(gen).startswith("data:")


def test_router_linked_datasets_delegates_to_graph_start():
    with patch(
        "backend.training.service.generate_graph_sse_events",
        return_value=iter(['data: {"type":"started"}\n\n']),
    ) as fn:
        req = ChatRequest(
            message="Build a classifier",
            linked_datasets=["csv/foo.csv"],
            experiment_id="exp-123",
        )
        tid, gen = chat(req)
        assert tid == ""
        fn.assert_called_once_with(
            "Build a classifier",
            ["csv/foo.csv"],
            None,
            experiment_id="exp-123",
        )
        assert next(gen).startswith("data:")


def test_router_mode_train_without_datasets():
    with patch(
        "backend.training.service.generate_graph_sse_events",
        return_value=iter([]),
    ) as fn:
        req = ChatRequest(message=" ", mode="train")
        chat(req)
        fn.assert_called_once()
        assert fn.call_args[0][0] == "Training run"


def test_router_default_is_orchestrator_chat():
    with patch(
        "backend.training.service.generate_graph_sse_events",
    ) as graph_fn:
        with patch("backend.chat.service.generate_chat_sse") as gcs:
            with patch(
                "backend.chat.service.training_repo.get_experiment", return_value=None
            ):
                with patch(
                    "backend.chat.service.should_route_to_training_graph",
                    return_value=False,
                ):
                    gcs.return_value = iter(['data: {"type":"start"}\n\n'])
                    req = ChatRequest(message="Hello")
                    tid, gen = chat(req)
                    assert tid.startswith("chat-") or len(tid) > 0
                    graph_fn.assert_not_called()
                    gcs.assert_called_once()
                    next(gen)


def test_router_llm_path_can_select_graph():
    with patch(
        "backend.training.service.generate_graph_sse_events",
        return_value=iter(['data: {"type":"started"}\n\n']),
    ) as fn:
        with patch(
            "backend.chat.service.should_route_to_training_graph",
            return_value=True,
        ):
            req = ChatRequest(message="Run the full training pipeline on my goal")
            tid, gen = chat(req)
            assert tid == ""
            fn.assert_called_once()
            next(gen)


def test_router_explicit_mode_chat_skips_graph():
    with patch(
        "backend.training.service.generate_graph_sse_events",
    ) as graph_fn:
        with patch("backend.chat.service.generate_chat_sse") as gcs:
            with patch(
                "backend.chat.service.training_repo.get_experiment", return_value=None
            ):
                gcs.return_value = iter(['data: {"type":"start"}\n\n'])
                req = ChatRequest(
                    message="Explain precision and recall",
                    mode="chat",
                )
                _, gen = chat(req)
                graph_fn.assert_not_called()
                gcs.assert_called_once()
                next(gen)


def test_chat_request_rejects_empty_without_resume_or_train():
    with pytest.raises(ValueError, match="message is required"):
        ChatRequest(message="   ")
