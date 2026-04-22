"""TDD-style router tests for unified POST /api/chat (no live LLM).

With the unified-agent refactor, ``chat()`` has only two branches:

1. Resume — delegates directly to the training sub-agent's resume stream.
2. Everything else — streams through ``generate_chat_sse`` (Jubilee).
   Training is now started by Jubilee calling ``run_training_pipeline``,
   which the chat generator detects and pivots on internally. The HTTP
   router therefore never calls ``generate_graph_sse_events`` directly.
"""

from unittest.mock import patch

import pandas as pd
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


def test_router_default_is_orchestrator_chat():
    with patch("backend.chat.service.generate_chat_sse") as gcs:
        with patch(
            "backend.chat.service.training_repo.get_experiment", return_value=None
        ):
            gcs.return_value = iter(['data: {"type":"start"}\n\n'])
            req = ChatRequest(message="Hello")
            tid, gen = chat(req)
            assert tid.startswith("chat-") or len(tid) > 0
            gcs.assert_called_once()
            next(gen)


def test_router_mode_train_with_refs_pivots_directly_to_training_graph():
    """``mode=train`` with refs is the plan-approval fast path: it bypasses
    Jubilee and goes straight to the training sub-agent so the UI never gets
    stuck re-rendering ``propose_training_plan`` cards."""
    with patch(
        "backend.training.service.generate_graph_sse_events",
        return_value=iter(['data: {"type":"start","training_graph":true}\n\n']),
    ) as graph_fn:
        with patch("backend.chat.service.generate_chat_sse") as gcs:
            with patch(
                "backend.chat.service.training_repo.get_experiment", return_value=None
            ):
                req = ChatRequest(
                    message="Build a classifier",
                    mode="train",
                    linked_datasets=["csv/foo.csv"],
                    experiment_id="exp-123",
                )
                _, gen = chat(req)
                gcs.assert_not_called()
                graph_fn.assert_called_once()
                args, kwargs = graph_fn.call_args
                assert args[0] == "Build a classifier"
                assert args[1] == ["csv/foo.csv"]
                assert kwargs.get("experiment_id") == "exp-123"
                next(gen)


def test_router_mode_train_without_datasets_requires_message():
    """Without linked datasets, ``mode=train`` still requires a non-empty message
    because Jubilee has to have something to work with."""
    with pytest.raises(ValueError, match="message is required"):
        ChatRequest(message="   ", mode="train")


def test_router_mode_train_without_datasets_passes_with_message():
    with patch("backend.chat.service.generate_chat_sse") as gcs:
        with patch(
            "backend.chat.service.training_repo.get_experiment", return_value=None
        ):
            gcs.return_value = iter(['data: {"type":"start"}\n\n'])
            req = ChatRequest(message="Train on the workspace data", mode="train")
            chat(req)
            gcs.assert_called_once()


def test_router_linked_datasets_without_mode_stays_in_orchestrator():
    """Attached datasets alone never start training — Jubilee decides."""
    with patch(
        "backend.training.service.generate_graph_sse_events",
    ) as graph_fn:
        with patch("backend.chat.service.generate_chat_sse") as gcs:
            with patch(
                "backend.chat.service.training_repo.get_experiment", return_value=None
            ):
                with patch(
                    "backend.training.service.resolve_linked_dataset",
                    return_value="health-insurance-prices",
                ):
                    with patch(
                        "utils.get_registered_dataset",
                        return_value=pd.DataFrame(
                            {"age": [39], "charges": [13270.0]}
                        ),
                    ):
                        gcs.return_value = iter(['data: {"type":"start"}\n\n'])
                        req = ChatRequest(
                            message="Which columns predict price?",
                            linked_datasets=["health-insurance-prices"],
                        )
                        _, gen = chat(req)
                        graph_fn.assert_not_called()
                        gcs.assert_called_once()
                        user_to_agent = gcs.call_args[0][1]
                        assert "health-insurance-prices" in user_to_agent
                        assert "[ATTACHED DATASETS" in user_to_agent
                        kwargs = gcs.call_args[1]
                        assert kwargs.get("attached_dataset_events")
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
