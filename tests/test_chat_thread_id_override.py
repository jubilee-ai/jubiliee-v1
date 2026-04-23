"""``chat_thread_id`` on :class:`backend.chat.schemas.ChatRequest` overrides experiment thread."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.chat.schemas import ChatRequest


def test_chat_passes_stripped_chat_thread_id_to_generate_chat_sse():
    pytest.importorskip("sklearn")
    import backend.chat.service as chat_service  # noqa: PLC0415 — after sklearn check

    with (
        patch.object(
            chat_service.training_repo,
            "get_latest_training_context",
            return_value=None,
        ),
        patch.object(chat_service, "generate_chat_sse", return_value=iter([])) as mock_sse,
    ):
        chat_service.chat(
            ChatRequest(
                message="hi",
                experiment_id="exp-1",
                chat_thread_id="  custom-tid  ",
            ),
            org_id=None,
        )

    first_positional = mock_sse.call_args[0][0]
    assert first_positional == "custom-tid"
