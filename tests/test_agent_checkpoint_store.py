"""Agent thread store: must not treat repr-serialized agents as runnable graphs."""

from unittest.mock import MagicMock

from backend.training import repository


def test_get_simple_agent_store_rejects_string_agent():
    tid = "test-thread-str-agent"
    repository.simple_agent_store[tid] = {
        "agent": "<CompiledStateGraph ...>",
        "state": {},
    }
    try:
        assert repository.get_simple_agent_store(tid) is None
        assert repository.thread_exists(tid) is False
    finally:
        repository.simple_agent_store.pop(tid, None)


def test_get_simple_agent_store_accepts_runnable_agent():
    tid = "test-thread-mock-agent"
    mock_agent = MagicMock()
    mock_agent.stream = lambda *a, **k: iter(())
    repository.simple_agent_store[tid] = {"agent": mock_agent, "state": {}}
    try:
        got = repository.get_simple_agent_store(tid)
        assert got is not None
        assert got["agent"] is mock_agent
        assert repository.thread_exists(tid) is True
    finally:
        repository.simple_agent_store.pop(tid, None)


def test_put_simple_agent_store_serializable_omits_agent(monkeypatch):
    """DB payload must not stringify the compiled graph (breaks resume)."""
    captured = {}

    class FakeSession:
        def get(self, model, key):
            return None

        def add(self, row):
            captured["row"] = row

    def fake_get_db_session():
        class CM:
            def __enter__(self):
                return FakeSession()

            def __exit__(self, *args):
                return None

        return CM()

    monkeypatch.setattr(repository, "get_db_session", fake_get_db_session)
    tid = "omit-agent-thread"
    graph = object()
    repository.put_simple_agent_store(
        tid,
        {"agent": graph, "checkpointer": object(), "mode": "graph", "emitted_steps": set()},
    )
    try:
        assert repository.simple_agent_store[tid]["agent"] is graph
        row = captured.get("row")
        assert row is not None
        st = row.state
        assert "agent" not in st
        assert "checkpointer" not in st
        assert st.get("mode") == "graph"
    finally:
        repository.simple_agent_store.pop(tid, None)
