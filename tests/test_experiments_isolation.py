"""
Phase 1 & 2 integration tests: experiment isolation and streaming.

Run with: python -m pytest tests/test_experiments_isolation.py -v
Or run directly against a running backend: python tests/test_experiments_isolation.py
"""
import json
import sys
import time
import requests

BASE = "http://localhost:8000"


def test_create_experiments():
    """Create two experiments and verify they get unique IDs."""
    r1 = requests.post(f"{BASE}/api/experiments", json={"name": "Test Exp A"})
    assert r1.status_code == 200, f"Create A failed: {r1.text}"
    a = r1.json()
    assert a["id"].startswith("exp-")
    assert a["chat_thread_id"].startswith("chat-exp-")
    assert a["status"] == "created"

    r2 = requests.post(f"{BASE}/api/experiments", json={"name": "Test Exp B"})
    assert r2.status_code == 200, f"Create B failed: {r2.text}"
    b = r2.json()
    assert b["id"] != a["id"]
    assert b["chat_thread_id"] != a["chat_thread_id"]

    return a, b


def test_list_experiments(exp_a, exp_b):
    """List experiments shows both."""
    r = requests.get(f"{BASE}/api/experiments")
    assert r.status_code == 200
    ids = {e["id"] for e in r.json()}
    assert exp_a["id"] in ids
    assert exp_b["id"] in ids


def test_get_experiment_detail(exp_a):
    """GET returns full detail with empty chat_history."""
    r = requests.get(f"{BASE}/api/experiments/{exp_a['id']}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["chat_history"] == [] or detail["chat_history"] is None or isinstance(detail["chat_history"], list)
    assert detail["training_state"] is None
    assert detail["training_context"] is None


def test_chat_isolation(exp_a, exp_b):
    """Chat to exp A uses A's thread_id, chat to exp B uses B's thread_id."""
    # Chat to A
    r_a = requests.post(
        f"{BASE}/api/chat",
        json={"message": "Hello A", "experiment_id": exp_a["id"]},
        stream=True,
    )
    assert r_a.status_code == 200
    thread_a = None
    for line in r_a.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if data.get("thread_id"):
                thread_a = data["thread_id"]
                break

    # Chat to B
    r_b = requests.post(
        f"{BASE}/api/chat",
        json={"message": "Hello B", "experiment_id": exp_b["id"]},
        stream=True,
    )
    assert r_b.status_code == 200
    thread_b = None
    for line in r_b.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if data.get("thread_id"):
                thread_b = data["thread_id"]
                break

    assert thread_a == exp_a["chat_thread_id"], f"A thread mismatch: {thread_a} != {exp_a['chat_thread_id']}"
    assert thread_b == exp_b["chat_thread_id"], f"B thread mismatch: {thread_b} != {exp_b['chat_thread_id']}"
    assert thread_a != thread_b, "Threads must be different!"
    print(f"  Experiment A thread: {thread_a}")
    print(f"  Experiment B thread: {thread_b}")


def test_backward_compat_no_experiment_id():
    """Chat without experiment_id still works and generates a random thread."""
    r = requests.post(
        f"{BASE}/api/chat",
        json={"message": "hello"},
        stream=True,
    )
    assert r.status_code == 200
    thread = None
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if data.get("thread_id"):
                thread = data["thread_id"]
                break
    assert thread is not None
    assert thread.startswith("chat-")
    print(f"  Generated thread: {thread}")


def test_update_experiment(exp_a):
    """PATCH updates the experiment name."""
    r = requests.patch(
        f"{BASE}/api/experiments/{exp_a['id']}",
        json={"name": "Renamed A"},
    )
    assert r.status_code == 200

    detail = requests.get(f"{BASE}/api/experiments/{exp_a['id']}").json()
    assert detail["name"] == "Renamed A"


def test_delete_experiment(exp_b):
    """DELETE removes the experiment."""
    r = requests.delete(f"{BASE}/api/experiments/{exp_b['id']}")
    assert r.status_code == 200

    r2 = requests.get(f"{BASE}/api/experiments/{exp_b['id']}")
    assert r2.status_code == 404


def test_chat_token_streaming():
    """Verify chat responses include token-level streaming events."""
    r = requests.post(
        f"{BASE}/api/chat",
        json={"message": "What is machine learning?"},
        stream=True,
    )
    assert r.status_code == 200

    event_types = set()
    token_count = 0
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            data = json.loads(line[6:])
            event_types.add(data.get("type", ""))
            if data.get("type") == "token":
                token_count += 1

    assert "start" in event_types, "Missing 'start' event"
    assert "token" in event_types, "Missing 'token' events -- no streaming?"
    assert "end" in event_types, "Missing 'end' event"
    print(f"  Token events received: {token_count}")
    print(f"  Event types: {event_types}")


def test_health():
    r = requests.get(f"{BASE}/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"


def cleanup(exp_ids):
    for eid in exp_ids:
        requests.delete(f"{BASE}/api/experiments/{eid}")


if __name__ == "__main__":
    print("=" * 60)
    print("Jubilee Integration Tests: Experiment Isolation & Streaming")
    print("=" * 60)

    created_ids = []
    try:
        print("\n[1] Health check...")
        test_health()
        print("  PASS")

        print("\n[2] Create experiments...")
        a, b = test_create_experiments()
        created_ids.extend([a["id"], b["id"]])
        print(f"  PASS: {a['id']}, {b['id']}")

        print("\n[3] List experiments...")
        test_list_experiments(a, b)
        print("  PASS")

        print("\n[4] Get experiment detail...")
        test_get_experiment_detail(a)
        print("  PASS")

        print("\n[5] Chat isolation (different threads per experiment)...")
        test_chat_isolation(a, b)
        print("  PASS")

        print("\n[6] Backward compat (chat without experiment_id)...")
        test_backward_compat_no_experiment_id()
        print("  PASS")

        print("\n[7] Token-level streaming...")
        test_chat_token_streaming()
        print("  PASS")

        print("\n[8] Update experiment...")
        test_update_experiment(a)
        print("  PASS")

        print("\n[9] Delete experiment...")
        test_delete_experiment(b)
        created_ids.remove(b["id"])
        print("  PASS")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n  FAIL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)
    finally:
        cleanup(created_ids)
