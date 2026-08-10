import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.sqlite_store import SQLiteStateStore  # noqa: E402


def workspace() -> Path:
    import uuid

    root = BACKEND_DIR / ".test-workspace" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def state_with_session(session_id: str = "s1") -> dict:
    return {
        "sessions": {
            session_id: {
                "id": session_id,
                "title": "SQLite session",
                "model": "qwen3.7-max",
                "workingDir": "/tmp/work",
                "createdAt": "2026-06-26T00:00:00Z",
                "updatedAt": "2026-06-26T00:00:01Z",
                "messages": [
                    {"id": "m1", "type": "user", "content": [{"type": "text", "text": "one"}], "usage": {"inputTokens": 1}},
                    {"id": "m2", "type": "assistant", "content": [{"type": "text", "text": "two"}], "usage": {"outputTokens": 2}},
                    {"id": "m3", "type": "user", "content": [{"type": "text", "text": "three"}]},
                ],
            }
        },
        "memories": [{"id": "mem1", "title": "memory", "content": "remember"}],
        "activities": {session_id: [{"id": "a1", "type": "query", "title": "activity"}]},
        "fileSnapshots": [],
        "costEvents": [],
        "anomalies": [],
        "evidence": {},
    }


def test_sqlite_store_migrates_syncs_and_reads_sessions() -> None:
    store = SQLiteStateStore(workspace() / "test.db")
    applied = store.migrate()
    assert applied
    store.sync_state(state_with_session())

    stats = store.stats().to_dict()
    assert stats["journalMode"].lower() == "wal"
    assert stats["foreignKeys"] is True
    assert stats["sessions"] == 1
    assert stats["messages"] == 3
    assert stats["memories"] == 1
    assert stats["activities"] == 1

    session = store.get_session("s1")
    assert session["model"] == "qwen3.7-max"
    messages = store.list_messages("s1")
    assert [message["id"] for message in messages] == ["m1", "m2", "m3"]


def test_sqlite_store_cascade_delete_and_message_rollback() -> None:
    store = SQLiteStateStore(workspace() / "test.db")
    state = state_with_session("s1")
    state["sessions"]["s2"] = {**state["sessions"]["s1"], "id": "s2", "messages": [{"id": "s2-m1", "type": "user", "content": "x"}]}
    store.sync_state(state)

    deleted = store.delete_messages_after("s1", 1)
    assert deleted == 2
    assert [message["id"] for message in store.list_messages("s1")] == ["m1"]
    assert [message["id"] for message in store.list_messages("s2")] == ["s2-m1"]

    state["sessions"].pop("s1")
    store.sync_state(state)
    assert store.get_session("s1") is None
    assert store.list_messages("s1") == []
    assert store.stats().messages == 1
