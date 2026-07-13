import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.websocket_runtime import MAX_PENDING_MESSAGES_PER_SESSION, WebSocketSessionManager  # noqa: E402


def test_client_ack_delivery_replay_ack_and_nack() -> None:
    manager = WebSocketSessionManager()
    first = manager.queue_message("s1", "notification", {"message": "one"})
    second = manager.queue_message("s1", "notification", {"message": "two"})

    delivered = manager.deliver_messages("s1", "sub-1", ack_mode="client-individual")
    assert [item["id"] for item in delivered] == [first["id"], second["id"]]
    assert delivered[0]["delivered"] is True
    assert manager.peek_messages("s1")[0]["id"] == first["id"]

    acked = manager.ack_messages("s1", [first["id"]])
    assert [item["id"] for item in acked] == [first["id"]]
    replay = manager.replay_messages("s1")
    assert [item["id"] for item in replay] == [second["id"]]

    nacked = manager.nack_messages("s1", [second["id"]], reason="retry later")
    assert nacked[0]["nacked"] is True
    replay_after_nack = manager.replay_messages("s1")
    assert replay_after_nack[0]["id"] == second["id"]
    assert replay_after_nack[0]["nackReason"] == "retry later"


def test_auto_ack_delivery_preserves_legacy_drain_behavior() -> None:
    manager = WebSocketSessionManager()
    manager.queue_message("s1", "notification", {"message": "one"})
    manager.queue_message("s1", "notification", {"message": "two"})

    delivered = manager.deliver_messages("s1", "sub-0", ack_mode="auto")

    assert len(delivered) == 2
    assert manager.peek_messages("s1") == []
    assert manager.replay_messages("s1") == []


def test_subscription_lifecycle_filters_delivery_and_replay_by_destination() -> None:
    manager = WebSocketSessionManager()
    user_sub = manager.register_subscription("s1", "sub-user", "/user/queue/messages", ack_mode="client-individual")
    topic_sub = manager.register_subscription("s1", "sub-topic", "/topic/swarm", ack_mode="client-individual")
    assert user_sub["destination"] == "/user/queue/messages"
    assert topic_sub["destination"] == "/topic/swarm"

    user_msg = manager.queue_message("s1", "notification", {"message": "user"})
    topic_msg = manager.queue_message("s1", "swarm_state_update", {"message": "topic"}, destination="/topic/swarm")

    user_delivery = manager.deliver_messages("s1", "sub-user", ack_mode="client-individual")
    topic_delivery = manager.deliver_messages("s1", "sub-topic", ack_mode="client-individual")

    assert [item["id"] for item in user_delivery] == [user_msg["id"]]
    assert [item["id"] for item in topic_delivery] == [topic_msg["id"]]
    assert user_delivery[0]["destination"] == "/user/queue/messages"
    assert topic_delivery[0]["destination"] == "/topic/swarm"

    topic_replay = manager.replay_messages("s1", subscription_id="sub-topic")
    assert [item["id"] for item in topic_replay] == [topic_msg["id"]]

    assert manager.unsubscribe("s1", "sub-user") is True
    manager.queue_message("s1", "notification", {"message": "after unsubscribe"})
    assert manager.deliver_messages("s1", "sub-user", ack_mode="client-individual") == []


def test_subscription_resume_replays_after_last_message_for_destination() -> None:
    manager = WebSocketSessionManager()
    first_topic = manager.queue_message("s1", "swarm_state_update", {"message": "first"}, destination="/topic/swarm")
    manager.register_subscription("s1", "sub-topic", "/topic/swarm", ack_mode="client-individual")
    delivered = manager.deliver_messages("s1", "sub-topic", ack_mode="client-individual")
    assert [item["id"] for item in delivered] == [first_topic["id"]]

    second_topic = manager.queue_message("s1", "swarm_state_update", {"message": "second"}, destination="/topic/swarm")
    manager.queue_message("s1", "notification", {"message": "user-only"})

    resumed = manager.resume_subscription(
        "s1",
        "sub-topic",
        "/topic/swarm",
        ack_mode="client-individual",
        since_id=first_topic["id"],
    )

    assert resumed["subscription"]["destination"] == "/topic/swarm"
    assert [item["id"] for item in resumed["messages"]] == [second_topic["id"]]
    assert resumed["replayedCount"] == 1
    assert resumed["messages"][0]["deliveryCount"] == 1


def test_queue_backpressure_preserves_order_and_reports_dropped_messages() -> None:
    manager = WebSocketSessionManager()

    first = manager.queue_message("s1", "notification", {"index": 0})
    last = first
    for index in range(1, MAX_PENDING_MESSAGES_PER_SESSION + 5):
        last = manager.queue_message("s1", "notification", {"index": index})

    stats = manager.queue_stats("s1")
    pending = manager.peek_messages("s1")

    assert stats["pendingCount"] == MAX_PENDING_MESSAGES_PER_SESSION
    assert stats["droppedCount"] == 5
    assert stats["firstPendingSequence"] == 6
    assert stats["lastSequence"] == MAX_PENDING_MESSAGES_PER_SESSION + 5
    assert first["sequence"] == 1
    assert last["sequence"] == MAX_PENDING_MESSAGES_PER_SESSION + 5
    assert [item["sequence"] for item in pending] == list(range(6, MAX_PENDING_MESSAGES_PER_SESSION + 6))
    assert pending[0]["payload"]["index"] == 5
