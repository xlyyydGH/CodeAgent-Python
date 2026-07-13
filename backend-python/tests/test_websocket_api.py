import sys
import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app import WS_SESSION_MANAGER, app  # noqa: E402
from zhikun_py.websocket_runtime import MAX_PENDING_MESSAGES_PER_SESSION  # noqa: E402


client = TestClient(app)


def stomp(command: str, headers: dict[str, str] | None = None, body: str = "") -> str:
    lines = [command]
    for key, value in (headers or {}).items():
        lines.append(f"{key}:{value}")
    return "\n".join(lines) + "\n\n" + body + "\x00"


def sockjs_payload(frame: str) -> str:
    return "a" + json.dumps([frame])


def decode_sockjs_frame(raw: str) -> str:
    if raw.startswith("a["):
        parsed = json.loads(raw[1:])
        return parsed[0]
    return raw


def parse_stomp_frame(raw: str) -> tuple[str, dict[str, str], str]:
    frame = decode_sockjs_frame(raw).rstrip("\x00")
    head, _, body = frame.partition("\n\n")
    lines = head.splitlines()
    command = lines[0]
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key] = value
    return command, headers, body


def test_ws_session_ack_nack_and_replay_rest_contract() -> None:
    session_id = f"ws-ack-{uuid4().hex}"
    principal = f"principal-{uuid4().hex}"
    assert client.post("/api/ws/sessions/bind", json={"principal": principal, "sessionId": session_id}).json()["success"] is True

    first = client.post(f"/api/ws/sessions/{session_id}/push", json={"type": "notification", "payload": {"message": "one"}}).json()["queued"]
    second = client.post(f"/api/ws/sessions/{session_id}/push", json={"type": "notification", "payload": {"message": "two"}}).json()["queued"]

    delivered = client.get(f"/api/ws/sessions/{session_id}/messages/deliver?subscriptionId=sub-1&ackMode=client-individual").json()
    assert [item["id"] for item in delivered["messages"]] == [first["id"], second["id"]]
    assert client.get(f"/api/ws/sessions/{session_id}/messages/peek").json()["count"] == 2

    acked = client.post(f"/api/ws/sessions/{session_id}/messages/ack", json={"messageIds": [first["id"]]}).json()
    assert acked["ackedCount"] == 1
    assert [item["id"] for item in client.get(f"/api/ws/sessions/{session_id}/messages/replay").json()["messages"]] == [second["id"]]

    nacked = client.post(f"/api/ws/sessions/{session_id}/messages/nack", json={"messageIds": [second["id"]], "reason": "retry"}).json()
    assert nacked["nackedCount"] == 1
    replay = client.get(f"/api/ws/sessions/{session_id}/messages/replay").json()["messages"]
    assert replay[0]["id"] == second["id"]
    assert replay[0]["nackReason"] == "retry"

    WS_SESSION_MANAGER.pending_messages.pop(session_id, None)
    WS_SESSION_MANAGER.message_journal.pop(session_id, None)


def test_sockjs_stomp_client_ack_removes_delivered_message() -> None:
    session_id = f"ws-stomp-{uuid4().hex}"
    queued = WS_SESSION_MANAGER.queue_message(session_id, "notification", {"type": "notification", "message": "queued"})

    with client.websocket_connect("/ws/websocket") as ws:
        assert ws.receive_text() == "o"
        ws.send_text(sockjs_payload(stomp("CONNECT", {"accept-version": "1.2", "X-Session-Id": session_id})))
        connected_command, _, _ = parse_stomp_frame(ws.receive_text())
        assert connected_command == "CONNECTED"

        ws.send_text(sockjs_payload(stomp("SUBSCRIBE", {"id": "sub-ack", "destination": "/user/queue/messages", "ack": "client-individual"})))
        message_command, headers, body = parse_stomp_frame(ws.receive_text())
        assert message_command == "MESSAGE"
        assert headers["message-id"] == queued["id"]
        assert json.loads(body)["message"] == "queued"
        assert WS_SESSION_MANAGER.peek_messages(session_id)

        ws.send_text(sockjs_payload(stomp("ACK", {"message-id": queued["id"]})))

    assert WS_SESSION_MANAGER.replay_messages(session_id) == []


def test_sockjs_stomp_subscription_filters_destination_and_unsubscribes() -> None:
    session_id = f"ws-topic-{uuid4().hex}"
    user_msg = WS_SESSION_MANAGER.queue_message(session_id, "notification", {"type": "notification", "message": "user-only"})
    topic_msg = WS_SESSION_MANAGER.queue_message(session_id, "swarm_state_update", {"type": "swarm_state_update", "message": "topic-only"}, destination="/topic/swarm")

    with client.websocket_connect("/ws/websocket") as ws:
        assert ws.receive_text() == "o"
        ws.send_text(sockjs_payload(stomp("CONNECT", {"accept-version": "1.2", "X-Session-Id": session_id})))
        connected_command, _, _ = parse_stomp_frame(ws.receive_text())
        assert connected_command == "CONNECTED"

        ws.send_text(sockjs_payload(stomp("SUBSCRIBE", {"id": "sub-topic", "destination": "/topic/swarm", "ack": "client-individual"})))
        message_command, headers, body = parse_stomp_frame(ws.receive_text())
        assert message_command == "MESSAGE"
        assert headers["message-id"] == topic_msg["id"]
        assert headers["destination"] == "/topic/swarm"
        assert json.loads(body)["message"] == "topic-only"

        ws.send_text(sockjs_payload(stomp("UNSUBSCRIBE", {"id": "sub-topic"})))

    assert WS_SESSION_MANAGER.list_subscriptions(session_id) == []
    replay = WS_SESSION_MANAGER.replay_messages(session_id)
    assert [item["id"] for item in replay] == [user_msg["id"], topic_msg["id"]]


def test_sockjs_stomp_subscription_resume_uses_last_message_id() -> None:
    session_id = f"ws-resume-{uuid4().hex}"
    first_topic = WS_SESSION_MANAGER.queue_message(session_id, "swarm_state_update", {"type": "swarm_state_update", "message": "first"}, destination="/topic/swarm")

    with client.websocket_connect("/ws/websocket") as ws:
        assert ws.receive_text() == "o"
        ws.send_text(sockjs_payload(stomp("CONNECT", {"accept-version": "1.2", "X-Session-Id": session_id})))
        connected_command, _, _ = parse_stomp_frame(ws.receive_text())
        assert connected_command == "CONNECTED"
        ws.send_text(sockjs_payload(stomp("SUBSCRIBE", {"id": "sub-topic", "destination": "/topic/swarm", "ack": "client-individual"})))
        message_command, headers, _ = parse_stomp_frame(ws.receive_text())
        assert message_command == "MESSAGE"
        assert headers["message-id"] == first_topic["id"]

    second_topic = WS_SESSION_MANAGER.queue_message(session_id, "swarm_state_update", {"type": "swarm_state_update", "message": "second"}, destination="/topic/swarm")
    WS_SESSION_MANAGER.queue_message(session_id, "notification", {"type": "notification", "message": "user-only"})

    with client.websocket_connect("/ws/websocket") as ws:
        assert ws.receive_text() == "o"
        ws.send_text(sockjs_payload(stomp("CONNECT", {"accept-version": "1.2", "X-Session-Id": session_id})))
        connected_command, _, _ = parse_stomp_frame(ws.receive_text())
        assert connected_command == "CONNECTED"
        ws.send_text(
            sockjs_payload(
                stomp(
                    "SUBSCRIBE",
                    {
                        "id": "sub-topic",
                        "destination": "/topic/swarm",
                        "ack": "client-individual",
                        "last-message-id": first_topic["id"],
                    },
                )
            )
        )
        message_command, headers, body = parse_stomp_frame(ws.receive_text())
        assert message_command == "MESSAGE"
        assert headers["message-id"] == second_topic["id"]
        assert headers["destination"] == "/topic/swarm"
        assert json.loads(body)["message"] == "second"


def test_ws_queue_stats_endpoint_reports_backpressure() -> None:
    session_id = f"ws-stats-{uuid4().hex}"
    for index in range(MAX_PENDING_MESSAGES_PER_SESSION + 3):
        client.post(f"/api/ws/sessions/{session_id}/push", json={"type": "notification", "payload": {"index": index}})

    response = client.get(f"/api/ws/sessions/{session_id}/messages/stats")

    assert response.status_code == 200
    payload = response.json()
    assert payload["sessionId"] == session_id
    assert payload["pendingCount"] == MAX_PENDING_MESSAGES_PER_SESSION
    assert payload["droppedCount"] == 3
    assert payload["lastSequence"] == MAX_PENDING_MESSAGES_PER_SESSION + 3
    assert payload["backpressureActive"] is True
