from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


STALE_ENTRY_TTL_SECONDS = 10 * 60
MAX_PENDING_MESSAGES_PER_SESSION = 200


@dataclass(slots=True)
class QueuedMessage:
    type: str
    payload: dict[str, Any]
    destination: str = "/user/queue/messages"
    sequence: int = 0
    id: str = field(default_factory=lambda: f"wsmsg-{uuid.uuid4().hex[:12]}")
    createdAt: float = field(default_factory=time.time)
    delivered: bool = False
    deliveredAt: float | None = None
    deliveryCount: int = 0
    subscriptionId: str | None = None
    acked: bool = False
    ackedAt: float | None = None
    nacked: bool = False
    nackedAt: float | None = None
    nackReason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WebSocketSessionManager:
    def __init__(self, ttl_seconds: int = STALE_ENTRY_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self.principal_to_session: dict[str, str] = {}
        self.session_to_principal: dict[str, str] = {}
        self.transport_to_principal: dict[str, str] = {}
        self.session_last_active: dict[str, float] = {}
        self.transport_last_active: dict[str, float] = {}
        self.pending_messages: dict[str, list[QueuedMessage]] = {}
        self.message_journal: dict[str, list[QueuedMessage]] = {}
        self.subscriptions: dict[str, dict[str, dict[str, Any]]] = {}
        self.session_sequences: dict[str, int] = {}
        self.session_dropped_counts: dict[str, int] = {}

    def connect(self, principal_name: str, transport_id: str, session_id: str | None = None) -> None:
        now = time.time()
        self.transport_to_principal[transport_id] = principal_name
        self.transport_last_active[transport_id] = now
        if session_id:
            self.bind_session(principal_name, session_id)

    def disconnect(self, transport_id: str) -> str | None:
        principal = self.transport_to_principal.pop(transport_id, None)
        self.transport_last_active.pop(transport_id, None)
        return principal

    def bind_session(self, principal_name: str, session_id: str) -> None:
        old_principal = self.session_to_principal.get(session_id)
        if old_principal and old_principal != principal_name:
            self.principal_to_session.pop(old_principal, None)
        self.principal_to_session[principal_name] = session_id
        self.session_to_principal[session_id] = principal_name
        self.session_last_active[session_id] = time.time()

    def refresh_activity(self, session_id: str) -> None:
        if session_id in self.session_to_principal:
            self.session_last_active[session_id] = time.time()

    def get_session_for_principal(self, principal_name: str) -> str | None:
        return self.principal_to_session.get(principal_name)

    def get_principal_for_session(self, session_id: str) -> str | None:
        return self.session_to_principal.get(session_id)

    def is_session_online(self, session_id: str) -> bool:
        return session_id in self.session_to_principal

    def get_active_session_ids(self) -> set[str]:
        return set(self.session_to_principal)

    def register_subscription(self, session_id: str, subscription_id: str, destination: str, ack_mode: str = "auto") -> dict[str, Any]:
        normalized_destination = self.normalize_destination(destination)
        subscription = {
            "sessionId": session_id,
            "subscriptionId": subscription_id,
            "destination": normalized_destination,
            "ackMode": (ack_mode or "auto").lower(),
            "createdAt": time.time(),
            "active": True,
        }
        self.subscriptions.setdefault(session_id, {})[subscription_id] = subscription
        self.refresh_activity(session_id)
        return dict(subscription)

    def unsubscribe(self, session_id: str, subscription_id: str) -> bool:
        subscriptions = self.subscriptions.get(session_id, {})
        if subscription_id not in subscriptions:
            return False
        subscriptions.pop(subscription_id, None)
        if not subscriptions:
            self.subscriptions.pop(session_id, None)
        return True

    def list_subscriptions(self, session_id: str) -> list[dict[str, Any]]:
        return [dict(item) for item in self.subscriptions.get(session_id, {}).values()]

    def normalize_destination(self, destination: str | None) -> str:
        text = str(destination or "/user/queue/messages").strip()
        if not text:
            return "/user/queue/messages"
        if text.startswith("/user/"):
            return "/user" + text[len("/user") :]
        return text

    def queue_message(self, session_id: str, message_type: str, payload: dict[str, Any], destination: str | None = None) -> dict[str, Any]:
        sequence = self.next_sequence(session_id)
        queued = QueuedMessage(message_type, payload, destination=self.normalize_destination(destination), sequence=sequence)
        queue = self.pending_messages.setdefault(session_id, [])
        queue.append(queued)
        overflow = max(0, len(queue) - MAX_PENDING_MESSAGES_PER_SESSION)
        if overflow:
            self.session_dropped_counts[session_id] = self.session_dropped_counts.get(session_id, 0) + overflow
            del queue[:overflow]
        journal = self.message_journal.setdefault(session_id, [])
        journal.append(queued)
        del journal[: -MAX_PENDING_MESSAGES_PER_SESSION * 5]
        return queued.to_dict()

    def next_sequence(self, session_id: str) -> int:
        current = self.session_sequences.get(session_id, 0) + 1
        self.session_sequences[session_id] = current
        return current

    def queue_stats(self, session_id: str) -> dict[str, Any]:
        pending = self.pending_messages.get(session_id, [])
        first_sequence = pending[0].sequence if pending else None
        last_pending_sequence = pending[-1].sequence if pending else None
        return {
            "sessionId": session_id,
            "pendingCount": len(pending),
            "journalCount": len(self.message_journal.get(session_id, [])),
            "droppedCount": self.session_dropped_counts.get(session_id, 0),
            "firstPendingSequence": first_sequence,
            "lastPendingSequence": last_pending_sequence,
            "lastSequence": self.session_sequences.get(session_id, 0),
            "maxPendingMessages": MAX_PENDING_MESSAGES_PER_SESSION,
            "backpressureActive": bool(pending and len(pending) >= MAX_PENDING_MESSAGES_PER_SESSION),
        }

    def publish_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        destination = str(payload.get("destination") or "/user/queue/messages") if isinstance(payload, dict) else "/user/queue/messages"
        return self.queue_message(session_id, event_type, {"type": event_type, **payload}, destination=destination)

    def broadcast_event(self, event_type: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [self.publish_event(session_id, event_type, payload) for session_id in sorted(self.get_active_session_ids())]

    def peek_messages(self, session_id: str) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.pending_messages.get(session_id, [])]

    def drain_messages(self, session_id: str) -> list[dict[str, Any]]:
        messages = [message.to_dict() for message in self.pending_messages.pop(session_id, [])]
        return messages

    def deliver_messages(self, session_id: str, subscription_id: str, ack_mode: str = "auto") -> list[dict[str, Any]]:
        subscription = self.subscriptions.get(session_id, {}).get(subscription_id)
        if self.subscriptions.get(session_id) and not subscription:
            return []
        destination = str(subscription.get("destination")) if subscription else None
        effective_ack_mode = str(subscription.get("ackMode") or ack_mode) if subscription else ack_mode
        normalized_ack = (ack_mode or "auto").lower()
        normalized_ack = (effective_ack_mode or "auto").lower()
        if normalized_ack == "auto":
            delivered: list[dict[str, Any]] = []
            retained: list[QueuedMessage] = []
            for message in self.pending_messages.get(session_id, []):
                if destination and not self.destination_matches(destination, message.destination):
                    retained.append(message)
                    continue
                delivered.append(message.to_dict())
            self.pending_messages[session_id] = retained
            return delivered
        now = time.time()
        delivered: list[dict[str, Any]] = []
        for message in self.pending_messages.get(session_id, []):
            if message.acked:
                continue
            if destination and not self.destination_matches(destination, message.destination):
                continue
            message.delivered = True
            message.deliveredAt = now
            message.deliveryCount += 1
            message.subscriptionId = subscription_id
            delivered.append(message.to_dict())
        return delivered

    def resume_subscription(
        self,
        session_id: str,
        subscription_id: str,
        destination: str,
        ack_mode: str = "auto",
        since_id: str | None = None,
    ) -> dict[str, Any]:
        subscription = self.register_subscription(session_id, subscription_id, destination, ack_mode=ack_mode)
        target_destination = str(subscription["destination"])
        candidates = self._pending_messages_after(session_id, since_id)
        matching = [message for message in candidates if not message.acked and self.destination_matches(target_destination, message.destination)]
        normalized_ack = str(subscription["ackMode"] or "auto").lower()
        if normalized_ack == "auto":
            retained = [message for message in self.pending_messages.get(session_id, []) if message not in matching]
            self.pending_messages[session_id] = retained
            messages = [message.to_dict() for message in matching]
        else:
            now = time.time()
            messages = []
            for message in matching:
                message.delivered = True
                message.deliveredAt = now
                message.deliveryCount += 1
                message.subscriptionId = subscription_id
                messages.append(message.to_dict())
        return {
            "sessionId": session_id,
            "subscription": subscription,
            "sinceId": since_id,
            "messages": messages,
            "replayedCount": len(messages),
        }

    def ack_messages(self, session_id: str, message_ids: list[str]) -> list[dict[str, Any]]:
        wanted = {str(message_id) for message_id in message_ids if str(message_id)}
        if not wanted:
            return []
        now = time.time()
        acked: list[dict[str, Any]] = []
        retained: list[QueuedMessage] = []
        for message in self.pending_messages.get(session_id, []):
            if message.id in wanted:
                message.acked = True
                message.ackedAt = now
                message.nacked = False
                message.nackReason = None
                acked.append(message.to_dict())
            else:
                retained.append(message)
        self.pending_messages[session_id] = retained
        return acked

    def nack_messages(self, session_id: str, message_ids: list[str], reason: str | None = None) -> list[dict[str, Any]]:
        wanted = {str(message_id) for message_id in message_ids if str(message_id)}
        if not wanted:
            return []
        now = time.time()
        nacked: list[dict[str, Any]] = []
        for message in self.pending_messages.get(session_id, []):
            if message.id not in wanted or message.acked:
                continue
            message.nacked = True
            message.nackedAt = now
            message.nackReason = reason
            nacked.append(message.to_dict())
        return nacked

    def replay_messages(self, session_id: str, include_acked: bool = False, since_id: str | None = None, subscription_id: str | None = None, destination: str | None = None) -> list[dict[str, Any]]:
        messages = self.message_journal.get(session_id, [])
        if since_id:
            for index, message in enumerate(messages):
                if message.id == since_id:
                    messages = messages[index + 1 :]
                    break
        target_destination = self.normalize_destination(destination) if destination else None
        if not target_destination and subscription_id:
            subscription = self.subscriptions.get(session_id, {}).get(subscription_id)
            target_destination = str(subscription.get("destination")) if subscription else None
        if target_destination:
            messages = [message for message in messages if self.destination_matches(target_destination, message.destination)]
        if not include_acked:
            messages = [message for message in messages if not message.acked and message in self.pending_messages.get(session_id, [])]
        return [message.to_dict() for message in messages]

    def _pending_messages_after(self, session_id: str, since_id: str | None = None) -> list[QueuedMessage]:
        pending = self.pending_messages.get(session_id, [])
        if not since_id:
            return list(pending)
        journal = self.message_journal.get(session_id, [])
        found = False
        allowed_ids: set[str] = set()
        for message in journal:
            if found:
                allowed_ids.add(message.id)
            elif message.id == since_id:
                found = True
        if not found:
            return list(pending)
        return [message for message in pending if message.id in allowed_ids]

    def destination_matches(self, subscription_destination: str, message_destination: str | None) -> bool:
        subscription = self.normalize_destination(subscription_destination)
        message = self.normalize_destination(message_destination)
        return subscription == message

    def cleanup_stale_entries(self, now: float | None = None) -> int:
        current = now or time.time()
        cleaned = 0
        for session_id, last_active in list(self.session_last_active.items()):
            if current - last_active <= self.ttl_seconds:
                continue
            principal = self.session_to_principal.pop(session_id, None)
            if principal and self.principal_to_session.get(principal) == session_id:
                self.principal_to_session.pop(principal, None)
            self.session_last_active.pop(session_id, None)
            cleaned += 1
        for transport_id, last_active in list(self.transport_last_active.items()):
            if current - last_active > self.ttl_seconds:
                self.transport_to_principal.pop(transport_id, None)
                self.transport_last_active.pop(transport_id, None)
                cleaned += 1
        return cleaned
