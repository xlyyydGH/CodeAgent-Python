from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


@dataclass(slots=True)
class Usage:
    inputTokens: int = 0
    outputTokens: int = 0
    cacheReadInputTokens: int = 0
    cacheCreationInputTokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "inputTokens": self.inputTokens,
            "outputTokens": self.outputTokens,
            "cacheReadInputTokens": self.cacheReadInputTokens,
            "cacheCreationInputTokens": self.cacheCreationInputTokens,
        }


@dataclass(slots=True)
class ContentBlock:
    type: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {"type": self.type}
        if self.text:
            payload["text"] = self.text
        payload.update(self.data)
        return payload


@dataclass(slots=True)
class Message:
    type: Literal["user", "assistant", "system"]
    content: list[ContentBlock] | str
    uuid: str = field(default_factory=lambda: new_id("msg"))
    timestamp: int = field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000))
    stopReason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    metadata: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(block.text for block in self.content if block.text)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "uuid": self.uuid,
            "timestamp": self.timestamp,
            "content": [block.to_dict() for block in self.content] if isinstance(self.content, list) else self.content,
        }
        if self.type == "assistant":
            payload["stopReason"] = self.stopReason
            payload["usage"] = self.usage.to_dict()
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass(slots=True)
class Session:
    id: str
    model: str
    workingDirectory: str = "."
    title: str | None = None
    messages: list[Message] = field(default_factory=list)
    costUsd: float = 0.0
    status: str = "idle"
    createdAt: str = field(default_factory=utc_now)
    updatedAt: str = field(default_factory=utc_now)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "model": self.model,
            "workingDirectory": self.workingDirectory,
            "messageCount": len(self.messages),
            "costUsd": self.costUsd,
            "createdAt": self.createdAt,
            "updatedAt": self.updatedAt,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "messages": [message.to_dict() for message in self.messages],
            "status": self.status,
        }
