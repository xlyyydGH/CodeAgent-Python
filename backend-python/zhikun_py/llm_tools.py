from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .tools import ToolRegistry


@dataclass(slots=True)
class ExecutedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    content: str
    is_error: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "content": self.content,
            "isError": self.is_error,
        }


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {"value": raw}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def normalize_tool_call(tool_call: dict[str, Any], index: int = 0) -> tuple[str, str, dict[str, Any]]:
    function = tool_call.get("function") or {}
    call_id = str(tool_call.get("id") or f"tool-call-{index}")
    name = str(function.get("name") or tool_call.get("name") or "")
    arguments = parse_tool_arguments(function.get("arguments") if function else tool_call.get("arguments"))
    return call_id, name, arguments


def assistant_tool_message(message: dict[str, Any]) -> dict[str, Any]:
    outgoing = {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls") or [],
    }
    return outgoing


def execute_tool_calls(registry: ToolRegistry, tool_calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[ExecutedToolCall]]:
    messages: list[dict[str, Any]] = []
    executed: list[ExecutedToolCall] = []
    for index, tool_call in enumerate(tool_calls):
        call_id, name, arguments = normalize_tool_call(tool_call, index)
        if not name:
            result = {"content": "Missing tool function name", "isError": True, "metadata": {}}
        else:
            result = registry.call(name, arguments).to_dict()
        content = json.dumps(result, ensure_ascii=False)
        messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": content})
        executed.append(
            ExecutedToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                content=str(result.get("content") or ""),
                is_error=bool(result.get("isError")),
            )
        )
    return messages, executed
