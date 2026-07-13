from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class PermissionDecision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(slots=True)
class PermissionRule:
    tool: str
    decision: PermissionDecision
    content: str | None = None
    scope: str = "session"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PermissionRule":
        return cls(
            tool=str(data.get("tool") or data.get("toolName") or "*"),
            decision=PermissionDecision(str(data.get("decision") or data.get("behavior") or "ask").lower()),
            content=data.get("content") or data.get("ruleContent"),
            scope=str(data.get("scope") or "session"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "decision": self.decision.value, "content": self.content, "scope": self.scope}


def _contains_unescaped_wildcard(value: str) -> bool:
    return any(char == "*" and (index == 0 or value[index - 1] != "\\") for index, char in enumerate(value))


def _wildcard_match(pattern: str, value: str) -> bool:
    marker = "\u0001"
    processed = pattern.replace("\\*", marker)
    processed = re.escape(processed).replace("\\*", ".*").replace(marker, "\\*")
    if pattern.count("*") == 1 and processed.endswith("\\ .*"):
        processed = processed[:-5] + r"( .*)?"
    try:
        return re.fullmatch(processed, value, flags=re.DOTALL) is not None
    except re.error:
        return False


def match_content(rule_content: str, command: str) -> bool:
    if rule_content.endswith(":*"):
        prefix = rule_content[:-2]
        return command == prefix or command.startswith(prefix + " ")
    if " " in rule_content:
        if rule_content.endswith(" *"):
            prefix = rule_content[:-2]
            return command == prefix or command.startswith(prefix + " ")
        if command == rule_content or command.startswith(rule_content + " "):
            return True
    if _contains_unescaped_wildcard(rule_content):
        return _wildcard_match(rule_content, command)
    return command == rule_content


class PermissionPolicy:
    def __init__(self, rules: list[PermissionRule] | None = None) -> None:
        self.rules = rules or []

    @classmethod
    def from_state(cls, raw_rules: list[dict[str, Any]] | None) -> "PermissionPolicy":
        parsed = []
        for item in raw_rules or []:
            try:
                parsed.append(PermissionRule.from_dict(item))
            except Exception:
                continue
        return cls(parsed)

    def decide(self, tool_name: str, tool_input: dict[str, Any] | None = None, risk: str = "low") -> PermissionDecision:
        command = str((tool_input or {}).get("command") or "")
        for decision in (PermissionDecision.DENY, PermissionDecision.ASK, PermissionDecision.ALLOW):
            for rule in self.rules:
                if rule.decision != decision:
                    continue
                if rule.tool not in {tool_name, "*"} and not (tool_name.startswith("mcp__") and tool_name.startswith(rule.tool + "__")):
                    continue
                if rule.content and not match_content(rule.content, command):
                    continue
                return decision
        return PermissionDecision.ASK if risk == "high" else PermissionDecision.ALLOW
