from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


MAX_ARG_LENGTH = 2000
MAX_FORK_NESTING_DEPTH = 3
DANGEROUS_ARG_PATTERNS = [
    re.compile(r"\$\(.*\)"),
    re.compile(r"`.*`"),
    re.compile(r";\s*\w+"),
    re.compile(r"\|\s*\w+"),
    re.compile(r"&&\s*rm\s"),
    re.compile(r"\$\{.*\}"),
]


@dataclass(slots=True)
class ValidationResult:
    allowed: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SkillToolValidator:
    def validate_tool(self, skill: dict[str, Any], tool_name: str) -> ValidationResult:
        allowed_tools = skill.get("allowedTools") or skill.get("allowed-tools") or []
        if not allowed_tools:
            return ValidationResult(True)
        if any(str(item).lower() == tool_name.lower() for item in allowed_tools):
            return ValidationResult(True)
        skill_name = str(skill.get("name") or skill.get("effectiveName") or "skill")
        return ValidationResult(False, f"Tool '{tool_name}' is not in allowed-tools list for skill '{skill_name}'. Allowed: {allowed_tools}")

    def validate_args(self, skill_name: str, args: dict[str, Any] | None) -> ValidationResult:
        if not args:
            return ValidationResult(True)
        for key, raw_value in args.items():
            if raw_value is None:
                continue
            value = str(raw_value)
            if len(value) > MAX_ARG_LENGTH:
                return ValidationResult(False, f"Argument '{key}' exceeds max length {MAX_ARG_LENGTH} for skill '{skill_name}'")
            for pattern in DANGEROUS_ARG_PATTERNS:
                if pattern.search(value):
                    return ValidationResult(False, f"Potentially dangerous content detected in argument '{key}' for skill '{skill_name}'")
        return ValidationResult(True)

    def validate_fork_permission(self, skill: dict[str, Any], nesting_depth: int = 0) -> ValidationResult:
        is_fork = bool(skill.get("isFork") or skill.get("fork") or str(skill.get("context") or "").lower() == "fork")
        if not is_fork:
            return ValidationResult(True)
        if nesting_depth >= MAX_FORK_NESTING_DEPTH:
            return ValidationResult(False, f"Fork rejected: nesting depth {nesting_depth} exceeds maximum {MAX_FORK_NESTING_DEPTH}")
        return ValidationResult(True)

    def validate(self, skill: dict[str, Any], tool_name: str, args: dict[str, Any] | None = None, nesting_depth: int = 0) -> ValidationResult:
        for result in (
            self.validate_tool(skill, tool_name),
            self.validate_args(str(skill.get("name") or "skill"), args),
            self.validate_fork_permission(skill, nesting_depth),
        ):
            if not result.allowed:
                return result
        return ValidationResult(True)
