from __future__ import annotations

import fnmatch
import contextlib
import difflib
import io
import json
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .permissions import PermissionDecision, PermissionPolicy
from .cron_runtime import CronTaskService, CronValidationError
from .file_recovery import FileEditRecoveryPolicy, FileVersionTracker
from .lsp_runtime import call_hierarchy, document_symbols, go_to_definition, hover, references, workspace_symbols
from .memdir_runtime import MemdirService
from .security import classify_command, command_parts, command_risk, filter_sensitive_data, sensitive_path_level
from .verify_runtime import CapabilityGate, verifier_for


IGNORED_DIRS = {".git", "node_modules", "venv", ".venv", "dist", "target", "__pycache__", ".pytest_cache"}
DANGEROUS_COMMAND_TOKENS = {"rm", "del", "erase", "format", "shutdown", "reboot", "reg", "diskpart", "mkfs"}
REPL_MAX_SESSIONS = 3
REPL_MAX_OUTPUT_BYTES = 100 * 1024
TASK_MAX_OUTPUT_BYTES = 1024 * 1024
MAX_CONCURRENT_AGENTS = 30
MAX_CONCURRENT_AGENTS_PER_SESSION = 10
MAX_AGENT_NESTING_DEPTH = 3
MAX_AGENT_RESULT_SIZE_CHARS = 100_000
COMMAND_MAX_OUTPUT_BYTES = 100_000
AGENT_TYPES = ["explore", "verification", "plan", "general-purpose", "guide"]
AGENT_MODEL_ALIASES = ["light", "standard", "premium", "default", "qwen3.7-max", "qwen3.7-plus", "deepseek-v4-pro", "glm-5.1"]
AGENT_ISOLATION_MODES = ["none", "worktree"]
AGENT_MODEL_ALIAS_MAP = {"light": "qwen3.7-plus", "standard": "qwen3.7-plus", "premium": "qwen3.7-max"}


@dataclass(slots=True)
class ToolResult:
    content: str
    isError: bool = False
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "isError": self.isError, "metadata": self.metadata or {}}


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult]
    group: str = "general"
    read_only: bool = True
    enabled: bool = True
    concurrency_safe: bool = False

    def api_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "group": self.group,
            "readOnly": self.read_only,
            "enabled": self.enabled,
            "concurrencySafe": self.concurrency_safe,
        }


def is_safe_relative_path(root: Path, requested: str) -> Path:
    candidate = (root / requested).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError("Path escapes workspace")
    return candidate


def blocked_command_reason(parts: list[str]) -> str | None:
    check = command_risk(parts)
    if check.blocked or check.needs_confirmation:
        return check.reason or f"blocked command: {check.level}"
    return None


def blocked_command_check(parts: list[str]) -> Any | None:
    check = command_risk(parts)
    if check.blocked or check.needs_confirmation:
        return check
    return None


def blocked_command_metadata(parts: list[str], check: Any) -> dict[str, Any]:
    classification = classify_command(" ".join(parts))
    reason = check.reason or f"blocked command: {check.level}"
    return {
        "blocked": True,
        "blockLevel": check.level.value,
        "blockReason": reason,
        "classification": asdict(classification),
        "sandbox": {
            "command": parts,
            "decision": "blocked",
            "blockLevel": check.level.value,
            "blockReason": reason,
            "readOnly": bool(classification.isReadOnly),
            "category": classification.category,
            "exitCode": None,
            "errorType": "blocked",
            "timedOut": False,
            "processTreeManaged": True,
        },
    }


class ToolRegistry:
    def __init__(
        self,
        root: Path,
        policy: PermissionPolicy | None = None,
        cron_service: CronTaskService | None = None,
        file_tracker: FileVersionTracker | None = None,
        file_recovery: FileEditRecoveryPolicy | None = None,
        memdir_service: MemdirService | None = None,
    ) -> None:
        self.root = root
        self.policy = policy or PermissionPolicy()
        self.cron_service = cron_service
        self.file_tracker = file_tracker or FileVersionTracker()
        self.file_recovery = file_recovery or FileEditRecoveryPolicy()
        self.memdir_service = memdir_service
        self._tools: dict[str, Tool] = {}
        self._repl_sessions: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, dict[str, Any]] = {}
        self._agent_lock = threading.Lock()
        self._active_agents: set[str] = set()
        self._session_agents: dict[str, set[str]] = {}
        self._active_agent_counts: dict[str, int] = {}
        self._session_agent_counts: dict[str, dict[str, int]] = {}
        self._external_roots: set[Path] = set()
        self.team_dispatcher: Callable[[dict[str, Any]], ToolResult] | None = None
        self.agent_dispatcher: Callable[[dict[str, Any]], ToolResult] | None = None
        self.background_agent_event_publisher: Callable[[str, str, dict[str, Any]], Any] | None = None
        self.register_builtin_tools()

    def set_team_dispatcher(self, dispatcher: Callable[[dict[str, Any]], ToolResult] | None) -> None:
        self.team_dispatcher = dispatcher

    def set_agent_dispatcher(self, dispatcher: Callable[[dict[str, Any]], ToolResult] | None) -> None:
        self.agent_dispatcher = dispatcher

    def set_background_agent_event_publisher(self, publisher: Callable[[str, str, dict[str, Any]], Any] | None) -> None:
        self.background_agent_event_publisher = publisher

    def _publish_background_agent_event(self, agent_id: str, event_type: str, data: dict[str, Any]) -> None:
        if self.background_agent_event_publisher is None:
            return
        try:
            self.background_agent_event_publisher(agent_id, event_type, data)
        except Exception:
            pass

    def allow_external_root(self, path: str | Path) -> None:
        self._external_roots.add(Path(path).resolve())

    def revoke_external_root(self, path: str | Path) -> None:
        self._external_roots.discard(Path(path).resolve())

    def _resolve_path(self, requested: str) -> Path:
        raw = Path(requested)
        if raw.is_absolute():
            candidate = raw.resolve()
            for root in self._external_roots:
                if candidate == root or root in candidate.parents:
                    return candidate
        return is_safe_relative_path(self.root, requested)

    def _display_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            for root in self._external_roots:
                try:
                    return path.relative_to(root).as_posix()
                except ValueError:
                    continue
        return str(path)

    def _atomic_write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _snapshot_before_write(self, path: Path, before_text: str, operation: str, agent_id: str | None = None) -> dict[str, Any]:
        encoded = before_text.encode("utf-8")
        return {
            "id": f"snapshot-{uuid.uuid4().hex}",
            "path": self._display_path(path),
            "operation": operation,
            "agentId": agent_id or "tool",
            "createdAt": time.time(),
            "contentHash": self.file_tracker.compute_hash(before_text),
            "bytes": len(encoded),
            "content": before_text,
            "truncated": False,
        }

    def _unified_diff(self, path: Path, before_text: str, after_text: str) -> str:
        display_path = self._display_path(path)
        return "\n".join(
            difflib.unified_diff(
                before_text.splitlines(),
                after_text.splitlines(),
                fromfile=f"{display_path} (before)",
                tofile=f"{display_path} (after)",
                lineterm="",
            )
        )

    def _find_whitespace_fuzzy_match(self, text: str, old: str) -> tuple[int, int] | None:
        old_lines = old.splitlines()
        if not old_lines:
            return None

        def normalize(line: str) -> str:
            return " ".join(line.strip().split())

        wanted = [normalize(line) for line in old_lines]
        text_lines = text.splitlines(keepends=True)
        if len(wanted) > len(text_lines):
            return None

        offsets: list[int] = []
        cursor = 0
        for line in text_lines:
            offsets.append(cursor)
            cursor += len(line)

        width = len(wanted)
        for index in range(0, len(text_lines) - width + 1):
            candidate = [normalize(line) for line in text_lines[index : index + width]]
            if candidate == wanted:
                start = offsets[index]
                end = offsets[index + width] if index + width < len(offsets) else len(text)
                return start, end
        return None

    def _find_trailing_whitespace_fuzzy_match(self, text: str, old: str) -> tuple[int, int] | None:
        if "\r" in old:
            return None
        old_lines = old.splitlines()
        if not old_lines:
            return None

        def normalize(line: str) -> str:
            return line.rstrip(" \t\r\n")

        wanted = [normalize(line) for line in old_lines]
        text_lines = text.splitlines(keepends=True)
        if len(wanted) > len(text_lines):
            return None

        offsets: list[int] = []
        cursor = 0
        for line in text_lines:
            offsets.append(cursor)
            cursor += len(line)

        width = len(wanted)
        for index in range(0, len(text_lines) - width + 1):
            candidate = [normalize(line) for line in text_lines[index : index + width]]
            if candidate == wanted:
                start = offsets[index]
                end = offsets[index + width] if index + width < len(offsets) else len(text)
                return start, end
        return None

    def _normalize_quotes_with_spans(self, value: str) -> tuple[str, list[tuple[int, int]]]:
        quote_map = {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
        }
        normalized: list[str] = []
        spans: list[tuple[int, int]] = []
        for index, char in enumerate(value):
            normalized.append(quote_map.get(char, char))
            spans.append((index, index + 1))
        return "".join(normalized), spans

    def _normalize_newlines_with_spans(self, value: str) -> tuple[str, list[tuple[int, int]]]:
        normalized: list[str] = []
        spans: list[tuple[int, int]] = []
        index = 0
        while index < len(value):
            if value[index : index + 2] == "\r\n":
                normalized.append("\n")
                spans.append((index, index + 2))
                index += 2
                continue
            if value[index] == "\r":
                normalized.append("\n")
            else:
                normalized.append(value[index])
            spans.append((index, index + 1))
            index += 1
        return "".join(normalized), spans

    def _normalize_tab_space_with_spans(self, value: str) -> tuple[str, list[tuple[int, int]]]:
        normalized: list[str] = []
        spans: list[tuple[int, int]] = []
        for index, char in enumerate(value):
            if char == "\t":
                normalized.extend("    ")
                spans.extend([(index, index + 1)] * 4)
            else:
                normalized.append(char)
                spans.append((index, index + 1))
        return "".join(normalized), spans

    def _find_normalized_match(
        self,
        text: str,
        old: str,
        normalizer: Callable[[str], tuple[str, list[tuple[int, int]]]],
    ) -> tuple[int, int] | None:
        normalized_text, spans = normalizer(text)
        normalized_old, _ = normalizer(old)
        if not normalized_old:
            return None
        index = normalized_text.find(normalized_old)
        if index < 0:
            return None
        end_index = index + len(normalized_old) - 1
        return spans[index][0], spans[end_index][1]

    def _replace_span(self, text: str, span: tuple[int, int], new: str) -> str:
        start, end = span
        matched_text = text[start:end]
        replacement = new
        if matched_text.endswith("\n") and not replacement.endswith(("\n", "\r")):
            replacement += "\n"
        return text[:start] + replacement + text[end:]

    def _replace_once_with_strategy(self, text: str, old: str, new: str) -> tuple[str, str] | None:
        if old in text:
            return text.replace(old, new, 1), "exact"
        fuzzy_strategies: list[tuple[str, Callable[[], tuple[int, int] | None]]] = [
            ("fuzzy_quotes", lambda: self._find_normalized_match(text, old, self._normalize_quotes_with_spans)),
            ("fuzzy_trailing_whitespace", lambda: self._find_trailing_whitespace_fuzzy_match(text, old)),
            ("fuzzy_newline", lambda: self._find_normalized_match(text, old, self._normalize_newlines_with_spans)),
            ("fuzzy_tab_space", lambda: self._find_normalized_match(text, old, self._normalize_tab_space_with_spans)),
            ("fuzzy_whitespace", lambda: self._find_whitespace_fuzzy_match(text, old)),
        ]
        for strategy, finder in fuzzy_strategies:
            match = finder()
            if match is not None:
                return self._replace_span(text, match, new), strategy
        return None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_mcp_tools(self, mcp_manager: Any) -> int:
        self._tools = {name: tool for name, tool in self._tools.items() if not name.startswith("mcp__")}
        registered = 0
        for wrapped in mcp_manager.discover_wrapped_tools():
            server_name = str(wrapped.get("serverName") or "")
            original_name = str(wrapped.get("originalName") or "")
            if not server_name or not original_name:
                continue

            def handler(payload: dict[str, Any], server: str = server_name, tool_name: str = original_name) -> ToolResult:
                result = mcp_manager.call_tool(server, tool_name, payload or {})
                metadata = dict(result.get("metadata") or {})
                metadata.update(
                    {
                        "mcpServer": server,
                        "mcpTool": tool_name,
                        "connectionType": result.get("connectionType"),
                        "cached": result.get("cached", False),
                        "request": result.get("request"),
                    }
                )
                content = str(result.get("content") if result.get("content") is not None else json.dumps(result, ensure_ascii=False))
                return ToolResult(content, isError=result.get("status") == "error", metadata=metadata)

            self.register(
                Tool(
                    name=str(wrapped.get("name")),
                    description=str(wrapped.get("description") or f"MCP tool: {original_name}"),
                    input_schema=wrapped.get("inputSchema") if isinstance(wrapped.get("inputSchema"), dict) else {"type": "object"},
                    handler=handler,
                    group="mcp",
                    read_only=True,
                    enabled=True,
                    concurrency_safe=False,
                )
            )
            registered += 1
        return registered

    def list(self) -> list[dict[str, Any]]:
        return [tool.api_dict() for tool in sorted(self._tools.values(), key=lambda item: item.name)]

    def llm_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
            if tool.enabled
        ]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def call(self, name: str, payload: dict[str, Any] | None = None) -> ToolResult:
        tool = self.get(name)
        if not tool or not tool.enabled:
            return ToolResult(f"Unknown or disabled tool: {name}", isError=True)
        risk = "low" if tool.read_only or tool.name == "Agent" else "high"
        decision = self.policy.decide(tool.name, payload or {}, risk=risk)
        if decision == PermissionDecision.DENY:
            return ToolResult(f"Permission denied for tool: {name}", isError=True, metadata={"decision": "deny"})
        if decision == PermissionDecision.ASK and risk == "high":
            return ToolResult(f"Permission required for tool: {name}", isError=True, metadata={"decision": "ask"})
        try:
            return tool.handler(payload or {})
        except Exception as exc:
            return ToolResult(str(exc), isError=True)

    def register_builtin_tools(self) -> None:
        self.register(
            Tool(
                name="list_files",
                description="List files in the workspace.",
                group="read",
                input_schema={"type": "object", "properties": {"pattern": {"type": "string"}, "limit": {"type": "integer"}}},
                handler=self._list_files,
            )
        )
        self.register(
            Tool(
                name="read_file",
                description="Read a UTF-8 text file from the workspace.",
                group="read",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                handler=self._read_file,
            )
        )
        self.register(
            Tool(
                name="search_files",
                description="Search text files in the workspace for a literal query.",
                group="search",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]},
                handler=self._search_files,
            )
        )
        self.register(
            Tool(
                name="write_file",
                description="Write a UTF-8 text file inside the workspace.",
                group="write",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
                handler=self._write_file,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="edit_file",
                description="Replace text in a UTF-8 file inside the workspace.",
                group="write",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
                    "required": ["path", "old", "new"],
                },
                handler=self._edit_file,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="run_command",
                description="Run a bounded non-destructive command inside the workspace.",
                group="execute",
                input_schema={"type": "object", "properties": {"command": {"oneOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}, "timeoutMs": {"type": "integer"}}},
                handler=self._run_command,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="Bash",
                description="Run a bounded shell command inside the workspace.",
                group="bash",
                input_schema={"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]},
                handler=self._bash,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="PowerShell",
                description="Run a bounded PowerShell command inside the workspace.",
                group="bash",
                input_schema={"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]},
                handler=self._powershell,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="REPL",
                description="Execute Python code in an interactive REPL session.",
                group="bash",
                input_schema={
                    "type": "object",
                    "properties": {"language": {"type": "string"}, "code": {"type": "string"}, "sessionId": {"type": "string"}},
                    "required": ["language", "code"],
                },
                handler=self._repl,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="NotebookEdit",
                description="Edit Jupyter Notebook cells.",
                group="edit",
                input_schema={
                    "type": "object",
                    "properties": {
                        "notebook_path": {"type": "string"},
                        "command": {"type": "string"},
                        "cell_index": {"type": "integer"},
                        "content": {"type": "string"},
                        "cell_type": {"type": "string"},
                        "direction": {"type": "string"},
                    },
                    "required": ["notebook_path", "command"],
                },
                handler=self._notebook_edit,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="Agent",
                description=(
                    "Launch a sub-agent to work on a specific task independently. "
                    "The sub-agent has its own conversation with the LLM and can use tools. "
                    "Use this when a task can be broken down into independent subtasks."
                ),
                group="agent",
                input_schema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Complete task description for the sub-agent"},
                        "description": {"type": "string", "description": "Short 3-5 word description of the task"},
                        "subagent_type": {"type": "string", "enum": AGENT_TYPES, "description": "Type of sub-agent to use"},
                        "model": {"type": "string", "enum": AGENT_MODEL_ALIASES, "description": "Model override for the sub-agent. Use aliases (light/standard/premium) or actual model names."},
                        "run_in_background": {"type": "boolean", "description": "Run the agent in the background"},
                        "isolation": {"type": "string", "enum": AGENT_ISOLATION_MODES, "description": "Isolation mode for the sub-agent"},
                        "teamName": {"type": "string"},
                        "fork": {"type": "boolean"},
                        "sessionId": {"type": "string"},
                        "nestingDepth": {"type": "integer"},
                    },
                    "required": ["prompt"],
                },
                handler=self._agent,
                read_only=False,
                concurrency_safe=True,
            )
        )
        self.register(
            Tool(
                name="TaskCreate",
                description="Create a tracked task.",
                group="task",
                input_schema={
                    "type": "object",
                    "properties": {"description": {"type": "string"}, "prompt": {"type": "string"}, "taskType": {"type": "string"}},
                    "required": ["description"],
                },
                handler=self._task_create,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="TaskList",
                description="List tracked tasks.",
                group="task",
                input_schema={"type": "object", "properties": {"status": {"type": "string"}, "sessionId": {"type": "string"}}},
                handler=self._task_list,
            )
        )
        self.register(
            Tool(
                name="TaskGet",
                description="Get a tracked task.",
                group="task",
                input_schema={"type": "object", "properties": {"taskId": {"type": "string"}}, "required": ["taskId"]},
                handler=self._task_get,
            )
        )
        self.register(
            Tool(
                name="TaskUpdate",
                description="Update a tracked task.",
                group="task",
                input_schema={"type": "object", "properties": {"taskId": {"type": "string"}, "status": {"type": "string"}, "output": {"type": "string"}, "error": {"type": "string"}}, "required": ["taskId"]},
                handler=self._task_update,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="TaskStop",
                description="Stop a tracked task.",
                group="task",
                input_schema={"type": "object", "properties": {"taskId": {"type": "string"}, "reason": {"type": "string"}}, "required": ["taskId"]},
                handler=self._task_stop,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="TaskOutput",
                description="Report output for a tracked task.",
                group="task",
                input_schema={"type": "object", "properties": {"taskId": {"type": "string"}, "output": {"type": "string"}, "isError": {"type": "boolean"}}, "required": ["output"]},
                handler=self._task_output,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="CronCreate",
                description="Create a scheduled cron task using a 5-field Unix cron expression.",
                group="cron",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cron": {"type": "string"},
                        "prompt": {"type": "string"},
                        "recurring": {"type": "boolean"},
                        "durable": {"type": "boolean"},
                        "agentId": {"type": "string"},
                    },
                    "required": ["cron", "prompt"],
                },
                handler=self._cron_create,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="CronList",
                description="List scheduled cron tasks.",
                group="cron",
                input_schema={"type": "object", "properties": {}},
                handler=self._cron_list,
            )
        )
        self.register(
            Tool(
                name="CronDelete",
                description="Delete a scheduled cron task by id.",
                group="cron",
                input_schema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
                handler=self._cron_delete,
                read_only=False,
            )
        )
        self.register(
            Tool(
                name="VerifyJourney",
                description="Verify a browser or HTTP journey and return step evidence.",
                group="verify",
                input_schema={
                    "type": "object",
                    "properties": {
                        "journey": {"type": "array", "items": {"type": "object"}},
                        "steps": {"type": "array", "items": {"type": "object"}},
                        "baseUrl": {"type": "string"},
                        "sessionId": {"type": "string"},
                    },
                    "required": ["journey"],
                },
                handler=self._verify_journey,
            )
        )
        self.register(
            Tool(
                name="LSP",
                description="Run local language-intelligence operations such as definitions, references, hover, and symbols.",
                group="lsp",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "filePath": {"type": "string"},
                        "path": {"type": "string"},
                        "line": {"type": "integer"},
                        "character": {"type": "integer"},
                        "query": {"type": "string"},
                        "symbol": {"type": "string"},
                    },
                    "required": ["operation"],
                },
                handler=self._lsp,
            )
        )
        self.register(
            Tool(
                name="Memory",
                description="Read, search, write, or delete persistent memories across sessions.",
                group="memory",
                input_schema={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["read", "search", "write", "delete"]},
                        "content": {"type": "string"},
                        "query": {"type": "string"},
                        "title": {"type": "string"},
                        "category": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["action"],
                },
                handler=self._memory,
                read_only=False,
            )
        )
        self.register_compatibility_tools()

    def register_compatibility_tools(self) -> None:
        tool_specs = [
            ("FileRead", "Read a file using source-compatible argument names.", "read", self._compat_file_read, True),
            ("Read", "Alias for FileRead.", "read", self._compat_file_read, True),
            ("View", "Alias for FileRead.", "read", self._compat_file_read, True),
            ("ListDir", "List one directory in the workspace.", "read", self._list_dir, True),
            ("Glob", "Find files by glob pattern.", "search", self._glob, True),
            ("GrepSearch", "Search files using a text or regex pattern.", "search", self._grep_search, True),
            ("FileWrite", "Write a file using source-compatible argument names.", "write", self._compat_file_write, False),
            ("FileEdit", "Edit a file using source-compatible argument names.", "write", self._compat_file_edit, False),
            ("MultiEdit", "Apply multiple literal replacements to one file.", "write", self._multi_edit, False),
            ("MakeDir", "Create a directory inside the workspace.", "write", self._make_dir, False),
            ("DeleteFile", "Delete a file inside the workspace.", "write", self._delete_file, False),
            ("GitStatus", "Run git status --short.", "git", self._git_status, True),
            ("GitLog", "Run git log --oneline.", "git", self._git_log, True),
            ("GitDiff", "Run git diff.", "git", self._git_diff, True),
            ("GitShow", "Run git show for a revision.", "git", self._git_show, True),
            ("GitBlame", "Run git blame for a file.", "git", self._git_blame, True),
            ("GitBranch", "Show the current git branch.", "git", self._git_branch, True),
            ("GitLsFiles", "List git-tracked files.", "git", self._git_ls_files, True),
            ("LspDefinition", "Find symbol definition.", "lsp", self._lsp_definition, True),
            ("LspReferences", "Find symbol references.", "lsp", self._lsp_references, True),
            ("LspHover", "Return hover information.", "lsp", self._lsp_hover, True),
            ("LspDocumentSymbols", "List document symbols.", "lsp", self._lsp_document_symbols, True),
            ("LspWorkspaceSymbols", "Search workspace symbols.", "lsp", self._lsp_workspace_symbols, True),
            ("MemoryRead", "Read persistent memories.", "memory", self._memory_read, True),
            ("MemorySearch", "Search persistent memories.", "memory", self._memory_search, True),
            ("MemoryWrite", "Write a persistent memory.", "memory", self._memory_write, False),
            ("HttpApiVerifier", "Verify an HTTP API journey.", "verify", self._http_api_verifier, True),
            ("BrowserVerifier", "Verify a browser-style journey.", "verify", self._browser_verifier, True),
            ("TokenCount", "Estimate token count for text.", "context", self._token_count, True),
            ("ContextStatus", "Report workspace and tool context status.", "context", self._context_status, True),
            ("ToolSearch", "Search registered tools.", "tooling", self._tool_search, True),
            ("CommandClassify", "Classify command risk and read-only category.", "security", self._command_classify, True),
            ("Sleep", "Sleep for a bounded number of milliseconds.", "utility", self._sleep, True),
            ("DateTime", "Return current local and epoch time.", "utility", self._date_time, True),
            ("JsonParse", "Parse JSON and report structure.", "utility", self._json_parse, True),
            ("RegexExtract", "Extract regex matches from text.", "utility", self._regex_extract, True),
            ("TodoWrite", "Create or update a lightweight tracked todo task.", "task", self._todo_write, False),
            ("AskUserQuestion", "Record a user question request for the session.", "task", self._ask_user_question, False),
            ("SyntheticOutput", "Emit bounded synthetic output for tests and dry runs.", "utility", self._synthetic_output, True),
        ]
        schema = {"type": "object", "properties": {}}
        for name, description, group, handler, read_only in tool_specs:
            self.register(Tool(name=name, description=description, group=group, input_schema=schema, handler=handler, read_only=read_only))

    def _compat_path_payload(self, payload: dict[str, Any]) -> str:
        return str(payload.get("path") or payload.get("file_path") or payload.get("filePath") or payload.get("notebook_path") or "")

    def _compat_file_read(self, payload: dict[str, Any]) -> ToolResult:
        return self._read_file({"path": self._compat_path_payload(payload)})

    def _compat_file_write(self, payload: dict[str, Any]) -> ToolResult:
        return self._write_file({"path": self._compat_path_payload(payload), "content": payload.get("content") or payload.get("text") or "", **payload})

    def _compat_file_edit(self, payload: dict[str, Any]) -> ToolResult:
        old = payload.get("old") or payload.get("old_string") or payload.get("oldString") or ""
        new = payload.get("new") or payload.get("new_string") or payload.get("newString") or ""
        return self._edit_file({"path": self._compat_path_payload(payload), "old": old, "new": new, **payload})

    def _list_dir(self, payload: dict[str, Any]) -> ToolResult:
        requested = self._compat_path_payload(payload) or "."
        path = self._resolve_path(requested)
        if not path.exists() or not path.is_dir():
            return ToolResult(f"Directory not found: {requested}", isError=True)
        limit = int(payload.get("limit") or 200)
        entries = []
        for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{child.name}{suffix}")
            if len(entries) >= limit:
                break
        return ToolResult("\n".join(entries), metadata={"path": self._display_path(path), "count": len(entries)})

    def _glob(self, payload: dict[str, Any]) -> ToolResult:
        pattern = str(payload.get("pattern") or payload.get("glob") or "*")
        return self._list_files({"pattern": pattern, "limit": payload.get("limit") or 200})

    def _grep_search(self, payload: dict[str, Any]) -> ToolResult:
        pattern = str(payload.get("pattern") or payload.get("query") or "")
        if not pattern:
            return ToolResult("Missing pattern", isError=True)
        limit = int(payload.get("limit") or 100)
        regex = bool(payload.get("regex"))
        compiled = re.compile(pattern) if regex else None
        matches: list[str] = []
        for path in self._iter_files():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                found = bool(compiled.search(line)) if compiled else pattern in line
                if found:
                    matches.append(f"{path.relative_to(self.root).as_posix()}:{line_no}: {line.strip()}")
                    break
            if len(matches) >= limit:
                break
        return ToolResult("\n".join(matches), metadata={"count": len(matches), "regex": regex})

    def _multi_edit(self, payload: dict[str, Any]) -> ToolResult:
        requested = self._compat_path_payload(payload)
        edits = payload.get("edits")
        if not isinstance(edits, list):
            return ToolResult("MultiEdit requires edits list", isError=True)
        path = self._resolve_path(requested)
        level = sensitive_path_level(requested)
        if level in {"forbidden", "protected"}:
            return ToolResult(f"Sensitive path blocked: {requested}", isError=True, metadata={"pathLevel": level})
        if not path.exists():
            return ToolResult(f"File not found: {requested}", isError=True)
        text = path.read_text(encoding="utf-8", errors="replace")
        agent_id = str(payload.get("agentId") or "tool")
        snapshot = self._snapshot_before_write(path, text, "multi_edit", agent_id)
        updated = text
        match_strategies: list[str] = []
        normalized_edits: list[dict[str, str]] = []
        for index, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return ToolResult(
                    "MultiEdit edit entry must be an object",
                    isError=True,
                    metadata={"failedEditIndex": index, "applied": 0, "atomic": True},
                )
            old = str(edit.get("old") or edit.get("old_string") or "")
            new = str(edit.get("new") or edit.get("new_string") or "")
            if not old:
                return ToolResult(
                    "MultiEdit old string is required",
                    isError=True,
                    metadata={"failedEditIndex": index, "applied": 0, "atomic": True},
                )
            replacement = self._replace_once_with_strategy(updated, old, new)
            if replacement is None:
                decision = self.file_recovery.recover("edit_file", "old_string not found in file")
                return ToolResult(
                    decision.message,
                    isError=True,
                    metadata={
                        "failedEditIndex": index,
                        "applied": 0,
                        "atomic": True,
                        "recovery": decision.to_dict(),
                        "matchStrategies": match_strategies,
                    },
                )
            updated, strategy = replacement
            match_strategies.append(strategy)
            normalized_edits.append({"old": old, "new": new})
        try:
            self._atomic_write_text(path, updated)
        except OSError as exc:
            decision = self.file_recovery.recover("edit_file", str(exc))
            return ToolResult(
                decision.message,
                isError=True,
                metadata={"recovery": decision.to_dict(), "applied": 0, "atomic": True, "path": self._display_path(path)},
            )
        content_hash = self.file_tracker.record_write(path, agent_id=agent_id)
        return ToolResult(
            f"Applied {len(normalized_edits)} edits to {requested}",
            metadata={
                "applied": len(normalized_edits),
                "contentHash": content_hash,
                "beforeHash": snapshot["contentHash"],
                "afterHash": content_hash,
                "snapshotBeforeWrite": snapshot,
                "diff": self._unified_diff(path, text, updated),
                "atomic": True,
                "allOrNothing": True,
                "matchStrategies": match_strategies,
            },
        )

    def _make_dir(self, payload: dict[str, Any]) -> ToolResult:
        requested = self._compat_path_payload(payload)
        path = self._resolve_path(requested)
        path.mkdir(parents=True, exist_ok=True)
        return ToolResult(f"Created directory {requested}", metadata={"path": self._display_path(path)})

    def _delete_file(self, payload: dict[str, Any]) -> ToolResult:
        requested = self._compat_path_payload(payload)
        path = self._resolve_path(requested)
        level = sensitive_path_level(requested)
        if level in {"forbidden", "protected"}:
            return ToolResult(f"Sensitive path blocked: {requested}", isError=True, metadata={"pathLevel": level})
        if not path.exists() or not path.is_file():
            return ToolResult(f"File not found: {requested}", isError=True)
        path.unlink()
        return ToolResult(f"Deleted {requested}", metadata={"path": requested})

    def _git_command(self, args: list[str]) -> ToolResult:
        result = self._run_command({"command": ["git", *args], "timeoutMs": 30_000})
        result.metadata = {**(result.metadata or {}), "command": "git " + " ".join(args)}
        return result

    def _git_status(self, payload: dict[str, Any]) -> ToolResult:
        return self._git_command(["status", "--short"])

    def _git_log(self, payload: dict[str, Any]) -> ToolResult:
        limit = str(int(payload.get("limit") or 20))
        return self._git_command(["log", "--oneline", "-n", limit])

    def _git_diff(self, payload: dict[str, Any]) -> ToolResult:
        path = self._compat_path_payload(payload)
        return self._git_command(["diff", "--", path] if path else ["diff"])

    def _git_show(self, payload: dict[str, Any]) -> ToolResult:
        rev = str(payload.get("rev") or payload.get("revision") or "HEAD")
        return self._git_command(["show", "--stat", rev])

    def _git_blame(self, payload: dict[str, Any]) -> ToolResult:
        path = self._compat_path_payload(payload)
        if not path:
            return ToolResult("GitBlame requires path", isError=True)
        return self._git_command(["blame", "--", path])

    def _git_branch(self, payload: dict[str, Any]) -> ToolResult:
        return self._git_command(["branch", "--show-current"])

    def _git_ls_files(self, payload: dict[str, Any]) -> ToolResult:
        return self._git_command(["ls-files"])

    def _lsp_definition(self, payload: dict[str, Any]) -> ToolResult:
        return self._lsp({**payload, "operation": "definition"})

    def _lsp_references(self, payload: dict[str, Any]) -> ToolResult:
        return self._lsp({**payload, "operation": "findReferences"})

    def _lsp_hover(self, payload: dict[str, Any]) -> ToolResult:
        return self._lsp({**payload, "operation": "hover"})

    def _lsp_document_symbols(self, payload: dict[str, Any]) -> ToolResult:
        return self._lsp({**payload, "operation": "documentSymbol"})

    def _lsp_workspace_symbols(self, payload: dict[str, Any]) -> ToolResult:
        return self._lsp({**payload, "operation": "workspaceSymbol"})

    def _memory_read(self, payload: dict[str, Any]) -> ToolResult:
        return self._memory({**payload, "action": "read"})

    def _memory_search(self, payload: dict[str, Any]) -> ToolResult:
        return self._memory({**payload, "action": "search"})

    def _memory_write(self, payload: dict[str, Any]) -> ToolResult:
        return self._memory({**payload, "action": "write"})

    def _http_api_verifier(self, payload: dict[str, Any]) -> ToolResult:
        journey = payload.get("journey") or payload.get("steps") or [{"action": "http_get", "url": payload.get("url") or "/"}]
        return self._verify_journey({**payload, "journey": journey})

    def _browser_verifier(self, payload: dict[str, Any]) -> ToolResult:
        journey = payload.get("journey") or payload.get("steps") or []
        return self._verify_journey({**payload, "journey": journey})

    def _token_count(self, payload: dict[str, Any]) -> ToolResult:
        from .query_runtime import TokenCounter

        text = str(payload.get("text") or "")
        ratio = float(payload.get("tokenCharRatio") or payload.get("token_char_ratio") or 3.5)
        tokens = TokenCounter().estimate_text_for_model(text, str(payload.get("model") or "default"), ratio)
        return ToolResult(str(tokens), metadata={"tokens": tokens, "chars": len(text), "tokenCharRatio": ratio})

    def _context_status(self, payload: dict[str, Any]) -> ToolResult:
        file_count = sum(1 for _ in self._iter_files())
        return ToolResult(
            f"Workspace: {self.root}\nTools: {len(self._tools)}\nFiles: {file_count}",
            metadata={"root": str(self.root), "tools": len(self._tools), "files": file_count},
        )

    def _tool_search(self, payload: dict[str, Any]) -> ToolResult:
        query = str(payload.get("query") or "").lower()
        matches = [
            tool.api_dict()
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
            if not query or query in tool.name.lower() or query in tool.description.lower() or query in tool.group.lower()
        ]
        limit = int(payload.get("limit") or 20)
        lines = [f"{tool['name']} ({tool['group']}): {tool['description']}" for tool in matches[:limit]]
        return ToolResult("\n".join(lines), metadata={"tools": matches[:limit], "count": len(matches)})

    def _command_classify(self, payload: dict[str, Any]) -> ToolResult:
        command = str(payload.get("command") or "")
        risk = command_risk(command)
        classification = classify_command(command)
        return ToolResult(
            f"{risk.level.value}: {risk.reason or classification.category}",
            metadata={"risk": asdict(risk), "classification": asdict(classification)},
        )

    def _sleep(self, payload: dict[str, Any]) -> ToolResult:
        ms = min(max(int(payload.get("milliseconds") or payload.get("ms") or 0), 0), 5_000)
        if ms:
            time.sleep(ms / 1000)
        return ToolResult(f"Slept {ms}ms", metadata={"milliseconds": ms})

    def _date_time(self, payload: dict[str, Any]) -> ToolResult:
        now = time.time()
        return ToolResult(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), metadata={"epochSeconds": now})

    def _json_parse(self, payload: dict[str, Any]) -> ToolResult:
        text = str(payload.get("text") or payload.get("json") or "")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return ToolResult(str(exc), isError=True)
        kind = "array" if isinstance(parsed, list) else "object" if isinstance(parsed, dict) else type(parsed).__name__
        size = len(parsed) if isinstance(parsed, (list, dict, str)) else None
        return ToolResult(json.dumps(parsed, ensure_ascii=False, indent=2)[:100_000], metadata={"type": kind, "size": size})

    def _regex_extract(self, payload: dict[str, Any]) -> ToolResult:
        text = str(payload.get("text") or "")
        pattern = str(payload.get("pattern") or "")
        if not pattern:
            return ToolResult("Missing pattern", isError=True)
        matches = re.findall(pattern, text)
        return ToolResult(json.dumps(matches, ensure_ascii=False), metadata={"count": len(matches), "matches": matches[:100]})

    def _todo_write(self, payload: dict[str, Any]) -> ToolResult:
        title = str(payload.get("title") or payload.get("description") or payload.get("content") or "todo")
        task = self._new_task(title, session_id=str(payload.get("sessionId") or "default"), task_type="todo")
        if payload.get("status"):
            task["status"] = str(payload.get("status"))
        return ToolResult(f"Todo tracked: {task['taskId']}", metadata={"task": task})

    def _ask_user_question(self, payload: dict[str, Any]) -> ToolResult:
        question = str(payload.get("question") or payload.get("prompt") or "")
        if not question:
            return ToolResult("Missing question", isError=True)
        task = self._new_task(question, session_id=str(payload.get("sessionId") or "default"), task_type="question")
        task["status"] = "WAITING_USER"
        return ToolResult(f"Question recorded: {question}", metadata={"task": task, "question": question})

    def _synthetic_output(self, payload: dict[str, Any]) -> ToolResult:
        content = str(payload.get("content") or payload.get("text") or "ok")
        return ToolResult(content[:10_000], metadata={"chars": min(len(content), 10_000), "synthetic": True})

    def _iter_files(self):
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            try:
                rel_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in IGNORED_DIRS for part in rel_parts):
                continue
            yield path

    def _list_files(self, payload: dict[str, Any]) -> ToolResult:
        pattern = str(payload.get("pattern") or "*")
        limit = int(payload.get("limit") or 200)
        files = []
        for path in self._iter_files():
            rel = path.relative_to(self.root).as_posix()
            if fnmatch.fnmatch(rel, pattern):
                files.append(rel)
            if len(files) >= limit:
                break
        return ToolResult("\n".join(files), metadata={"count": len(files)})

    def _read_file(self, payload: dict[str, Any]) -> ToolResult:
        requested = str(payload.get("path") or "")
        if not requested:
            return ToolResult("Missing path", isError=True)
        path = self._resolve_path(requested)
        level = sensitive_path_level(requested)
        if level == "forbidden":
            return ToolResult(f"Sensitive path blocked: {requested}", isError=True, metadata={"pathLevel": level})
        if not path.exists() or not path.is_file():
            return ToolResult(f"File not found: {requested}", isError=True)
        text = path.read_text(encoding="utf-8", errors="replace")
        content_hash = self.file_tracker.record_read(path)
        filtered = filter_sensitive_data(text) or ""
        return ToolResult(
            filtered[:200_000],
            metadata={
                "path": self._display_path(path),
                "truncated": len(filtered) > 200_000,
                "pathLevel": level,
                "contentHash": content_hash,
            },
        )

    def _search_files(self, payload: dict[str, Any]) -> ToolResult:
        query = str(payload.get("query") or "")
        limit = int(payload.get("limit") or 100)
        if not query:
            return ToolResult("Missing query", isError=True)
        matches = []
        for path in self._iter_files():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    rel = path.relative_to(self.root).as_posix()
                    matches.append(f"{rel}:{line_no}: {line.strip()}")
                    break
            if len(matches) >= limit:
                break
        return ToolResult("\n".join(matches), metadata={"count": len(matches)})

    def _write_file(self, payload: dict[str, Any]) -> ToolResult:
        requested = str(payload.get("path") or "")
        level = sensitive_path_level(requested)
        if level in {"forbidden", "protected"}:
            return ToolResult(f"Sensitive path blocked: {requested}", isError=True, metadata={"pathLevel": level})
        path = self._resolve_path(requested)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(payload.get("content") or "")
        expected_hash = payload.get("expectedHash") or payload.get("expected_hash")
        conflict = self.file_tracker.check_before_write(path, str(expected_hash) if expected_hash else None)
        if conflict.hasConflict:
            decision = self.file_recovery.recover("write_file", "conflict detected")
            return ToolResult(
                decision.message,
                isError=True,
                metadata={"conflict": conflict.to_dict(), "recovery": decision.to_dict()},
            )
        before_text = path.read_text(encoding="utf-8", errors="replace") if path.exists() and path.is_file() else ""
        agent_id = str(payload.get("agentId") or payload.get("agent_id") or "tool")
        snapshot = self._snapshot_before_write(path, before_text, "write_file", agent_id)
        try:
            self._atomic_write_text(path, content)
        except OSError as exc:
            decision = self.file_recovery.recover("write_file", str(exc))
            return ToolResult(
                decision.message,
                isError=True,
                metadata={"recovery": decision.to_dict(), "atomic": True, "path": self._display_path(path)},
            )
        content_hash = self.file_tracker.record_write(path, agent_id=agent_id)
        return ToolResult(
            f"Wrote {requested}",
            metadata={
                "path": self._display_path(path),
                "bytes": len(content.encode("utf-8")),
                "contentHash": content_hash,
                "beforeHash": snapshot["contentHash"],
                "afterHash": content_hash,
                "snapshotBeforeWrite": snapshot,
                "diff": self._unified_diff(path, before_text, content),
                "atomic": True,
            },
        )

    def _edit_file(self, payload: dict[str, Any]) -> ToolResult:
        requested = str(payload.get("path") or "")
        level = sensitive_path_level(requested)
        if level in {"forbidden", "protected"}:
            return ToolResult(f"Sensitive path blocked: {requested}", isError=True, metadata={"pathLevel": level})
        path = self._resolve_path(requested)
        old = str(payload.get("old") or "")
        new = str(payload.get("new") or "")
        if not path.exists():
            decision = self.file_recovery.recover("edit_file", f"File not found: {requested}")
            return ToolResult(decision.message, isError=True, metadata={"recovery": decision.to_dict()})
        expected_hash = payload.get("expectedHash") or payload.get("expected_hash")
        conflict = self.file_tracker.check_before_write(path, str(expected_hash) if expected_hash else None)
        if conflict.hasConflict:
            decision = self.file_recovery.recover("edit_file", "conflict detected")
            return ToolResult(
                decision.message,
                isError=True,
                metadata={"conflict": conflict.to_dict(), "recovery": decision.to_dict()},
            )
        text = path.read_text(encoding="utf-8", errors="replace")
        agent_id = str(payload.get("agentId") or payload.get("agent_id") or "tool")
        snapshot = self._snapshot_before_write(path, text, "edit_file", agent_id)
        match_strategy = "exact"
        replacement = self._replace_once_with_strategy(text, old, new)
        if replacement is None:
            decision = self.file_recovery.recover("edit_file", "old_string not found in file")
            return ToolResult(decision.message, isError=True, metadata={"recovery": decision.to_dict()})
        updated, match_strategy = replacement
        try:
            self._atomic_write_text(path, updated)
        except OSError as exc:
            decision = self.file_recovery.recover("edit_file", str(exc))
            return ToolResult(
                decision.message,
                isError=True,
                metadata={"recovery": decision.to_dict(), "atomic": True, "path": self._display_path(path)},
            )
        content_hash = self.file_tracker.record_write(path, agent_id=agent_id)
        return ToolResult(
            f"Edited {requested}",
            metadata={
                "path": self._display_path(path),
                "contentHash": content_hash,
                "beforeHash": snapshot["contentHash"],
                "afterHash": content_hash,
                "snapshotBeforeWrite": snapshot,
                "diff": self._unified_diff(path, text, updated),
                "matchStrategy": match_strategy,
                "atomic": True,
            },
        )

    def _run_command(self, payload: dict[str, Any]) -> ToolResult:
        parts = command_parts(payload.get("command"))
        blocked = blocked_command_check(parts)
        if blocked:
            reason = blocked.reason or f"blocked command: {blocked.level}"
            return ToolResult(reason, isError=True, metadata=blocked_command_metadata(parts, blocked))
        classification = classify_command(" ".join(parts))
        timeout = min(max(int(payload.get("timeoutMs") or 30_000), 100), 120_000) / 1000
        started = time.time()
        try:
            completed = subprocess.run(parts, cwd=self.root, text=True, capture_output=True, timeout=timeout)
            content = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")
            filtered = filter_sensitive_data(content) or ""
            visible, output_truncated, total_output_bytes = self._bounded_command_output(filtered)
            sandbox = self._command_sandbox_metadata(
                parts,
                classification,
                started,
                timeout,
                exit_code=completed.returncode,
                output_bytes=len(visible.encode("utf-8")),
                total_output_bytes=total_output_bytes,
                output_truncated=output_truncated,
            )
            return ToolResult(
                visible,
                isError=completed.returncode != 0,
                metadata={"exitCode": completed.returncode, "classification": asdict(classification), "sandbox": sandbox},
            )
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or "") + (("\n" + exc.stderr) if exc.stderr else "")
            filtered = filter_sensitive_data(str(partial)) or ""
            visible, output_truncated, total_output_bytes = self._bounded_command_output(filtered)
            sandbox = self._command_sandbox_metadata(
                parts,
                classification,
                started,
                timeout,
                exit_code=124,
                output_bytes=len(visible.encode("utf-8")),
                total_output_bytes=total_output_bytes,
                output_truncated=output_truncated,
                timed_out=True,
            )
            message = "Command timed out" + (("\n" + visible) if visible else "")
            return ToolResult(message, isError=True, metadata={"exitCode": 124, "classification": asdict(classification), "sandbox": sandbox})
        except OSError as exc:
            sandbox = self._command_sandbox_metadata(
                parts,
                classification,
                started,
                timeout,
                exit_code=127,
                output_bytes=0,
                total_output_bytes=0,
                output_truncated=False,
                spawn_error=str(exc),
            )
            return ToolResult(str(exc), isError=True, metadata={"exitCode": 127, "classification": asdict(classification), "sandbox": sandbox})

    def _bounded_command_output(self, content: str) -> tuple[str, bool, int]:
        encoded = content.encode("utf-8")
        total = len(encoded)
        if total <= COMMAND_MAX_OUTPUT_BYTES:
            return content, False, total
        visible = encoded[-COMMAND_MAX_OUTPUT_BYTES:].decode("utf-8", errors="replace")
        return visible, True, total

    def _command_sandbox_metadata(
        self,
        parts: list[str],
        classification: Any,
        started: float,
        timeout_seconds: float,
        exit_code: int,
        output_bytes: int,
        total_output_bytes: int,
        output_truncated: bool,
        timed_out: bool = False,
        spawn_error: str | None = None,
    ) -> dict[str, Any]:
        if timed_out:
            error_type = "timeout"
        elif spawn_error is not None:
            error_type = "spawn_error"
        elif exit_code < 0:
            error_type = "signal"
        elif exit_code > 0:
            error_type = "exit_code"
        else:
            error_type = None
        return {
            "command": parts,
            "cwd": str(self.root),
            "readOnly": bool(getattr(classification, "isReadOnly", False)),
            "category": str(getattr(classification, "category", "unknown")),
            "exitCode": exit_code,
            "errorType": error_type,
            "timedOut": timed_out,
            "timeoutMs": int(timeout_seconds * 1000),
            "durationMs": max(0, int((time.time() - started) * 1000)),
            "outputBytes": output_bytes,
            "totalOutputBytes": total_output_bytes,
            "outputTruncated": output_truncated,
            "spawnError": spawn_error,
            "processTreeManaged": True,
        }

    def _bash(self, payload: dict[str, Any]) -> ToolResult:
        command = str(payload.get("command") or "")
        parts = command_parts(command)
        blocked = blocked_command_check(parts)
        if blocked:
            reason = blocked.reason or f"blocked command: {blocked.level}"
            return ToolResult(reason, isError=True, metadata=blocked_command_metadata(parts, blocked))
        timeout = int(payload.get("timeout") or payload.get("timeoutMs") or 120_000)
        if sys.platform.startswith("win"):
            return self._run_command({"command": ["cmd.exe", "/c", command], "timeoutMs": timeout})
        return self._run_command({"command": ["bash", "-lc", command], "timeoutMs": timeout})

    def _powershell(self, payload: dict[str, Any]) -> ToolResult:
        command = str(payload.get("command") or "")
        parts = command_parts(command)
        blocked = blocked_command_check(parts)
        if blocked:
            reason = blocked.reason or f"blocked command: {blocked.level}"
            return ToolResult(reason, isError=True, metadata=blocked_command_metadata(parts, blocked))
        timeout = int(payload.get("timeout") or payload.get("timeoutMs") or 120_000)
        executable = "powershell.exe" if sys.platform.startswith("win") else "pwsh"
        return self._run_command({"command": [executable, "-NoLogo", "-NoProfile", "-Command", command], "timeoutMs": timeout})

    def _repl(self, payload: dict[str, Any]) -> ToolResult:
        language = str(payload.get("language") or "python").lower()
        if language != "python":
            return ToolResult(f"Unsupported REPL language: {language}", isError=True)
        code = str(payload.get("code") or "")
        session_id = str(payload.get("sessionId") or "default")
        if session_id not in self._repl_sessions:
            if len(self._repl_sessions) >= REPL_MAX_SESSIONS:
                return ToolResult("REPL session limit reached", isError=True, metadata={"maxSessions": REPL_MAX_SESSIONS})
            self._repl_sessions[session_id] = {"namespace": {}, "createdAt": time.time(), "language": language}
        session = self._repl_sessions[session_id]
        session["lastActive"] = time.time()
        namespace = session["namespace"]
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
                try:
                    compiled = compile(code, f"<repl:{session_id}>", "eval")
                except SyntaxError:
                    compiled = compile(code, f"<repl:{session_id}>", "exec")
                    exec(compiled, namespace, namespace)
                    value = None
                else:
                    value = eval(compiled, namespace, namespace)
                    if value is not None:
                        print(repr(value))
        except Exception as exc:
            output = stdout.getvalue()
            return ToolResult((output + f"{type(exc).__name__}: {exc}")[-REPL_MAX_OUTPUT_BYTES:], isError=True, metadata={"sessionId": session_id})
        output = stdout.getvalue()
        truncated = len(output.encode("utf-8")) > REPL_MAX_OUTPUT_BYTES
        return ToolResult(output[-REPL_MAX_OUTPUT_BYTES:], metadata={"sessionId": session_id, "truncated": truncated})

    def _source_array(self, content: str | None) -> list[str]:
        if not content:
            return []
        lines = str(content).splitlines(keepends=True)
        return lines

    def _create_cell(self, cell_type: str, content: str | None) -> dict[str, Any]:
        cell_type = cell_type if cell_type in {"code", "markdown", "raw"} else "code"
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": self._source_array(content),
            "id": uuid.uuid4().hex[:8],
        }
        if cell_type == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        return cell

    def _notebook_edit(self, payload: dict[str, Any]) -> ToolResult:
        requested = str(payload.get("notebook_path") or payload.get("path") or "")
        if not requested:
            return ToolResult("Missing notebook_path", isError=True)
        path = self._resolve_path(requested)
        if not path.exists():
            return ToolResult(f"Notebook not found: {requested}", isError=True)
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook.setdefault("cells", [])
        command = str(payload.get("command") or "")
        index = payload.get("cell_index")
        content = payload.get("content")
        cell_type = str(payload.get("cell_type") or "code")
        if command == "add_cell":
            insert_at = len(cells) if index is None else max(0, min(int(index), len(cells)))
            cells.insert(insert_at, self._create_cell(cell_type, content))
        elif command == "edit_cell":
            if index is None or int(index) < 0 or int(index) >= len(cells):
                return ToolResult("Invalid cell_index", isError=True)
            cell = cells[int(index)]
            cell["source"] = self._source_array(content)
            if cell.get("cell_type") == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        elif command == "delete_cell":
            if index is None or int(index) < 0 or int(index) >= len(cells):
                return ToolResult("Invalid cell_index", isError=True)
            del cells[int(index)]
        elif command == "move_cell":
            if index is None or int(index) < 0 or int(index) >= len(cells):
                return ToolResult("Invalid cell_index", isError=True)
            current = int(index)
            direction = str(payload.get("direction") or "down")
            target = current - 1 if direction == "up" else current + 1
            if target < 0 or target >= len(cells):
                return ToolResult("Cannot move cell beyond notebook bounds", isError=True)
            cells[current], cells[target] = cells[target], cells[current]
        elif command == "change_cell_type":
            if index is None or int(index) < 0 or int(index) >= len(cells):
                return ToolResult("Invalid cell_index", isError=True)
            if cell_type not in {"code", "markdown", "raw"}:
                return ToolResult(f"Invalid cell_type: {cell_type}", isError=True)
            cell = cells[int(index)]
            cell["cell_type"] = cell_type
            if cell_type == "code":
                cell.setdefault("outputs", [])
                cell["execution_count"] = None
            else:
                cell.pop("outputs", None)
                cell.pop("execution_count", None)
        else:
            return ToolResult(f"Unsupported notebook command: {command}", isError=True)
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
        return ToolResult(f"Notebook updated: {requested}", metadata={"path": path.relative_to(self.root).as_posix(), "cells": len(cells)})

    def _truncate_task_output(self, output: str | None) -> str:
        text = str(output or "")
        if len(text.encode("utf-8")) <= TASK_MAX_OUTPUT_BYTES:
            return text
        return text[:TASK_MAX_OUTPUT_BYTES] + "[Output truncated at 1MB limit]"

    def _new_task(self, description: str, session_id: str = "default", task_type: str = "general") -> dict[str, Any]:
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        now = time.time()
        task = {
            "taskId": task_id,
            "sessionId": session_id,
            "description": description,
            "taskType": task_type,
            "status": "RUNNING",
            "createdAt": now,
            "updatedAt": now,
            "output": None,
            "error": None,
            "childTaskIds": [],
            "childPids": [],
        }
        self._tasks[task_id] = task
        return task

    def _acquire_agent_slot(self, agent_id: str, session_id: str, nesting_depth: int) -> str | None:
        if nesting_depth > MAX_AGENT_NESTING_DEPTH:
            return f"Agent nesting depth {nesting_depth} exceeds max {MAX_AGENT_NESTING_DEPTH}"
        with self._agent_lock:
            session_counts = self._session_agent_counts.get(session_id, {})
            active_total = sum(self._active_agent_counts.values())
            session_total = sum(session_counts.values())
            if active_total >= MAX_CONCURRENT_AGENTS:
                return f"Concurrent agent limit reached: {MAX_CONCURRENT_AGENTS}"
            if session_total >= MAX_CONCURRENT_AGENTS_PER_SESSION:
                return f"Session {session_id} concurrent agent limit reached: {MAX_CONCURRENT_AGENTS_PER_SESSION}"
            session_agents = self._session_agents.setdefault(session_id, set())
            session_counts = self._session_agent_counts.setdefault(session_id, {})
            self._active_agent_counts[agent_id] = self._active_agent_counts.get(agent_id, 0) + 1
            session_counts[agent_id] = session_counts.get(agent_id, 0) + 1
            self._active_agents.add(agent_id)
            session_agents.add(agent_id)
        return None

    def _release_agent_slot(self, agent_id: str, session_id: str) -> None:
        with self._agent_lock:
            active_count = self._active_agent_counts.get(agent_id, 0)
            if active_count <= 1:
                self._active_agent_counts.pop(agent_id, None)
                self._active_agents.discard(agent_id)
            else:
                self._active_agent_counts[agent_id] = active_count - 1
            session_agents = self._session_agents.get(session_id)
            session_counts = self._session_agent_counts.get(session_id)
            if session_counts is not None:
                count = session_counts.get(agent_id, 0)
                if count <= 1:
                    session_counts.pop(agent_id, None)
                else:
                    session_counts[agent_id] = count - 1
                if not session_counts:
                    self._session_agent_counts.pop(session_id, None)
            if session_agents is not None:
                if not session_counts or agent_id not in session_counts:
                    session_agents.discard(agent_id)
                if not session_agents:
                    self._session_agents.pop(session_id, None)

    def _run_agent_prompt(self, prompt: str, agent_type: str, model: str, team_name: str | None, fork: bool, isolation: str, hierarchy: str) -> str:
        prefix = f"Agent completed ({agent_type}, {model})"
        if team_name:
            prefix = f"Agent routed to team {team_name} ({agent_type}, {model})"
        mode_bits = []
        if fork:
            mode_bits.append("fork")
        if isolation and isolation.lower() != "none":
            mode_bits.append(f"isolation={isolation}")
        mode_bits.append(f"hierarchy={hierarchy}")
        result = f"{prefix}: {prompt}\nContext: {', '.join(mode_bits)}"
        if len(result) > MAX_AGENT_RESULT_SIZE_CHARS:
            result = result[:MAX_AGENT_RESULT_SIZE_CHARS] + "\n...[truncated]"
        return result

    def _complete_background_agent(
        self,
        task_id: str,
        agent_id: str,
        session_id: str,
        prompt: str,
        description: str,
        agent_type: str,
        model: str,
        team_name: str | None,
        fork: bool,
        isolation: str,
        hierarchy: str,
        output_file: str,
        tool_calls: list[dict[str, Any]],
        turns: list[dict[str, Any]],
        worktree_merge_strategy: str | None,
        timeout_ms: int | str | None,
    ) -> None:
        try:
            metadata: dict[str, Any] = {}
            is_error = False
            dispatcher = self.team_dispatcher if team_name and self.team_dispatcher is not None else self.agent_dispatcher
            if dispatcher is not None:
                dispatched = dispatcher(
                    {
                        "prompt": prompt,
                        "description": description,
                        "agentType": agent_type,
                        "model": model,
                        "teamName": team_name,
                        "sessionId": session_id,
                        "agentId": agent_id,
                        "agentHierarchy": hierarchy,
                        "fork": fork,
                        "isolation": isolation,
                        "toolCalls": tool_calls,
                        "turns": turns,
                        "worktreeMergeStrategy": worktree_merge_strategy,
                        "timeoutMs": timeout_ms,
                    }
                )
                result = dispatched.content
                metadata = dispatched.metadata or {}
                is_error = dispatched.isError
            else:
                result = self._run_agent_prompt(prompt, agent_type, model, team_name, fork, isolation, hierarchy)
            path = Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result, encoding="utf-8")
            task = self._tasks.get(task_id)
            if task is not None:
                task["status"] = "FAILED" if is_error else "COMPLETED"
                task["output"] = result
                if metadata:
                    task["metadata"] = {**(task.get("metadata") or {}), **metadata}
                    for key in ["swarmId", "workerId", "childSessionId", "turnCount"]:
                        if key in metadata:
                            task[key] = metadata[key]
                if is_error:
                    task["error"] = result
                task["updatedAt"] = time.time()
            if is_error:
                self._publish_background_agent_event(agent_id, "agent_failed", {"error": result, "childSessionId": metadata.get("childSessionId")})
            else:
                self._publish_background_agent_event(agent_id, "agent_completed", {"resultPreview": result[:500], "childSessionId": metadata.get("childSessionId")})
        except Exception as exc:
            task = self._tasks.get(task_id)
            if task is not None:
                task["status"] = "FAILED"
                task["error"] = str(exc)
                task["updatedAt"] = time.time()
            self._publish_background_agent_event(agent_id, "agent_failed", {"error": str(exc)})
        finally:
            self._release_agent_slot(agent_id, session_id)

    def _agent(self, payload: dict[str, Any]) -> ToolResult:
        prompt = str(payload.get("prompt") or "")
        if not prompt:
            return ToolResult("Missing prompt", isError=True)
        description = str(payload.get("description") or "sub-agent task")
        agent_type = str(payload.get("subagent_type") or payload.get("agentType") or "general-purpose")
        raw_model = str(payload.get("model") or ("light" if agent_type in {"explore", "guide"} else "standard"))
        model = raw_model if raw_model == "default" else AGENT_MODEL_ALIAS_MAP.get(raw_model, raw_model)
        background = bool(payload.get("run_in_background") or payload.get("runInBackground"))
        team_name = str(payload.get("teamName") or payload.get("team_name") or "").strip() or None
        fork = bool(payload.get("fork"))
        isolation = str(payload.get("isolation") or "NONE")
        session_id = str(payload.get("sessionId") or payload.get("session_id") or "default")
        nesting_depth = int(payload.get("nestingDepth") or payload.get("nesting_depth") or 0) + 1
        agent_id = str(payload.get("agentId") or payload.get("agent_id") or f"agent-{uuid.uuid4().hex[:8]}")
        parent_hierarchy = str(payload.get("agentHierarchy") or payload.get("agent_hierarchy") or "main")
        hierarchy = f"{parent_hierarchy} > subagent-{agent_id}"
        limit_error = self._acquire_agent_slot(agent_id, session_id, nesting_depth)
        if limit_error:
            return ToolResult(limit_error, isError=True, metadata={"status": "limit_exceeded", "agentId": agent_id, "sessionId": session_id, "nestingDepth": nesting_depth})
        if background:
            task = self._new_task(prompt[:200], task_type=f"agent:{agent_type}")
            child_session_id = f"{'fork' if fork else 'subagent'}-{agent_id}"
            task["status"] = "RUNNING"
            task["outputFile"] = str(self.root / "backend-python" / "data" / "agents" / f"{task['taskId']}.txt")
            task["agentId"] = agent_id
            task["childSessionId"] = child_session_id
            task["sessionId"] = session_id
            task["agentHierarchy"] = hierarchy
            task["agentType"] = agent_type
            task["model"] = model
            task["isolation"] = isolation
            task["fork"] = fork
            task["teamName"] = team_name
            task["agentDescription"] = description
            task["toolCalls"] = payload.get("toolCalls") if isinstance(payload.get("toolCalls"), list) else []
            task["turns"] = payload.get("turns") if isinstance(payload.get("turns"), list) else []
            task["worktreeMergeStrategy"] = payload.get("worktreeMergeStrategy") or payload.get("mergeStrategy")
            task["timeoutMs"] = payload.get("timeoutMs") or payload.get("timeout_ms")
            thread = threading.Thread(
                target=self._complete_background_agent,
                args=(
                    task["taskId"],
                    agent_id,
                    session_id,
                    prompt,
                    description,
                    agent_type,
                    model,
                    team_name,
                    fork,
                    isolation,
                    hierarchy,
                    task["outputFile"],
                    task["toolCalls"],
                    task["turns"],
                    task["worktreeMergeStrategy"],
                    task["timeoutMs"],
                ),
                daemon=True,
                name=f"zhiku-agent-{agent_id}",
            )
            self._publish_background_agent_event(agent_id, "agent_started", {"prompt": prompt, "taskId": task["taskId"], "childSessionId": child_session_id})
            thread.start()
            return ToolResult(
                "Agent launched in background.\n"
                f"Agent ID: {agent_id}\n"
                f"Output file: {task['outputFile']}\n"
                f"Description: {description}\n"
                f"Prompt: {prompt}",
                metadata={"status": "async_launched", "taskId": task["taskId"], "outputFile": task["outputFile"], "agentId": agent_id, "agentType": agent_type, "model": model, "teamName": team_name, "sessionId": session_id, "nestingDepth": nesting_depth, "agentHierarchy": hierarchy, "description": description},
            )
        try:
            dispatcher = self.team_dispatcher if team_name and self.team_dispatcher is not None else self.agent_dispatcher
            if dispatcher is not None:
                return dispatcher(
                    {
                        "prompt": prompt,
                        "description": description,
                        "agentType": agent_type,
                        "model": model,
                        "teamName": team_name,
                        "sessionId": session_id,
                        "agentId": agent_id,
                        "agentHierarchy": hierarchy,
                        "fork": fork,
                        "isolation": isolation,
                        "nestingDepth": nesting_depth,
                        "toolCalls": payload.get("toolCalls") if isinstance(payload.get("toolCalls"), list) else [],
                        "turns": payload.get("turns") if isinstance(payload.get("turns"), list) else [],
                        "worktreeMergeStrategy": payload.get("worktreeMergeStrategy") or payload.get("mergeStrategy"),
                        "timeoutMs": payload.get("timeoutMs") or payload.get("timeout_ms"),
                    }
                )
            result = self._run_agent_prompt(prompt, agent_type, model, team_name, fork, isolation, hierarchy)
            return ToolResult(result, metadata={"status": "completed", "agentId": agent_id, "agentType": agent_type, "model": model, "teamName": team_name, "sessionId": session_id, "nestingDepth": nesting_depth, "agentHierarchy": hierarchy})
        finally:
            self._release_agent_slot(agent_id, session_id)

    def _task_create(self, payload: dict[str, Any]) -> ToolResult:
        description = str(payload.get("description") or payload.get("prompt") or "")
        if not description:
            return ToolResult("Missing description", isError=True)
        session_id = str(payload.get("sessionId") or "default")
        task = self._new_task(description, session_id=session_id, task_type=str(payload.get("taskType") or "general"))
        if payload.get("prompt"):
            task["prompt"] = str(payload.get("prompt"))
        return ToolResult(f"Task {task['taskId']} created successfully", metadata={"task": task})

    def _task_list(self, payload: dict[str, Any]) -> ToolResult:
        status = str(payload.get("status") or "").upper()
        session_id = str(payload.get("sessionId") or "")
        tasks = list(self._tasks.values())
        if status:
            tasks = [task for task in tasks if str(task.get("status")).upper() == status]
        if session_id:
            tasks = [task for task in tasks if task.get("sessionId") == session_id]
        if not tasks:
            return ToolResult("No tasks found", metadata={"tasks": []})
        lines = [f"Tasks ({len(tasks)})"]
        for task in tasks:
            lines.append(f"- {task['taskId']} [{task['status']}] {task['description']}")
        return ToolResult("\n".join(lines), metadata={"tasks": tasks})

    def _task_get(self, payload: dict[str, Any]) -> ToolResult:
        task_id = str(payload.get("taskId") or payload.get("task_id") or "")
        task = self._tasks.get(task_id)
        if not task:
            return ToolResult(f"Task not found: {task_id}", isError=True)
        return ToolResult(f"Task: {task_id}\nStatus: {task['status']}\nDescription: {task['description']}", metadata={"task": task})

    def _task_update(self, payload: dict[str, Any]) -> ToolResult:
        task_id = str(payload.get("taskId") or payload.get("task_id") or "")
        task = self._tasks.get(task_id)
        if not task:
            return ToolResult(f"Task not found: {task_id}", isError=True)
        for key in ("status", "error"):
            if key in payload:
                task[key] = str(payload.get(key))
        if "output" in payload:
            task["output"] = self._truncate_task_output(str(payload.get("output") or ""))
        task["updatedAt"] = time.time()
        return ToolResult(f"Task {task_id} updated to {task['status']}", metadata={"task": task})

    def _task_stop(self, payload: dict[str, Any]) -> ToolResult:
        task_id = str(payload.get("taskId") or payload.get("task_id") or "")
        task = self._tasks.get(task_id)
        if not task:
            return ToolResult(f"Task not found: {task_id}", isError=True)
        if str(task.get("status")).upper() in {"COMPLETED", "FAILED", "CANCELLED", "KILLED"}:
            return ToolResult(f"Task {task_id} is already in terminal state {task['status']}", isError=True)
        task["status"] = "CANCELLED"
        task["error"] = str(payload.get("reason") or "cancelled")
        task["updatedAt"] = time.time()
        return ToolResult(f"Task {task_id} cancelled", metadata={"task": task})

    def _task_output(self, payload: dict[str, Any]) -> ToolResult:
        task_id = str(payload.get("taskId") or payload.get("task_id") or "")
        if not task_id:
            return ToolResult("TaskOutput requires taskId in Python backend", isError=True)
        task = self._tasks.get(task_id)
        if not task:
            return ToolResult(f"Task not found: {task_id}", isError=True)
        output = self._truncate_task_output(str(payload.get("output") or ""))
        task["output"] = output
        if bool(payload.get("isError")):
            task["error"] = output
        task["updatedAt"] = time.time()
        return ToolResult(f"Output reported to parent task {task_id}" + (" (error)" if payload.get("isError") else ""), metadata={"task": task})

    def _cron_create(self, payload: dict[str, Any]) -> ToolResult:
        if not self.cron_service:
            return ToolResult("Cron service is not configured", isError=True)
        try:
            task = self.cron_service.add_task(
                str(payload.get("cron") or ""),
                str(payload.get("prompt") or ""),
                recurring=bool(payload.get("recurring", True)),
                durable=bool(payload.get("durable", False)),
                agent_id=str(payload.get("agentId") or payload.get("agent_id") or "") or None,
            )
        except CronValidationError as exc:
            return ToolResult(str(exc), isError=True)
        summary = task.prompt[:80] + ("..." if len(task.prompt) > 80 else "")
        return ToolResult(
            json.dumps({**task.to_dict(), "prompt": summary, "total_tasks": self.cron_service.task_count()}, ensure_ascii=False),
            metadata={"task": task.to_dict(), "total": self.cron_service.task_count()},
        )

    def _cron_list(self, payload: dict[str, Any]) -> ToolResult:
        if not self.cron_service:
            return ToolResult("No scheduled tasks.")
        tasks = [task.to_dict() for task in self.cron_service.list_all()]
        if not tasks:
            return ToolResult("No scheduled tasks.", metadata={"total": 0, "tasks": []})
        return ToolResult(json.dumps({"total": len(tasks), "tasks": tasks}, ensure_ascii=False), metadata={"total": len(tasks), "tasks": tasks})

    def _cron_delete(self, payload: dict[str, Any]) -> ToolResult:
        if not self.cron_service:
            return ToolResult("Cron service is not configured", isError=True)
        task_id = str(payload.get("id") or "")
        if not task_id:
            return ToolResult("Task id is required", isError=True)
        removed = self.cron_service.remove(task_id)
        if not removed:
            return ToolResult(f"No scheduled task found with id: {task_id}", isError=True)
        return ToolResult(
            f"Deleted scheduled task: id={removed.id}, cron='{removed.cron}', remaining={self.cron_service.task_count()}",
            metadata={"deleted": removed.to_dict(), "remaining": self.cron_service.task_count()},
        )

    def _memory(self, payload: dict[str, Any]) -> ToolResult:
        if not self.memdir_service:
            return ToolResult("Memory service is not configured", isError=True)
        action = str(payload.get("action") or "read")
        result = self.memdir_service.tool(
            action,
            content=str(payload.get("content") or ""),
            title=str(payload.get("title") or ""),
            category=str(payload.get("category") or "semantic"),
            query=str(payload.get("query") or ""),
            limit=int(payload.get("limit") or 5),
        )
        if "error" in result:
            return ToolResult(str(result["error"]), isError=True, metadata=result)
        return ToolResult(json.dumps(result, ensure_ascii=False), metadata=result)

    def _verify_journey(self, payload: dict[str, Any]) -> ToolResult:
        gate = CapabilityGate(payload.get("featureFlags"), payload.get("capabilities"))
        if not gate.verify_enabled():
            return ToolResult("Runtime verification capability unavailable", isError=True, metadata={"verdict": "unavailable"})
        result = verifier_for(payload, gate).verify(payload)
        return ToolResult(
            "Journey verified" if result.passed else result.errorMessage or "Journey failed",
            isError=not result.passed,
            metadata=result.to_dict(),
        )

    def _lsp(self, payload: dict[str, Any]) -> ToolResult:
        operation = str(payload.get("operation") or "")
        file_path = str(payload.get("filePath") or payload.get("path") or "")
        line = int(payload.get("line") or 1)
        character = int(payload.get("character") or 0)
        if operation in {"goToDefinition", "definition"}:
            if not file_path:
                return ToolResult("filePath is required", isError=True)
            return ToolResult("goToDefinition complete", metadata=go_to_definition(self.root, file_path, line, character))
        if operation in {"findReferences", "references"}:
            symbol = str(payload.get("symbol") or payload.get("query") or "")
            if not symbol:
                return ToolResult("symbol or query is required", isError=True)
            return ToolResult("findReferences complete", metadata={"references": references(self.root, symbol, file_path or None)})
        if operation == "hover":
            if not file_path:
                return ToolResult("filePath is required", isError=True)
            return ToolResult("hover complete", metadata={"hover": hover(self.root, file_path, line, character)})
        if operation == "documentSymbol":
            if not file_path:
                return ToolResult("filePath is required", isError=True)
            return ToolResult("documentSymbol complete", metadata={"symbols": document_symbols(self.root, file_path)})
        if operation == "workspaceSymbol":
            query = str(payload.get("query") or "")
            if not query:
                return ToolResult("query is required", isError=True)
            return ToolResult(f"workspaceSymbol: {query}", metadata={"symbols": workspace_symbols(self.root, query)})
        if operation in {"incomingCalls", "outgoingCalls", "callHierarchy"}:
            symbol = str(payload.get("symbol") or payload.get("query") or "")
            if not symbol:
                return ToolResult("symbol or query is required", isError=True)
            return ToolResult("callHierarchy complete", metadata=call_hierarchy(self.root, symbol, file_path or None))
        return ToolResult(f"Unknown LSP operation: {operation}", isError=True)
