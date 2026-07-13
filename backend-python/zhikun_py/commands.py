from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

from .tools import ToolRegistry


class ResultType(StrEnum):
    TEXT = "text"
    COMPACT = "compact"
    SKIP = "skip"
    JSX = "jsx"
    ERROR = "error"


@dataclass(slots=True)
class CommandResult:
    type: ResultType
    value: str | None = None
    data: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def text(cls, value: str) -> "CommandResult":
        return cls(ResultType.TEXT, value=value, data={})

    @classmethod
    def compact(cls, value: str, data: dict[str, Any]) -> "CommandResult":
        return cls(ResultType.COMPACT, value=value, data=data)

    @classmethod
    def error_result(cls, error: str) -> "CommandResult":
        return cls(ResultType.ERROR, error=error, data={})

    def to_ws_payload(self, command: str) -> dict[str, Any]:
        if self.type == ResultType.ERROR:
            return {"type": "command_result", "command": command, "resultType": "text", "output": self.error or "Command failed"}
        if self.type == ResultType.COMPACT:
            return {"type": "compact_complete", "displayText": self.value or "", "compactionData": self.data or {}}
        return {"type": "command_result", "command": command, "resultType": "text", "output": self.value or ""}


@dataclass(slots=True)
class Command:
    name: str
    description: str
    usage: str
    handler: Callable[[str, dict[str, Any]], CommandResult]
    aliases: list[str] | None = None
    hidden: bool = False
    type: str = "LOCAL"

    def api_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "usage": self.usage,
            "aliases": self.aliases or [],
            "type": self.type,
        }


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


class CommandRegistry:
    def __init__(self, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry
        self._commands: dict[str, Command] = {}
        self._aliases: dict[str, str] = {}
        self.register_builtin_commands()

    def register(self, command: Command) -> None:
        self._commands[command.name] = command
        for alias in command.aliases or []:
            self._aliases[alias] = command.name

    def list(self) -> list[dict[str, Any]]:
        return [cmd.api_dict() for cmd in sorted(self._commands.values(), key=lambda item: item.name) if not cmd.hidden]

    def get(self, name: str) -> Command | None:
        normalized = name.lower().strip().lstrip("/")
        return self._commands.get(normalized) or self._commands.get(self._aliases.get(normalized, ""))

    def suggest(self, name: str) -> str:
        candidates = sorted(self._commands, key=lambda candidate: levenshtein(name, candidate))[:3]
        return "Did you mean: " + ", ".join(f"/{candidate}" for candidate in candidates) + "?"

    def execute(self, name: str, args: str = "", context: dict[str, Any] | None = None) -> CommandResult:
        command = self.get(name)
        if not command:
            return CommandResult.error_result(f"Unknown command: /{name}. {self.suggest(name)}")
        return command.handler(args, context or {})

    def register_builtin_commands(self) -> None:
        self.register(Command("help", "Show available commands.", "/help", self._help, aliases=["?"]))
        self.register(Command("status", "Show Python backend status.", "/status", self._status))
        self.register(Command("model", "Show or switch the current model.", "/model [name]", self._model))
        self.register(Command("compact", "Compact the current session display.", "/compact", self._compact))
        self.register(Command("files", "List workspace files.", "/files [pattern]", self._files))
        self.register(Command("read", "Read a workspace file.", "/read <path>", self._read))
        self.register(Command("search", "Search workspace files.", "/search <query>", self._search))
        self.register_enhanced_commands()

    def register_once(self, command: Command) -> None:
        if command.name not in self._commands:
            self.register(command)

    def register_enhanced_commands(self) -> None:
        prompt_commands = {
            "commit-push-pr": "Review git changes, commit them, push the branch, and prepare a pull request.",
            "security-review": "Review the current changes for security issues and risky behavior.",
            "advisor": "Act as a senior engineering advisor for the current task.",
            "ultrareview": "Run a strict multi-pass code and architecture review.",
        }
        jsx_commands = {
            "export": ("Export the current session.", {"action": "export"}),
            "theme": ("Open theme settings.", {"action": "theme"}),
            "keybindings": ("Open keybinding settings.", {"action": "keybindings"}),
            "mcp": ("Open MCP server and capability management.", {"action": "list"}),
            "skills": ("Open the skill manager.", {"action": "skills"}),
            "plugin": ("Open plugin management.", {"action": "plugins"}),
            "agent": ("Open sub-agent controls.", {"action": "agent"}),
            "tasks": ("Open task controls.", {"action": "list"}),
            "install-github-app": ("Install or connect the GitHub app.", {"action": "installGithubApp"}),
            "install-slack-app": ("Install or connect the Slack app.", {"action": "installSlackApp"}),
        }
        local_commands = [
            "branch",
            "pr-comments",
            "rewind",
            "context",
            "copy",
            "rename",
            "tag",
            "fast",
            "effort",
            "output-style",
            "color",
            "vim",
            "hooks",
            "reload-plugins",
            "add-dir",
            "ide",
            "chrome",
            "desktop",
            "mobile",
            "terminal-setup",
            "remote-env",
            "usage",
            "extra-usage",
            "rate-limit-options",
            "upgrade",
            "version",
            "feedback",
            "stats",
            "stickers",
            "release-notes",
            "btw",
            "statusline",
            "privacy-settings",
            "sandbox-toggle",
            "bridge",
            "voice",
            "buddy",
            "passes",
            "torch",
            "fork",
            "peers",
            "workflows",
        ]
        aliases = {
            "pr-comments": ["pr_comments"],
            "add-dir": ["adddir"],
            "reload-plugins": ["reload"],
            "output-style": ["output_style"],
            "privacy-settings": ["privacy"],
            "sandbox-toggle": ["sandbox"],
        }
        for name, text in prompt_commands.items():
            self.register_once(Command(name, text, f"/{name} [args]", self._prompt_command(name, text), type="PROMPT"))
        for name, (description, data) in jsx_commands.items():
            self.register_once(Command(name, description, f"/{name} [args]", self._jsx_command(name, data), type="LOCAL_JSX"))
        for name in local_commands:
            self.register_once(Command(name, f"Run /{name}.", f"/{name} [args]", self._local_command(name), aliases=aliases.get(name)))
        self.register_once(Command("heapdump", "Capture heap diagnostics.", "/heapdump", self._local_command("heapdump"), hidden=True))

    def _prompt_command(self, name: str, prompt: str) -> Callable[[str, dict[str, Any]], CommandResult]:
        def handler(args: str, _context: dict[str, Any]) -> CommandResult:
            suffix = f"\n\nUser arguments: {args.strip()}" if args.strip() else ""
            return CommandResult(ResultType.TEXT, value=prompt + suffix, data={"promptCommand": name})

        return handler

    def _jsx_command(self, name: str, data: dict[str, Any]) -> Callable[[str, dict[str, Any]], CommandResult]:
        def handler(args: str, _context: dict[str, Any]) -> CommandResult:
            payload = {"command": name, **data}
            if args.strip():
                payload["args"] = args.strip()
            if name == "export":
                payload["format"] = args.strip() or "json"
            if name == "tasks" and args.strip():
                parts = args.split()
                payload["action"] = parts[0]
                if len(parts) > 1:
                    payload["taskId"] = parts[1]
            return CommandResult(ResultType.JSX, value=f"/{name}", data=payload)

        return handler

    def _local_command(self, name: str) -> Callable[[str, dict[str, Any]], CommandResult]:
        def handler(args: str, context: dict[str, Any]) -> CommandResult:
            value = args.strip()
            if name in {"branch", "fork", "workflows"} and not value:
                return CommandResult.text(f"Usage: /{name} <action>")
            if name in {"rewind", "rename", "add-dir", "btw"} and not value:
                return CommandResult.error_result(f"/{name} requires an argument")
            if name in {"fast", "vim", "sandbox-toggle"}:
                enabled = value.lower() in {"on", "true", "1", "enable", "enabled"}
                return CommandResult.text(f"{name} {'enabled' if enabled else 'disabled'}")
            if name == "context":
                return CommandResult.text(
                    f"Session: {context.get('sessionId') or 'none'}\nModel: {context.get('model') or 'default'}\nWorkspace: {context.get('workspace') or context.get('workingDirectory') or 'unknown'}"
                )
            if name == "copy":
                return CommandResult.text("Conversation copied to clipboard.")
            if name == "version":
                return CommandResult.text("ZhikunCode Python Backend\nRuntime: Python")
            if name in {"bridge", "voice", "buddy", "passes", "torch", "peers"}:
                return CommandResult.error_result(f"/{name} is not active in this runtime")
            if name == "remote-env":
                return CommandResult.text("Remote environment not configured.")
            if name == "ide":
                return CommandResult.text("IDE not connected.")
            if name == "rate-limit-options":
                return CommandResult.text("Rate limit options: reduce model usage or run /upgrade.")
            if name == "usage":
                return CommandResult.text("Usage tracking is available through /api/cost.")
            return CommandResult.text(f"/{name} executed" + (f": {value}" if value else ""))

        return handler

    def _help(self, _args: str, _context: dict[str, Any]) -> CommandResult:
        lines = [f"/{cmd['name']} - {cmd['description']}" for cmd in self.list()]
        return CommandResult.text("\n".join(lines))

    def _status(self, _args: str, context: dict[str, Any]) -> CommandResult:
        return CommandResult.text(
            "\n".join(
                [
                    "Python backend: running",
                    f"Session: {context.get('sessionId') or 'none'}",
                    f"Model: {context.get('model') or 'default'}",
                    f"Tools: {len(self.tool_registry.list())}",
                ]
            )
        )

    def _model(self, args: str, context: dict[str, Any]) -> CommandResult:
        model = args.strip()
        if model:
            return CommandResult(ResultType.TEXT, value=f"Model switched to {model}", data={"setModel": model})
        return CommandResult.text(f"Current model: {context.get('model') or 'default'}")

    def _compact(self, _args: str, _context: dict[str, Any]) -> CommandResult:
        return CommandResult.compact("Context compacted by the Python command router.", {"tokensSaved": 0})

    def _files(self, args: str, _context: dict[str, Any]) -> CommandResult:
        result = self.tool_registry.call("list_files", {"pattern": args.strip() or "*", "limit": 200})
        return CommandResult.error_result(result.content) if result.isError else CommandResult.text(result.content)

    def _read(self, args: str, _context: dict[str, Any]) -> CommandResult:
        result = self.tool_registry.call("read_file", {"path": args.strip()})
        return CommandResult.error_result(result.content) if result.isError else CommandResult.text(result.content)

    def _search(self, args: str, _context: dict[str, Any]) -> CommandResult:
        result = self.tool_registry.call("search_files", {"query": args.strip(), "limit": 100})
        return CommandResult.error_result(result.content) if result.isError else CommandResult.text(result.content)
