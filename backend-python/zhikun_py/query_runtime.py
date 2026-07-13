from __future__ import annotations

import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Callable


DEFAULT_CHARS_PER_TOKEN = 3.5
JSON_CHARS_PER_TOKEN = 2.0
CODE_CHARS_PER_TOKEN = 3.5
NATURAL_LANGUAGE_CHARS_PER_TOKEN = 4.0
CHINESE_CHARS_PER_TOKEN = 2.0
DEFAULT_MAX_OUTPUT_TOKENS = 8192
ESCALATED_MAX_OUTPUT_TOKENS = 65_536
MAX_OUTPUT_TOKENS_RECOVERY_LIMIT = 3
MAX_TOKENS_RECOVERY_MESSAGE = (
    "Output token limit hit. Resume directly -- no apology, no recap of what you were doing. "
    "Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces."
)


ExactTokenCounter = Callable[[str, str], int]


class TokenCounter:
    CODE_KEYWORDS = (
        "import ",
        "function ",
        "class ",
        "def ",
        "public ",
        "private ",
        "const ",
        "let ",
        "var ",
        "return ",
    )

    def __init__(self, exact_counter: ExactTokenCounter | None = None, precise_enabled: bool = False) -> None:
        self.exact_counter = exact_counter
        self.precise_enabled = precise_enabled

    def estimate_text(self, text: str | None) -> int:
        if not text:
            return 0
        if self.precise_enabled and self.exact_counter:
            try:
                exact = int(self.exact_counter(text, "default"))
                if exact >= 0:
                    return exact
            except Exception:
                pass
        return int(len(text) / self.detect_chars_per_token(text))

    def estimate_text_for_model(self, text: str | None, model_id: str | None = None, token_char_ratio: float = DEFAULT_CHARS_PER_TOKEN) -> int:
        if not text:
            return 0
        if not model_id:
            return self.estimate_text(text)
        ratio = max(float(token_char_ratio or DEFAULT_CHARS_PER_TOKEN), 1.0)
        chinese_ratio = self.chinese_ratio(text)
        if chinese_ratio > 0.3:
            adjusted_ratio = ratio * (1.0 - chinese_ratio * 0.3)
            return int(len(text) / max(adjusted_ratio, 1.0))
        return int(len(text) / ratio)

    def estimate_text_with_type(self, text: str | None, content_type: str | None = None) -> int:
        if not text:
            return 0
        normalized = (content_type or "").lower()
        if normalized == "json":
            ratio = JSON_CHARS_PER_TOKEN
        elif normalized in {"code", "java", "python", "javascript", "typescript"}:
            ratio = CODE_CHARS_PER_TOKEN
        elif normalized in {"text", "markdown"}:
            ratio = NATURAL_LANGUAGE_CHARS_PER_TOKEN
        else:
            ratio = self.detect_chars_per_token(text)
        return int(len(text) / ratio)

    def estimate_messages(self, messages: list[dict[str, Any]] | None, model_id: str | None = None, token_char_ratio: float = DEFAULT_CHARS_PER_TOKEN) -> int:
        if not messages:
            return 0
        total_chars = sum(self.estimate_message_chars(message) for message in messages)
        ratio = max(float(token_char_ratio or DEFAULT_CHARS_PER_TOKEN), 1.0) if model_id else DEFAULT_CHARS_PER_TOKEN
        return int(total_chars / ratio) + len(messages) * 4

    def estimate_message_chars(self, message: dict[str, Any]) -> int:
        total = 0
        content = message.get("content")
        if isinstance(content, list):
            total += sum(self.estimate_block_chars(block) for block in content)
        elif isinstance(content, str):
            total += len(content)
        tool_use_result = message.get("toolUseResult")
        if tool_use_result:
            total += len(str(tool_use_result))
        return total

    def estimate_block_chars(self, block: Any) -> int:
        if isinstance(block, str):
            return len(block)
        if not isinstance(block, dict):
            return len(str(block)) if block is not None else 0
        block_type = str(block.get("type") or "").lower()
        if block_type == "text":
            return len(str(block.get("text") or ""))
        if block_type in {"tool_use", "tooluse"}:
            name = str(block.get("toolName") or block.get("name") or "")
            return len(name) + len(jsonish(block.get("input") or {})) + 20
        if block_type in {"tool_result", "toolresult"}:
            return len(str(block.get("content") or "")) + 10
        if block_type == "image":
            width = int(block.get("width") or 0)
            height = int(block.get("height") or 0)
            return int(self.estimate_image_tokens(width, height) * DEFAULT_CHARS_PER_TOKEN)
        if block_type == "thinking":
            return len(str(block.get("thinking") or block.get("text") or ""))
        if block_type == "redacted":
            return 10
        return len(jsonish(block))

    def detect_content_type(self, text: str | None) -> str:
        if not text or len(text) < 10:
            return "text"
        trimmed = text.strip()
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
            return "json"
        if self.chinese_ratio(text) > 0.3:
            return "chinese"
        if self.looks_like_code(trimmed):
            return "code"
        return "text"

    def detect_chars_per_token(self, text: str | None) -> float:
        if not text or len(text) < 10:
            return DEFAULT_CHARS_PER_TOKEN
        trimmed = text.strip()
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
            return JSON_CHARS_PER_TOKEN
        chinese_ratio = self.chinese_ratio(text)
        if chinese_ratio > 0.3:
            return CHINESE_CHARS_PER_TOKEN * chinese_ratio + DEFAULT_CHARS_PER_TOKEN * (1 - chinese_ratio)
        if self.looks_like_code(trimmed):
            return CODE_CHARS_PER_TOKEN
        return DEFAULT_CHARS_PER_TOKEN

    def looks_like_code(self, text: str | None) -> bool:
        if not text:
            return False
        sample = text[:500]
        indicators = 0
        if sample.count("{") + sample.count("}") > 2 or sample.count(";") > 3:
            indicators += 1
        if any(keyword in sample for keyword in self.CODE_KEYWORDS):
            indicators += 1
        if len(re.findall(r"(?m)^(?: {4}|\t)", sample)) > 3:
            indicators += 1
        return indicators >= 2

    def estimate_image_tokens(self, width: int, height: int) -> int:
        if width <= 0 or height <= 0:
            return 85
        return math.ceil((width * height) / 750)

    def chinese_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        return chinese_chars / len(text)


def estimate_tokens(text: str | None, token_char_ratio: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    return TokenCounter().estimate_text_for_model(text, "default", token_char_ratio)


class RecoveryEventType(StrEnum):
    PROMPT_TOO_LONG = "prompt_too_long"
    COMPACT_APPLIED = "compact_applied"
    TOOL_RETRY = "tool_retry"
    ABORTED = "aborted"


class QueryPhase(StrEnum):
    CREATED = "created"
    PREPARE = "prepare"
    COMPACTING = "compacting"
    MODEL_CALL = "model_call"
    STREAMING = "streaming"
    TOOL_RUNNING = "tool_running"
    WAITING_PERMISSION = "waiting_permission"
    SELF_CORRECTING = "self_correcting"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class TerminationAction(StrEnum):
    CONTINUE = "continue"
    WAIT = "wait"
    STOP = "stop"
    ABORT = "abort"


FRONTEND_QUERY_EVENT_TYPES = {
    "stream_delta",
    "thinking_delta",
    "tool_use_start",
    "tool_use_input",
    "tool_use_progress",
    "tool_result",
    "compact_start",
    "compact_event",
    "compact_complete",
    "token_warning",
    "token_budget_nudge",
    "permission_request",
    "tool_permission_denied",
    "cost_update",
    "message_complete",
    "error",
    "interrupt_ack",
    "termination_decision",
}


@dataclass(slots=True)
class QueryEvent:
    type: str
    sessionId: str
    loopId: str
    phase: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: int = field(default_factory=lambda: int(time.time() * 1000))
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "sessionId": self.sessionId,
            "loopId": self.loopId,
            "phase": self.phase,
            "ts": self.ts,
            "seq": self.seq,
            **self.payload,
        }


@dataclass(slots=True)
class QueryTransition:
    fromPhase: str
    toPhase: str
    reason: str
    ts: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TerminationDecision:
    action: TerminationAction
    reason: str
    stopReason: str | None = None
    retryable: bool = False
    phase: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action"] = self.action.value
        return data


class DefaultTerminationStrategy:
    MAX_TOKEN_STOP_REASONS = {"max_tokens", "length"}
    NORMAL_STOP_REASONS = {"end_turn", "stop", "completed"}

    def decide(
        self,
        loop: "QueryLoopState",
        requested_stop_reason: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TerminationDecision:
        payload = dict(metadata or {})
        if loop.phase == QueryPhase.ABORTED or loop.status == "aborted" or loop.error == "USER_INTERRUPT":
            return TerminationDecision(TerminationAction.ABORT, "user_interrupt", "aborted", False, loop.phase.value, loop.error, payload)
        if loop.phase == QueryPhase.WAITING_PERMISSION:
            return TerminationDecision(TerminationAction.WAIT, "permission_wait", None, True, loop.phase.value, None, payload)
        if requested_stop_reason == "withhold" or loop.promptTooLongWithheld or loop.withheldErrors:
            if loop.withheldErrors and not loop.incrementalCollapseNeeded:
                loop.incrementalCollapseNeeded = True
            payload.update(
                {
                    "withheld": True,
                    "withheldErrorCount": len(loop.withheldErrors),
                    "withheldErrors": list(loop.withheldErrors),
                    "promptTooLongWithheld": loop.promptTooLongWithheld,
                    "incrementalCollapseNeeded": loop.incrementalCollapseNeeded,
                }
            )
            return TerminationDecision(TerminationAction.CONTINUE, "withhold", None, True, loop.phase.value, None, payload)
        if error:
            return TerminationDecision(TerminationAction.STOP, "error", "error", False, loop.phase.value, error, payload)
        if loop.turns >= loop.maxTurns:
            return TerminationDecision(TerminationAction.STOP, "max_turns", "max_turns", False, loop.phase.value, None, payload)
        if requested_stop_reason:
            payload.setdefault("modelStopReason", requested_stop_reason)
            if requested_stop_reason in self.MAX_TOKEN_STOP_REASONS:
                if loop.maxOutputTokensRecoveryCount >= MAX_OUTPUT_TOKENS_RECOVERY_LIMIT:
                    payload.update(
                        {
                            "recoveryAction": "stop",
                            "recoveryCount": loop.maxOutputTokensRecoveryCount,
                            "recoveryLimit": MAX_OUTPUT_TOKENS_RECOVERY_LIMIT,
                        }
                    )
                    return TerminationDecision(
                        TerminationAction.STOP,
                        "max_tokens_recovery_limit",
                        "max_tokens",
                        False,
                        loop.phase.value,
                        None,
                        payload,
                    )
                if loop.maxTokensOverride is None:
                    previous_max_tokens = int(payload.get("maxTokens") or DEFAULT_MAX_OUTPUT_TOKENS)
                    loop.set_max_tokens_override(ESCALATED_MAX_OUTPUT_TOKENS)
                    payload.update(
                        {
                            "recoveryAction": "escalate_max_tokens",
                            "previousMaxTokens": previous_max_tokens,
                            "maxTokensOverride": ESCALATED_MAX_OUTPUT_TOKENS,
                            "recoveryLimit": MAX_OUTPUT_TOKENS_RECOVERY_LIMIT,
                        }
                    )
                else:
                    loop.increment_recovery_count()
                    payload.update(
                        {
                            "recoveryAction": "inject_recovery_prompt",
                            "recoveryPrompt": MAX_TOKENS_RECOVERY_MESSAGE,
                            "recoveryCount": loop.maxOutputTokensRecoveryCount,
                            "maxTokensOverride": loop.maxTokensOverride,
                            "recoveryLimit": MAX_OUTPUT_TOKENS_RECOVERY_LIMIT,
                        }
                    )
                return TerminationDecision(TerminationAction.CONTINUE, "max_tokens_recovery", None, True, loop.phase.value, None, payload)
            if requested_stop_reason == "tool_use":
                payload.setdefault("requiresToolExecution", True)
                return TerminationDecision(TerminationAction.CONTINUE, "tool_use", None, True, loop.phase.value, None, payload)
            reason = requested_stop_reason if requested_stop_reason in self.NORMAL_STOP_REASONS else "model_stop_reason"
            return TerminationDecision(TerminationAction.STOP, reason, requested_stop_reason, False, loop.phase.value, None, payload)
        return TerminationDecision(TerminationAction.CONTINUE, "continue", None, True, loop.phase.value, None, payload)


@dataclass(slots=True)
class RecoveryEvent:
    type: RecoveryEventType
    message: str
    createdAt: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data


@dataclass(slots=True)
class TokenBudget:
    contextWindow: int
    threshold: float
    usedTokens: int = 0
    reservedOutputTokens: int = 8192

    @property
    def max_input_tokens(self) -> int:
        return max(1, int(self.contextWindow * self.threshold) - self.reservedOutputTokens)

    @property
    def exceeded(self) -> bool:
        return self.usedTokens > self.max_input_tokens

    def usage_ratio(self) -> float:
        return min(1.0, self.usedTokens / max(1, self.max_input_tokens))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contextWindow": self.contextWindow,
            "threshold": self.threshold,
            "usedTokens": self.usedTokens,
            "reservedOutputTokens": self.reservedOutputTokens,
            "maxInputTokens": self.max_input_tokens,
            "exceeded": self.exceeded,
            "usageRatio": self.usage_ratio(),
        }


class PromptTooLongRecovery:
    def __init__(self, max_chars: int = 120_000) -> None:
        self.max_chars = max_chars

    def recover(self, prompt: str, budget: TokenBudget, cause: str | None = None) -> tuple[str, RecoveryEvent | None]:
        if not budget.exceeded and len(prompt) <= self.max_chars:
            return prompt, None
        keep = max(1_000, min(len(prompt), self.max_chars))
        recovered = prompt[: keep // 2] + "\n\n[...prompt compacted by Python QueryEngine...]\n\n" + prompt[-keep // 2 :]
        event = RecoveryEvent(
            RecoveryEventType.PROMPT_TOO_LONG,
            "Prompt exceeded token budget and was compacted.",
            metadata={
                "originalChars": len(prompt),
                "recoveredChars": len(recovered),
                "usedTokens": budget.usedTokens,
                "cause": cause or ("token_budget_exceeded" if budget.exceeded else "prompt_chars_exceeded"),
                "recoveryStage": "prompt_too_long",
            },
        )
        return recovered, event


class ToolCallTracker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, name: str, status: str, duration_ms: int = 0, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        call = {
            "id": f"tool-{uuid.uuid4().hex[:12]}",
            "name": name,
            "status": status,
            "durationMs": duration_ms,
            "createdAt": time.time(),
            "metadata": metadata or {},
        }
        self.calls.append(call)
        return call

    def to_list(self) -> list[dict[str, Any]]:
        return list(self.calls)


class SideQueryService:
    def query(self, system_prompt: str, user_content: str, max_tokens: int = 512, timeout_ms: int = 3000) -> dict[str, Any]:
        text = (user_content or "").strip()
        summary = text[: max(1, min(max_tokens * 4, 2000))]
        return {
            "status": "completed",
            "systemPrompt": system_prompt,
            "answer": summary,
            "maxTokens": max_tokens,
            "timeoutMs": timeout_ms,
        }


class MicroCompactService:
    def __init__(self, max_tool_result_chars: int = 2_000) -> None:
        self.max_tool_result_chars = max_tool_result_chars

    def compact_tool_results(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        compacted: list[dict[str, Any]] = []
        count = 0
        freed = 0
        for message in messages:
            copied = dict(message)
            result = copied.get("toolUseResult")
            if isinstance(result, str) and len(result) > self.max_tool_result_chars:
                copied["toolUseResult"] = result[: self.max_tool_result_chars] + f"\n...[micro-compact: {len(result)} chars]"
                count += 1
                freed += len(result) - len(copied["toolUseResult"])
            compacted.append(copied)
        return {"messages": compacted, "compactedCount": count, "estimatedCharsFreed": freed}


class ToolResultSummarizer:
    def summarize(self, tool_name: str, content: str | None, max_chars: int = 500) -> dict[str, Any]:
        text = str(content or "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        summary = "\n".join(lines[:10])
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "...[truncated]"
        return {
            "toolName": tool_name,
            "summary": summary,
            "originalChars": len(text),
            "lineCount": len(lines),
            "truncated": len(summary) < len(text),
        }


class AbortController:
    def __init__(self) -> None:
        self.aborted: dict[str, dict[str, Any]] = {}

    def abort(self, session_id: str, reason: str = "USER_INTERRUPT") -> dict[str, Any]:
        record = {"sessionId": session_id, "reason": reason, "abortedAt": time.time()}
        self.aborted[session_id] = record
        return record

    def is_aborted(self, session_id: str) -> bool:
        return session_id in self.aborted

    def clear(self, session_id: str) -> None:
        self.aborted.pop(session_id, None)


class ToolPriorityScheduler:
    PRIORITY_MAP = {
        "FileRead": 0,
        "read_file": 0,
        "GrepSearch": 0,
        "search_files": 0,
        "ListDir": 0,
        "list_files": 0,
        "LspDefinition": 1,
        "LspReferences": 1,
        "Lsp": 1,
        "LSP": 1,
        "Bash": 2,
        "PowerShell": 2,
        "run_command": 2,
        "FileWrite": 3,
        "write_file": 3,
        "FileEdit": 3,
        "edit_file": 3,
    }

    def get_priority(self, tool_name: str) -> int:
        return self.PRIORITY_MAP.get(tool_name, 2)

    def sort_by_priority(self, tool_calls: list[Any], name_getter=None) -> list[Any]:
        if len(tool_calls) <= 1:
            return tool_calls
        getter = name_getter or self._default_name
        ordered = sorted(enumerate(tool_calls), key=lambda item: (self.get_priority(str(getter(item[1]))), item[0]))
        return [item for _, item in ordered]

    def ordered_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.sort_by_priority(tool_calls, self._default_name)

    def has_conflict(self, tool_names: list[str], file_paths: list[str | None]) -> bool:
        if len(tool_names) != len(file_paths):
            return False
        read_files: set[str] = set()
        write_files: set[str] = set()
        for tool_name, file_path in zip(tool_names, file_paths):
            if not file_path:
                continue
            priority = self.get_priority(tool_name)
            if priority == 0:
                read_files.add(file_path)
            elif priority == 3:
                write_files.add(file_path)
        return bool(read_files & write_files)

    def _default_name(self, item: Any) -> str:
        if isinstance(item, dict):
            if "name" in item:
                return str(item["name"])
            function = item.get("function")
            if isinstance(function, dict):
                return str(function.get("name") or "")
        return str(item)


class ContextCollapseService:
    def __init__(self, protected_tail: int = 6, threshold: int = 2_000, keep: int = 500) -> None:
        self.protected_tail = protected_tail
        self.threshold = threshold
        self.keep = keep

    def collapse_messages(self, messages: list[dict[str, Any]] | None, protected_tail: int | None = None) -> dict[str, Any]:
        if not messages:
            return {"messages": [], "collapsedCount": 0, "estimatedCharsFreed": 0}
        tail = max(protected_tail or self.protected_tail, 0)
        collapse_end = max(0, len(messages) - tail)
        collapsed: list[dict[str, Any]] = []
        collapsed_count = 0
        freed = 0
        for index, message in enumerate(messages):
            if index >= collapse_end or self._is_user_prompt(message):
                collapsed.append(message)
                continue
            replacement, before, after = self._collapse_message(message)
            collapsed.append(replacement)
            if before > after:
                collapsed_count += 1
                freed += before - after
        return {"messages": collapsed, "collapsedCount": collapsed_count, "estimatedCharsFreed": freed}

    def _is_user_prompt(self, message: dict[str, Any]) -> bool:
        return message.get("type") == "user" and not message.get("toolUseResult")

    def _collapse_message(self, message: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
        copied = dict(message)
        before = len(jsonish(message))
        if copied.get("toolUseResult"):
            copied["toolUseResult"] = "[collapsed]"
        elif isinstance(copied.get("content"), list):
            copied["content"] = [self._collapse_block(block) for block in copied["content"]]
        after = len(jsonish(copied))
        return copied, before, after

    def _collapse_block(self, block: Any) -> Any:
        if not isinstance(block, dict) or block.get("type") != "text":
            return block
        text = str(block.get("text") or "")
        if len(text) <= self.threshold:
            return block
        keep = min(self.keep, len(text))
        return {**block, "text": text[:keep] + f"\n...[collapsed: {len(text)} chars]"}


class ContextCascadeService:
    LAYER_NAMES = (
        "snip_selection",
        "micro_compact",
        "auto_compact",
        "collapse_drain",
        "reactive_compact",
    )

    def __init__(
        self,
        max_prompt_chars: int = 120_000,
        max_message_chars: int = 40_000,
        token_counter: TokenCounter | None = None,
        micro_compact: MicroCompactService | None = None,
        context_collapse: ContextCollapseService | None = None,
        prompt_recovery: PromptTooLongRecovery | None = None,
    ) -> None:
        self.max_prompt_chars = max_prompt_chars
        self.max_message_chars = max_message_chars
        self.token_counter = token_counter or TokenCounter()
        self.micro_compact = micro_compact or MicroCompactService()
        self.context_collapse = context_collapse or ContextCollapseService()
        self.prompt_recovery = prompt_recovery or PromptTooLongRecovery(max_chars=max_prompt_chars)

    def apply(
        self,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        budget: TokenBudget,
        model_id: str | None = None,
        token_char_ratio: float = DEFAULT_CHARS_PER_TOKEN,
        protected_tail: int = 6,
        force: bool = False,
        recovery_cause: str | None = None,
    ) -> dict[str, Any]:
        current_prompt = str(prompt or "")
        current_messages = [dict(message) for message in (messages or [])]
        layers: list[dict[str, Any]] = []
        events: list[RecoveryEvent] = []
        original_chars = len(current_prompt) + len(jsonish(current_messages))

        current_prompt, current_messages, layer = self._run_snip_selection(current_prompt, current_messages)
        layers.append(layer)
        if layer["changed"]:
            events.append(self._compact_event(layer))
            if (layer.get("metadata") or {}).get("promptSnipped"):
                events.append(
                    RecoveryEvent(
                        RecoveryEventType.PROMPT_TOO_LONG,
                        "Prompt exceeded cascade max chars and was compacted.",
                        metadata={
                            "layer": "snip_selection",
                            "cause": "prompt_chars_exceeded",
                            "recoveryStage": "snip_selection",
                            "originalChars": layer["beforeChars"],
                            "recoveredChars": layer["afterChars"],
                        },
                    )
                )

        before_chars = len(jsonish(current_messages))
        micro = self.micro_compact.compact_tool_results(current_messages)
        current_messages = micro["messages"]
        layers.append(
            self._layer(
                "micro_compact",
                before_chars,
                len(jsonish(current_messages)),
                bool(micro.get("compactedCount")),
                {
                    "compactedCount": int(micro.get("compactedCount") or 0),
                    "estimatedCharsFreed": int(micro.get("estimatedCharsFreed") or 0),
                },
            )
        )
        if layers[-1]["changed"]:
            events.append(self._compact_event(layers[-1]))

        should_compact = force or budget.exceeded or budget.usage_ratio() >= 0.75
        before_chars = len(jsonish(current_messages))
        if should_compact:
            collapse = self.context_collapse.collapse_messages(current_messages, protected_tail)
            current_messages = collapse["messages"]
            metadata = {
                "collapsedCount": int(collapse.get("collapsedCount") or 0),
                "estimatedCharsFreed": int(collapse.get("estimatedCharsFreed") or 0),
            }
        else:
            metadata = {"collapsedCount": 0, "estimatedCharsFreed": 0}
        layers.append(self._layer("auto_compact", before_chars, len(jsonish(current_messages)), bool(metadata["collapsedCount"]), metadata))
        if layers[-1]["changed"]:
            events.append(self._compact_event(layers[-1]))

        before_chars = len(jsonish(current_messages))
        current_used_tokens = self._estimate_tokens(current_prompt, current_messages, model_id, token_char_ratio)
        drain_needed = force or current_used_tokens > budget.max_input_tokens
        drain_metadata = {"drainedCount": 0, "estimatedCharsFreed": 0}
        if drain_needed:
            drained = self._drain_messages(current_messages, protected_tail)
            current_messages = drained["messages"]
            drain_metadata = {"drainedCount": drained["drainedCount"], "estimatedCharsFreed": drained["estimatedCharsFreed"]}
        layers.append(self._layer("collapse_drain", before_chars, len(jsonish(current_messages)), bool(drain_metadata["drainedCount"]), drain_metadata))
        if layers[-1]["changed"]:
            events.append(self._compact_event(layers[-1]))

        before_chars = len(current_prompt)
        before_message_chars = len(jsonish(current_messages))
        media_metadata = {"mediaStrippedCount": 0, "estimatedMediaTokensFreed": 0, "estimatedCharsFreed": 0}
        if recovery_cause == "http_413" or budget.exceeded:
            media_strip = self._strip_media_blocks(current_messages)
            current_messages = media_strip["messages"]
            media_metadata = {
                "mediaStrippedCount": int(media_strip.get("mediaStrippedCount") or 0),
                "estimatedMediaTokensFreed": int(media_strip.get("estimatedMediaTokensFreed") or 0),
                "estimatedCharsFreed": int(media_strip.get("estimatedCharsFreed") or 0),
            }
        reactive_budget = TokenBudget(
            contextWindow=budget.contextWindow,
            threshold=budget.threshold,
            usedTokens=self._estimate_tokens(current_prompt, current_messages, model_id, token_char_ratio),
            reservedOutputTokens=budget.reservedOutputTokens,
        )
        reactive = self._reactive_recovery_for(reactive_budget, token_char_ratio)
        recovered_prompt, recovery_event = reactive.recover(current_prompt, reactive_budget, cause=recovery_cause)
        current_prompt = recovered_prompt
        reactive_metadata = {
            "usedTokens": reactive_budget.usedTokens,
            "maxInputTokens": reactive_budget.max_input_tokens,
            "recoveryCause": recovery_cause,
            **media_metadata,
        }
        layers.append(
            self._layer(
                "reactive_compact",
                before_chars + before_message_chars,
                len(current_prompt) + len(jsonish(current_messages)),
                recovery_event is not None or bool(media_metadata["mediaStrippedCount"]),
                reactive_metadata,
            )
        )
        if recovery_event:
            recovery_event.metadata = {**recovery_event.metadata, "layer": "reactive_compact", **media_metadata}
            events.append(recovery_event)
        elif media_metadata["mediaStrippedCount"]:
            events.append(self._compact_event(layers[-1]))

        used_tokens = self._estimate_tokens(current_prompt, current_messages, model_id, token_char_ratio)
        final_chars = len(current_prompt) + len(jsonish(current_messages))
        return {
            "prompt": current_prompt,
            "messages": current_messages,
            "layers": layers,
            "events": events,
            "changed": any(layer["changed"] for layer in layers),
            "usedTokens": used_tokens,
            "estimatedCharsFreed": max(0, original_chars - final_chars),
        }

    def _run_snip_selection(self, prompt: str, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        before_chars = len(prompt) + len(jsonish(messages))
        prompt_snipped = len(prompt) > self.max_prompt_chars
        snipped_prompt = self._snip_text(prompt, self.max_prompt_chars, "prompt")
        snipped_messages = [self._snip_message(message) for message in messages]
        after_chars = len(snipped_prompt) + len(jsonish(snipped_messages))
        return snipped_prompt, snipped_messages, self._layer(
            "snip_selection",
            before_chars,
            after_chars,
            after_chars < before_chars,
            {"maxPromptChars": self.max_prompt_chars, "maxMessageChars": self.max_message_chars, "promptSnipped": prompt_snipped},
        )

    def _snip_message(self, message: dict[str, Any]) -> dict[str, Any]:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            copied["content"] = [self._snip_block(block) for block in content]
        elif isinstance(content, str):
            copied["content"] = self._snip_text(content, self.max_message_chars, "message")
        return copied

    def _snip_block(self, block: Any) -> Any:
        if not isinstance(block, dict):
            return block
        if block.get("type") != "text":
            return block
        text = str(block.get("text") or "")
        return {**block, "text": self._snip_text(text, self.max_message_chars, "message")}

    def _snip_text(self, text: str, max_chars: int, label: str) -> str:
        if len(text) <= max_chars:
            return text
        if label == "prompt":
            marker = f"\n...[prompt compacted by context cascade: {len(text)} chars]...\n"
        else:
            marker = f"\n...[{label} snipped by context cascade: {len(text)} chars]...\n"
        body_budget = max(200, max_chars - len(marker))
        head = body_budget // 2
        tail = body_budget - head
        return text[:head] + marker + text[-tail:]

    def _drain_messages(self, messages: list[dict[str, Any]], protected_tail: int) -> dict[str, Any]:
        tail = max(protected_tail, 0)
        drain_end = max(0, len(messages) - tail)
        drained: list[dict[str, Any]] = []
        drained_count = 0
        freed = 0
        for index, message in enumerate(messages):
            before = len(jsonish(message))
            if index >= drain_end or self._is_user_intent(message):
                drained.append(message)
                continue
            replacement = {
                "type": message.get("type", "assistant"),
                "content": [{"type": "text", "text": f"[context drained: {before} chars retained as compressed history]"}],
                "contextDrained": True,
            }
            after = len(jsonish(replacement))
            drained.append(replacement)
            if before > after:
                drained_count += 1
                freed += before - after
        return {"messages": drained, "drainedCount": drained_count, "estimatedCharsFreed": freed}

    def _strip_media_blocks(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        stripped: list[dict[str, Any]] = []
        stripped_count = 0
        freed_tokens = 0
        freed_chars = 0
        for message in messages:
            copied = dict(message)
            content = copied.get("content")
            if isinstance(content, list):
                new_content: list[Any] = []
                for block in content:
                    replacement, did_strip, block_tokens, block_chars = self._strip_media_block(block)
                    new_content.append(replacement)
                    if did_strip:
                        stripped_count += 1
                        freed_tokens += block_tokens
                        freed_chars += block_chars
                copied["content"] = new_content
            stripped.append(copied)
        return {
            "messages": stripped,
            "mediaStrippedCount": stripped_count,
            "estimatedMediaTokensFreed": freed_tokens,
            "estimatedCharsFreed": freed_chars,
        }

    def _strip_media_block(self, block: Any) -> tuple[Any, bool, int, int]:
        if not isinstance(block, dict):
            return block, False, 0, 0
        block_type = str(block.get("type") or "").lower()
        if block_type not in {"image", "media", "video", "audio", "file"}:
            return block, False, 0, 0
        width = int(block.get("width") or 0)
        height = int(block.get("height") or 0)
        media_tokens = self.token_counter.estimate_image_tokens(width, height) if block_type == "image" else 85
        before_chars = len(jsonish(block))
        placeholder = {
            "type": "text",
            "text": f"[media stripped during context recovery: type={block_type}, estimatedTokens={media_tokens}]",
            "mediaStripped": True,
            "originalType": block_type,
            "estimatedTokens": media_tokens,
        }
        return placeholder, True, media_tokens, max(0, before_chars - len(jsonish(placeholder)))

    def _is_user_intent(self, message: dict[str, Any]) -> bool:
        return message.get("type") == "user" or bool(message.get("toolUseResult"))

    def _reactive_recovery_for(self, budget: TokenBudget, token_char_ratio: float) -> PromptTooLongRecovery:
        if not budget.exceeded:
            return self.prompt_recovery
        target_chars = int(max(1_000, min(self.max_prompt_chars, budget.max_input_tokens * max(token_char_ratio, 1.0) * 0.6)))
        return PromptTooLongRecovery(max_chars=target_chars)

    def _estimate_tokens(self, prompt: str, messages: list[dict[str, Any]], model_id: str | None, token_char_ratio: float) -> int:
        prompt_tokens = self.token_counter.estimate_text_for_model(prompt, model_id or "default", token_char_ratio)
        message_tokens = self.token_counter.estimate_messages(messages, model_id or "default", token_char_ratio)
        return prompt_tokens + message_tokens

    def _layer(self, name: str, before_chars: int, after_chars: int, changed: bool, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "name": name,
            "changed": bool(changed),
            "beforeChars": before_chars,
            "afterChars": after_chars,
            "estimatedCharsFreed": max(0, before_chars - after_chars),
            "metadata": metadata or {},
        }

    def _compact_event(self, layer: dict[str, Any]) -> RecoveryEvent:
        return RecoveryEvent(
            RecoveryEventType.COMPACT_APPLIED,
            f"Context cascade layer applied: {layer['name']}.",
            metadata={
                "layer": layer["name"],
                "estimatedCharsFreed": layer["estimatedCharsFreed"],
                **dict(layer.get("metadata") or {}),
            },
        )


def jsonish(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


@dataclass(slots=True)
class QueryLoopState:
    id: str
    sessionId: str
    userInput: str
    model: str
    startedAt: float = field(default_factory=time.time)
    status: str = "running"
    phase: QueryPhase = QueryPhase.CREATED
    turns: int = 0
    maxTurns: int = 20
    maxOutputTokensRecoveryCount: int = 0
    maxTokensOverride: int | None = None
    autoCompactEnabled: bool = True
    autoCompactFailures: int = 0
    hasAttemptedReactiveCompact: bool = False
    stopHookActive: bool = False
    lastTransitionReason: str | None = None
    correctionAttempts: int = 0
    previousToolOutput: str | None = None
    promptTooLongWithheld: bool = False
    incrementalCollapseNeeded: bool = False
    tokenBudget: TokenBudget | None = None
    tokenBudgetBreakdown: dict[str, int] = field(default_factory=dict)
    recoveryEvents: list[RecoveryEvent] = field(default_factory=list)
    toolCalls: list[dict[str, Any]] = field(default_factory=list)
    events: list[QueryEvent] = field(default_factory=list)
    transitions: list[QueryTransition] = field(default_factory=list)
    withheldErrors: list[dict[str, Any]] = field(default_factory=list)
    terminationDecision: TerminationDecision | None = None
    contextCascade: dict[str, Any] | None = None
    stopReason: str | None = None
    error: str | None = None

    @classmethod
    def start(
        cls,
        session_id: str,
        user_input: str,
        model: str,
        context_window: int,
        threshold: float,
        ratio: float,
        used_tokens: int | None = None,
    ) -> "QueryLoopState":
        budget = TokenBudget(
            contextWindow=context_window,
            threshold=threshold,
            usedTokens=used_tokens if used_tokens is not None else estimate_tokens(user_input, ratio),
        )
        loop = cls(id=f"loop-{uuid.uuid4().hex[:12]}", sessionId=session_id, userInput=user_input, model=model, tokenBudget=budget)
        loop.transition(QueryPhase.PREPARE, "query_start")
        return loop

    def transition(self, phase: QueryPhase | str, reason: str = "") -> QueryTransition:
        next_phase = phase if isinstance(phase, QueryPhase) else QueryPhase(str(phase))
        previous = self.phase
        self.phase = next_phase
        self.lastTransitionReason = reason or next_phase.value
        transition = QueryTransition(previous.value, next_phase.value, self.lastTransitionReason)
        self.transitions.append(transition)
        if next_phase == QueryPhase.WAITING_PERMISSION:
            self.status = "waiting_permission"
        elif next_phase in {QueryPhase.COMPLETED, QueryPhase.FAILED, QueryPhase.ABORTED}:
            self.status = next_phase.value
        elif self.status not in {"failed", "aborted", "completed"}:
            self.status = "running"
        return transition

    def event(self, event_type: str, payload: dict[str, Any] | None = None, phase: QueryPhase | str | None = None) -> QueryEvent:
        phase_value = (phase.value if isinstance(phase, QueryPhase) else str(phase)) if phase else self.phase.value
        event = QueryEvent(
            type=event_type,
            sessionId=self.sessionId,
            loopId=self.id,
            phase=phase_value,
            payload=payload or {},
            seq=len(self.events) + 1,
        )
        self.events.append(event)
        return event

    def add_recovery(self, event: RecoveryEvent | None) -> None:
        if event:
            self.recoveryEvents.append(event)
            if event.type == RecoveryEventType.PROMPT_TOO_LONG:
                self.promptTooLongWithheld = True
                self.incrementalCollapseNeeded = True
                self.add_withheld_error("prompt_too_long", event.message, retryable=True)

    def release_withheld(self) -> None:
        self.promptTooLongWithheld = False
        self.incrementalCollapseNeeded = False
        self.withheldErrors.clear()

    def add_withheld_error(self, code: str, message: str, retryable: bool = True) -> None:
        self.withheldErrors.append({"code": code, "message": message, "retryable": retryable, "ts": int(time.time() * 1000)})

    def record_tool_call(self, tool_use_id: str, tool_name: str, input_payload: dict[str, Any] | None = None, status: str = "running") -> dict[str, Any]:
        call = {
            "toolUseId": tool_use_id,
            "toolName": tool_name,
            "input": input_payload or {},
            "status": status,
            "startTime": int(time.time() * 1000),
            "durationMs": None,
            "result": None,
            "progress": [],
        }
        self.toolCalls.append(call)
        return call

    def update_tool_call(self, tool_use_id: str, status: str | None = None, result: dict[str, Any] | None = None, progress: str | None = None) -> dict[str, Any] | None:
        for call in reversed(self.toolCalls):
            if call.get("toolUseId") != tool_use_id:
                continue
            if status:
                call["status"] = status
            if progress:
                call.setdefault("progress", []).append(progress)
            if result is not None:
                call["result"] = result
                content = result.get("content") if isinstance(result, dict) else None
                if isinstance(content, str) and len(content) > 500:
                    call["summary"] = ToolResultSummarizer().summarize(str(call.get("toolName") or ""), content)
                started = int(call.get("startTime") or int(time.time() * 1000))
                call["durationMs"] = max(0, int(time.time() * 1000) - started)
            return call
        return None

    def increment_recovery_count(self) -> None:
        self.maxOutputTokensRecoveryCount += 1

    def set_max_tokens_override(self, value: int | None) -> None:
        self.maxTokensOverride = value

    def effective_max_tokens(self, default_max_tokens: int) -> int:
        return self.maxTokensOverride or default_max_tokens

    def set_termination_decision(self, decision: TerminationDecision | None) -> None:
        self.terminationDecision = decision

    def finish(self, stop_reason: str = "end_turn", error: str | None = None) -> None:
        if self.phase != QueryPhase.ABORTED:
            self.transition(QueryPhase.FAILED if error else QueryPhase.COMPLETED, error or stop_reason)
        self.status = "failed" if error else ("aborted" if self.phase == QueryPhase.ABORTED else "completed")
        self.stopReason = stop_reason
        self.error = error
        self.turns += 1

    def abort(self, reason: str = "USER_INTERRUPT") -> None:
        self.transition(QueryPhase.ABORTED, reason)
        self.stopReason = "aborted"
        self.error = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sessionId": self.sessionId,
            "model": self.model,
            "startedAt": self.startedAt,
            "status": self.status,
            "phase": self.phase.value,
            "turns": self.turns,
            "maxTurns": self.maxTurns,
            "maxOutputTokensRecoveryCount": self.maxOutputTokensRecoveryCount,
            "maxTokensOverride": self.maxTokensOverride,
            "autoCompactEnabled": self.autoCompactEnabled,
            "autoCompactFailures": self.autoCompactFailures,
            "hasAttemptedReactiveCompact": self.hasAttemptedReactiveCompact,
            "stopHookActive": self.stopHookActive,
            "lastTransitionReason": self.lastTransitionReason,
            "correctionAttempts": self.correctionAttempts,
            "previousToolOutput": self.previousToolOutput,
            "promptTooLongWithheld": self.promptTooLongWithheld,
            "incrementalCollapseNeeded": self.incrementalCollapseNeeded,
            "tokenBudget": self.tokenBudget.to_dict() if self.tokenBudget else None,
            "tokenBudgetBreakdown": dict(self.tokenBudgetBreakdown),
            "recoveryEvents": [event.to_dict() for event in self.recoveryEvents],
            "toolCalls": self.toolCalls,
            "events": [event.to_dict() for event in self.events],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "withheldErrors": list(self.withheldErrors),
            "terminationDecision": self.terminationDecision.to_dict() if self.terminationDecision else None,
            "contextCascade": self.contextCascade,
            "stopReason": self.stopReason,
            "error": self.error,
        }
