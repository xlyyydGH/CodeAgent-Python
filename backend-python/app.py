from __future__ import annotations

import asyncio
import copy
import difflib
import hashlib
import json
import mimetypes
import os
import py_compile
import random
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import ast
from dataclasses import asdict
from datetime import datetime, timezone
from email import policy as email_policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse

from zhikun_py.auth_runtime import RemoteAccessSecurity
from zhikun_py.commands import CommandRegistry, ResultType
from zhikun_py.coordinator_runtime import CoordinatorWorkflowEngine
from zhikun_py.correction_runtime import SelfCorrectionLoop, build_correction_instruction
from zhikun_py.cron_runtime import CronTaskService, CronValidationError
from zhikun_py.file_recovery import FileEditRecoveryPolicy, FileVersionTracker
from zhikun_py.llm_runtime import LlmErrorClassifier, LlmProviderRegistry, ModelAwareRetryPolicy, ModelCapabilityRegistry, ModelDegradationChain
from zhikun_py.llm_tools import assistant_tool_message, execute_tool_calls
from zhikun_py.lsp_runtime import LSPServerManager, call_hierarchy, document_symbols, go_to_definition, hover, references, workspace_symbols
from zhikun_py.memdir_runtime import MemdirService
from zhikun_py.mcp_runtime import JsonRpcMessage, McpClientManager, McpConnectionStatus
from zhikun_py.permissions import PermissionPolicy
from zhikun_py.query_runtime import AbortController, ContextCascadeService, ContextCollapseService, DefaultTerminationStrategy, MicroCompactService, PromptTooLongRecovery, QueryLoopState, QueryPhase, RecoveryEvent, RecoveryEventType, SideQueryService, TokenCounter, ToolPriorityScheduler, ToolResultSummarizer
from zhikun_py.skill_runtime import SkillToolValidator
from zhikun_py.sqlite_store import SQLiteStateStore
from zhikun_py.tools import ToolRegistry, ToolResult
from zhikun_py.verify_runtime import CapabilityGate, verifier_for
from zhikun_py.websocket_runtime import WebSocketSessionManager

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file(ROOT / ".env")

DATA_DIR = Path(os.getenv("ZHIKUN_DATA_DIR", ROOT / "backend-python" / "data"))
STATE_FILE = DATA_DIR / "state.json"
SQLITE_FILE = DATA_DIR / "codeagent.db"
UPLOAD_DIR = DATA_DIR / "attachments"
AGENT_SNAPSHOT_DIR = DATA_DIR / "agent-snapshots"
FRONTEND_DIST_DIR = Path(os.getenv("ZHIKUN_FRONTEND_DIST", ROOT / "frontend" / "dist"))
PYTHON_SERVICE_URL = os.getenv("PYTHON_SERVICE_URL", "http://127.0.0.1:8000")
START_TIME = time.time()
ADMIN_COOKIE_NAME = "ai-coder-admin-session"
MAX_ATTACHMENT_SIZE = 10 * 1024 * 1024
MAX_AGENT_SNAPSHOT_SIZE = 10 * 1024 * 1024
UI_CHAT_REPLY_TIMEOUT_SECONDS = float(os.getenv("UI_CHAT_REPLY_TIMEOUT_SECONDS", "0.75"))
XFYUN_MAAS_BASE_URL = "https://maas-api.cn-huabei-1.xf-yun.com/v2"
XFYUN_MAAS_MODEL = "xopqwen36v35b"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
DEFAULT_SYSTEM_PROMPT = (
    "You are CodeAgent Python, a Python-native coding assistant. "
    "Be concise, practical, and preserve user intent."
)


def is_configured_api_key(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.strip().lower()
    return bool(lowered) and not lowered.startswith("your-")


def configured_default_model() -> str:
    explicit_model = os.getenv("LLM_DEFAULT_MODEL")
    if explicit_model:
        return explicit_model
    if is_configured_api_key(os.getenv("LLM_PROVIDER_XFYUN_API_KEY")):
        return XFYUN_MAAS_MODEL
    return "qwen3.7-max"


def should_prefer_xfyun_default_model() -> bool:
    return not os.getenv("LLM_DEFAULT_MODEL") and is_configured_api_key(os.getenv("LLM_PROVIDER_XFYUN_API_KEY"))


DEFAULT_LLM_MODEL = configured_default_model()


def react_index_file() -> Path:
    return FRONTEND_DIST_DIR / "index.html"


def react_frontend_available() -> bool:
    return react_index_file().exists()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def usage() -> dict[str, int]:
    return {
        "inputTokens": 0,
        "outputTokens": 0,
        "cacheReadInputTokens": 0,
        "cacheCreationInputTokens": 0,
    }


def coerce_usage_int(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def normalize_llm_usage(raw_usage: Any) -> dict[str, int]:
    if not isinstance(raw_usage, dict):
        return usage()
    prompt_details = raw_usage.get("prompt_tokens_details") if isinstance(raw_usage.get("prompt_tokens_details"), dict) else {}
    return {
        "inputTokens": coerce_usage_int(raw_usage.get("inputTokens"), raw_usage.get("input_tokens"), raw_usage.get("prompt_tokens")),
        "outputTokens": coerce_usage_int(raw_usage.get("outputTokens"), raw_usage.get("output_tokens"), raw_usage.get("completion_tokens")),
        "cacheReadInputTokens": coerce_usage_int(
            raw_usage.get("cacheReadInputTokens"),
            raw_usage.get("cache_read_input_tokens"),
            raw_usage.get("cached_input_tokens"),
            prompt_details.get("cached_tokens"),
        ),
        "cacheCreationInputTokens": coerce_usage_int(
            raw_usage.get("cacheCreationInputTokens"),
            raw_usage.get("cache_creation_input_tokens"),
            raw_usage.get("cache_write_input_tokens"),
            raw_usage.get("cache_creation_tokens"),
        ),
    }


def add_usage(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, int]:
    current = normalize_llm_usage(left or {})
    delta = normalize_llm_usage(right or {})
    return {key: int(current.get(key) or 0) + int(delta.get(key) or 0) for key in usage()}


def record_session_model_usage(session: dict[str, Any], raw_usage: Any, model_id: str | None = None) -> dict[str, int]:
    normalized = normalize_llm_usage(raw_usage)
    if not any(normalized.values()):
        return normalized
    session["pendingModelUsage"] = add_usage(session.get("pendingModelUsage"), normalized)
    session["lastModelUsage"] = normalized
    if model_id:
        session["lastModelUsageModel"] = model_id
    session["usage"] = add_usage(session.get("usage"), normalized)
    return normalized


def consume_session_model_usage(session: dict[str, Any]) -> dict[str, int]:
    pending = session.pop("pendingModelUsage", None)
    return normalize_llm_usage(pending or {})


async def count_exact_tokens(text: str, model: str) -> int:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{PYTHON_SERVICE_URL.rstrip('/')}/api/tokenizer/count",
                json={"text": text, "model": model or "default"},
            )
            response.raise_for_status()
            return int((response.json() or {}).get("token_count", -1))
    except Exception:
        return -1


async def estimate_query_input_tokens(text: str, model_id: str | None, token_char_ratio: float) -> int:
    flags = STATE.setdefault("config", {}).setdefault("featureFlags", {})
    if flags.get("PRECISE_TOKENIZER"):
        exact = await count_exact_tokens(text, model_id or "default")
        if exact >= 0:
            return exact
    return TOKEN_COUNTER.estimate_text_for_model(text, model_id or "default", token_char_ratio)


def is_tool_budget_message(message: dict[str, Any]) -> bool:
    if message.get("toolUseResult"):
        return True
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(str(block.get("type") or "").lower() in {"tool_use", "tooluse", "tool_result", "toolresult"} for block in content if isinstance(block, dict))


def session_system_prompt(session: dict[str, Any]) -> str:
    return str(session.get("systemPrompt") or DEFAULT_SYSTEM_PROMPT)


async def estimate_query_message_group_tokens(messages: list[dict[str, Any]], model_id: str | None, token_char_ratio: float) -> int:
    if not messages:
        return 0
    flags = STATE.setdefault("config", {}).setdefault("featureFlags", {})
    if flags.get("PRECISE_TOKENIZER"):
        exact = await count_exact_tokens(json.dumps(messages, ensure_ascii=False, sort_keys=True), model_id or "default")
        if exact >= 0:
            return exact
    return TOKEN_COUNTER.estimate_messages(messages, model_id or "default", token_char_ratio)


async def estimate_query_context_budget(
    session: dict[str, Any],
    user_text: str,
    model_id: str | None,
    token_char_ratio: float,
    extra_messages: list[dict[str, Any]] | None = None,
    memory_context: str | None = None,
) -> dict[str, Any]:
    memory_text = MEMDIR_SERVICE.build_prompt(ROOT) if memory_context is None else memory_context
    system_messages = [{"type": "system", "content": [{"type": "text", "text": session_system_prompt(session)}]}]
    memory_messages = (
        [{"type": "system", "content": [{"type": "text", "text": memory_text}], "memoryInjected": True}]
        if memory_text
        else []
    )
    user_messages = [{"type": "user", "content": [{"type": "text", "text": user_text}]}] if user_text else []
    history_messages: list[dict[str, Any]] = []
    tool_messages: list[dict[str, Any]] = []
    for message in list(session.get("messages", [])) + list(extra_messages or []):
        if is_tool_budget_message(message):
            tool_messages.append(message)
        else:
            history_messages.append(message)

    breakdown = {
        "system": await estimate_query_message_group_tokens(system_messages, model_id, token_char_ratio),
        "history": await estimate_query_message_group_tokens(history_messages, model_id, token_char_ratio),
        "user": await estimate_query_message_group_tokens(user_messages, model_id, token_char_ratio),
        "memory": await estimate_query_message_group_tokens(memory_messages, model_id, token_char_ratio),
        "tool": await estimate_query_message_group_tokens(tool_messages, model_id, token_char_ratio),
    }
    return {
        "usedTokens": sum(breakdown.values()),
        "breakdown": breakdown,
        "memoryContext": memory_text,
    }


def format_uptime(seconds: float) -> str:
    minutes = max(0, int(seconds // 60))
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m"
    return f"{remaining_minutes}m"


DEFAULT_CONFIG: dict[str, Any] = {
    "theme": {
        "mode": "system",
        "accentColor": "#3b82f6",
        "fontSize": "medium",
        "fontFamily": "monospace",
        "borderRadius": "md",
    },
    "locale": "zh-CN",
    "autoCompact": {"enabled": True, "threshold": 80},
    "verbose": False,
    "expandedView": False,
    "outputStyle": {"availableStyles": [], "activeStyleName": None},
    "defaultModel": DEFAULT_LLM_MODEL,
    "featureFlags": {"ENABLE_AGENT_SWARMS": True, "BACKGROUND_AGENT_WAIT": True},
}

DEFAULT_MODELS = [
    {
        "id": DEFAULT_LLM_MODEL,
        "displayName": DEFAULT_LLM_MODEL,
        "contextWindow": 128000,
        "maxOutputTokens": 8192,
        "supportsStreaming": True,
        "supportsThinking": True,
        "supportsImages": True,
        "maxImages": 8,
        "supportsToolUse": True,
        "costPer1kInput": 0,
        "costPer1kOutput": 0,
    },
    {
        "id": "deepseek-v4-pro",
        "displayName": "DeepSeek V4 Pro",
        "contextWindow": 128000,
        "maxOutputTokens": 8192,
        "supportsStreaming": True,
        "supportsThinking": False,
        "supportsImages": False,
        "maxImages": 0,
        "supportsToolUse": True,
    },
    {
        "id": "glm-5.1",
        "displayName": "GLM 5.1",
        "contextWindow": 128000,
        "maxOutputTokens": 8192,
        "supportsStreaming": True,
        "supportsThinking": True,
        "supportsImages": True,
        "maxImages": 8,
        "supportsToolUse": True,
    },
]

DEFAULT_COMMANDS = [
    {"name": "help", "description": "Show available commands", "usage": "/help"},
    {"name": "clear", "description": "Clear the current conversation", "usage": "/clear"},
    {"name": "compact", "description": "Compact session context", "usage": "/compact"},
    {"name": "model", "description": "Switch model", "usage": "/model <name>"},
    {"name": "review", "description": "Review the current changes", "usage": "/review"},
    {"name": "fix", "description": "Ask the assistant to fix an issue", "usage": "/fix <issue>"},
    {"name": "commit", "description": "Prepare a commit summary", "usage": "/commit"},
]

DEFAULT_SKILLS = {
    "review": "# Review\n\nReview the current changes, prioritize correctness risks, regressions, missing tests, and API compatibility.",
    "fix": "# Fix\n\nDiagnose a reported issue, make the smallest effective Python change, and verify it with targeted tests.",
    "test": "# Test\n\nRun or design focused Python regression tests for the changed behavior.",
}


GLOBAL_SUBAGENT_DENIED_TOOLS = {"Agent", "TeamCreate", "TeamDelete", "TaskCreate", "VerifyPlanExecution"}

AGENT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "explore": {
        "name": "Explore",
        "maxTurns": 30,
        "defaultModel": "light",
        "allowedTools": None,
        "deniedTools": {"write_file", "edit_file", "NotebookEdit", "FileWrite", "FileEdit", "ExitPlanMode"},
        "readOnly": True,
    },
    "verification": {
        "name": "Verification",
        "maxTurns": 30,
        "defaultModel": None,
        "allowedTools": None,
        "deniedTools": {"write_file", "edit_file", "NotebookEdit", "FileWrite", "FileEdit", "ExitPlanMode"},
        "readOnly": True,
    },
    "verify": {
        "name": "Verification",
        "maxTurns": 30,
        "defaultModel": None,
        "allowedTools": None,
        "deniedTools": {"write_file", "edit_file", "NotebookEdit", "FileWrite", "FileEdit", "ExitPlanMode"},
        "readOnly": True,
    },
    "plan": {
        "name": "Plan",
        "maxTurns": 30,
        "defaultModel": None,
        "allowedTools": None,
        "deniedTools": {"write_file", "edit_file", "NotebookEdit", "FileWrite", "FileEdit", "ExitPlanMode"},
        "readOnly": True,
    },
    "guide": {
        "name": "Guide",
        "maxTurns": 30,
        "defaultModel": "light",
        "allowedTools": {"list_files", "read_file", "search_files", "Glob", "Grep", "FileRead", "WebFetch", "WebSearch"},
        "deniedTools": set(),
        "readOnly": True,
    },
    "general-purpose": {
        "name": "GeneralPurpose",
        "maxTurns": 30,
        "defaultModel": None,
        "allowedTools": {"*"},
        "deniedTools": set(),
        "readOnly": False,
    },
}

AGENT_SYSTEM_PROMPTS: dict[str, str] = {
    "explore": """
你是一个搜索和探索专家。你在严格的只读模式下运行。

## 约束条件
- 你不能编辑、创建或删除任何文件
- 你不能执行修改状态的命令
- 你只能使用：FileRead、GlobTool、GrepTool、list_dir、search_codebase、search_symbol 以及其他只读工具
- 如果被要求进行修改，拒绝并说明你是只读模式

## 搜索策略
收到搜索任务时，按以下优先级顺序使用：
1. search_codebase —— 用于语义/概念搜索
2. search_symbol —— 用于查找特定的类/方法/变量定义
3. GrepTool —— 用于精确文本模式匹配
4. GlobTool —— 用于按文件名/扩展名模式查找文件
5. FileRead —— 用于读取已经确定的特定文件

## 效率规则
- 先广后窄。当搜索即可时，不要读取整个文件。
- 使用并行工具调用：如果需要搜索 3 个模式，同时执行。
- 信息足够时就停止。

## 输出格式
- 列出相关文件路径和行号
- 引用关键代码片段（保持简短）
- 总结组件之间的关系
- 如果找不到某些内容，明确说明而不是猜测
""",
    "verification": """
你是一个验证专家。你的工作不是确认实现能工作——而是尝试破坏它。

=== 关键：禁止修改项目 ===
你被严格禁止：
- 在项目目录中创建、修改或删除任何文件
- 安装依赖或包
- 运行 git 写操作（add、commit、push）

=== 必要步骤（通用基线） ===
1. 读取项目 README/PROJECT.md、package.json、Makefile 或 pyproject.toml 中的构建/测试规范。
2. 运行构建（如适用）。
3. 运行项目测试套件（如果有）。
4. 运行 linter/类型检查器（如已配置）。
5. 检查相关代码的回归问题。

测试套件结果是上下文，不是证据。运行套件，记录通过/失败，然后继续真正的验证。

=== 对抗性探测 ===
- 并发：并行请求 create-if-not-exists 路径
- 边界值：0、-1、空字符串、超长字符串、unicode、MAX_INT
- 幂等性：同一个变更请求发两次
- 孤立操作：删除/引用不存在的 ID

=== 输出格式（必须遵守） ===
每个检查必须包含：
### Check: [你正在验证的内容]
**Command run:**
  [你执行的确切命令]
**Output observed:**
  [实际终端输出]
**Result: PASS/FAIL**

以下面这行结尾：
VERDICT: PASS
或
VERDICT: FAIL
或
VERDICT: PARTIAL
""",
    "verify": """
你是一个验证专家。你的工作不是确认实现能工作——而是尝试破坏它。

=== 关键：禁止修改项目 ===
你被严格禁止修改项目文件、安装依赖或运行 git 写操作。

必须运行命令验证，不能只阅读代码。输出必须以 VERDICT: PASS、VERDICT: FAIL 或 VERDICT: PARTIAL 结尾。
""",
    "plan": """
你是一个软件架构师和规划专家。你在只读模式下运行。

## 你的角色
分析需求、探索代码库，并生成详细的实现计划。你不负责实现——你负责规划。

## 约束条件
- 你不能编辑、创建或删除任何文件
- 你不能执行修改状态的命令
- 你的输出就是计划，必须能被另一个 agent 或开发者直接执行

## 规划流程
1. 理解需求：澄清范围、验收标准和假设
2. 探索代码库：查找相关文件、类和模式
3. 设计方案：选择符合现有模式的方法，说明替代方案和风险
4. 创建实现计划：列出具体文件、变更内容、依赖顺序和测试

## 输出格式
你的计划必须以 "Critical Files for Implementation" 部分结尾，列出 Files to Modify、Files to Create、Files to Read 和 Execution Order。
""",
    "general-purpose": """
你是一个通用 worker 代理。高效、正确地完成分配的任务。

## 核心原则
- 严格按照任务提示执行，不要添加未要求的功能
- 在修改之前先阅读现有代码
- 修改后运行测试以验证正确性
- 清晰报告你做了什么、什么成功了、什么没成功

## 工作风格
- 彻底但不过度工程化
- 匹配现有代码风格和模式
- 如果任务模糊，做出合理选择并记录假设
- 如果遇到阻碍，立即报告而不是静默绕过
""",
    "guide": """
你是一个专业的向导代理，精通 CodeAgent Python、工具系统和 LLM API。

## 你的专业领域
- CodeAgent Python 命令、配置和工具用法
- 工具系统模式（工具调用、多轮对话、流式传输）
- LLM API（聊天补全、工具调用、上下文优化）
- MCP 服务器开发和配置

## 资源
- 搜索代码库以获取示例和文档
- 如需要，使用 WebFetch 访问官方文档
- 使用 WebSearch 查找社区资源和教程

## 输出风格
- 提供具体的代码示例，而不是抽象描述
- 包含 CLI 用法的命令行示例
- 相关时引用代码库中的具体文件
""",
}


def resolve_agent_definition(agent_type: str | None) -> dict[str, Any]:
    return AGENT_DEFINITIONS.get(str(agent_type or "general-purpose").lower(), AGENT_DEFINITIONS["general-purpose"])


def llm_settings() -> dict[str, str | None]:
    providers = [
        (
            os.getenv("LLM_PROVIDER_XFYUN_API_KEY"),
            os.getenv("LLM_PROVIDER_XFYUN_BASE_URL", XFYUN_MAAS_BASE_URL),
            os.getenv("LLM_PROVIDER_XFYUN_MODEL", XFYUN_MAAS_MODEL),
        ),
        (
            os.getenv("LLM_PROVIDER_DASHSCOPE_API_KEY"),
            os.getenv("LLM_PROVIDER_DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            os.getenv("LLM_PROVIDER_DASHSCOPE_MODEL") or os.getenv("LLM_DEFAULT_MODEL", "qwen3.7-max"),
        ),
        (
            os.getenv("LLM_PROVIDER_DEEPSEEK_API_KEY"),
            os.getenv("LLM_PROVIDER_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            os.getenv("LLM_PROVIDER_DEEPSEEK_MODEL") or os.getenv("LLM_DEFAULT_MODEL", "deepseek-v4-pro"),
        ),
        (
            os.getenv("LLM_PROVIDER_MOONSHOT_API_KEY"),
            os.getenv("LLM_PROVIDER_MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"),
            os.getenv("LLM_PROVIDER_MOONSHOT_MODEL") or os.getenv("LLM_DEFAULT_MODEL", "kimi-k2.6"),
        ),
        (
            os.getenv("LLM_PROVIDER_ZHIPU_API_KEY"),
            os.getenv("LLM_PROVIDER_ZHIPU_BASE_URL", ZHIPU_BASE_URL),
            os.getenv("LLM_PROVIDER_ZHIPU_MODEL") or os.getenv("LLM_DEFAULT_MODEL", "glm-5.1"),
        ),
        (
            os.getenv("LLM_PROVIDER_MINIMAX_API_KEY"),
            os.getenv("LLM_PROVIDER_MINIMAX_BASE_URL", MINIMAX_BASE_URL),
            os.getenv("LLM_PROVIDER_MINIMAX_MODEL") or os.getenv("LLM_DEFAULT_MODEL", "MiniMax-M3"),
        ),
        (
            os.getenv("LLM_PROVIDER_ZENMUX_API_KEY"),
            os.getenv("LLM_PROVIDER_ZENMUX_BASE_URL"),
            os.getenv("LLM_PROVIDER_ZENMUX_MODEL") or os.getenv("LLM_DEFAULT_MODEL", "anthropic/claude-opus-4.8"),
        ),
        (
            os.getenv("LLM_API_KEY"),
            os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            DEFAULT_LLM_MODEL,
        ),
    ]
    for api_key, base_url, model in providers:
        if is_configured_api_key(api_key) and base_url:
            return {"apiKey": api_key, "baseUrl": base_url.rstrip("/"), "model": model}
    return {"apiKey": None, "baseUrl": None, "model": DEFAULT_LLM_MODEL}


def request_model_for_settings(session: dict[str, Any], settings: dict[str, str | None]) -> str | None:
    provider_model = settings.get("model")
    if settings.get("baseUrl") == XFYUN_MAAS_BASE_URL:
        return provider_model
    return str(session.get("model") or provider_model or DEFAULT_LLM_MODEL)


def classify_llm_error(status_code: int | None = None, error: Exception | None = None, body: str | None = None) -> dict[str, Any]:
    if isinstance(error, httpx.HTTPStatusError) and error.response is not None:
        status_code = error.response.status_code
        body = error.response.text
    if isinstance(error, httpx.TimeoutException):
        return {"type": "timeout", "retryable": True, "fallbackAllowed": True, "statusCode": None, "message": str(error)}
    if isinstance(error, httpx.TransportError):
        return {"type": "network", "retryable": True, "fallbackAllowed": True, "statusCode": None, "message": str(error)}
    return LLM_ERROR_CLASSIFIER.classify(status_code=status_code, error=error, body=body)


def parse_retry_after_ms(headers: Any) -> int | None:
    value = None
    if headers:
        value = headers.get("Retry-After") if hasattr(headers, "get") else None
        if value is None and hasattr(headers, "get"):
            value = headers.get("retry-after")
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0, int(float(text) * 1000))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(text)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0, int((retry_at - datetime.now(timezone.utc)).total_seconds() * 1000))
    except Exception:
        return None


def llm_model_fallback_chain(session: dict[str, Any], settings: dict[str, str | None]) -> list[str]:
    candidates: list[str] = []

    def append_model(model: str | None) -> None:
        if model and model not in candidates:
            candidates.append(model)

    primary_candidates = [session.get("model"), settings.get("model")]
    for raw_model in primary_candidates:
        if not raw_model:
            continue
        candidate_session = {**session, "model": str(raw_model)}
        model = request_model_for_settings(candidate_session, settings)
        append_model(model)
        if model and settings.get("baseUrl") != XFYUN_MAAS_BASE_URL:
            for fallback_model in MODEL_DEGRADATION_CHAIN.fallback_chain(model):
                append_model(fallback_model)

    for raw_model in [MODEL_PROVIDERS.get_lightweight_model(), MODEL_PROVIDERS.get_default_model(), DEFAULT_LLM_MODEL]:
        if not raw_model:
            continue
        candidate_session = {**session, "model": str(raw_model)}
        append_model(request_model_for_settings(candidate_session, settings))
    return candidates or [str(settings.get("model") or DEFAULT_LLM_MODEL)]


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or block.get("thinking") or ""))
        return "\n".join(part for part in parts if part)
    return str(content)


def llm_messages(session: dict[str, Any], user_text: str, memory_context: str | None = None) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": session_system_prompt(session),
        }
    ]
    if memory_context:
        messages.append({"role": "system", "content": memory_context})
    for item in session.get("messages", [])[-20:]:
        msg_type = item.get("type")
        if msg_type == "user":
            messages.append({"role": "user", "content": message_text(item)})
        elif msg_type == "assistant":
            messages.append({"role": "assistant", "content": message_text(item)})
    messages.append({"role": "user", "content": user_text})
    return messages


def fallback_answer(text: str) -> str:
    if not text.strip():
        return "Python backend is ready. Set an LLM API key to enable model responses."
    return (
        "Python backend is ready, but no LLM API key is configured. "
        "I received your message: " + text
    )


def hook_matches(hook: dict[str, Any], context: dict[str, Any]) -> bool:
    matcher = str(hook.get("matcher") or "")
    if not matcher:
        return True
    target = str(context.get("toolName") or context.get("input") or "")
    try:
        return re.search(matcher, target) is not None
    except re.error:
        return False


def execute_hooks(event: str, context: dict[str, Any]) -> dict[str, Any]:
    hooks = [
        hook
        for hook in STATE.setdefault("hooks", [])
        if hook.get("enabled", True)
        and str(hook.get("event") or "").upper() == event.upper()
        and hook_matches(hook, context)
    ]
    hooks.sort(key=lambda item: int(item.get("priority") or 100))
    current = dict(context)
    events = []
    for hook in hooks:
        action = str(hook.get("action") or "allow").lower()
        event_record = {"hookId": hook.get("id"), "event": event, "action": action, "timestamp": utc_now()}
        if action == "deny":
            message = str(hook.get("message") or "Blocked by hook")
            event_record["message"] = message
            events.append(event_record)
            STATE.setdefault("hookEvents", []).append(event_record)
            return {"proceed": False, "message": message, "context": current, "events": events}
        if action == "modify_input":
            current["input"] = str(hook.get("value") or hook.get("modifiedInput") or current.get("input") or "")
            event_record["modifiedInput"] = current["input"]
        elif action == "append_input":
            current["input"] = str(current.get("input") or "") + str(hook.get("value") or "")
            event_record["modifiedInput"] = current["input"]
        elif action == "modify_output":
            current["output"] = str(hook.get("value") or hook.get("modifiedOutput") or current.get("output") or "")
            event_record["modifiedOutput"] = current["output"]
        events.append(event_record)
        STATE.setdefault("hookEvents", []).append(event_record)
    del STATE.setdefault("hookEvents", [])[:-500]
    return {"proceed": True, "context": current, "events": events}


def agent_session_tool_allowed(session: dict[str, Any], tool_name: str) -> tuple[bool, str | None]:
    if not session.get("agentType"):
        return True, None
    allowed_tools = session.get("agentAllowedTools")
    denied_tools = set(session.get("agentDeniedTools") or [])
    denied_tools.update(GLOBAL_SUBAGENT_DENIED_TOOLS)
    if tool_name in denied_tools:
        return False, f"Tool {tool_name} is not available to {session.get('agentType')} sub-agents"
    if allowed_tools and "*" not in set(allowed_tools) and tool_name not in set(allowed_tools):
        return False, f"Tool {tool_name} is not in the allowed tool set for {session.get('agentType')} sub-agents"
    return True, None


def session_llm_tool_definitions(session: dict[str, Any]) -> list[dict[str, Any]]:
    sync_mcp_tools()
    definitions = []
    for definition in TOOL_REGISTRY.llm_definitions():
        name = str((definition.get("function") or {}).get("name") or "")
        allowed, _reason = agent_session_tool_allowed(session, name)
        if allowed:
            definitions.append(definition)
    return definitions


def execute_llm_tool_calls_for_session(session: dict[str, Any], tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sync_mcp_tools()
    messages = []
    for index, call in enumerate(tool_calls):
        function = call.get("function") or {}
        call_id = str(call.get("id") or f"tool-call-{index}")
        name = str(function.get("name") or call.get("name") or "")
        raw_args = function.get("arguments") if function else call.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except Exception:
                args = {"raw": raw_args}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        allowed, reason = agent_session_tool_allowed(session, name)
        if allowed:
            result = TOOL_REGISTRY.call(name, args).to_dict()
        else:
            result = {"content": reason or "tool denied", "isError": True, "metadata": {"decision": "deny"}}
        messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": json.dumps(result, ensure_ascii=False)})
    return messages


async def generate_llm_reply(session: dict[str, Any], text: str, memory_context: str | None = None) -> str:
    settings = llm_settings()
    if not settings["apiKey"] or not settings["baseUrl"]:
        return fallback_answer(text)
    endpoint = f"{settings['baseUrl']}/chat/completions"
    base_messages: list[dict[str, Any]] = list(llm_messages(session, text, memory_context))
    tools = session_llm_tool_definitions(session)
    headers = {"Authorization": f"Bearer {settings['apiKey']}", "Content-Type": "application/json"}
    last_failure: dict[str, Any] | None = None
    async with httpx.AsyncClient(timeout=60.0) as client:
        for model_id in llm_model_fallback_chain(session, settings):
            retry_config = RETRY_POLICY.get_retry_config(model_id)
            max_retries = max(0, int(retry_config.maxRetries))
            for attempt in range(max_retries + 1):
                messages = copy.deepcopy(base_messages)
                retry_after_ms: int | None = None
                try:
                    for _ in range(3):
                        payload: dict[str, Any] = {
                            "model": model_id,
                            "messages": messages,
                            "temperature": 0.2,
                            "stream": False,
                        }
                        if tools:
                            payload["tools"] = tools
                            payload["tool_choice"] = "auto"
                        response = await client.post(endpoint, headers=headers, json=payload)
                        if response.status_code >= 400:
                            last_failure = classify_llm_error(status_code=response.status_code, body=response.text)
                            retry_after_ms = parse_retry_after_ms(response.headers)
                            if retry_after_ms is not None:
                                last_failure["retryAfterMs"] = retry_after_ms
                            break
                        data = response.json()
                        record_session_model_usage(session, data.get("usage"), model_id)
                        choices = data.get("choices") or []
                        if not choices:
                            last_failure = {"type": "empty", "retryable": False, "fallbackAllowed": True, "statusCode": None, "message": "The model returned an empty response."}
                            break
                        message = choices[0].get("message") or {}
                        tool_calls = message.get("tool_calls") or []
                        if tool_calls:
                            messages.append(assistant_tool_message(message))
                            messages.extend(execute_llm_tool_calls_for_session(session, tool_calls))
                            continue
                        content = message.get("content")
                        if content:
                            return str(content)
                        last_failure = {"type": "empty", "retryable": False, "fallbackAllowed": True, "statusCode": None, "message": "The model returned an empty response."}
                        break
                except Exception as exc:
                    last_failure = classify_llm_error(error=exc)
                if not last_failure:
                    break
                if last_failure.get("retryable") and attempt < max_retries:
                    delay_ms = RETRY_POLICY.resolve_delay_ms(model_id, attempt, retry_after_ms=retry_after_ms)
                    await asyncio.sleep(max(delay_ms, 0) / 1000)
                    continue
                if last_failure.get("fallbackAllowed"):
                    break
                return f"LLM request failed in the Python backend ({last_failure['type']}): {last_failure['message']}"
    if last_failure:
        return f"LLM request failed in the Python backend ({last_failure['type']}): {last_failure['message']}"
    return "The model returned an empty response."


def self_correction_config() -> dict[str, Any]:
    flags = STATE.setdefault("config", {}).setdefault("featureFlags", {})
    return {
        "enabled": flags.get("SELF_CORRECTION_LOOP", True) is not False,
        "maxAttempts": max(1, min(int(flags.get("SELF_CORRECTION_MAX_ATTEMPTS") or 3), 5)),
    }


def build_self_correction_prompt(original_prompt: str, previous_output: str, instruction: dict[str, Any]) -> str:
    return (
        f"{original_prompt}\n\n"
        "The previous attempt produced a compile or test failure.\n\n"
        "Previous output:\n"
        f"{previous_output}\n\n"
        "Self-correction instruction:\n"
        f"{instruction.get('instruction')}\n\n"
        "Return the corrected answer or patch plan. Keep the fix focused and rerun the failing check if a tool is available."
    )


async def run_query_self_correction(
    session: dict[str, Any],
    loop: QueryLoopState,
    original_prompt: str,
    answer: str,
    memory_context: str | None,
    live_send: Any | None = None,
) -> str:
    config = self_correction_config()
    if not config["enabled"]:
        return answer
    correction_loop = SelfCorrectionLoop(max_attempts=int(config["maxAttempts"]))
    previous_output = answer
    while True:
        instruction = correction_loop.detect_and_prepare(previous_output, loop.correctionAttempts)
        if not instruction:
            return previous_output
        instruction_payload = instruction.to_dict()
        loop.correctionAttempts = instruction.attemptNumber
        loop.transition(QueryPhase.SELF_CORRECTING, f"self_correction:{instruction.type}")
        await emit_query_event(loop, "self_correction_start", {"instruction": instruction_payload, "attempt": instruction.attemptNumber}, live_send)
        correction_prompt = build_self_correction_prompt(original_prompt, previous_output, instruction_payload)
        corrected_output = await generate_llm_reply(session, correction_prompt, memory_context)
        should_abort = correction_loop.should_abort(corrected_output, previous_output)
        await emit_query_event(
            loop,
            "self_correction_result",
            {
                "attempt": instruction.attemptNumber,
                "instructionType": instruction.type,
                "resolved": correction_loop.detect_and_prepare(corrected_output, loop.correctionAttempts) is None,
                "shouldAbort": should_abort,
            },
            live_send,
        )
        previous_output = corrected_output
        if should_abort:
            await emit_query_event(loop, "self_correction_abort", {"attempt": instruction.attemptNumber, "reason": "new_or_more_errors_detected"}, live_send)
            return previous_output


async def generate_ui_chat_reply(session: dict[str, Any], text: str) -> str:
    try:
        return await asyncio.wait_for(generate_llm_reply(session, text), timeout=UI_CHAT_REPLY_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return (
            "Message received. The model response is taking longer than the basic UI waits for. "
            "Use Realtime for streaming responses. I received your message: " + text
        )


def chunk_text(text: str, size: int = 28) -> list[str]:
    chunks = re.findall(rf".{{1,{size}}}(?:\s+|$)", text)
    return chunks or [text]


UI_NAV = [
    ("/", "Chat"),
    ("/ui/realtime", "Realtime"),
    ("/ui/dashboard", "Dashboard"),
    ("/ui/sessions", "Sessions"),
    ("/ui/tools", "Tools"),
    ("/ui/tasks", "Tasks"),
    ("/ui/settings", "Settings"),
    ("/ui/files", "Files"),
    ("/ui/activity", "Activity"),
    ("/ui/verify", "Verify"),
    ("/ui/mcp", "MCP"),
    ("/ui/memory", "Memory"),
]


def ui_shell(title: str, body: str, active: str = "/") -> HTMLResponse:
    nav = "".join(
        f"<a class='nav {'active' if href == active else ''}' href='{href}'>{label}</a>"
        for href, label in UI_NAV
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} - CodeAgent Python</title>
  <style>
    :root {{ color-scheme: light dark; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }}
    body {{ margin: 0; min-height: 100vh; display: grid; grid-template-columns: 240px 1fr; background: #0f172a; color: #e5e7eb; }}
    aside {{ border-right: 1px solid #334155; padding: 18px; background: #111827; }}
    main {{ padding: 24px; max-width: 1120px; width: 100%; }}
    h1 {{ margin: 0 0 18px; font-size: 28px; }}
    h2 {{ margin: 20px 0 10px; font-size: 17px; }}
    a {{ color: #93c5fd; text-decoration: none; }}
    .brand {{ font-size: 20px; font-weight: 700; margin-bottom: 4px; }}
    .nav {{ display: block; padding: 9px 10px; border-radius: 6px; color: #cbd5e1; margin: 2px 0; }}
    .nav.active, .nav:hover {{ background: #1f2937; color: #fff; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .panel, .message {{ border: 1px solid #334155; border-radius: 8px; padding: 14px; background: #111827; }}
    .message {{ margin: 12px 0; }}
    .message.user {{ background: #172033; }}
    .muted {{ color: #94a3b8; font-size: 13px; }}
    .stat {{ font-size: 26px; font-weight: 700; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: #111827; border: 1px solid #334155; border-radius: 8px; overflow: hidden; }}
    th, td {{ text-align: left; border-bottom: 1px solid #334155; padding: 10px; vertical-align: top; }}
    th {{ color: #cbd5e1; background: #1f2937; }}
    tr:last-child td {{ border-bottom: 0; }}
    textarea, input, select, button {{ font: inherit; }}
    textarea, input, select {{ width: 100%; box-sizing: border-box; border-radius: 8px; border: 1px solid #475569; background: #020617; color: #e5e7eb; padding: 10px; }}
    textarea {{ min-height: 120px; }}
    button {{ margin-top: 10px; border: 0; border-radius: 8px; padding: 10px 14px; background: #2563eb; color: white; cursor: pointer; }}
    pre {{ white-space: pre-wrap; margin: 8px 0 0; line-height: 1.5; }}
    code {{ color: #bfdbfe; }}
  </style>
</head>
<body>
  <aside>
    <div class="brand">CodeAgent Python</div>
    <p class="muted">Python-native UI</p>
    <nav>{nav}</nav>
  </aside>
  <main>{body}</main>
</body>
</html>"""
    return HTMLResponse(html)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    if not rows:
        body = f"<tr><td colspan='{len(headers)}' class='muted'>No data</td></tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_python_ui(session_id: str | None = None) -> HTMLResponse:
    session = get_or_create_session(session_id) if session_id else None
    if session is None:
        session = next(
            (
                item
                for item in sorted(STATE["sessions"].values(), key=lambda value: value.get("updatedAt", ""), reverse=True)
                if item.get("messages")
            ),
            None,
        )
    messages = session.get("messages", []) if session else []
    items = []
    for msg in messages:
        role = "User" if msg.get("type") == "user" else "Assistant"
        items.append(
            f"<article class='message {escape(msg.get('type', 'system'))}'>"
            f"<strong>{role}</strong><pre>{escape(message_text(msg))}</pre></article>"
        )
    session_options = "".join(
        f"<li><a href='/?session_id={escape(sid)}'>{escape(s.get('title') or sid)}</a></li>"
        for sid, s in sorted(STATE["sessions"].items(), key=lambda pair: pair[1].get("updatedAt", ""), reverse=True)[:20]
    )
    body = f"""
    <h1>Chat</h1>
    <div class="grid">
      <section class="panel"><div class="muted">Sessions</div><div class="stat">{len(STATE.get('sessions', {}))}</div></section>
      <section class="panel"><div class="muted">Tools</div><div class="stat">{len(TOOL_REGISTRY.list())}</div></section>
      <section class="panel"><div class="muted">Memories</div><div class="stat">{len(STATE.get('memories', []))}</div></section>
    </div>
    <h2>Recent Sessions</h2>
    <ul>{session_options or "<li class='muted'>No sessions yet</li>"}</ul>
    <section>{''.join(items) or "<p class='muted'>Start a conversation.</p>"}</section>
    <form method="post" action="/ui/chat">
      <input type="hidden" name="session_id" value="{escape(session['id']) if session else ''}">
      <textarea name="text" placeholder="Ask CodeAgent Python..."></textarea>
      <button type="submit">Send</button>
    </form>"""
    return ui_shell("Chat", body, "/")


def render_realtime_ui(session_id: str | None = None) -> HTMLResponse:
    sid = session_id or new_id("session")
    body = f"""
    <h1>Realtime Workspace</h1>
    <section class="grid">
      <div class="panel"><div class="muted">Connection</div><div class="stat" id="conn">offline</div></div>
      <div class="panel"><div class="muted">Session</div><div class="stat" style="font-size:16px" id="sid">{escape(sid)}</div></div>
      <div class="panel"><div class="muted">Token Budget</div><div class="stat" id="token">0%</div></div>
      <div class="panel"><div class="muted">Tools</div><div class="stat" id="tool-count">0</div></div>
    </section>
    <section class="panel" style="margin-top:12px">
      <textarea id="prompt" placeholder="Ask CodeAgent Python..."></textarea>
      <button id="send" type="button">Send</button>
      <button id="stop" type="button" style="background:#475569">Interrupt</button>
    </section>
    <section class="grid" style="margin-top:12px">
      <div class="panel"><h2>Messages</h2><div id="messages"></div></div>
      <div class="panel"><h2>Tool Calls</h2><div id="tools"></div></div>
      <div class="panel"><h2>Notifications</h2><div id="notifications"></div></div>
      <div class="panel"><h2>Query Events</h2><div id="events"></div></div>
    </section>
    <script>
    const sessionId = {json.dumps(sid)};
    const conn = document.getElementById('conn');
    const messages = document.getElementById('messages');
    const tools = document.getElementById('tools');
    const notifications = document.getElementById('notifications');
    const events = document.getElementById('events');
    const token = document.getElementById('token');
    const toolCount = document.getElementById('tool-count');
    let ws;
    let stream = '';
    let activeTools = 0;
    function stomp(command, headers, body) {{
      const hs = Object.entries(headers || {{}}).map(([k, v]) => `${{k}}:${{v}}`).join('\\n');
      return command + (hs ? '\\n' + hs : '') + '\\n\\n' + (body || '') + '\\u0000';
    }}
    function sendFrame(frame) {{
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify([frame]));
    }}
    function append(target, html) {{
      const node = document.createElement('div');
      node.className = 'message';
      node.innerHTML = html;
      target.prepend(node);
    }}
    function parseSock(raw) {{
      if (!raw || raw === 'o' || raw === 'h') return [];
      if (raw.startsWith('a[')) {{
        try {{ return JSON.parse(raw.slice(1)); }} catch {{ return []; }}
      }}
      return [raw];
    }}
    function parseStomp(frame) {{
      const body = frame.split('\\n\\n').slice(1).join('\\n\\n').replace(/\\u0000$/, '');
      try {{ return JSON.parse(body); }} catch {{ return null; }}
    }}
    function handle(data) {{
      if (!data || !data.type) return;
      append(events, `<strong>${{data.type}}</strong><pre>${{JSON.stringify(data, null, 2).slice(0, 1200)}}</pre>`);
      if (data.type === 'stream_delta') {{
        stream += data.delta || '';
        messages.innerHTML = `<article class="message"><strong>Assistant</strong><pre>${{stream}}</pre></article>` + messages.innerHTML.replace(/^<article class="message"><strong>Assistant<\\/strong>[\\s\\S]*?<\\/article>/, '');
      }} else if (data.type === 'message_complete') {{
        stream = '';
      }} else if (data.type === 'tool_use_start') {{
        activeTools += 1; toolCount.textContent = String(activeTools);
        append(tools, `<strong>${{data.toolName}}</strong><pre>${{JSON.stringify(data.input || {{}}, null, 2)}}</pre>`);
      }} else if (data.type === 'tool_result') {{
        append(tools, `<strong>${{data.toolUseId}}</strong><pre>${{(data.content || '').slice(0, 1200)}}</pre>`);
      }} else if (data.type === 'notification' || data.type === 'error' || data.type === 'interrupt_ack') {{
        append(notifications, `<strong>${{data.level || data.type}}</strong><pre>${{data.message || data.reason || ''}}</pre>`);
      }} else if (data.type === 'token_budget_nudge' || data.type === 'token_warning') {{
        token.textContent = `${{data.pct || data.usagePercent || 0}}%`;
      }}
    }}
    function connect() {{
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(`${{proto}}//${{location.host}}/ws/websocket`);
      ws.onopen = () => conn.textContent = 'opening';
      ws.onmessage = (ev) => {{
        for (const frame of parseSock(ev.data)) {{
          if (frame === 'o') continue;
          if (frame.startsWith('CONNECTED')) {{
            conn.textContent = 'connected';
            sendFrame(stomp('SUBSCRIBE', {{id:'sub-0', destination:'/user/queue/messages'}}, ''));
            sendFrame(stomp('SEND', {{destination:'/app/bind-session'}}, JSON.stringify({{sessionId}})));
            continue;
          }}
          handle(parseStomp(frame));
        }}
        if (ev.data === 'o') sendFrame(stomp('CONNECT', {{'accept-version':'1.2', 'X-Session-Id': sessionId}}, ''));
      }};
      ws.onclose = () => {{ conn.textContent = 'reconnecting'; setTimeout(connect, 1500); }};
    }}
    document.getElementById('send').onclick = () => {{
      const prompt = document.getElementById('prompt');
      const text = prompt.value.trim();
      if (!text) return;
      append(messages, `<strong>User</strong><pre>${{text}}</pre>`);
      prompt.value = '';
      sendFrame(stomp('SEND', {{destination:'/app/chat'}}, JSON.stringify({{text, sessionId}})));
    }};
    document.getElementById('stop').onclick = () => sendFrame(stomp('SEND', {{destination:'/app/interrupt'}}, JSON.stringify({{sessionId}})));
    connect();
    </script>
    """
    return ui_shell("Realtime", body, "/ui/realtime")


def render_dashboard_ui() -> HTMLResponse:
    active_sessions = sum(1 for item in STATE.get("sessions", {}).values() if item.get("status") in {"running", "streaming", "processing"} or item.get("online"))
    body = f"""
    <h1>Dashboard</h1>
    <section class="grid">
      <div class="panel"><div class="muted">Sessions</div><div class="stat">{len(STATE.get('sessions', {}))}</div></div>
      <div class="panel"><div class="muted">Active</div><div class="stat">{active_sessions}</div></div>
      <div class="panel"><div class="muted">Tasks</div><div class="stat">{len(STATE.get('tasks', {}))}</div></div>
      <div class="panel"><div class="muted">Memories</div><div class="stat">{len(STATE.get('memories', []))}</div></div>
      <div class="panel"><div class="muted">MCP Servers</div><div class="stat">{len(STATE.get('mcpServers', []))}</div></div>
      <div class="panel"><div class="muted">Tools</div><div class="stat">{len(TOOL_REGISTRY.list())}</div></div>
    </section>
    <h2>Runtime</h2>
    {html_table(['Area', 'Status'], [
        ['Python backend', 'ready'],
        ['Database', SQLITE_STORE.stats().journalMode],
        ['Remote control', os.getenv('AUTH_MODE', 'localhost')],
        ['WebSocket sessions', str(len(WS_SESSION_MANAGER.get_active_session_ids()))],
    ])}"""
    return ui_shell("Dashboard", body, "/ui/dashboard")


def render_sessions_ui() -> HTMLResponse:
    rows = []
    for session in sorted(STATE["sessions"].values(), key=lambda item: item.get("updatedAt", ""), reverse=True)[:100]:
        sid = escape(session["id"])
        rows.append([
            f"<a href='/?session_id={sid}'>{escape(session.get('title') or session['id'])}</a>",
            escape(session.get("model") or ""),
            str(len(session.get("messages", []))),
            escape(session.get("status") or "idle"),
            escape(session.get("updatedAt") or ""),
        ])
    body = f"<h1>Sessions</h1>{html_table(['Title', 'Model', 'Messages', 'Status', 'Updated'], rows)}"
    return ui_shell("Sessions", body, "/ui/sessions")


def render_tools_ui() -> HTMLResponse:
    rows = [[escape(item["name"]), escape(item.get("group") or ""), escape(str(item.get("readOnly"))), escape(str(item.get("enabled"))), escape(item.get("description") or "")] for item in TOOL_REGISTRY.list()]
    body = f"<h1>Tools</h1>{html_table(['Name', 'Group', 'Read only', 'Enabled', 'Description'], rows)}"
    return ui_shell("Tools", body, "/ui/tools")


def render_tasks_ui() -> HTMLResponse:
    task_rows = [[escape(item.get("title") or item.get("id") or ""), escape(item.get("status") or ""), escape(item.get("type") or ""), escape(item.get("updatedAt") or "")] for item in STATE.get("tasks", {}).values()]
    swarm_rows = [[escape(item.get("id") or ""), escape(item.get("teamName") or ""), escape(item.get("phase") or ""), str(item.get("activeWorkers") or 0)] for item in STATE.get("swarms", {}).values()]
    body = f"<h1>Tasks</h1><h2>Tasks</h2>{html_table(['Title', 'Status', 'Type', 'Updated'], task_rows)}<h2>Swarms</h2>{html_table(['ID', 'Team', 'Phase', 'Active workers'], swarm_rows)}"
    return ui_shell("Tasks", body, "/ui/tasks")


def render_settings_ui() -> HTMLResponse:
    config_rows = [[escape(str(key)), f"<code>{escape(json.dumps(value, ensure_ascii=False))}</code>"] for key, value in STATE.get("config", {}).items()]
    project_rows = [[escape(str(key)), f"<code>{escape(json.dumps(value, ensure_ascii=False))}</code>"] for key, value in STATE.get("projectConfig", {}).items()]
    body = f"""
    <h1>Settings</h1>
    <section class="grid">
      <div class="panel"><div class="muted">Default model</div><div class="stat">{escape(str(STATE.get('config', {}).get('defaultModel', '')))}</div></div>
      <div class="panel"><div class="muted">Locale</div><div class="stat">{escape(str(STATE.get('config', {}).get('locale', '')))}</div></div>
    </section>
    <h2>Runtime Config</h2>{html_table(['Key', 'Value'], config_rows)}
    <h2>Project Config</h2>{html_table(['Key', 'Value'], project_rows)}"""
    return ui_shell("Settings", body, "/ui/settings")


def render_files_ui(query: str = "") -> HTMLResponse:
    results = []
    if query:
        q = query.lower()
        for path in ROOT.rglob("*"):
            if len(results) >= 80:
                break
            if not path.is_file() or any(part in FILE_TREE_IGNORES for part in path.relative_to(ROOT).parts):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if q in rel.lower():
                results.append([f"<code>{escape(rel)}</code>", escape(path.suffix or ""), str(path.stat().st_size)])
    body = f"""
    <h1>Files</h1>
    <form method="get" action="/ui/files">
      <input name="q" value="{escape(query)}" placeholder="Search files">
      <button type="submit">Search</button>
    </form>
    <h2>Results</h2>{html_table(['Path', 'Type', 'Bytes'], results)}"""
    return ui_shell("Files", body, "/ui/files")


def render_activity_ui() -> HTMLResponse:
    activities = []
    for sid, items in STATE.get("activities", {}).items():
        for item in items[-20:]:
            activities.append((item.get("timestamp") or item.get("createdAt") or "", sid, item))
    rows = [
        [escape(str(stamp)), escape(sid), escape(str(item.get("type") or item.get("event") or "")), escape(str(item.get("title") or item.get("message") or ""))]
        for stamp, sid, item in sorted(activities, key=lambda row: row[0], reverse=True)[:100]
    ]
    body = f"<h1>Activity</h1>{html_table(['Time', 'Session', 'Type', 'Summary'], rows)}"
    return ui_shell("Activity", body, "/ui/activity")


def render_verify_ui() -> HTMLResponse:
    rows = []
    for bundle in sorted(STATE.get("evidence", {}).values(), key=lambda item: item.get("createdAt", ""), reverse=True)[:100]:
        rows.append([
            f"<code>{escape(bundle.get('bundleId', ''))}</code>",
            escape(bundle.get("sessionId") or ""),
            escape(bundle.get("verdict") or ""),
            escape(bundle.get("claim") or ""),
            escape(bundle.get("createdAt") or ""),
        ])
    body = f"<h1>Verify</h1>{html_table(['Bundle', 'Session', 'Verdict', 'Claim', 'Created'], rows)}"
    return ui_shell("Verify", body, "/ui/verify")


def render_mcp_ui() -> HTMLResponse:
    server_rows = [[escape(str(item.get("name") or "")), escape(str(item.get("status") or "")), str(len(item.get("tools") or [])), str(len(item.get("resources") or []))] for item in STATE.get("mcpServers", [])]
    cap_rows = [[escape(str(item.get("id") or "")), escape(str(item.get("toolName") or "")), escape(str(item.get("domain") or "")), escape(str(item.get("enabled", True)))] for item in STATE.get("mcpCapabilities", [])]
    body = f"<h1>MCP</h1><h2>Servers</h2>{html_table(['Name', 'Status', 'Tools', 'Resources'], server_rows)}<h2>Capabilities</h2>{html_table(['ID', 'Tool', 'Domain', 'Enabled'], cap_rows)}"
    return ui_shell("MCP", body, "/ui/mcp")


def render_memory_ui(query: str = "") -> HTMLResponse:
    rows = [[escape(str(item.get("title") or item.get("id") or "")), escape(str(item.get("category") or "")), escape(str(item.get("content") or ""))[:220]] for item in STATE.get("memories", [])]
    categories = ", ".join(item["tag"] for item in MEMDIR_SERVICE.categories())
    search_rows: list[list[str]] = []
    if query:
        search_rows = [
            [
                escape(item["title"]),
                escape(item["category"]),
                f"{item['score']:.3f}",
                escape(item["content"])[:260],
            ]
            for item in (result.to_dict() for result in MEMDIR_SERVICE.search(query, 10, rerank=True))
        ]
    body = f"""
    <h1>Memory</h1>
    <p class="muted">Categories: {escape(categories)}</p>
    <form method="get" action="/ui/memory">
      <input name="q" value="{escape(query)}" placeholder="Search memories">
      <button type="submit">Search</button>
    </form>
    {"<h2>Search Results</h2>" + html_table(['Title', 'Category', 'Score', 'Content'], search_rows) if query else ""}
    <form method="post" action="/ui/memory">
      <input name="title" placeholder="Title">
      <textarea name="content" placeholder="Memory content"></textarea>
      <button type="submit">Save Memory</button>
    </form>
    <h2>Entries</h2>{html_table(['Title', 'Category', 'Content'], rows)}"""
    return ui_shell("Memory", body, "/ui/memory")


def default_state() -> dict[str, Any]:
    return {
        "config": dict(DEFAULT_CONFIG),
        "projectConfig": {"workspace": str(ROOT / "workspace"), "pythonBackend": True},
        "sessions": {},
        "sessionSnapshots": {},
        "activities": {},
        "permissions": [],
        "memories": [],
        "tools": [],
        "plugins": [],
        "hooks": [],
        "hookEvents": [],
        "dialogDecisions": {},
        "fileSnapshots": [],
        "browserReplay": {},
        "mcpServers": [],
        "mcpCapabilities": [],
        "permissionResponses": {},
        "elicitations": {},
        "swarms": {},
        "teams": {},
        "tasks": {},
        "sandboxRuns": {},
        "costEvents": [],
        "anomalies": [],
        "bridge": {"devices": [], "messages": []},
        "keybindings": {},
        "replSessions": {},
        "queryLoops": {},
        "queryEvents": [],
        "notifications": [],
        "journeyResults": {},
        "correctionReports": {},
        "evidence": {},
        "evidenceBlobs": {},
    }


def load_state() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        return default_state()
    try:
        loaded = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return default_state()
    state = default_state()
    for key, value in loaded.items():
        if key == "config":
            merged = dict(DEFAULT_CONFIG)
            merged.update(value or {})
            if should_prefer_xfyun_default_model():
                merged["defaultModel"] = XFYUN_MAAS_MODEL
            state[key] = merged
        else:
            state[key] = value
    return state


STATE = load_state()
PERMISSION_POLICY = PermissionPolicy.from_state(STATE.get("permissions"))
CRON_SERVICE = CronTaskService(DATA_DIR / "scheduled_tasks.json")
FILE_VERSION_TRACKER = FileVersionTracker()
FILE_RECOVERY_POLICY = FileEditRecoveryPolicy()
WS_SESSION_MANAGER = WebSocketSessionManager()
COORDINATOR_ENGINE = CoordinatorWorkflowEngine()
MEMDIR_SERVICE = MemdirService(ROOT, STATE)
TOOL_REGISTRY = ToolRegistry(ROOT, PERMISSION_POLICY, CRON_SERVICE, FILE_VERSION_TRACKER, FILE_RECOVERY_POLICY, MEMDIR_SERVICE)
COMMAND_REGISTRY = CommandRegistry(TOOL_REGISTRY)
MODEL_CAPABILITIES = ModelCapabilityRegistry()
MODEL_PROVIDERS = LlmProviderRegistry(models=[model["id"] for model in DEFAULT_MODELS], default_model=DEFAULT_CONFIG["defaultModel"])
RETRY_POLICY = ModelAwareRetryPolicy()
LLM_ERROR_CLASSIFIER = LlmErrorClassifier()
MODEL_DEGRADATION_CHAIN = ModelDegradationChain()
TOOL_SCHEDULER = ToolPriorityScheduler()
CONTEXT_COLLAPSE = ContextCollapseService()
CONTEXT_CASCADE = ContextCascadeService(context_collapse=CONTEXT_COLLAPSE)
SIDE_QUERY_SERVICE = SideQueryService()
MICRO_COMPACT = MicroCompactService()
TOOL_RESULT_SUMMARIZER = ToolResultSummarizer()
TOKEN_COUNTER = TokenCounter()
QUERY_ABORTS = AbortController()
TERMINATION_STRATEGY = DefaultTerminationStrategy()
SKILL_VALIDATOR = SkillToolValidator()
SQLITE_STORE = SQLiteStateStore(SQLITE_FILE)
SQLITE_STORE.migrate()
MCP_MANAGER = McpClientManager()
MCP_MANAGER.load_state(STATE.get("mcpServers", []))


def sync_mcp_tools() -> int:
    MCP_MANAGER.load_state(STATE.setdefault("mcpServers", []))
    return TOOL_REGISTRY.register_mcp_tools(MCP_MANAGER)


sync_mcp_tools()
LSP_MANAGER = LSPServerManager(ROOT)
REMOTE_SECURITY = RemoteAccessSecurity(DATA_DIR / "access-token")
SWARM_TASKS: dict[str, dict[str, asyncio.Task[Any]]] = {}
SWARM_MAILBOXES: dict[str, list[dict[str, Any]]] = {}
SWARM_PERMISSION_WAITERS: dict[str, asyncio.Future[dict[str, Any]]] = {}
WORKER_MESSAGE_CAP = 50
STATE_SAVE_LOCK = threading.RLock()


def stable_state_snapshot(retries: int = 5) -> dict[str, Any]:
    last_error: RuntimeError | None = None
    for _ in range(retries):
        try:
            return copy.deepcopy(STATE)
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.01)
    if last_error:
        raise last_error
    return copy.deepcopy(STATE)


def save_state() -> None:
    with STATE_SAVE_LOCK:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        snapshot = stable_state_snapshot()
        tmp = STATE_FILE.with_name(f"{STATE_FILE.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        last_error: Exception | None = None
        for _ in range(5):
            try:
                tmp.replace(STATE_FILE)
                SQLITE_STORE.sync_state(snapshot)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05)
        try:
            tmp.unlink(missing_ok=True)
        finally:
            if last_error:
                raise last_error


def rebuild_runtime_registries() -> None:
    global PERMISSION_POLICY, TOOL_REGISTRY, COMMAND_REGISTRY
    PERMISSION_POLICY = PermissionPolicy.from_state(STATE.get("permissions"))
    TOOL_REGISTRY = ToolRegistry(ROOT, PERMISSION_POLICY, CRON_SERVICE, FILE_VERSION_TRACKER, FILE_RECOVERY_POLICY, MEMDIR_SERVICE)
    if "dispatch_direct_agent_to_team" in globals():
        TOOL_REGISTRY.set_team_dispatcher(dispatch_direct_agent_to_team)
    if "dispatch_direct_agent" in globals():
        TOOL_REGISTRY.set_agent_dispatcher(dispatch_direct_agent)
    if "MCP_MANAGER" in globals():
        sync_mcp_tools()
    COMMAND_REGISTRY = CommandRegistry(TOOL_REGISTRY)


SQLITE_STORE.sync_state(STATE)


app = FastAPI(
    title="CodeAgent Python Backend",
    version="0.1.0-python",
    description="Python replacement for the original Spring Boot backend surface.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index() -> Response:
    if react_frontend_available():
        return FileResponse(react_index_file())
    return render_python_ui()


@app.get("/ui")
async def ui(session_id: str | None = None) -> HTMLResponse:
    return render_python_ui(session_id)


@app.get("/ui/realtime")
async def ui_realtime(session_id: str | None = None) -> HTMLResponse:
    return render_realtime_ui(session_id)


@app.get("/ui/dashboard")
async def ui_dashboard() -> HTMLResponse:
    return render_dashboard_ui()


@app.get("/ui/sessions")
async def ui_sessions() -> HTMLResponse:
    return render_sessions_ui()


@app.get("/ui/tools")
async def ui_tools() -> HTMLResponse:
    return render_tools_ui()


@app.get("/ui/tasks")
async def ui_tasks() -> HTMLResponse:
    return render_tasks_ui()


@app.get("/ui/settings")
async def ui_settings() -> HTMLResponse:
    return render_settings_ui()


@app.get("/ui/files")
async def ui_files(q: str = "") -> HTMLResponse:
    return render_files_ui(q)


@app.get("/ui/activity")
async def ui_activity() -> HTMLResponse:
    return render_activity_ui()


@app.get("/ui/verify")
async def ui_verify() -> HTMLResponse:
    return render_verify_ui()


@app.get("/ui/mcp")
async def ui_mcp() -> HTMLResponse:
    return render_mcp_ui()


@app.get("/ui/memory")
async def ui_memory(q: str = "") -> HTMLResponse:
    return render_memory_ui(q)


@app.post("/ui/memory")
async def ui_memory_create(request: Request) -> RedirectResponse:
    body = (await request.body()).decode("utf-8", errors="ignore")
    form = parse_qs(body)
    title = (form.get("title") or [""])[0].strip() or "Memory"
    content = (form.get("content") or [""])[0].strip()
    if content:
        STATE.setdefault("memories", []).append(normalize_memory_entry({"title": title, "content": content}))
        save_state()
    return RedirectResponse(url="/ui/memory", status_code=303)


@app.post("/ui/chat")
async def ui_chat(request: Request) -> RedirectResponse:
    body = (await request.body()).decode("utf-8", errors="ignore")
    form = parse_qs(body)
    text = (form.get("text") or [""])[0]
    session_id = (form.get("session_id") or [""])[0] or None
    session = get_or_create_session(session_id, persist=False)
    hook_result = execute_hooks("USER_PROMPT_SUBMIT", {"input": text, "sessionId": session["id"]})
    if not hook_result.get("proceed", True):
        text = str(hook_result.get("message") or "Blocked by hook")
        answer = text
    else:
        text = str((hook_result.get("context") or {}).get("input") or text)
        answer = ""
    now_ms = int(time.time() * 1000)
    session["messages"].append({"type": "user", "uuid": new_id("user"), "timestamp": now_ms, "content": [{"type": "text", "text": text}]})
    if not session.get("title"):
        session["title"] = text[:60] if text else "New session"
    if not answer:
        answer = await generate_ui_chat_reply(session, text)
    session["messages"].append(
        {
            "type": "assistant",
            "uuid": new_id("assistant"),
            "timestamp": int(time.time() * 1000),
            "content": [{"type": "text", "text": answer}],
            "stopReason": "end_turn",
            "usage": usage(),
        }
    )
    session["status"] = "idle"
    session["updatedAt"] = utc_now()
    return RedirectResponse(url=f"/ui?session_id={session['id']}", status_code=303)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "codeagent-python-backend",
        "version": app.version,
        "pythonServiceUrl": PYTHON_SERVICE_URL,
    }


@app.get("/api/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "UP"}


@app.get("/api/health/ready")
async def health_ready() -> dict[str, Any]:
    python_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{PYTHON_SERVICE_URL}/api/health")
        python_status = "ok" if resp.status_code < 500 else "degraded"
    except Exception:
        python_status = "unavailable"
    db_stats = SQLITE_STORE.stats().to_dict()
    return {"status": "UP", "pythonService": python_status, "database": {"status": "ok", **db_stats}}


@app.get("/api/doctor")
async def doctor() -> dict[str, Any]:
    db_stats = SQLITE_STORE.stats().to_dict()
    return {
        "status": "ok",
        "checks": [
            {"name": "stateFile", "status": "ok", "path": str(STATE_FILE)},
            {"name": "sqlite", "status": "ok", **db_stats},
            {"name": "workspace", "status": "ok", "path": str(ROOT)},
        ],
    }


@app.get("/api/database/status")
async def database_status() -> dict[str, Any]:
    return {"status": "ok", **SQLITE_STORE.stats().to_dict()}


@app.get("/api/database/migrations")
async def database_migrations() -> dict[str, Any]:
    return {"migrations": SQLITE_STORE.migration_status()}


@app.get("/api/database/sessions/{session_id}")
async def database_session(session_id: str) -> dict[str, Any]:
    session = SQLITE_STORE.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session, "messages": SQLITE_STORE.list_messages(session_id)}


@app.post("/api/database/sessions/{session_id}/messages/delete-after/{seq_num}")
async def database_delete_messages_after(session_id: str, seq_num: int) -> dict[str, Any]:
    deleted = SQLITE_STORE.delete_messages_after(session_id, seq_num)
    return {"success": True, "deleted": deleted, "sessionId": session_id, "seqNum": seq_num}


@app.get("/actuator/health")
async def actuator_health() -> dict[str, str]:
    return {"status": "UP"}


@app.get("/api/auth/status")
async def auth_status(request: Request) -> dict[str, Any]:
    auth_mode = os.getenv("AUTH_MODE", "localhost")
    remote_addr = "192.168.1.2" if auth_mode != "localhost" else "127.0.0.1"
    decision = REMOTE_SECURITY.validate(
        remote_addr,
        headers=dict(request.headers),
        cookies=request.cookies,
        query={key: value[-1] for key, value in parse_qs(request.url.query).items()},
    )
    return {
        "authenticated": decision.authenticated,
        "authMode": auth_mode,
        "mode": auth_mode,
        "username": "local-user" if decision.authenticated else None,
        "user": "local-user" if decision.authenticated else None,
        "reason": decision.reason,
        "tokenPreview": REMOTE_SECURITY.token_preview(),
    }


@app.get("/api/auth/token")
async def auth_token() -> dict[str, str]:
    token = REMOTE_SECURITY.access_token()
    return {"token": token, "accessToken": token, "tokenType": "Bearer"}


def admin_password_hash() -> str | None:
    password = os.getenv("ADMIN_PASSWORD") or os.getenv("ZHIKUN_ADMIN_PASSWORD")
    if not password:
        return None
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


@app.post("/api/admin/login")
async def admin_login(request: Request) -> JSONResponse:
    payload = await request.json()
    configured_hash = admin_password_hash()
    if not configured_hash:
        return JSONResponse({"success": False, "message": "Admin authentication not configured"}, status_code=503)
    provided = str(payload.get("password") or "")
    if hashlib.sha256(provided.encode("utf-8")).hexdigest() != configured_hash:
        return JSONResponse({"success": False, "message": "Invalid password"}, status_code=401)
    response = JSONResponse({"success": True, "message": "Login successful", "admin": True, "timestamp": utc_now()})
    response.set_cookie(ADMIN_COOKIE_NAME, configured_hash, httponly=True, samesite="lax", max_age=8 * 60 * 60)
    return response


@app.get("/api/admin/status")
async def admin_status(request: Request) -> dict[str, Any]:
    configured_hash = admin_password_hash()
    cookie = request.cookies.get(ADMIN_COOKIE_NAME)
    header = request.headers.get("authorization", "")
    token_hash = hashlib.sha256(header[7:].encode("utf-8")).hexdigest() if header.startswith("Bearer ") else None
    authenticated = bool(configured_hash and (cookie == configured_hash or token_hash == configured_hash))
    return {"authenticated": authenticated, "configured": configured_hash is not None, "timestamp": utc_now()}


@app.post("/api/admin/logout")
async def admin_logout() -> JSONResponse:
    response = JSONResponse({"success": True, "message": "Logged out successfully"})
    response.delete_cookie(ADMIN_COOKIE_NAME)
    return response


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return STATE["config"]


@app.put("/api/config")
async def update_config(request: Request) -> dict[str, Any]:
    updates = await request.json()
    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="Expected JSON object")
    STATE["config"].update(updates)
    save_state()
    return STATE["config"]


@app.get("/api/config/project")
async def get_project_config() -> dict[str, Any]:
    return STATE["projectConfig"]


@app.put("/api/config/project")
async def update_project_config(request: Request) -> dict[str, Any]:
    updates = await request.json()
    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="Expected JSON object")
    STATE["projectConfig"].update(updates)
    save_state()
    return STATE["projectConfig"]


@app.get("/api/models")
async def get_models() -> dict[str, Any]:
    default_model = STATE["config"].get("defaultModel") or DEFAULT_MODELS[0]["id"]
    models = [dict(m) for m in DEFAULT_MODELS]
    if default_model not in {m["id"] for m in models}:
        models.insert(0, {**DEFAULT_MODELS[0], "id": default_model, "displayName": default_model})
    capabilities = {model["id"]: asdict(MODEL_CAPABILITIES.get_capability(model["id"])) for model in models}
    return {
        "models": models,
        "defaultModel": default_model,
        "aliases": dict(MODEL_PROVIDERS.aliases),
        "capabilities": capabilities,
    }


@app.get("/api/models/{model_id}/capability")
async def get_model_capability(model_id: str) -> dict[str, Any]:
    capability = MODEL_CAPABILITIES.get_capability(model_id)
    retry = RETRY_POLICY.get_retry_config(model_id)
    return {
        "capability": asdict(capability),
        "compactThreshold": MODEL_CAPABILITIES.compact_threshold(model_id),
        "bufferTokens": MODEL_CAPABILITIES.buffer_tokens(model_id),
        "retry": asdict(retry),
    }


@app.get("/api/models/resolve/{alias}")
async def resolve_model_alias(alias: str) -> dict[str, str]:
    return {"alias": alias, "model": MODEL_PROVIDERS.resolve_model_alias(alias)}


@app.get("/api/commands")
async def get_commands() -> list[dict[str, Any]]:
    return COMMAND_REGISTRY.list()


def session_summary(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": session["id"],
        "title": session.get("title"),
        "model": session.get("model", STATE["config"].get("defaultModel")),
        "workingDirectory": session.get("workingDirectory", "."),
        "messageCount": len(session.get("messages", [])),
        "costUsd": session.get("costUsd", 0),
        "createdAt": session["createdAt"],
        "updatedAt": session["updatedAt"],
    }


def get_or_create_session(session_id: str | None = None, persist: bool = True) -> dict[str, Any]:
    if session_id and session_id in STATE["sessions"]:
        return STATE["sessions"][session_id]
    sid = session_id or new_id("session")
    now = utc_now()
    session = {
        "id": sid,
        "title": None,
        "model": STATE["config"].get("defaultModel", "qwen3.7-max"),
        "workingDirectory": ".",
        "messages": [],
        "costUsd": 0,
        "createdAt": now,
        "updatedAt": now,
        "status": "idle",
    }
    STATE["sessions"][sid] = session
    STATE["activities"].setdefault(sid, [])
    if persist:
        save_state()
    return session


@app.get("/api/sessions/snapshots")
async def list_session_snapshots() -> dict[str, Any]:
    snapshots = [
        {
            "sessionId": snapshot.get("sessionId"),
            "model": snapshot.get("model"),
            "turnCount": snapshot.get("turnCount", 0),
            "messageCount": len(snapshot.get("messages", [])),
            "createdAt": snapshot.get("createdAt"),
        }
        for snapshot in STATE.setdefault("sessionSnapshots", {}).values()
    ]
    return {"snapshots": sorted(snapshots, key=lambda item: item.get("createdAt", ""), reverse=True)}


@app.get("/api/sessions")
async def list_sessions(limit: int = 50, cursor: str | None = None) -> dict[str, Any]:
    sessions = sorted(STATE["sessions"].values(), key=lambda s: s.get("updatedAt", ""), reverse=True)
    start = int(cursor or 0)
    selected = sessions[start:start + limit]
    next_cursor = str(start + limit) if start + limit < len(sessions) else None
    return {
        "sessions": [session_summary(s) for s in selected],
        "hasMore": next_cursor is not None,
        "nextCursor": next_cursor,
    }


@app.post("/api/sessions")
async def create_session(request: Request) -> dict[str, Any]:
    payload = await request.json()
    sid = new_id("session")
    now = utc_now()
    session = {
        "id": sid,
        "title": payload.get("title"),
        "model": payload.get("model") or STATE["config"].get("defaultModel", "qwen3.7-max"),
        "workingDirectory": payload.get("dir") or payload.get("workingDirectory") or ".",
        "messages": [],
        "costUsd": 0,
        "createdAt": now,
        "updatedAt": now,
        "status": "idle",
    }
    STATE["sessions"][sid] = session
    STATE["activities"][sid] = []
    save_state()
    return {"sessionId": sid, "session": session_summary(session)}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = STATE["sessions"].get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session_summary(session), "messages": session.get("messages", [])}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    active_agents = active_background_agent_ids(session_id)
    for agent_id in active_agents:
        task = next((item for item in TOOL_REGISTRY._tasks.values() if item.get("agentId") == agent_id), {})
        child_session_id = str(task.get("childSessionId") or f"subagent-{agent_id}")
        QUERY_ABORTS.abort(child_session_id, "SESSION_DISCONNECTED")
    if active_agents:
        await await_background_agents(session_id, timeout_ms=5000)
    STATE["sessions"].pop(session_id, None)
    STATE["activities"].pop(session_id, None)
    removed_background_agents = remove_background_agent_session(session_id)
    save_state()
    return {"success": True, "activeBackgroundAgents": active_agents, "removedBackgroundAgents": removed_background_agents}


def session_snapshot_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "sessionId": snapshot.get("sessionId"),
        "model": snapshot.get("model"),
        "turnCount": snapshot.get("turnCount", 0),
        "messageCount": len(snapshot.get("messages", [])),
        "createdAt": snapshot.get("createdAt"),
    }


@app.post("/api/sessions/{session_id}/snapshot")
async def save_session_snapshot(session_id: str) -> JSONResponse:
    session = STATE["sessions"].get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = list(session.get("messages", []))
    snapshot = {
        "sessionId": session_id,
        "messages": messages,
        "model": session.get("model"),
        "turnCount": sum(1 for message in messages if message.get("type") == "user"),
        "createdAt": utc_now(),
        "metadata": {
            "title": session.get("title"),
            "workingDir": session.get("workingDirectory"),
            "status": session.get("status"),
            "costUsd": session.get("costUsd", 0),
        },
    }
    STATE.setdefault("sessionSnapshots", {})[session_id] = snapshot
    save_state()
    return JSONResponse(session_snapshot_summary(snapshot), status_code=201)


@app.post("/api/sessions/{session_id}/snapshot/resume")
async def resume_session_snapshot(session_id: str) -> dict[str, Any]:
    snapshot = STATE.setdefault("sessionSnapshots", {}).get(session_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    session = get_or_create_session(session_id)
    session["messages"] = list(snapshot.get("messages", []))
    session["model"] = snapshot.get("model") or session.get("model")
    metadata = snapshot.get("metadata") or {}
    session["title"] = metadata.get("title")
    session["workingDirectory"] = metadata.get("workingDir") or session.get("workingDirectory")
    session["status"] = "idle"
    session["updatedAt"] = utc_now()
    save_state()
    return session_snapshot_summary(snapshot)


@app.delete("/api/sessions/snapshots/{session_id}")
async def delete_session_snapshot(session_id: str) -> dict[str, bool]:
    existed = session_id in STATE.setdefault("sessionSnapshots", {})
    STATE["sessionSnapshots"].pop(session_id, None)
    save_state()
    return {"success": existed}


@app.post("/api/sessions/{session_id}/resume")
async def resume_session(session_id: str) -> dict[str, Any]:
    session = get_or_create_session(session_id)
    session["status"] = "idle"
    session["updatedAt"] = utc_now()
    save_state()
    return {
        "type": "session_restored",
        "messages": session.get("messages", []),
        "metadata": {"sessionId": session_id, "model": session.get("model"), "status": "idle"},
    }


@app.post("/api/sessions/{session_id}/compact")
async def compact_session(session_id: str) -> dict[str, Any]:
    session = get_or_create_session(session_id)
    messages = session.get("messages", [])
    before_tokens = max(1, len(json.dumps(messages, ensure_ascii=False)) // 4) if messages else 0
    if len(messages) > 6:
        older = messages[:-4]
        summary_lines = []
        for item in older[-20:]:
            role = item.get("type", "message")
            text = message_text(item).replace("\n", " ").strip()
            if text:
                summary_lines.append(f"{role}: {text[:180]}")
        compact_message = {
            "type": "system",
            "uuid": new_id("compact"),
            "timestamp": int(time.time() * 1000),
            "content": [{"type": "text", "text": "Compacted prior context:\n" + "\n".join(summary_lines)}],
        }
        session["messages"] = [compact_message, *messages[-4:]]
    after_tokens = max(1, len(json.dumps(session.get("messages", []), ensure_ascii=False)) // 4) if session.get("messages") else 0
    session["updatedAt"] = utc_now()
    save_state()
    return {
        "success": True,
        "beforeTokens": before_tokens,
        "afterTokens": after_tokens,
        "tokensSaved": max(0, before_tokens - after_tokens),
        "summary": "Context compacted by the Python backend.",
    }


@app.post("/api/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = "json") -> Response:
    session = STATE["sessions"].get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    export_payload = {"sessionId": session_id, "session": session, "messages": session.get("messages", []), "exportedAt": utc_now()}
    if format.lower() in {"markdown", "md"}:
        lines = [f"# Session {session_id}", ""]
        for item in session.get("messages", []):
            role = item.get("type", "message").title()
            lines.extend([f"## {role}", "", message_text(item), ""])
        content = "\n".join(lines).encode("utf-8")
        filename = f"session-{session_id}.md"
        media_type = "text/plain; charset=utf-8"
    else:
        content = json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8")
        filename = f"session-{session_id}.json"
        media_type = "application/json"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/sessions/{session_id}/messages")
async def list_messages(session_id: str) -> dict[str, Any]:
    session = get_or_create_session(session_id)
    return {"messages": session.get("messages", []), "total": len(session.get("messages", []))}


@app.get("/api/sessions/{session_id}/activities")
async def list_activities(session_id: str, offset: int = 0, limit: int = 50) -> dict[str, Any]:
    activities = STATE["activities"].get(session_id, [])
    return {
        "activities": activities[offset:offset + limit],
        "total": len(activities),
        "hasMore": offset + limit < len(activities),
    }


def snapshot_rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def file_snapshots_for(session_id: str, message_id: str | None = None) -> list[dict[str, Any]]:
    snapshots = [item for item in STATE.setdefault("fileSnapshots", []) if item.get("sessionId") == session_id]
    if message_id:
        snapshots = [item for item in snapshots if item.get("messageId") == message_id]
    return snapshots


def save_file_snapshot(session_id: str, message_id: str, file_path: str, operation: str = "snapshot") -> dict[str, Any]:
    path = safe_workspace_path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(file_path)
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("File is larger than 10MB")
    content = path.read_text(encoding="utf-8", errors="replace")
    snapshot = {
        "id": new_id("snapshot"),
        "sessionId": session_id,
        "messageId": message_id,
        "filePath": snapshot_rel_path(path),
        "content": content,
        "operation": operation,
        "createdAt": utc_now(),
    }
    STATE.setdefault("fileSnapshots", []).append(snapshot)
    return snapshot


def save_tool_file_snapshot(session_id: str, tool_use_id: str, tool_name: str, result_payload: dict[str, Any]) -> dict[str, Any] | None:
    if result_payload.get("isError"):
        return None
    metadata = result_payload.get("metadata") if isinstance(result_payload.get("metadata"), dict) else {}
    snapshot = metadata.get("snapshotBeforeWrite") if isinstance(metadata.get("snapshotBeforeWrite"), dict) else None
    if not snapshot:
        return None
    file_path = str(snapshot.get("path") or metadata.get("path") or "")
    if not file_path:
        return None
    content = snapshot.get("content")
    if content is None:
        content = ""
    normalized_path = file_path.replace("\\", "/")
    snapshots = STATE.setdefault("fileSnapshots", [])
    for existing in snapshots:
        if (
            existing.get("sessionId") == session_id
            and existing.get("messageId") == tool_use_id
            and str(existing.get("filePath") or "").replace("\\", "/") == normalized_path
        ):
            return existing
    persisted = {
        "id": str(snapshot.get("id") or new_id("snapshot")),
        "sessionId": session_id,
        "messageId": tool_use_id,
        "toolUseId": tool_use_id,
        "toolName": tool_name,
        "filePath": normalized_path,
        "content": str(content),
        "operation": str(snapshot.get("operation") or metadata.get("operation") or tool_name),
        "createdAt": snapshot.get("createdAt") or utc_now(),
        "contentHash": snapshot.get("contentHash") or metadata.get("beforeHash"),
        "beforeHash": metadata.get("beforeHash"),
        "afterHash": metadata.get("afterHash"),
        "diff": metadata.get("diff"),
        "agentId": snapshot.get("agentId"),
        "source": "tool_result",
    }
    snapshots.append(persisted)
    return persisted


def latest_snapshot_by_file(snapshots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        latest[str(snapshot.get("filePath") or "")] = snapshot
    return latest


def current_snapshot_by_file(file_paths: set[str]) -> dict[str, dict[str, Any]]:
    current: dict[str, dict[str, Any]] = {}
    for file_path in file_paths:
        try:
            target = safe_workspace_path(file_path)
        except Exception:
            continue
        if not target.exists() or not target.is_file():
            continue
        current[file_path] = {
            "filePath": file_path,
            "content": target.read_text(encoding="utf-8", errors="replace"),
            "operation": "current",
            "createdAt": utc_now(),
        }
    return current


def rewind_files(session_id: str, message_id: str, file_paths: list[str] | None = None) -> dict[str, Any]:
    snapshots = file_snapshots_for(session_id, message_id)
    if file_paths:
        wanted = {str(item).replace("\\", "/") for item in file_paths}
        snapshots = [
            item
            for item in snapshots
            if str(item.get("filePath") or "").replace("\\", "/") in wanted or Path(str(item.get("filePath"))).name in wanted
        ]
    latest = latest_snapshot_by_file(snapshots)
    if not latest:
        return {"success": False, "restoredFiles": [], "skippedFiles": file_paths or [], "errors": [f"No snapshots found for messageId: {message_id}"]}
    restored: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    rewind_message_id = new_id("rewind")
    for rel_path, snapshot in latest.items():
        try:
            target = safe_workspace_path(rel_path)
            if target.exists() and target.is_file():
                save_file_snapshot(session_id, rewind_message_id, rel_path, "rewind-before")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".rewind.tmp")
            tmp.write_text(str(snapshot.get("content") or ""), encoding="utf-8")
            tmp.replace(target)
            restored.append(rel_path)
        except Exception as exc:
            errors.append(f"{rel_path}: {exc}")
    if file_paths:
        restored_set = {item.replace("\\", "/") for item in restored}
        skipped = [item for item in file_paths if str(item).replace("\\", "/") not in restored_set]
    return {"success": not errors, "restoredFiles": restored, "skippedFiles": skipped, "errors": errors}


@app.get("/api/sessions/{session_id}/history/snapshots")
async def history_snapshots(session_id: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for snapshot in file_snapshots_for(session_id):
        message_id = str(snapshot.get("messageId") or "")
        grouped.setdefault(message_id, []).append({"filePath": snapshot.get("filePath"), "timestamp": snapshot.get("createdAt")})
    return {
        "snapshots": [
            {
                "messageId": message_id,
                "trackedFiles": [item["filePath"] for item in items],
                "timestamp": items[-1].get("timestamp"),
                "files": items,
            }
            for message_id, items in grouped.items()
        ],
        "byMessage": grouped,
    }


@app.post("/api/sessions/{session_id}/history/snapshot")
async def create_history_snapshot(session_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    message_id = str(payload.get("messageId") or new_id("message"))
    file_paths = payload.get("filePaths") or payload.get("files") or []
    snapshots = []
    errors = []
    for file_path in file_paths:
        try:
            snapshots.append(save_file_snapshot(session_id, message_id, str(file_path), payload.get("operation") or "manual"))
        except Exception as exc:
            errors.append({"filePath": file_path, "error": str(exc)})
    save_state()
    return {"success": not errors, "messageId": message_id, "snapshots": snapshots, "errors": errors}


@app.get("/api/sessions/{session_id}/history/diff")
async def history_diff(session_id: str, fromMessageId: str | None = None, toMessageId: str | None = None) -> dict[str, Any]:
    from_is_current = str(fromMessageId or "").lower() == "current"
    to_is_current = str(toMessageId or "").lower() == "current"
    from_map = {} if from_is_current else (latest_snapshot_by_file(file_snapshots_for(session_id, fromMessageId)) if fromMessageId else {})
    to_map = {} if to_is_current else (latest_snapshot_by_file(file_snapshots_for(session_id, toMessageId)) if toMessageId else {})
    if from_is_current:
        from_map = current_snapshot_by_file(set(to_map))
    if to_is_current:
        to_map = current_snapshot_by_file(set(from_map))
    all_files = sorted(set(from_map) | set(to_map))
    files = []
    diff_parts = []
    for file_path in all_files:
        before = str((from_map.get(file_path) or {}).get("content") or "")
        after = str((to_map.get(file_path) or {}).get("content") or "")
        if before == after:
            continue
        status = "modified" if file_path in from_map and file_path in to_map else ("added" if file_path in to_map else "deleted")
        files.append({"path": file_path, "status": status})
        diff_parts.extend(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"{fromMessageId or 'empty'}/{file_path}",
                tofile=f"{toMessageId or 'empty'}/{file_path}",
                lineterm="",
            )
        )
    return {"sessionId": session_id, "fromMessageId": fromMessageId, "toMessageId": toMessageId, "files": files, "diff": "\n".join(diff_parts)}


@app.post("/api/sessions/{session_id}/history/rewind")
async def history_rewind(session_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    result = rewind_files(session_id, str(payload.get("messageId") or ""), payload.get("filePaths") or payload.get("files"))
    save_state()
    return {"sessionId": session_id, "messageId": payload.get("messageId"), "files": payload.get("filePaths", []), **result}


def skill_search_roots() -> list[tuple[Path, str]]:
    return [
        (ROOT / "skills", "LOCAL"),
        (ROOT / ".codex" / "skills", "LOCAL"),
    ]


def skill_description(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.strip("# ").strip() or fallback
        if stripped:
            return stripped[:160]
    return fallback


def discover_skills() -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {
        name: {
            "name": name,
            "title": skill_description(content, name),
            "description": skill_description(content, name),
            "source": "BUILTIN",
            "filePath": None,
            "content": content,
        }
        for name, content in DEFAULT_SKILLS.items()
    }
    for root, source in skill_search_roots():
        if not root.exists():
            continue
        paths = list(root.glob("*.md")) + list(root.glob("*/SKILL.md"))
        for path in sorted(paths):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            name = path.stem if path.name.lower() != "skill.md" else path.parent.name
            found[name] = {
                "name": name,
                "title": skill_description(content, name),
                "description": skill_description(content, name),
                "source": source,
                "filePath": snapshot_rel_path(path),
                "content": content,
            }
    return sorted(found.values(), key=lambda item: item["name"])


@app.get("/api/skills")
async def list_skills() -> list[dict[str, Any]]:
    return [{key: value for key, value in skill.items() if key != "content"} for skill in discover_skills()]


@app.get("/api/skills/{name}")
async def get_skill(name: str) -> dict[str, Any]:
    for skill in discover_skills():
        if skill["name"] == name:
            return skill
    raise HTTPException(status_code=404, detail="Skill not found")


@app.post("/api/skills/validate-tool")
async def validate_skill_tool(request: Request) -> dict[str, Any]:
    payload = await request.json()
    skill_payload = payload.get("skill") if isinstance(payload.get("skill"), dict) else None
    skill_name = str(payload.get("skillName") or payload.get("skill_name") or (skill_payload or {}).get("name") or "skill")
    skill = skill_payload or next((item for item in discover_skills() if item["name"] == skill_name), {"name": skill_name})
    if payload.get("allowedTools") is not None:
        skill = {**skill, "allowedTools": payload.get("allowedTools")}
    if payload.get("context") is not None:
        skill = {**skill, "context": payload.get("context")}
    tool_name = str(payload.get("toolName") or payload.get("tool_name") or "")
    if not tool_name:
        raise HTTPException(status_code=400, detail="toolName is required")
    result = SKILL_VALIDATOR.validate(
        skill,
        tool_name,
        args=payload.get("args") if isinstance(payload.get("args"), dict) else None,
        nesting_depth=int(payload.get("nestingDepth") or payload.get("nesting_depth") or 0),
    )
    return result.to_dict()


@app.get("/api/tools")
async def list_tools() -> list[dict[str, Any]]:
    sync_mcp_tools()
    return TOOL_REGISTRY.list()


@app.get("/api/tools/{tool_name}")
async def get_tool(tool_name: str) -> dict[str, Any]:
    sync_mcp_tools()
    tool = TOOL_REGISTRY.get(tool_name)
    if tool:
        return tool.api_dict()
    raise HTTPException(status_code=404, detail="Tool not found")


@app.patch("/api/tools/{tool_name}")
async def patch_tool(tool_name: str, request: Request) -> dict[str, Any]:
    updates = await request.json()
    sync_mcp_tools()
    tool = TOOL_REGISTRY.get(tool_name)
    if tool:
        if "enabled" in updates:
            tool.enabled = bool(updates["enabled"])
        return tool.api_dict()
    raise HTTPException(status_code=404, detail="Tool not found")


def plugin_search_roots() -> list[Path]:
    return [ROOT / "plugins", ROOT / ".codex" / "plugins", ROOT / "backend-python" / "plugins"]


SAFE_PLUGIN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def plugin_info_from_manifest(manifest_path: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    plugin_root = manifest_path.parents[1] if manifest_path.parent.name == ".codex-plugin" else manifest_path.parent
    name = str(manifest.get("name") or manifest.get("id") or plugin_root.name)
    commands = manifest.get("commands") or []
    tools = manifest.get("tools") or []
    hooks = manifest.get("hooks") or []
    return {
        "id": name,
        "name": name,
        "version": str(manifest.get("version") or "unknown"),
        "description": str(manifest.get("description") or ""),
        "enabled": bool(manifest.get("enabled", True)),
        "isBuiltin": False,
        "sourceType": "LOCAL",
        "commandCount": len(commands) if isinstance(commands, list) else 0,
        "toolCount": len(tools) if isinstance(tools, list) else 0,
        "hookCount": len(hooks) if isinstance(hooks, list) else 0,
        "path": snapshot_rel_path(plugin_root),
    }


def discover_plugins() -> list[dict[str, Any]]:
    plugins: dict[str, dict[str, Any]] = {}
    for plugin in STATE.setdefault("plugins", []):
        name = str(plugin.get("name") or plugin.get("id") or new_id("plugin"))
        normalized = {
            "id": plugin.get("id") or name,
            "name": name,
            "version": str(plugin.get("version") or "unknown"),
            "description": str(plugin.get("description") or ""),
            "enabled": bool(plugin.get("enabled", True)),
            "isBuiltin": bool(plugin.get("isBuiltin", False)),
            "sourceType": str(plugin.get("sourceType") or "LOCAL"),
            "commandCount": int(plugin.get("commandCount") or 0),
            "toolCount": int(plugin.get("toolCount") or 0),
            "hookCount": int(plugin.get("hookCount") or 0),
            **plugin,
        }
        plugins[name] = normalized
    for root in plugin_search_roots():
        if not root.exists():
            continue
        for manifest_path in sorted(root.glob("*/.codex-plugin/plugin.json")) + sorted(root.glob("*/plugin.json")):
            info = plugin_info_from_manifest(manifest_path)
            if info:
                plugins[info["name"]] = {**plugins.get(info["name"], {}), **info}
    return sorted(plugins.values(), key=lambda item: item["name"])


@app.get("/api/plugins")
async def list_plugins() -> dict[str, Any]:
    return {"plugins": discover_plugins()}


@app.post("/api/plugins/install")
async def install_plugin(request: Request) -> dict[str, Any]:
    plugin = await request.json()
    plugin.setdefault("id", new_id("plugin"))
    plugin.setdefault("name", plugin["id"])
    plugin_id = str(plugin.get("id") or "")
    plugin_name = str(plugin.get("name") or "")
    if not SAFE_PLUGIN_ID_PATTERN.fullmatch(plugin_id) or not SAFE_PLUGIN_ID_PATTERN.fullmatch(plugin_name):
        raise HTTPException(status_code=400, detail="Invalid plugin id or name")
    file_name = plugin.get("fileName") or plugin.get("filename")
    if file_name:
        safe_name = Path(str(file_name)).name
        if safe_name != str(file_name) or not safe_name.endswith(".jar"):
            raise HTTPException(status_code=400, detail="Only safe .jar file names are accepted")
    plugin.setdefault("enabled", True)
    plugin.setdefault("sourceType", "LOCAL")
    plugin.setdefault("installedAt", utc_now())
    STATE["plugins"].append(plugin)
    save_state()
    return plugin


@app.post("/api/plugins/reload")
async def reload_plugins() -> dict[str, Any]:
    plugins = discover_plugins()
    return {
        "success": True,
        "plugins": plugins,
        "loaded": len(plugins),
        "enabled": sum(1 for plugin in plugins if plugin.get("enabled")),
        "disabled": sum(1 for plugin in plugins if not plugin.get("enabled")),
    }


@app.delete("/api/plugins/{plugin_id}")
async def delete_plugin(plugin_id: str) -> dict[str, Any]:
    plugin = next((item for item in discover_plugins() if item.get("id") == plugin_id or item.get("name") == plugin_id), None)
    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")
    if plugin.get("isBuiltin"):
        raise HTTPException(status_code=400, detail="Cannot delete builtin plugin")
    before = len(STATE["plugins"])
    STATE["plugins"] = [p for p in STATE["plugins"] if p.get("id") != plugin_id and p.get("name") != plugin_id]
    save_state()
    return {"success": True, "deleted": plugin_id, "removed": before - len(STATE["plugins"])}


@app.get("/api/hooks")
async def list_hooks(event: str | None = None) -> dict[str, Any]:
    hooks = STATE.setdefault("hooks", [])
    if event:
        hooks = [hook for hook in hooks if str(hook.get("event") or "").upper() == event.upper()]
    return {"hooks": hooks, "total": len(hooks)}


@app.post("/api/hooks")
async def add_hook(request: Request) -> dict[str, Any]:
    hook = await request.json()
    hook.setdefault("id", new_id("hook"))
    hook.setdefault("event", "USER_PROMPT_SUBMIT")
    hook.setdefault("priority", 100)
    hook.setdefault("action", "allow")
    hook.setdefault("enabled", True)
    hook.setdefault("source", "USER")
    hook.setdefault("createdAt", utc_now())
    STATE.setdefault("hooks", []).append(hook)
    save_state()
    return hook


@app.post("/api/hooks/execute")
async def execute_hook_endpoint(request: Request) -> dict[str, Any]:
    payload = await request.json()
    result = execute_hooks(str(payload.get("event") or ""), payload.get("context") or {})
    save_state()
    return result


@app.delete("/api/hooks/{hook_id}")
async def delete_hook(hook_id: str) -> dict[str, Any]:
    hooks = STATE.setdefault("hooks", [])
    before = len(hooks)
    STATE["hooks"] = [hook for hook in hooks if hook.get("id") != hook_id]
    save_state()
    return {"success": len(STATE["hooks"]) < before}


@app.get("/api/hooks/events")
async def list_hook_events(limit: int = 100) -> dict[str, Any]:
    return {"events": STATE.setdefault("hookEvents", [])[-limit:]}


@app.get("/api/permissions/rules")
async def get_permission_rules() -> list[dict[str, Any]]:
    return STATE["permissions"]


@app.put("/api/permissions/rules")
async def put_permission_rules(request: Request) -> list[dict[str, Any]]:
    rules = await request.json()
    STATE["permissions"] = rules if isinstance(rules, list) else []
    save_state()
    rebuild_runtime_registries()
    return STATE["permissions"]


@app.post("/api/permissions/rules")
async def add_permission_rule(request: Request) -> dict[str, Any]:
    rule = await request.json()
    rule.setdefault("id", new_id("rule"))
    STATE["permissions"].append(rule)
    save_state()
    rebuild_runtime_registries()
    return rule


@app.delete("/api/permissions/rules/{rule_id}")
async def delete_permission_rule(rule_id: str) -> dict[str, Any]:
    STATE["permissions"] = [r for r in STATE["permissions"] if r.get("id") != rule_id]
    save_state()
    rebuild_runtime_registries()
    return {"success": True}


def normalize_memory_entry(item: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    entry = dict(item)
    entry.setdefault("id", new_id("memory"))
    entry.setdefault("category", "general")
    entry.setdefault("title", entry.get("summary") or str(entry.get("content") or "")[:60] or "Memory")
    entry.setdefault("content", "")
    entry.setdefault("keywords", "")
    entry.setdefault("scope", "global")
    entry.setdefault("source", "USER")
    entry.setdefault("createdAt", entry.get("created_at") or now)
    entry.setdefault("updatedAt", entry.get("updated_at") or entry["createdAt"])
    return entry


@app.get("/api/memory")
async def get_memory() -> dict[str, Any]:
    entries = [normalize_memory_entry(item) for item in STATE.setdefault("memories", [])]
    return {"entries": entries, "memories": entries}


@app.put("/api/memory")
async def put_memory(request: Request) -> dict[str, Any]:
    payload = await request.json()
    raw_entries = payload.get("entries") or payload.get("memories") if isinstance(payload, dict) else []
    STATE["memories"] = [normalize_memory_entry(item) for item in (raw_entries or [])]
    save_state()
    return {"success": True, "entries": STATE["memories"], "memories": STATE["memories"]}


@app.post("/api/memory")
async def add_memory(request: Request) -> dict[str, Any]:
    item = normalize_memory_entry(await request.json())
    item["createdAt"] = item.get("createdAt") or utc_now()
    item["updatedAt"] = utc_now()
    STATE["memories"].append(item)
    save_state()
    return {"success": True, "id": item["id"], **item}


@app.get("/api/memory/all")
async def get_all_memory() -> dict[str, Any]:
    entries = [normalize_memory_entry(item) for item in STATE.setdefault("memories", [])]
    memory_md = []
    for name in ("MEMORY.md", "PROJECT.md"):
        path = ROOT / name
        if path.exists():
            memory_md.append({"source": name, "category": "file", "timestamp": utc_now(), "content": path.read_text(encoding="utf-8", errors="ignore")})
    return {"sqlite": entries, "memoryMd": memory_md, "entries": entries, "memories": entries}


@app.get("/api/memory/categories")
async def memory_categories() -> dict[str, Any]:
    return {"categories": MEMDIR_SERVICE.categories()}


@app.get("/api/memory/search")
async def memory_search(q: str = "", query: str = "", limit: int = 5, category: str | None = None, rerank: bool = True) -> dict[str, Any]:
    text = query or q
    results = MEMDIR_SERVICE.search(text, max(1, min(limit, 50)), category=category, rerank=rerank)
    return {"query": text, "results": [item.to_dict() for item in results], "total": len(results), "rerank": rerank}


@app.post("/api/memory/search")
async def memory_search_post(request: Request) -> dict[str, Any]:
    payload = await request.json()
    text = str(payload.get("query") or payload.get("q") or "")
    limit = int(payload.get("limit") or payload.get("topK") or 5)
    results = MEMDIR_SERVICE.search(text, max(1, min(limit, 50)), category=payload.get("category"), rerank=bool(payload.get("rerank", True)))
    return {"query": text, "results": [item.to_dict() for item in results], "total": len(results), "rerank": bool(payload.get("rerank", True))}


@app.get("/api/memory/category/{category}")
async def memory_by_category(category: str, limit: int = 10) -> dict[str, Any]:
    entries = MEMDIR_SERVICE.search_by_category(category, max(1, min(limit, 100)))
    return {"category": category, "entries": [item.to_dict() for item in entries], "total": len(entries)}


@app.get("/api/memory/prompt")
async def memory_prompt() -> dict[str, Any]:
    prompt = MEMDIR_SERVICE.build_prompt(ROOT)
    return {"prompt": prompt, "bytes": len(prompt.encode("utf-8"))}


@app.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str) -> dict[str, Any]:
    STATE["memories"] = [m for m in STATE["memories"] if m.get("id") != memory_id]
    save_state()
    return {"success": True}


@app.get("/api/mcp/servers")
async def list_mcp_servers() -> dict[str, Any]:
    MCP_MANAGER.load_state(STATE.setdefault("mcpServers", []))
    return {"servers": STATE.setdefault("mcpServers", []), "connections": MCP_MANAGER.list()}


def normalize_mcp_server(server: dict[str, Any]) -> dict[str, Any]:
    name = str(server.get("name") or server.get("id") or new_id("mcp"))
    normalized = dict(server)
    normalized.setdefault("id", name)
    normalized.setdefault("name", name)
    normalized.setdefault("status", "configured")
    normalized.setdefault("type", normalized.get("transport") or "sse")
    normalized.setdefault("scope", "dynamic")
    normalized.setdefault("resources", [])
    normalized.setdefault("prompts", [])
    normalized.setdefault("tools", [])
    normalized.setdefault("logs", [])
    return normalized


def find_mcp_server(name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    for server in STATE.setdefault("mcpServers", []):
        if server.get("name") == name or server.get("id") == name:
            return server
    return None


@app.post("/api/mcp/servers")
async def add_mcp_server(request: Request) -> dict[str, Any]:
    server = normalize_mcp_server(await request.json())
    server["updatedAt"] = utc_now()
    server.setdefault("createdAt", server["updatedAt"])
    server.setdefault("logs", []).append(f"{utc_now()} configured")
    servers = STATE.setdefault("mcpServers", [])
    servers[:] = [item for item in servers if item.get("name") != server["name"] and item.get("id") != server["name"]]
    servers.append(server)
    MCP_MANAGER.load_state(servers)
    sync_mcp_tools()
    save_state()
    return {"success": True, "server": server, "connection": MCP_MANAGER.get(server["name"]).to_dict() if MCP_MANAGER.get(server["name"]) else None}


@app.delete("/api/mcp/servers/{name}")
async def delete_mcp_server(name: str) -> dict[str, Any]:
    servers = STATE.setdefault("mcpServers", [])
    servers[:] = [item for item in servers if item.get("name") != name and item.get("id") != name]
    MCP_MANAGER.remove_server(name)
    sync_mcp_tools()
    save_state()
    return {"success": True, "server": name}


@app.post("/api/mcp/servers/{name}/restart")
async def restart_mcp_server(name: str) -> dict[str, Any]:
    server = find_mcp_server(name)
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    MCP_MANAGER.load_state(STATE.setdefault("mcpServers", []))
    connection = MCP_MANAGER.connect_server(name)
    server["status"] = connection.status.value
    server["tools"] = list(connection.tools)
    server["resources"] = list(connection.resources)
    server["prompts"] = list(connection.prompts)
    server["lastError"] = connection.lastError
    server["nextRetryAt"] = connection.nextRetryAt
    server["reconnectAttempts"] = int(connection.reconnectAttempts or int(server.get("reconnectAttempts") or 0) + 1)
    server["restartedAt"] = utc_now()
    server.setdefault("logs", []).append(f"{utc_now()} restarted status={connection.status.value}")
    sync_mcp_tools()
    save_state()
    return {
        "success": connection.status == McpConnectionStatus.CONNECTED,
        "server": name,
        "status": connection.status.value,
        "backoffMs": MCP_MANAGER.calculate_backoff(max(1, int(server["reconnectAttempts"]))),
        "connection": connection.to_dict(),
    }


@app.get("/api/mcp/servers/{name}/logs")
async def get_mcp_logs(name: str) -> dict[str, Any]:
    server = find_mcp_server(name)
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return {"server": name, "logs": server.get("logs", [])[-100:]}


@app.get("/api/mcp/status")
async def mcp_status() -> dict[str, Any]:
    MCP_MANAGER.load_state(STATE.setdefault("mcpServers", []))
    schema_validation = MCP_MANAGER.validate_tool_schemas()
    return {
        "running": True,
        "connectionCount": len(MCP_MANAGER.connections),
        "connections": MCP_MANAGER.list(),
        "authFailures": MCP_MANAGER.authFailures.to_dict(),
        "trustedServers": sorted(MCP_MANAGER.approvals._trusted),
        "schemaValidation": schema_validation,
    }


@app.get("/api/mcp/tools/wrapped")
async def mcp_wrapped_tools() -> dict[str, Any]:
    MCP_MANAGER.load_state(STATE.setdefault("mcpServers", []))
    sync_mcp_tools()
    tools = MCP_MANAGER.discover_wrapped_tools()
    return {"tools": tools, "totalCount": len(tools)}


@app.post("/api/mcp/json-rpc")
async def mcp_json_rpc(request: Request) -> dict[str, Any]:
    message = JsonRpcMessage.parse(await request.json())
    return {"message": message.to_dict()}


@app.get("/api/mcp/auth-failures")
async def mcp_auth_failures() -> dict[str, Any]:
    return MCP_MANAGER.authFailures.to_dict()


@app.post("/api/mcp/auth-failures/{server}")
async def mcp_record_auth_failure(server: str) -> dict[str, Any]:
    MCP_MANAGER.authFailures.record(server)
    item = find_mcp_server(server)
    if item is not None:
        item["status"] = "failed"
        item["lastError"] = "auth_failure"
        item["updatedAt"] = utc_now()
        item["reconnectAttempts"] = int(item.get("reconnectAttempts") or 0) + 1
        item["nextRetryAt"] = time.time() + (MCP_MANAGER.calculate_backoff(item["reconnectAttempts"]) / 1000)
        item.setdefault("logs", []).append(f"{utc_now()} auth failure cached")
        save_state()
    return {"success": True, "cached": MCP_MANAGER.authFailures.is_cached(server), **MCP_MANAGER.authFailures.to_dict()}


@app.delete("/api/mcp/auth-failures/{server}")
async def mcp_clear_auth_failure(server: str) -> dict[str, Any]:
    MCP_MANAGER.authFailures.clear(server)
    return {"success": True, "cached": MCP_MANAGER.authFailures.is_cached(server)}


@app.post("/api/mcp/approvals/{server}")
async def mcp_approval(server: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    decision = str(payload.get("decision") or "deny").lower()
    if decision not in {"allow", "deny"}:
        raise HTTPException(status_code=400, detail="decision must be allow or deny")
    record = MCP_MANAGER.approvals.decide(server, decision, payload.get("reason"))
    return {"success": True, "trusted": MCP_MANAGER.approvals.is_trusted(server), "decision": record}


@app.post("/api/mcp/tokens/encrypt")
async def mcp_encrypt_token(request: Request) -> dict[str, Any]:
    payload = await request.json()
    token = str(payload.get("token") or "")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    encrypted = MCP_MANAGER.tokens.encrypt(token)
    return {"encrypted": encrypted, "roundTrip": MCP_MANAGER.tokens.decrypt(encrypted) == token}


@app.get("/api/mcp/resources")
async def list_mcp_resources(server: str | None = None) -> dict[str, Any]:
    servers = [find_mcp_server(server)] if server else STATE.setdefault("mcpServers", [])
    grouped: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for item in servers:
        if not item:
            continue
        name = str(item.get("name") or item.get("id"))
        resources = []
        for resource in item.get("resources") or []:
            if isinstance(resource, str):
                resource = {"uri": resource, "name": resource}
            resources.append(
                {
                    "uri": resource.get("uri") or resource.get("path") or resource.get("name"),
                    "name": resource.get("name") or resource.get("uri") or "resource",
                    "description": resource.get("description") or "",
                    "mimeType": resource.get("mimeType") or resource.get("mime_type") or "text/plain",
                    "serverName": name,
                    "content": resource.get("content"),
                }
            )
        grouped[name] = resources
        total += len(resources)
    return {"resources": grouped, "totalCount": total}


@app.get("/api/mcp/resources/read")
async def read_mcp_resource(uri: str, server: str) -> dict[str, str]:
    item = find_mcp_server(server)
    if not item:
        raise HTTPException(status_code=404, detail="MCP server not found")
    for resource in item.get("resources") or []:
        if isinstance(resource, str):
            resource = {"uri": resource}
        if resource.get("uri") == uri or resource.get("path") == uri:
            if "content" in resource:
                return {"uri": uri, "serverName": server, "content": str(resource.get("content") or "")}
            path_value = resource.get("path")
            if path_value:
                path = safe_workspace_path(str(path_value))
                return {"uri": uri, "serverName": server, "content": path.read_text(encoding="utf-8", errors="replace")[:200_000]}
    raise HTTPException(status_code=404, detail="MCP resource not found")


@app.get("/api/mcp/prompts")
async def list_mcp_prompts(server: str | None = None) -> dict[str, Any]:
    servers = [find_mcp_server(server)] if server else STATE.setdefault("mcpServers", [])
    grouped: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for item in servers:
        if not item:
            continue
        name = str(item.get("name") or item.get("id"))
        prompts = []
        for prompt in item.get("prompts") or []:
            prompts.append(
                {
                    "name": prompt.get("name") or "prompt",
                    "description": prompt.get("description") or "",
                    "serverName": name,
                    "arguments": prompt.get("arguments") or [],
                    "template": prompt.get("template") or prompt.get("content") or "",
                }
            )
        grouped[name] = prompts
        total += len(prompts)
    return {"prompts": grouped, "totalCount": total}


@app.post("/api/mcp/prompts/execute")
async def execute_mcp_prompt(request: Request) -> dict[str, Any]:
    payload = await request.json()
    server_name = str(payload.get("server") or "")
    prompt_name = str(payload.get("promptName") or "")
    item = find_mcp_server(server_name)
    if not item:
        return {"success": False, "serverName": server_name, "promptName": prompt_name, "error": "MCP server not found"}
    arguments = payload.get("arguments") or {}
    for prompt in item.get("prompts") or []:
        if prompt.get("name") != prompt_name:
            continue
        template = str(prompt.get("template") or prompt.get("content") or "")
        for key, value in arguments.items():
            template = template.replace("{{" + str(key) + "}}", str(value))
        return {
            "success": True,
            "serverName": server_name,
            "promptName": prompt_name,
            "messages": [{"role": "user", "content": template}],
        }
    return {"success": False, "serverName": server_name, "promptName": prompt_name, "error": "Prompt not found"}


@app.post("/api/mcp/reconnect")
async def reconnect_mcp(server: str | None = None) -> dict[str, Any]:
    item = find_mcp_server(server)
    if not item:
        return {"success": False, "serverName": server, "status": "not_found"}
    MCP_MANAGER.load_state(STATE.setdefault("mcpServers", []))
    server_name = str(item.get("name") or item.get("id"))
    reconnect_plan = MCP_MANAGER.plan_reconnect(server_name)
    if not reconnect_plan.get("allowed"):
        item["status"] = "failed"
        item["lastError"] = reconnect_plan.get("reason")
        item["nextRetryAt"] = reconnect_plan.get("nextRetryAt")
        item.setdefault("logs", []).append(f"{utc_now()} reconnect blocked: {reconnect_plan.get('reason')}")
        save_state()
        return {
            "success": False,
            "serverName": server_name,
            "status": "blocked",
            "reason": reconnect_plan.get("reason"),
            "backoffMs": reconnect_plan.get("backoffMs"),
            "nextRetryAt": reconnect_plan.get("nextRetryAt"),
        }
    item["status"] = "connected"
    item["updatedAt"] = utc_now()
    item["reconnectAttempts"] = int(item.get("reconnectAttempts") or 0) + 1
    item["nextRetryAt"] = None
    item["lastError"] = None
    item.setdefault("logs", []).append(f"{utc_now()} reconnected")
    MCP_MANAGER.load_state(STATE.setdefault("mcpServers", []))
    save_state()
    connection = MCP_MANAGER.get(server_name)
    return {
        "success": True,
        "serverName": server,
        "status": "CONNECTED",
        "backoffMs": MCP_MANAGER.calculate_backoff(item["reconnectAttempts"]),
        "connection": connection.to_dict() if connection else None,
    }


@app.get("/api/mcp/capabilities/domains")
async def capability_domains() -> dict[str, Any]:
    domains = sorted({c.get("domain", "general") for c in STATE["mcpCapabilities"]})
    return {"domains": domains}


@app.get("/api/mcp/capabilities")
async def list_capabilities(domain: str | None = None, enabled: bool | None = None) -> dict[str, Any]:
    caps = STATE["mcpCapabilities"]
    if domain:
        caps = [c for c in caps if c.get("domain") == domain]
    if enabled is not None:
        caps = [c for c in caps if bool(c.get("enabled")) == enabled]
    return {"capabilities": caps, "total": len(caps), "enabledCount": sum(1 for c in caps if c.get("enabled"))}


@app.post("/api/mcp/capabilities")
async def add_capability(request: Request) -> dict[str, Any]:
    cap = await request.json()
    cap.setdefault("id", new_id("cap"))
    cap.setdefault("name", cap.get("toolName") or cap["id"])
    cap.setdefault("domain", "general")
    cap.setdefault("category", "tool")
    cap.setdefault("timeoutMs", 30000)
    cap.setdefault("enabled", True)
    STATE["mcpCapabilities"].append(cap)
    save_state()
    return cap


@app.get("/api/mcp/capabilities/{cap_id}")
async def get_capability(cap_id: str) -> dict[str, Any]:
    for cap in STATE["mcpCapabilities"]:
        if cap.get("id") == cap_id:
            return cap
    raise HTTPException(status_code=404, detail="Capability not found")


@app.put("/api/mcp/capabilities/{cap_id}")
async def update_capability(cap_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    for index, cap in enumerate(STATE["mcpCapabilities"]):
        if cap.get("id") == cap_id:
            payload["id"] = cap_id
            STATE["mcpCapabilities"][index] = payload
            save_state()
            return payload
    raise HTTPException(status_code=404, detail="Capability not found")


@app.patch("/api/mcp/capabilities/{cap_id}/toggle")
async def toggle_capability(cap_id: str, enabled: bool = True) -> dict[str, Any]:
    for cap in STATE["mcpCapabilities"]:
        if cap.get("id") == cap_id:
            cap["enabled"] = enabled
            save_state()
            return {"id": cap_id, "status": "enabled" if enabled else "disabled", "enabled": enabled}
    raise HTTPException(status_code=404, detail="Capability not found")


@app.delete("/api/mcp/capabilities/{cap_id}")
async def delete_capability(cap_id: str) -> dict[str, Any]:
    STATE["mcpCapabilities"] = [c for c in STATE["mcpCapabilities"] if c.get("id") != cap_id]
    save_state()
    return {"success": True}


@app.get("/api/mcp/capabilities/{cap_id}/server-tools")
async def capability_tools(cap_id: str) -> dict[str, Any]:
    cap = next((item for item in STATE["mcpCapabilities"] if item.get("id") == cap_id), None)
    if not cap:
        raise HTTPException(status_code=404, detail="Capability not found")
    server_key = cap.get("server") or cap.get("serverName") or cap.get("serverKey")
    server = find_mcp_server(str(server_key)) if server_key else None
    tools = list(server.get("tools") or []) if server else []
    local_tool = TOOL_REGISTRY.get(str(cap.get("toolName") or ""))
    if local_tool:
        tools.append(local_tool.api_dict())
    return {"id": cap_id, "capabilityId": cap_id, "serverKey": server_key, "tools": tools, "status": "connected" if server else "local"}


@app.post("/api/mcp/capabilities/{cap_id}/test")
async def test_capability(cap_id: str) -> dict[str, Any]:
    cap = next((item for item in STATE["mcpCapabilities"] if item.get("id") == cap_id), None)
    if not cap:
        raise HTTPException(status_code=404, detail="Capability not found")
    tool_name = str(cap.get("toolName") or "")
    server_key = cap.get("server") or cap.get("serverName") or cap.get("serverKey")
    if TOOL_REGISTRY.get(tool_name):
        return {"id": cap_id, "status": "reachable", "serverKey": server_key or "local", "toolName": tool_name}
    server = find_mcp_server(str(server_key)) if server_key else None
    if server:
        exists = any((tool.get("name") if isinstance(tool, dict) else str(tool)) == tool_name for tool in server.get("tools") or [])
        return {"id": cap_id, "status": "reachable" if exists or not tool_name else "configured", "serverKey": server_key}
    return {"id": cap_id, "status": "error", "error": "No local tool or MCP server configured"}


@app.post("/api/mcp/capabilities/{cap_id}/invoke")
async def invoke_capability(cap_id: str, request: Request) -> dict[str, Any]:
    cap = next((item for item in STATE["mcpCapabilities"] if item.get("id") == cap_id), None)
    if not cap:
        raise HTTPException(status_code=404, detail="Capability not found")
    body = await request.json()
    arguments = body.get("arguments") if isinstance(body, dict) and "arguments" in body else body
    tool_name = str(cap.get("toolName") or "")
    local_tool = TOOL_REGISTRY.get(tool_name)
    if local_tool:
        result = TOOL_REGISTRY.call(tool_name, arguments if isinstance(arguments, dict) else {})
        return {
            "id": cap_id,
            "status": "success" if not result.isError else "error",
            "toolName": tool_name,
            "connectionType": "local",
            "result": result.to_dict(),
        }
    server_key = cap.get("server") or cap.get("serverName") or cap.get("serverKey")
    server = find_mcp_server(str(server_key)) if server_key else None
    if server:
        for tool in server.get("tools") or []:
            if isinstance(tool, dict) and tool.get("name") == tool_name:
                MCP_MANAGER.load_state(STATE.setdefault("mcpServers", []))
                result = MCP_MANAGER.call_tool(str(server_key), tool_name, arguments if isinstance(arguments, dict) else {}, int(cap.get("timeoutMs") or 30000))
                server.setdefault("logs", []).append(f"{utc_now()} tools/call {tool_name} status={result.get('status')}")
                save_state()
                return {
                    "id": cap_id,
                    "status": result.get("status") or "error",
                    "toolName": tool_name,
                    "connectionType": result.get("connectionType") or server.get("type") or "mcp",
                    "result": result,
                }
    return {"id": cap_id, "status": "error", "error": "Invocation target not available", "toolName": tool_name}


@app.post("/api/dialogs/snapshot-update/{request_id}/decision")
async def snapshot_decision(request_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    action = str(payload.get("action") or payload.get("decision") or "merge")
    decision = {"requestId": request_id, "type": "snapshot-update", "action": action, "payload": payload, "resolvedAt": utc_now()}
    STATE.setdefault("dialogDecisions", {})[request_id] = decision
    save_state()
    return {"success": True, "requestId": request_id, "action": action, "decision": decision}


@app.post("/api/dialogs/plugin-permission/{request_id}/decision")
async def plugin_permission_decision(request_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    allowed = bool(payload.get("allowed", payload.get("approved", payload.get("allow", False))))
    decision = "allow" if allowed else "deny"
    record = {"requestId": request_id, "type": "plugin-permission", "allowed": allowed, "decision": decision, "payload": payload, "resolvedAt": utc_now()}
    STATE.setdefault("dialogDecisions", {})[request_id] = record
    STATE.setdefault("permissionResponses", {})[request_id] = record
    save_state()
    return {"success": True, "requestId": request_id, "allowed": allowed, "decision": record}


BROWSER_REPLAY_SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def validate_browser_replay_session(session_id: str) -> None:
    if not BROWSER_REPLAY_SESSION_PATTERN.fullmatch(session_id or ""):
        raise HTTPException(status_code=400, detail="Invalid browser replay sessionId")


@app.get("/api/browser/replay/{session_id}")
async def browser_replay(session_id: str) -> dict[str, Any]:
    validate_browser_replay_session(session_id)
    return {"sessionId": session_id, "snapshots": STATE.setdefault("browserReplay", {}).get(session_id, [])}


@app.delete("/api/browser/replay/{session_id}")
async def delete_browser_replay(session_id: str) -> dict[str, Any]:
    validate_browser_replay_session(session_id)
    STATE.setdefault("browserReplay", {}).pop(session_id, None)
    save_state()
    return {"success": True, "sessionId": session_id}


@app.post("/api/browser/replay/{session_id}")
async def add_browser_replay(session_id: str, request: Request) -> dict[str, Any]:
    validate_browser_replay_session(session_id)
    payload = await request.json()
    snapshot = {
        "id": payload.get("id") or new_id("browser"),
        "sessionId": session_id,
        "url": payload.get("url") or "",
        "title": payload.get("title") or "",
        "selector": payload.get("selector"),
        "html": payload.get("html") or payload.get("dom") or "",
        "screenshot": payload.get("screenshot"),
        "createdAt": payload.get("createdAt") or utc_now(),
        "meta": payload.get("meta") or {},
    }
    timeline = STATE.setdefault("browserReplay", {}).setdefault(session_id, [])
    timeline.append(snapshot)
    del timeline[:-100]
    save_state()
    return {"success": True, "snapshot": snapshot}


@app.get("/api/files/search")
async def search_files(query: str = "", limit: int = 20) -> dict[str, Any]:
    results = []
    if query:
        q = query.lower()
        ignored = {".git", "node_modules", "venv", ".venv", "dist", "target", "__pycache__"}
        for path in ROOT.rglob("*"):
            if len(results) >= limit:
                break
            if not path.is_file() or any(part in ignored for part in path.parts):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if q in rel.lower():
                results.append({"path": rel, "name": path.name})
    return {"files": results, "results": results}


FILE_TREE_IGNORES = {".git", "node_modules", "venv", ".venv", "dist", "target", "__pycache__", ".pytest_cache"}


def safe_workspace_path(value: str | None, default: Path | None = None) -> Path:
    base = default or ROOT
    raw = value or str(base)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    root_resolved = ROOT.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise HTTPException(status_code=400, detail="Path escapes workspace")
    return resolved


def build_file_tree(path: Path, depth: int = 0, max_depth: int = 4, max_children: int = 250) -> dict[str, Any]:
    rel = "." if path == ROOT else path.relative_to(ROOT).as_posix()
    if path.is_file():
        return {
            "name": path.name,
            "path": rel,
            "type": "file",
            "size": path.stat().st_size,
            "extension": path.suffix.lstrip("."),
        }
    children = []
    if depth < max_depth:
        try:
            entries = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        except OSError:
            entries = []
        for child in entries:
            if child.name in FILE_TREE_IGNORES:
                continue
            children.append(build_file_tree(child, depth + 1, max_depth, max_children))
            if len(children) >= max_children:
                break
    return {"name": path.name or str(path), "path": rel, "type": "dir", "children": children}


@app.get("/api/files/version")
async def file_version(path: str) -> dict[str, Any]:
    resolved = safe_workspace_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    content_hash = FILE_VERSION_TRACKER.record_read(resolved)
    return {"path": snapshot_rel_path(resolved), "contentHash": content_hash, "version": FILE_VERSION_TRACKER.version_for(resolved)}


@app.post("/api/files/recovery")
async def file_recovery(request: Request) -> dict[str, Any]:
    payload = await request.json()
    decision = FILE_RECOVERY_POLICY.recover(
        str(payload.get("toolName") or payload.get("tool_name") or "edit_file"),
        str(payload.get("error") or payload.get("errorMessage") or ""),
    )
    return decision.to_dict()


@app.post("/api/files/tree")
async def files_tree(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        root_path = safe_workspace_path(payload.get("root_path") or payload.get("rootPath"))
    except HTTPException as exc:
        return {"success": False, "data": None, "error_code": "PATH_DENIED", "error_message": exc.detail}
    if not root_path.exists():
        return {"success": False, "data": None, "error_code": "NOT_FOUND", "error_message": "Root path not found"}
    return {"success": True, "data": build_file_tree(root_path)}


def api_success(data: Any) -> dict[str, Any]:
    return {"success": True, "data": data, "error_code": None, "error_message": None}


def api_error(code: str, message: str) -> dict[str, Any]:
    return {"success": False, "data": None, "error_code": code, "error_message": message}


def run_git(repo_path: Path, args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def git_repo_from_payload(payload: dict[str, Any]) -> Path:
    repo_path = safe_workspace_path(str(payload.get("repo_path") or payload.get("repoPath") or "."))
    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError("Repository path not found")
    probe = run_git(repo_path, ["rev-parse", "--git-dir"], timeout=5.0)
    if probe.returncode != 0:
        raise ValueError("Path is not a Git repository")
    return repo_path


def git_error_response(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, subprocess.TimeoutExpired):
        return api_error("GIT_TIMEOUT", "Git command timed out")
    return api_error("GIT_ERROR", str(exc))


@app.post("/api/git/log")
async def git_log(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        repo_path = git_repo_from_payload(payload)
        max_count = max(1, min(int(payload.get("max_count") or payload.get("maxCount") or 20), 100))
        branch = str(payload.get("branch") or "").strip()
        args = ["log", f"-n{max_count}", "--pretty=format:%H%x1f%an%x1f%aI%x1f%s", "--name-only"]
        if branch:
            args.append(branch)
        result = run_git(repo_path, args)
        if result.returncode != 0:
            return api_error("GIT_LOG_FAILED", result.stderr.strip() or result.stdout.strip() or "git log failed")
        commits: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for line in result.stdout.splitlines():
            if "\x1f" in line:
                if current:
                    commits.append(current)
                sha, author, date, message = (line.split("\x1f", 3) + ["", "", "", ""])[:4]
                current = {"sha": sha, "author": author, "date": date, "message": message, "files": []}
            elif current is not None and line.strip():
                current["files"].append(line.strip())
        if current:
            commits.append(current)
        return api_success({"commits": commits, "total": len(commits)})
    except Exception as exc:
        return git_error_response(exc)


@app.post("/api/git/diff")
async def git_diff(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        repo_path = git_repo_from_payload(payload)
        ref1 = str(payload.get("ref1") or "HEAD")
        ref2 = str(payload.get("ref2") or "")
        refs = [ref1, ref2] if ref2 else [ref1]
        stat = run_git(repo_path, ["diff", "--stat", *refs])
        detailed = run_git(repo_path, ["diff", *refs, "--", "."])
        if detailed.returncode != 0:
            return api_error("GIT_DIFF_FAILED", detailed.stderr.strip() or detailed.stdout.strip() or "git diff failed")
        changed = sum(1 for line in stat.stdout.splitlines() if " | " in line)
        return api_success({"summary": stat.stdout.strip(), "detailed": detailed.stdout[:200_000], "files_changed": changed})
    except Exception as exc:
        return git_error_response(exc)


@app.post("/api/git/blame")
async def git_blame(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        repo_path = git_repo_from_payload(payload)
        file_path = str(payload.get("file_path") or payload.get("filePath") or "")
        if not file_path:
            return api_error("MISSING_FILE", "file_path is required")
        checked = safe_workspace_path((repo_path / file_path).as_posix())
        if repo_path.resolve() != checked.resolve() and repo_path.resolve() not in checked.resolve().parents:
            return api_error("PATH_DENIED", "File path escapes repository")
        ref = str(payload.get("ref") or "").strip()
        args = ["blame", "--line-porcelain"]
        if ref:
            args.append(ref)
        args.extend(["--", file_path])
        result = run_git(repo_path, args)
        if result.returncode != 0:
            return api_error("GIT_BLAME_FAILED", result.stderr.strip() or result.stdout.strip() or "git blame failed")
        lines: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        for raw in result.stdout.splitlines():
            if re.match(r"^[0-9a-f]{40} ", raw):
                parts = raw.split()
                current = {"sha": parts[0], "line_no": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else len(lines) + 1}
            elif raw.startswith("author "):
                current["author"] = raw.removeprefix("author ")
            elif raw.startswith("author-time "):
                current["date"] = raw.removeprefix("author-time ")
            elif raw.startswith("\t"):
                current["content"] = raw[1:]
                current.setdefault("author", "")
                current.setdefault("date", "")
                lines.append(current)
                current = {}
        return api_success({"file_path": file_path, "lines": lines, "total_lines": len(lines)})
    except Exception as exc:
        return git_error_response(exc)


def safe_attachment_extension(filename: str | None) -> str:
    suffix = Path(filename or "").suffix
    if suffix and re.fullmatch(r"\.[A-Za-z0-9]{1,16}", suffix):
        return suffix
    return ""


def parse_attachment_upload(content_type: str, body: bytes) -> tuple[str, bytes, str]:
    if "multipart/form-data" not in content_type.lower():
        return "attachment", body, content_type or "application/octet-stream"
    raw = b"Content-Type: " + content_type.encode("utf-8", errors="ignore") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    message = BytesParser(policy=email_policy.default).parsebytes(raw)
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        if part.get_param("name", header="content-disposition") != "file":
            continue
        filename = part.get_filename() or "attachment"
        payload = part.get_payload(decode=True) or b""
        return filename, payload, part.get_content_type() or "application/octet-stream"
    raise HTTPException(status_code=400, detail="Missing multipart file field")


def find_attachment_path(file_uuid: str) -> Path | None:
    for path in UPLOAD_DIR.glob(f"{file_uuid}*"):
        if path.is_file():
            return path
    return None


@app.post("/api/attachments/upload")
async def upload_attachment(request: Request) -> JSONResponse:
    content_type = request.headers.get("content-type", "application/octet-stream")
    body = await request.body()
    filename, content, mime_type = parse_attachment_upload(content_type, body)
    if len(content) > MAX_ATTACHMENT_SIZE:
        return JSONResponse(
            {"fileUuid": None, "fileName": filename, "size": len(content), "error": "File too large (max 10MB)"},
            status_code=400,
        )
    file_uuid = uuid.uuid4().hex
    file_path = UPLOAD_DIR / f"{file_uuid}{safe_attachment_extension(filename)}"
    file_path.write_bytes(content)
    return JSONResponse(
        {"fileUuid": file_uuid, "fileName": filename, "size": len(content), "mimeType": mime_type, "error": None},
        status_code=201,
    )


@app.get("/api/attachments/{file_uuid}")
async def get_attachment(file_uuid: str) -> Response:
    path = find_attachment_path(file_uuid)
    if not path:
        raise HTTPException(status_code=404, detail="Attachment not found")
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream", filename=path.name)


@app.post("/api/code-diagrams/generate")
async def generate_code_diagram(request: Request) -> dict[str, Any]:
    payload = await request.json()
    api_path = payload.get("apiPath") or payload.get("path") or "/api/example"
    endpoints = scan_api_endpoints(safe_workspace_path(payload.get("project_root") or payload.get("projectRoot") or "."))
    matched = next((item for item in endpoints if item["path"] == api_path), None)
    if payload.get("type", "sequence") == "flowchart":
        diagram = "flowchart TD\n    Client[Client]\n    Backend[Python Backend]\n"
        if matched:
            diagram += f"    Backend --> Handler[{matched['handler_function']}]\n    Handler --> File[{matched['file_path']}:{matched['line_number']}]\n"
        else:
            diagram += f"    Client --> Backend\n    Backend --> Missing[{api_path}]\n"
    else:
        handler = matched["handler_function"] if matched else "route handler"
        diagram = f"sequenceDiagram\n    participant Client\n    participant Backend\n    participant Handler\n    Client->>Backend: {api_path}\n    Backend->>Handler: {handler}\n    Handler-->>Client: JSON response"
    return {"success": True, "diagram": diagram, "mermaidSyntax": diagram, "type": payload.get("type", "sequence"), "apiPath": api_path, "matchedEndpoint": matched}


ROUTE_PATTERNS = [
    ("python", re.compile(r"@(?:app|router)\.(get|post|put|patch|delete|api_route)\(\s*[\"']([^\"']+)[\"']")),
    ("java", re.compile(r"@(GetMapping|PostMapping|PutMapping|PatchMapping|DeleteMapping|RequestMapping)\(\s*(?:value\s*=\s*)?[\"']([^\"']+)[\"']")),
    ("typescript", re.compile(r"\b(?:app|router)\.(get|post|put|patch|delete)\(\s*[\"'`]([^\"'`]+)[\"'`]")),
]


def scan_api_endpoints(project_root: Path, languages: list[str] | None = None) -> list[dict[str, Any]]:
    allowed = {lang.lower() for lang in languages or []}
    endpoints: list[dict[str, Any]] = []
    suffix_lang = {".py": "python", ".java": "java", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript"}
    for path in project_root.rglob("*"):
        if len(endpoints) >= 2000:
            break
        if not path.is_file() or path.suffix.lower() not in suffix_lang:
            continue
        rel_parts = path.relative_to(project_root).parts
        if any(part in FILE_TREE_IGNORES for part in rel_parts):
            continue
        lang = suffix_lang[path.suffix.lower()]
        if allowed and lang not in allowed:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for index, line in enumerate(lines, start=1):
            for pattern_lang, pattern in ROUTE_PATTERNS:
                if pattern_lang != lang and not (pattern_lang == "typescript" and lang == "javascript"):
                    continue
                match = pattern.search(line)
                if not match:
                    continue
                method_raw, route = match.group(1), match.group(2)
                method = method_raw.replace("Mapping", "").replace("Request", "").upper() or "ANY"
                handler = "handler"
                for next_line in lines[index:index + 6]:
                    handler_match = re.search(r"(?:def|function|const|public|private|protected)\s+([A-Za-z_][A-Za-z0-9_]*)", next_line)
                    if handler_match:
                        handler = handler_match.group(1)
                        break
                endpoints.append(
                    {
                        "http_method": method,
                        "method": method,
                        "path": route,
                        "handler_function": handler,
                        "handlerFunction": handler,
                        "handler_class": "",
                        "file_path": path.relative_to(project_root).as_posix(),
                        "filePath": path.relative_to(project_root).as_posix(),
                        "line_number": index,
                        "lineNumber": index,
                        "language": lang,
                        "parameters": [],
                    }
                )
    return endpoints


@app.post("/api/code-path/endpoints")
async def code_path_endpoints(request: Request) -> dict[str, Any]:
    payload = await request.json()
    root = safe_workspace_path(payload.get("project_root") or payload.get("projectRoot") or ".")
    endpoints = scan_api_endpoints(root, payload.get("languages"))
    return {"success": True, "endpoints": endpoints, "total": len(endpoints)}


@app.post("/api/code-path/trace")
async def code_path_trace(request: Request) -> dict[str, Any]:
    payload = await request.json()
    root = safe_workspace_path(payload.get("project_root") or payload.get("projectRoot") or ".")
    target_path = payload.get("path") or payload.get("apiPath")
    entry_file = payload.get("entry_file") or payload.get("entryFile")
    entry_function = payload.get("entry_function") or payload.get("entryFunction")
    endpoints = scan_api_endpoints(root, payload.get("languages"))
    matched = next((item for item in endpoints if item["path"] == target_path), None)
    if not matched and entry_file:
        matched = {"file_path": entry_file, "handler_function": entry_function or Path(str(entry_file)).stem, "line_number": 1, "path": target_path or entry_file, "language": Path(str(entry_file)).suffix.lstrip(".")}
    if not matched:
        return {"success": False, "error": "Entry point not found", "nodes": [], "edges": [], "layers": [], "steps": []}
    node = {
        "id": "entry",
        "name": matched.get("handler_function") or matched.get("handlerFunction"),
        "class_name": matched.get("handler_class", ""),
        "file_path": matched.get("file_path"),
        "line_range": [matched.get("line_number", 1), matched.get("line_number", 1)],
        "layer": "controller",
        "node_type": "endpoint",
        "annotations": [str(matched.get("method") or matched.get("http_method") or "")],
        "parameters": [],
        "return_type": "JSON",
    }
    return {"success": True, "path": matched.get("path"), "nodes": [node], "edges": [], "layers": [{"layer": "controller", "node_count": 1, "description": "API route handler"}], "steps": [{"name": node["name"], "file": node["file_path"], "line": node["line_range"][0]}]}


LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".java": "java",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
}


def complexity_risk(cc: int, loc: int) -> str:
    if cc >= 30 or loc >= 800:
        return "E"
    if cc >= 20 or loc >= 500:
        return "D"
    if cc >= 10 or loc >= 250:
        return "C"
    if cc >= 5 or loc >= 100:
        return "B"
    return "A"


def simple_complexity(text: str) -> int:
    branch_pattern = r"\b(if|elif|for|while|case|catch|except|and|or|switch|try)\b|\?|&&|\|\|"
    return 1 + len(re.findall(branch_pattern, text))


def maintainability_index(loc: int, cc: int) -> int:
    score = 100 - min(60, cc * 3) - min(35, loc // 25)
    return max(0, min(100, score))


def analyze_file_complexity(path: Path, root: Path) -> dict[str, Any] | None:
    language = LANGUAGE_EXTENSIONS.get(path.suffix.lower())
    if not language:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    loc = sum(1 for line in text.splitlines() if line.strip())
    cc = simple_complexity(text)
    return {
        "name": path.name,
        "type": "file",
        "loc": loc,
        "cc": cc,
        "mi": maintainability_index(loc, cc),
        "risk_level": complexity_risk(cc, loc),
        "file_path": path.relative_to(root).as_posix(),
        "language": language,
    }


@app.post("/api/code-quality/complexity")
async def code_quality_complexity(request: Request) -> dict[str, Any]:
    start = time.time()
    payload = await request.json()
    try:
        project_root = safe_workspace_path(payload.get("project_root") or payload.get("projectRoot") or ".")
        target_raw = payload.get("target_path") or payload.get("targetPath")
        target_path = safe_workspace_path(target_raw, project_root) if target_raw else project_root
        languages = set(payload.get("languages") or LANGUAGE_EXTENSIONS.values())
        nodes: list[dict[str, Any]] = []
        if target_path.is_file():
            node = analyze_file_complexity(target_path, project_root)
            if node and node.get("language") in languages:
                nodes.append(node)
        else:
            for path in target_path.rglob("*"):
                if len(nodes) >= 1000:
                    break
                if not path.is_file() or any(part in FILE_TREE_IGNORES for part in path.relative_to(project_root).parts):
                    continue
                node = analyze_file_complexity(path, project_root)
                if node and node.get("language") in languages:
                    nodes.append(node)
        total_loc = sum(int(node["loc"]) for node in nodes)
        total_cc = sum(int(node["cc"]) for node in nodes)
        avg_cc = round(total_cc / len(nodes), 2) if nodes else 0
        root_node = {
            "name": project_root.name or "project",
            "type": "project",
            "loc": total_loc,
            "cc": total_cc,
            "mi": maintainability_index(total_loc, total_cc),
            "risk_level": complexity_risk(total_cc, total_loc),
            "children": nodes,
            "file_path": ".",
        }
        stats = {
            "total_files": len(nodes),
            "avg_cc": avg_cc,
            "high_risk_count": sum(1 for node in nodes if node["risk_level"] in {"C", "D", "E"}),
            "analysis_time_ms": int((time.time() - start) * 1000),
        }
        return {"success": True, "data": {"root": root_node, "stats": stats, "cached": False}, "error_message": None, "elapsed_ms": stats["analysis_time_ms"]}
    except Exception as exc:
        return {"success": False, "data": None, "error_message": str(exc), "elapsed_ms": int((time.time() - start) * 1000)}


def impact_node(node_id: str, name: str, file_path: str, impact_level: str, confidence: str = "medium") -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "function",
        "name": name,
        "file_path": file_path,
        "line_range": [1, 1],
        "impact_level": impact_level,
        "confidence": confidence,
        "language": LANGUAGE_EXTENSIONS.get(Path(file_path).suffix.lower(), "text"),
    }


@app.post("/api/analysis/change-impact")
async def analysis_change_impact(request: Request) -> dict[str, Any]:
    start = time.time()
    payload = await request.json()
    try:
        project_root = safe_workspace_path(payload.get("project_root") or payload.get("projectRoot") or ".")
        file_value = str(payload.get("file_path") or payload.get("filePath") or "")
        if not file_value:
            return {"success": False, "data": None, "error_code": "MISSING_FILE", "error_message": "file_path is required", "elapsed_ms": 0}
        changed_file = safe_workspace_path((project_root / file_value).as_posix(), project_root)
        if project_root.resolve() != changed_file.resolve() and project_root.resolve() not in changed_file.resolve().parents:
            return {"success": False, "data": None, "error_code": "PATH_DENIED", "error_message": "file_path escapes project_root", "elapsed_ms": 0}
        changed_lines = [int(line) for line in payload.get("changed_lines") or payload.get("changedLines") or []]
        rel_changed = changed_file.relative_to(project_root).as_posix()
        stem = changed_file.stem
        nodes = [impact_node("changed-file", stem, rel_changed, "direct", "high")]
        edges: list[dict[str, Any]] = []
        affected_apis: list[str] = []
        for path in project_root.rglob("*"):
            if len(nodes) >= 50:
                break
            if path == changed_file or not path.is_file() or path.suffix.lower() not in LANGUAGE_EXTENSIONS:
                continue
            if any(part in FILE_TREE_IGNORES for part in path.relative_to(project_root).parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if stem not in text and rel_changed not in text:
                continue
            node_id = f"impact-{len(nodes)}"
            rel_path = path.relative_to(project_root).as_posix()
            nodes.append(impact_node(node_id, path.stem, rel_path, "indirect", "medium"))
            edges.append({"source": "changed-file", "target": node_id, "type": "dependency", "weight": 1})
            if re.search(r"(@app\.|@GetMapping|@PostMapping|router\.)", text):
                affected_apis.append(rel_path)
        summary = {
            "direct_count": 1,
            "indirect_count": len(nodes) - 1,
            "potential_count": 0,
            "affected_apis": affected_apis,
            "affected_tasks": [],
        }
        data = {
            "changed_file": rel_changed,
            "changed_lines": changed_lines,
            "impact_nodes": nodes,
            "impact_edges": edges,
            "summary": summary,
        }
        return {"success": True, "data": data, "error_code": None, "error_message": None, "elapsed_ms": round((time.time() - start) * 1000, 1)}
    except Exception as exc:
        return {"success": False, "data": None, "error_code": "ANALYSIS_ERROR", "error_message": str(exc), "elapsed_ms": round((time.time() - start) * 1000, 1)}


def check_issue(line: int, column: int, rule: str | None, severity: str, message: str, code: str | None = None) -> dict[str, Any]:
    return {"line": line, "column": column, "rule": rule, "severity": severity, "message": message, "code": code}


def check_detail(status: str, errors: int = 0, warnings: int = 0, issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"status": status, "errorCount": errors, "warningCount": warnings, "issues": issues or []}


def test_check_detail(
    status: str,
    passed: int = 0,
    failed: int = 0,
    coverage: float | None = None,
    failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {"status": status, "passedCount": passed, "failedCount": failed, "coveragePercent": coverage, "failures": failures or []}


def skipped_check() -> dict[str, Any]:
    return check_detail("skipped")


def skipped_test_check() -> dict[str, Any]:
    return test_check_detail("skipped")


def command_workspace(path: Path, fallback: Path) -> Path:
    current = path.parent if path.is_file() else path
    fallback = fallback.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "package.json").exists() or (candidate / "pyproject.toml").exists() or (candidate / "pytest.ini").exists():
            return candidate
        if candidate == fallback:
            break
    return fallback


def run_verification_command(args: list[str], cwd: Path, timeout_ms: int = 60_000) -> subprocess.CompletedProcess[str] | None:
    executable = args[0]
    if not shutil.which(executable) and not shutil.which(f"{executable}.cmd"):
        return None
    try:
        return subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout_ms / 1000)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(args, 124, (exc.stdout or "") if isinstance(exc.stdout, str) else "", "Timed out")
    except OSError as exc:
        return subprocess.CompletedProcess(args, 127, "", str(exc))


def parse_tsc_issues(output: str) -> list[dict[str, Any]]:
    issues = []
    pattern = re.compile(r"^(.*)\((\d+),(\d+)\):\s+error\s+(TS\d+):\s+(.*)$")
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if match:
            issues.append(check_issue(int(match.group(2)), int(match.group(3)), match.group(4), "error", match.group(5), match.group(4)))
    if not issues and output.strip():
        issues.append(check_issue(0, 0, "typescript", "error", output.strip()[:1000]))
    return issues


def parse_eslint_issues(output: str) -> tuple[list[dict[str, Any]], int, int]:
    issues: list[dict[str, Any]] = []
    try:
        parsed = json.loads(output or "[]")
        for file_result in parsed if isinstance(parsed, list) else []:
            for item in file_result.get("messages", []):
                severity = "error" if item.get("severity") == 2 else "warning"
                issues.append(
                    check_issue(
                        int(item.get("line") or 0),
                        int(item.get("column") or 0),
                        item.get("ruleId"),
                        severity,
                        item.get("message") or "",
                        item.get("ruleId"),
                    )
                )
    except json.JSONDecodeError:
        if output.strip():
            issues.append(check_issue(0, 0, "eslint", "error", output.strip()[:1000]))
    errors = sum(1 for item in issues if item["severity"] == "error")
    warnings = sum(1 for item in issues if item["severity"] == "warning")
    return issues, errors, warnings


def run_python_syntax_check(file_path: Path) -> dict[str, Any]:
    try:
        py_compile.compile(str(file_path), doraise=True)
        return check_detail("pass")
    except py_compile.PyCompileError as exc:
        message = str(exc)
        line_match = re.search(r"line\s+(\d+)", message)
        line = int(line_match.group(1)) if line_match else 0
        return check_detail("fail", 1, 0, [check_issue(line, 0, "py_compile", "error", message[:1000], "py_compile")])


def run_typescript_check(file_path: Path, workspace: Path) -> dict[str, Any]:
    if file_path.suffix.lower() not in {".ts", ".tsx", ".js", ".jsx"}:
        return skipped_check()
    cwd = command_workspace(file_path, workspace)
    result = run_verification_command(["npx", "tsc", "--noEmit", "--pretty", "false", str(file_path)], cwd)
    if result is None or result.returncode == 127:
        return skipped_check()
    if result.returncode == 0:
        return check_detail("pass")
    issues = parse_tsc_issues((result.stdout or "") + "\n" + (result.stderr or ""))
    return check_detail("fail", max(1, len(issues)), 0, issues)


def run_eslint_check(file_path: Path, workspace: Path) -> dict[str, Any]:
    if file_path.suffix.lower() not in {".ts", ".tsx", ".js", ".jsx"}:
        return skipped_check()
    cwd = command_workspace(file_path, workspace)
    result = run_verification_command(["npx", "eslint", "--format", "json", str(file_path)], cwd)
    if result is None or result.returncode == 127:
        return skipped_check()
    issues, errors, warnings = parse_eslint_issues(result.stdout or result.stderr or "")
    if result.returncode == 0:
        return check_detail("pass", 0, warnings, issues)
    return check_detail("fail" if errors else "pass", errors, warnings, issues)


def run_vitest_check(file_path: Path, workspace: Path) -> dict[str, Any]:
    if file_path.suffix.lower() not in {".ts", ".tsx", ".js", ".jsx"}:
        return skipped_test_check()
    cwd = command_workspace(file_path, workspace)
    result = run_verification_command(["npx", "vitest", "run", "--reporter=json", str(file_path)], cwd)
    if result is None or result.returncode == 127:
        return skipped_test_check()
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if "No test files found" in output:
        return test_check_detail("no_tests")
    if result.returncode == 0:
        return test_check_detail("pass", passed=1)
    return test_check_detail("fail", failed=1, failures=[{"testName": file_path.name, "message": output.strip()[:1000]}])


def run_pytest_check(file_path: Path, workspace: Path) -> dict[str, Any]:
    if file_path.suffix.lower() != ".py":
        return skipped_test_check()
    cwd = command_workspace(file_path, workspace)
    result = run_verification_command(["python", "-m", "pytest", str(file_path), "-q"], cwd, timeout_ms=60_000)
    if result is None or result.returncode == 127:
        return skipped_test_check()
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if "no tests ran" in output.lower() or "not found" in output.lower():
        return test_check_detail("no_tests")
    if result.returncode == 0:
        passed_match = re.search(r"(\d+)\s+passed", output)
        return test_check_detail("pass", passed=int(passed_match.group(1)) if passed_match else 1)
    return test_check_detail("fail", failed=1, failures=[{"testName": file_path.name, "message": output.strip()[:1000]}])


def verify_single_file(file_value: str, checks: list[str], workspace: Path) -> dict[str, Any]:
    file_path = safe_workspace_path(file_value, workspace)
    rel = file_path.relative_to(workspace).as_posix() if workspace.resolve() in [file_path.resolve(), *file_path.resolve().parents] else file_path.name
    if not file_path.exists():
        missing = check_detail("fail", 1, 0, [check_issue(0, 0, "file", "error", "File not found", "ENOENT")])
        return {"filePath": rel, "typescript": missing, "eslint": skipped_check(), "vitest": skipped_test_check(), "python": missing, "pytest": skipped_test_check()}
    normalized = {item.lower() for item in checks}
    if file_path.suffix.lower() == ".py":
        python_result = run_python_syntax_check(file_path) if normalized & {"python", "py_compile", "typescript"} else skipped_check()
        pytest_result = run_pytest_check(file_path, workspace) if normalized & {"pytest", "vitest", "test"} else skipped_test_check()
        return {
            "filePath": rel,
            "typescript": python_result,
            "eslint": skipped_check(),
            "vitest": pytest_result,
            "python": python_result,
            "pytest": pytest_result,
        }
    ts_result = run_typescript_check(file_path, workspace) if "typescript" in normalized else skipped_check()
    eslint_result = run_eslint_check(file_path, workspace) if "eslint" in normalized else skipped_check()
    vitest_result = run_vitest_check(file_path, workspace) if "vitest" in normalized else skipped_test_check()
    return {"filePath": rel, "typescript": ts_result, "eslint": eslint_result, "vitest": vitest_result}


def heuristic_for_files(file_paths: list[str], workspace: Path) -> dict[str, Any]:
    impacted: set[str] = set()
    affected_api_count = 0
    for file_value in file_paths[:20]:
        try:
            changed = safe_workspace_path(file_value, workspace)
            stem = changed.stem
            rel = changed.relative_to(workspace).as_posix()
        except Exception:
            continue
        for path in workspace.rglob("*"):
            if len(impacted) >= 100:
                break
            if path == changed or not path.is_file() or path.suffix.lower() not in LANGUAGE_EXTENSIONS:
                continue
            if any(part in FILE_TREE_IGNORES for part in path.relative_to(workspace).parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if stem in text or rel in text:
                impacted.add(path.relative_to(workspace).as_posix())
                if re.search(r"(@app\.|@GetMapping|@PostMapping|router\.)", text):
                    affected_api_count += 1
    return {
        "affectedApiCount": affected_api_count,
        "indirectImpactCount": len(impacted),
        "potentialImpactCount": 0,
        "hasHighConfidenceImpact": bool(impacted),
        "truncated": len(impacted) >= 100,
        "filesAffected": sorted(impacted)[:100],
    }


def compute_verify_signal(results: list[dict[str, Any]], heuristic: dict[str, Any]) -> tuple[str, str, str]:
    has_errors = any(
        item["typescript"]["errorCount"] > 0
        or item["eslint"]["errorCount"] > 0
        or item["vitest"]["failedCount"] > 0
        for item in results
    )
    has_warnings = any(item["eslint"]["warningCount"] > 0 for item in results)
    no_tests = any(item["vitest"]["status"] == "no_tests" for item in results)
    if has_errors or heuristic.get("truncated"):
        return "blocked", "Verification found errors or truncated impact analysis.", "fail"
    if has_warnings or no_tests or heuristic.get("affectedApiCount", 0) > 0 or heuristic.get("indirectImpactCount", 0) > 3:
        return "review_recommended", "Checks passed with warnings, missing tests, or notable impact.", "partial" if has_warnings or no_tests else "pass"
    if results and all(
        item["typescript"]["status"] in {"pass", "skipped"}
        and item["eslint"]["status"] in {"pass", "skipped"}
        and item["vitest"]["status"] in {"pass", "skipped", "no_tests"}
        for item in results
    ):
        return "auto_approve", "All requested checks passed with low impact.", "pass"
    return "manual_required", "Insufficient deterministic evidence for automatic approval.", "partial"


@app.post("/api/verify/run-checks")
async def run_checks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    start = time.time()
    bundle_id = new_id("evidence")
    session_id = str(payload.get("sessionId") or payload.get("session_id") or "default")
    claim = payload.get("claim") or payload.get("summary") or "Python verification request"
    workspace = safe_workspace_path(payload.get("workingDirectory") or payload.get("working_directory") or ".")
    file_paths = [str(item) for item in payload.get("filePaths") or payload.get("file_paths") or []]
    checks = [str(item) for item in payload.get("checks") or (["python", "pytest"] if any(path.endswith(".py") for path in file_paths) else ["typescript", "eslint", "vitest"])]
    results = [verify_single_file(path, checks, workspace) for path in file_paths]
    heuristic = heuristic_for_files(file_paths, workspace)
    signal, signal_reason, overall_status = compute_verify_signal(results, heuristic)
    duration_ms = int((time.time() - start) * 1000)
    timestamp = utc_now()
    response_core = {
        "results": results,
        "heuristic": heuristic,
        "signal": signal,
        "signalReason": signal_reason,
        "overallStatus": overall_status,
        "duration": duration_ms,
        "timestamp": timestamp,
    }
    blob_text = json.dumps({"request": payload, "response": response_core, "checkedAt": timestamp}, ensure_ascii=False, indent=2)
    sha256 = hashlib.sha256(blob_text.encode("utf-8")).hexdigest()
    STATE.setdefault("evidenceBlobs", {})[sha256] = blob_text
    bundle = {
        "bundleId": bundle_id,
        "sessionId": session_id,
        "agentId": payload.get("agentId"),
        "kind": payload.get("kind") or "qa",
        "claim": claim,
        "verdict": "verified" if overall_status != "fail" else "failed",
        "items": [
            {
                "id": new_id("evidence-item"),
                "type": "test",
                "summary": signal_reason,
                "blobSha256": sha256,
                "meta": {"status": overall_status, "checks": checks, "signal": signal},
            }
        ],
        "createdAt": timestamp,
    }
    STATE["evidence"][bundle_id] = bundle
    save_state()
    return {
        **response_core,
        "success": True,
        "status": "passed" if overall_status != "fail" else "failed",
        "bundleId": bundle_id,
        "summary": signal_reason,
        "checks": [
            {
                "name": name,
                "status": "passed" if overall_status != "fail" else "failed",
            }
            for name in checks
        ],
        "bundle": bundle,
    }


@app.post("/api/verify/legacy-checks")
async def legacy_checks(request: Request) -> dict[str, Any]:
    payload = await request.json()
    operation_id = str(payload.get("operationId") or payload.get("operation_id") or new_id("operation"))
    modern_payload = {
        "sessionId": payload.get("sessionId"),
        "filePaths": payload.get("filePaths") or [],
        "checks": payload.get("checks") or ["typescript", "eslint", "vitest"],
        "claim": f"legacy verification {operation_id}",
    }
    class _RequestAdapter:
        async def json(self) -> dict[str, Any]:
            return modern_payload

    modern = await run_checks(_RequestAdapter())  # type: ignore[arg-type]
    results = []
    for name in modern_payload["checks"]:
        passed = modern["overallStatus"] != "fail"
        results.append({"check": name, "passed": passed, "errors": [] if passed else [{"file": "", "line": 0, "column": 0, "message": modern["signalReason"], "rule": name}], "warnings": [], "duration": modern["duration"]})
    return {
        "operationId": operation_id,
        "status": "all_pass" if modern["overallStatus"] == "pass" else "has_error" if modern["overallStatus"] == "fail" else "has_warning",
        "results": results,
        "totalDuration": modern["duration"],
        "timestamp": modern["timestamp"],
        "signal": modern["signal"],
        "signalReason": modern["signalReason"],
    }


@app.get("/api/evidence/{bundle_id}")
async def get_evidence(bundle_id: str) -> dict[str, Any]:
    bundle = STATE["evidence"].get(bundle_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="Evidence bundle not found")
    return bundle


@app.get("/api/evidence/session/{session_id}")
async def get_session_evidence(session_id: str) -> list[dict[str, Any]]:
    bundles = [bundle for bundle in STATE["evidence"].values() if bundle.get("sessionId") == session_id]
    return sorted(bundles, key=lambda item: item.get("createdAt", ""), reverse=True)


@app.get("/api/evidence/blob/{sha256}")
async def get_evidence_blob(sha256: str) -> Response:
    blob = STATE.setdefault("evidenceBlobs", {}).get(sha256)
    if blob is None:
        raise HTTPException(status_code=404, detail="Evidence blob not found")
    return Response(
        content=blob.encode("utf-8"),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{sha256}"'},
    )


@app.get("/api/remote/status")
async def remote_status() -> dict[str, Any]:
    active_statuses = {"running", "streaming", "processing", "waiting_permission"}
    sessions = []
    websocket_sessions = WS_SESSION_MANAGER.get_active_session_ids()
    for session in STATE.setdefault("sessions", {}).values():
        status = session.get("status", "idle")
        if status in active_statuses or session.get("online") or session.get("id") in websocket_sessions:
            sessions.append(
                {
                    "sessionId": session["id"],
                    "online": bool(session.get("online", status in active_statuses) or session["id"] in websocket_sessions),
                    "status": status,
                    "principal": WS_SESSION_MANAGER.get_principal_for_session(session["id"]),
                    "title": session.get("title"),
                    "updatedAt": session.get("updatedAt"),
                }
            )
    return {
        "enabled": True,
        "mode": os.getenv("AUTH_MODE", "localhost"),
        "status": "ready",
        "activeSessions": len(sessions),
        "sessions": sorted(sessions, key=lambda item: item.get("updatedAt") or "", reverse=True),
        "serverUptime": format_uptime(time.time() - START_TIME),
    }


@app.get("/api/ws/sessions")
async def ws_sessions() -> dict[str, Any]:
    sessions = sorted(WS_SESSION_MANAGER.get_active_session_ids())
    return {
        "activeSessions": len(sessions),
        "sessions": [{"sessionId": session_id, "principal": WS_SESSION_MANAGER.get_principal_for_session(session_id)} for session_id in sessions],
    }


@app.post("/api/ws/sessions/bind")
async def ws_bind_session(request: Request) -> dict[str, Any]:
    payload = await request.json()
    principal = str(payload.get("principal") or "")
    session_id = str(payload.get("sessionId") or payload.get("session_id") or "")
    if not principal or not session_id:
        raise HTTPException(status_code=400, detail="principal and sessionId are required")
    WS_SESSION_MANAGER.bind_session(principal, session_id)
    return {"success": True, "principal": principal, "sessionId": session_id}


@app.post("/api/ws/sessions/{session_id}/push")
async def ws_push_message(session_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    queued = WS_SESSION_MANAGER.queue_message(session_id, str(payload.get("type") or "notification"), payload.get("payload") if isinstance(payload.get("payload"), dict) else payload)
    return {"success": True, "queued": queued, "queueStats": WS_SESSION_MANAGER.queue_stats(session_id)}


@app.get("/api/ws/sessions/{session_id}/messages")
async def ws_drain_messages(session_id: str) -> dict[str, Any]:
    messages = WS_SESSION_MANAGER.drain_messages(session_id)
    return {"sessionId": session_id, "messages": messages, "count": len(messages)}


@app.get("/api/ws/sessions/{session_id}/messages/deliver")
async def ws_deliver_messages(session_id: str, subscriptionId: str = "sub-0", ackMode: str = "auto") -> dict[str, Any]:
    messages = WS_SESSION_MANAGER.deliver_messages(session_id, subscriptionId, ack_mode=ackMode)
    return {"sessionId": session_id, "messages": messages, "count": len(messages), "ackMode": ackMode, "subscriptionId": subscriptionId}


@app.get("/api/ws/sessions/{session_id}/messages/peek")
async def ws_peek_messages(session_id: str) -> dict[str, Any]:
    messages = WS_SESSION_MANAGER.peek_messages(session_id)
    return {"sessionId": session_id, "messages": messages, "count": len(messages)}


@app.get("/api/ws/sessions/{session_id}/messages/stats")
async def ws_message_stats(session_id: str) -> dict[str, Any]:
    return WS_SESSION_MANAGER.queue_stats(session_id)


@app.get("/api/ws/sessions/{session_id}/messages/replay")
async def ws_replay_messages(session_id: str, includeAcked: bool = False, sinceId: str | None = None) -> dict[str, Any]:
    messages = WS_SESSION_MANAGER.replay_messages(session_id, include_acked=includeAcked, since_id=sinceId)
    return {"sessionId": session_id, "messages": messages, "count": len(messages), "includeAcked": includeAcked, "sinceId": sinceId}


@app.post("/api/ws/sessions/{session_id}/messages/ack")
async def ws_ack_messages(session_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    message_ids = payload.get("messageIds") or payload.get("message_ids") or payload.get("ids") or []
    if isinstance(message_ids, str):
        message_ids = [message_ids]
    acked = WS_SESSION_MANAGER.ack_messages(session_id, [str(item) for item in message_ids])
    return {"success": True, "sessionId": session_id, "acked": acked, "ackedCount": len(acked)}


@app.post("/api/ws/sessions/{session_id}/messages/nack")
async def ws_nack_messages(session_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    message_ids = payload.get("messageIds") or payload.get("message_ids") or payload.get("ids") or []
    if isinstance(message_ids, str):
        message_ids = [message_ids]
    nacked = WS_SESSION_MANAGER.nack_messages(session_id, [str(item) for item in message_ids], reason=str(payload.get("reason") or "nack"))
    return {"success": True, "sessionId": session_id, "nacked": nacked, "nackedCount": len(nacked)}


@app.post("/api/ws/sessions/{session_id}/events")
async def ws_publish_event(session_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    event = WS_SESSION_MANAGER.publish_event(session_id, str(payload.get("type") or "event"), payload.get("payload") if isinstance(payload.get("payload"), dict) else payload)
    return {"success": True, "event": event}


@app.post("/api/ws/events/broadcast")
async def ws_broadcast_event(request: Request) -> dict[str, Any]:
    payload = await request.json()
    events = WS_SESSION_MANAGER.broadcast_event(str(payload.get("type") or "event"), payload.get("payload") if isinstance(payload.get("payload"), dict) else payload)
    return {"success": True, "count": len(events), "events": events}


@app.get("/api/notifications")
async def list_notifications(sessionId: str | None = None, includeDismissed: bool = False, limit: int = 100) -> dict[str, Any]:
    items = list(STATE.setdefault("notifications", []))
    if sessionId:
        items = [item for item in items if item.get("sessionId") in {None, sessionId}]
    if not includeDismissed:
        items = [item for item in items if not item.get("dismissed")]
    items = sorted(items, key=lambda item: item.get("createdAt", 0), reverse=True)[: max(1, min(limit, 500))]
    return {"notifications": items, "total": len(items)}


@app.post("/api/notifications")
async def post_notification(request: Request) -> dict[str, Any]:
    payload = await request.json()
    notification = await create_notification(payload)
    save_state()
    return {"success": True, "notification": notification}


@app.delete("/api/notifications/{key}")
async def dismiss_notification(key: str) -> dict[str, Any]:
    dismissed = None
    for item in STATE.setdefault("notifications", []):
        if item.get("key") == key:
            item["dismissed"] = True
            item["dismissedAt"] = int(time.time() * 1000)
            dismissed = item
            break
    if not dismissed:
        raise HTTPException(status_code=404, detail="Notification not found")
    save_state()
    return {"success": True, "notification": dismissed}


@app.post("/api/remote/interrupt")
async def remote_interrupt() -> dict[str, Any]:
    active_statuses = {"running", "streaming", "processing", "waiting_permission"}
    count = 0
    now = utc_now()
    interrupted_sessions = []
    for session in STATE.setdefault("sessions", {}).values():
        status = session.get("status", "idle")
        if status in active_statuses or session.get("online"):
            session["status"] = "idle"
            session["interruptedAt"] = now
            session["updatedAt"] = now
            QUERY_ABORTS.abort(session["id"], "USER_INTERRUPT")
            WS_SESSION_MANAGER.publish_event(session["id"], "interrupt_ack", {"reason": "USER_INTERRUPT"})
            interrupted_sessions.append(session["id"])
            count += 1

    for key, response in list(STATE.setdefault("permissionResponses", {}).items()):
        if response.get("decision") in {None, "pending"}:
            STATE["permissionResponses"][key] = {**response, "decision": "deny", "reason": "USER_INTERRUPT", "updatedAt": now}

    STATE["remoteInterruptedAt"] = now
    save_state()
    return {
        "success": True,
        "interrupted": count > 0 or not STATE.get("sessions"),
        "sessionCount": count,
        "sessions": interrupted_sessions,
    }


async def emit_query_event(
    loop: QueryLoopState,
    event_type: str,
    payload: dict[str, Any] | None = None,
    live_send: Any | None = None,
) -> dict[str, Any]:
    event = loop.event(event_type, payload or {})
    event_payload = event.to_dict()
    STATE.setdefault("queryEvents", []).append(event_payload)
    del STATE["queryEvents"][:-1000]
    WS_SESSION_MANAGER.publish_event(loop.sessionId, event_type, event_payload)
    session = STATE.setdefault("sessions", {}).get(loop.sessionId) or {}
    parent_session_id = session.get("parentSessionId")
    if event_type == "permission_request" and parent_session_id and parent_session_id != loop.sessionId:
        bubbled_payload = {
            **event_payload,
            "sessionId": str(parent_session_id),
            "parentSessionId": str(parent_session_id),
            "childSessionId": loop.sessionId,
            "bubbledFromSessionId": loop.sessionId,
        }
        WS_SESSION_MANAGER.publish_event(str(parent_session_id), event_type, bubbled_payload)
    if live_send is not None:
        await live_send(event_payload)
    return event_payload


def tool_call_summary(loop: QueryLoopState, tool_use_id: str) -> dict[str, Any] | None:
    for call in reversed(loop.toolCalls):
        if call.get("toolUseId") == tool_use_id:
            summary = call.get("summary")
            return summary if isinstance(summary, dict) else None
    return None


async def record_termination_decision(
    loop: QueryLoopState,
    requested_stop_reason: str | None = None,
    error: str | None = None,
    live_send: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = TERMINATION_STRATEGY.decide(loop, requested_stop_reason=requested_stop_reason, error=error, metadata=metadata)
    loop.set_termination_decision(decision)
    return await emit_query_event(loop, "termination_decision", {"decision": decision.to_dict()}, live_send)


def persist_query_loop(loop: QueryLoopState) -> None:
    STATE.setdefault("queryLoops", {})[loop.id] = loop.to_dict()


async def create_notification(payload: dict[str, Any], live_send: Any | None = None) -> dict[str, Any]:
    item = {
        "key": str(payload.get("key") or new_id("notification")),
        "level": str(payload.get("level") or "info"),
        "message": str(payload.get("message") or ""),
        "priority": str(payload.get("priority") or "normal"),
        "timeout": int(payload.get("timeout") or 5000),
        "sessionId": payload.get("sessionId") or payload.get("session_id"),
        "createdAt": int(time.time() * 1000),
        "dismissed": False,
    }
    STATE.setdefault("notifications", []).append(item)
    del STATE["notifications"][:-500]
    event = {"type": "notification", **item}
    session_id = item.get("sessionId")
    if session_id:
        WS_SESSION_MANAGER.publish_event(str(session_id), "notification", event)
    else:
        for active_session_id in WS_SESSION_MANAGER.get_active_session_ids():
            WS_SESSION_MANAGER.publish_event(active_session_id, "notification", event)
    if live_send is not None:
        await live_send(event)
    return item


def normalize_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_calls, list):
        return []
    normalized = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        raw_args = item.get("arguments", item.get("input", function.get("arguments", {})))
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) if raw_args.strip() else {}
            except Exception:
                raw_args = {"raw": raw_args}
        if not isinstance(raw_args, dict):
            raw_args = {}
        name = str(item.get("name") or function.get("name") or item.get("toolName") or "")
        if not name:
            continue
        normalized.append({"id": str(item.get("id") or item.get("toolUseId") or new_id("tool")), "name": name, "arguments": raw_args})
    return normalized


async def await_permission_decision(tool_use_id: str, session_id: str, timeout_ms: int = 120_000) -> dict[str, Any]:
    deadline = time.time() + max(1, timeout_ms) / 1000
    wait_started = time.time()
    session = STATE.setdefault("sessions", {}).get(session_id)
    if session is not None:
        session["permissionWaitStartedAt"] = wait_started
    try:
        while time.time() < deadline:
            if QUERY_ABORTS.is_aborted(session_id):
                return {"decision": "deny", "reason": "Session interrupted"}
            stored = STATE.setdefault("permissionResponses", {}).pop(tool_use_id, None)
            if stored is not None:
                decision = str(stored.get("decision") or ("allow" if stored.get("allowed") else "deny")).lower()
                return {**stored, "decision": decision}
            await asyncio.sleep(0.05)
        return {"decision": "deny", "reason": "Permission request timed out"}
    finally:
        session = STATE.setdefault("sessions", {}).get(session_id)
        if session is not None:
            elapsed_ms = int((time.time() - wait_started) * 1000)
            session["permissionWaitMs"] = int(session.get("permissionWaitMs") or 0) + max(0, elapsed_ms)
            if session.get("permissionWaitStartedAt") == wait_started:
                session.pop("permissionWaitStartedAt", None)


def execute_approved_tool(tool_name: str, arguments: dict[str, Any]) -> ToolResult:
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool or not tool.enabled:
        return ToolResult(f"Unknown or disabled tool: {tool_name}", isError=True)
    try:
        return tool.handler(arguments)
    except Exception as exc:
        return ToolResult(str(exc), isError=True)


def session_permission_wait_ms(session_id: str) -> int:
    session = STATE.setdefault("sessions", {}).get(session_id) or {}
    total = int(session.get("permissionWaitMs") or 0)
    active_started = session.get("permissionWaitStartedAt")
    if active_started:
        try:
            total += max(0, int((time.time() - float(active_started)) * 1000))
        except Exception:
            pass
    return total


async def wait_for_agent_with_permission_budget(coro: Any, child_session_id: str, timeout_ms: int) -> Any:
    task = asyncio.create_task(coro)
    started = time.time()
    try:
        while not task.done():
            elapsed_ms = int((time.time() - started) * 1000)
            effective_ms = elapsed_ms - session_permission_wait_ms(child_session_id)
            if effective_ms > timeout_ms:
                task.cancel()
                raise asyncio.TimeoutError()
            await asyncio.wait({task}, timeout=0.05)
        return await task
    except Exception:
        if not task.done():
            task.cancel()
        raise


def swarm_worker_permission_wait_ms(swarm: dict[str, Any], worker_id: str) -> int:
    worker = swarm.setdefault("workers", {}).get(worker_id) or {}
    total = int(worker.get("permissionWaitMs") or 0)
    for item in swarm.setdefault("pendingPermissions", []):
        if item.get("workerId") != worker_id or item.get("status") != "pending":
            continue
        try:
            created_at = datetime.fromisoformat(str(item.get("createdAt")).replace("Z", "+00:00")).timestamp()
            total += max(0, int((time.time() - created_at) * 1000))
        except Exception:
            pass
    return total


async def wait_for_swarm_worker_with_permission_budget(task: asyncio.Task[Any], swarm: dict[str, Any], worker_id: str, timeout_ms: int) -> None:
    started = time.time()
    while not task.done():
        elapsed_ms = int((time.time() - started) * 1000)
        effective_ms = elapsed_ms - swarm_worker_permission_wait_ms(swarm, worker_id)
        if effective_ms > timeout_ms:
            task.cancel()
            raise asyncio.TimeoutError()
        await asyncio.wait({task}, timeout=0.05)
    await task


def subagent_system_prompt(prompt: str, agent_type: str, parent_session_id: str, working_directory: str, fork: bool) -> str:
    agent_key = str(agent_type or "general-purpose").lower()
    template = AGENT_SYSTEM_PROMPTS.get(agent_key, AGENT_SYSTEM_PROMPTS["general-purpose"])
    constraints = [
        template.strip(),
        "",
        "你的任务：",
        prompt,
        "",
        "上下文：",
        f"Parent session: {parent_session_id}",
        f"Working directory: {working_directory}",
        f"Agent type: {agent_type or 'general-purpose'}",
        "你无法使用：AgentTool、TaskCreateTool、TeamTools。",
        "完成任务并返回清晰、简洁的结果。",
        "如果你修改了文件，在最终回复中列出所有修改的文件路径。",
        "不要尝试超出上述范围的任务。",
    ]
    if fork:
        constraints.extend(
            [
                "",
                "[FORK 模式]",
                "你是一个 fork 实例，可以访问父会话的完整对话历史。",
                "父会话提供上下文——专注于完成下方的新任务。",
                "你继承了父会话的文件状态和对话历史。",
                "LLM 提供者应从公共消息前缀中复用共享上下文。",
            ]
        )
    return "\n".join(constraints)


def resolve_agent_timeout_ms(agent_type: str, payload: dict[str, Any]) -> int:
    if payload.get("timeoutMs") or payload.get("timeout_ms"):
        return max(1, int(payload.get("timeoutMs") or payload.get("timeout_ms")))
    base_ms = 300_000
    kind = agent_type.lower()
    if kind in {"coding", "frontend-dev", "backend-dev"}:
        return base_ms * 2
    if kind in {"verify", "qa", "verification"}:
        return base_ms * 3
    return base_ms


def create_agent_worktree(agent_id: str) -> Path:
    safe_agent_id = re.sub(r"[^A-Za-z0-9_.-]", "-", agent_id)[:80]
    git_base = Path(tempfile.gettempdir()) / "zhikun-agent-worktrees"
    git_path = (git_base / safe_agent_id).resolve()
    git_base.mkdir(parents=True, exist_ok=True)
    if git_path.exists():
        shutil.rmtree(git_path, ignore_errors=True)
    branch_name = f"agent-{safe_agent_id}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
    try:
        subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"], text=True, capture_output=True, check=True, timeout=10)
        added = subprocess.run(
            ["git", "-C", str(ROOT), "worktree", "add", "-b", branch_name, str(git_path), "HEAD"],
            text=True,
            capture_output=True,
            timeout=60,
        )
        if added.returncode != 0:
            raise RuntimeError((added.stderr or added.stdout or "git worktree add failed").strip())
        TOOL_REGISTRY.allow_external_root(git_path)
        STATE.setdefault("agentWorktrees", {})[agent_id] = {
            "path": str(git_path),
            "branchName": branch_name,
            "mode": "git",
            "createdAt": utc_now(),
            "status": "active",
        }
        return git_path
    except Exception as exc:
        if git_path.exists():
            shutil.rmtree(git_path, ignore_errors=True)
        base = DATA_DIR / "agent-worktrees"
        path = (base / safe_agent_id).resolve()
        base_resolved = base.resolve()
        if path != base_resolved and base_resolved not in path.parents:
            raise ValueError("Invalid agent worktree path")
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        STATE.setdefault("agentWorktrees", {})[agent_id] = {
            "path": str(path),
            "mode": "lightweight",
            "fallbackReason": str(exc),
            "createdAt": utc_now(),
            "status": "active",
        }
        return path


def _git_worktree_changed_paths(worktree_path: Path) -> list[str]:
    result = subprocess.run(["git", "-C", str(worktree_path), "status", "--porcelain"], text=True, capture_output=True, timeout=30)
    if result.returncode != 0:
        return []
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        raw_path = line[3:] if len(line) > 3 else ""
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        if raw_path:
            paths.append(raw_path.strip().replace("\\", "/"))
    return paths


def _copy_worktree_changes(source_root: Path, relative_paths: list[str] | None = None, target_root: Path | None = None) -> list[str]:
    target_root = (target_root or ROOT).resolve()
    merged: list[str] = []
    if relative_paths is None:
        candidates = [source for source in source_root.rglob("*") if source.is_file()]
    else:
        candidates = [source_root / rel for rel in relative_paths]
    for source in candidates:
        relative = source.relative_to(source_root)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        target = (target_root / relative).resolve()
        if target != target_root and target_root not in target.parents:
            continue
        if source.exists() and source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            merged.append(relative.as_posix())
        elif relative_paths is not None and target.exists():
            target.unlink()
            merged.append(relative.as_posix())
    return sorted(dict.fromkeys(merged))


def _git_commit_and_merge_worktree(agent_id: str, worktree_path: Path, repo_root: Path) -> list[str]:
    record = STATE.setdefault("agentWorktrees", {}).setdefault(agent_id, {"path": str(worktree_path)})
    branch = str(record.get("branchName") or "")
    if not branch:
        raise RuntimeError("Missing agent worktree branch name")
    changed_paths = _git_worktree_changed_paths(worktree_path)
    if not changed_paths:
        record["changedPaths"] = []
        record["mergeMode"] = "git"
        return []
    env = os.environ.copy()
    env.setdefault("GIT_AUTHOR_NAME", "Zhikun Agent")
    env.setdefault("GIT_AUTHOR_EMAIL", "agent@localhost")
    env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
    env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
    add = subprocess.run(["git", "-C", str(worktree_path), "add", "-A"], text=True, capture_output=True, timeout=60, env=env)
    if add.returncode != 0:
        raise RuntimeError((add.stderr or add.stdout or "git add failed").strip())
    commit = subprocess.run(
        ["git", "-C", str(worktree_path), "commit", "-m", f"Agent work: {branch}"],
        text=True,
        capture_output=True,
        timeout=60,
        env=env,
    )
    if commit.returncode != 0:
        raise RuntimeError((commit.stderr or commit.stdout or "git commit failed").strip())
    original_branch = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"], text=True, capture_output=True, timeout=30)
    if original_branch.returncode != 0:
        raise RuntimeError((original_branch.stderr or original_branch.stdout or "git rev-parse failed").strip())
    checkout = subprocess.run(["git", "-C", str(repo_root), "checkout", original_branch.stdout.strip()], text=True, capture_output=True, timeout=60, env=env)
    if checkout.returncode != 0:
        raise RuntimeError((checkout.stderr or checkout.stdout or "git checkout failed").strip())
    merge = subprocess.run(["git", "-C", str(repo_root), "merge", branch, "--no-edit"], text=True, capture_output=True, timeout=120, env=env)
    if merge.returncode != 0:
        raise RuntimeError((merge.stderr or merge.stdout or "git merge failed").strip())
    record["changedPaths"] = changed_paths
    record["mergeMode"] = "git"
    record["mergeCommitOutput"] = commit.stdout[-2000:]
    record["mergeOutput"] = merge.stdout[-2000:]
    return sorted(dict.fromkeys(changed_paths))


def merge_agent_worktree(agent_id: str, worktree_path: str | Path, strategy: str = "copy", repo_root: Path | None = None) -> list[str]:
    source_root = Path(worktree_path).resolve()
    if not source_root.exists():
        return []
    record = STATE.setdefault("agentWorktrees", {}).setdefault(agent_id, {"path": str(source_root)})
    repo_root = (repo_root or ROOT).resolve()
    if record.get("mode") == "git":
        changed_paths = _git_worktree_changed_paths(source_root)
        record["changedPaths"] = changed_paths
        if str(strategy).lower() == "git":
            try:
                merged = _git_commit_and_merge_worktree(agent_id, source_root, repo_root)
            except Exception as exc:
                record["mergeMode"] = "copy_fallback"
                record["mergeError"] = str(exc)
                merged = _copy_worktree_changes(source_root, changed_paths, repo_root)
        else:
            record["mergeMode"] = "copy"
            merged = _copy_worktree_changes(source_root, changed_paths, repo_root)
    else:
        record["mergeMode"] = "copy"
        merged = _copy_worktree_changes(source_root, target_root=repo_root)
    record["mergeStrategy"] = strategy
    record["mergedFiles"] = merged
    record["mergedAt"] = utc_now()
    record["status"] = "merged"
    return merged


def cleanup_agent_worktree(agent_id: str, worktree_path: str | Path) -> None:
    path = Path(worktree_path).resolve()
    record = STATE.setdefault("agentWorktrees", {}).setdefault(agent_id, {"path": str(path)})
    if record.get("mode") == "git":
        TOOL_REGISTRY.revoke_external_root(path)
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(path)], text=True, capture_output=True, timeout=60)
        branch = str(record.get("branchName") or "")
        if branch:
            subprocess.run(["git", "-C", str(ROOT), "branch", "-D", branch], text=True, capture_output=True, timeout=30)
    else:
        base = (DATA_DIR / "agent-worktrees").resolve()
        if path.exists() and (path == base or base in path.parents):
            shutil.rmtree(path, ignore_errors=True)
    record["cleanedAt"] = utc_now()
    record["status"] = "cleaned"


def path_relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved == root:
        return "."
    if root not in resolved.parents:
        raise ValueError("Path escapes workspace")
    return resolved.relative_to(root).as_posix()


def rewrite_tool_arguments_for_session(session_id: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    session = STATE.setdefault("sessions", {}).get(session_id) or {}
    working_dir = str(session.get("workingDirectory") or ".")
    if not working_dir:
        return arguments
    try:
        working_path = Path(working_dir)
        if not working_path.is_absolute():
            working_path = ROOT / working_path
        working_path = working_path.resolve()
    except Exception:
        return arguments
    allowed_roots = [(DATA_DIR / "agent-worktrees").resolve()]
    if session.get("agentWorktreePath"):
        allowed_roots.append(Path(str(session["agentWorktreePath"])).resolve())
    if not any(working_path == root or root in working_path.parents for root in allowed_roots):
        return arguments
    rewritten = dict(arguments)
    for key in ("path", "notebook_path"):
        value = rewritten.get(key)
        if not value or not isinstance(value, str):
            continue
        requested = Path(value)
        if requested.is_absolute():
            try:
                requested = requested.resolve()
                if ROOT.resolve() in requested.parents:
                    requested = requested.relative_to(ROOT.resolve())
                else:
                    continue
            except Exception:
                continue
        target = (working_path / requested).resolve()
        if target != working_path and working_path not in target.parents:
            continue
        try:
            rewritten[key] = path_relative_to_root(target)
        except ValueError:
            rewritten[key] = str(target)
    return rewritten


def update_session_file_state(session_id: str, tool_name: str, arguments: dict[str, Any], result_payload: dict[str, Any]) -> None:
    session = STATE.setdefault("sessions", {}).get(session_id)
    if not session or result_payload.get("isError"):
        return
    path_value = arguments.get("path") or arguments.get("notebook_path")
    if not path_value or not isinstance(path_value, str):
        return
    cache = session.setdefault("fileStateCache", {})
    normalized = os.path.normpath(path_value).replace("\\", "/")
    now = int(time.time() * 1000)
    if tool_name == "read_file":
        cache[normalized] = {
            "content": str(result_payload.get("content") or ""),
            "timestamp": now,
            "offset": arguments.get("offset"),
            "limit": arguments.get("limit"),
            "isPartialView": False,
            "state": "read",
        }
    elif tool_name in {"write_file", "edit_file", "NotebookEdit"}:
        previous = cache.get(normalized, {})
        cache[normalized] = {
            "content": str(arguments.get("content") or previous.get("content") or ""),
            "timestamp": now,
            "offset": previous.get("offset"),
            "limit": previous.get("limit"),
            "isPartialView": True,
            "state": "modified",
        }
    session["updatedAt"] = utc_now()


def clone_file_state_cache(parent_session_id: str, child_session: dict[str, Any]) -> None:
    parent = STATE.setdefault("sessions", {}).get(parent_session_id)
    if not parent:
        return
    cache = parent.get("fileStateCache") or {}
    child_session["fileStateCache"] = json.loads(json.dumps(cache, ensure_ascii=False))


def merge_file_state_cache(child_session_id: str, parent_session_id: str) -> list[str]:
    parent = STATE.setdefault("sessions", {}).get(parent_session_id)
    child = STATE.setdefault("sessions", {}).get(child_session_id)
    if not parent or not child:
        return []
    parent_cache = parent.setdefault("fileStateCache", {})
    child_cache = child.get("fileStateCache") or {}
    merged: list[str] = []
    for path, state in child_cache.items():
        existing = parent_cache.get(path)
        if not existing or int(state.get("timestamp") or 0) >= int(existing.get("timestamp") or 0):
            parent_cache[path] = json.loads(json.dumps(state, ensure_ascii=False))
            merged.append(path)
    parent["updatedAt"] = utc_now()
    return merged


def agent_child_session_id(agent_id: str, fork: bool = False) -> str:
    return f"{'fork' if fork else 'subagent'}-{agent_id}"


def prepare_subagent_session(
    agent_id: str,
    prompt: str,
    agent_type: str,
    model: str,
    parent_session_id: str,
    working_directory: str,
    fork: bool,
    hierarchy: str,
) -> dict[str, Any]:
    child_session_id = agent_child_session_id(agent_id, fork)
    session = get_or_create_session(child_session_id)
    if fork and parent_session_id in STATE.setdefault("sessions", {}):
        parent_messages = STATE["sessions"][parent_session_id].get("messages", [])
        session["messages"] = [json.loads(json.dumps(message, ensure_ascii=False)) for message in parent_messages]
        clone_file_state_cache(parent_session_id, session)
    session["title"] = session.get("title") or f"Subagent {agent_id}"
    if model and model != "default":
        session["model"] = model
    session["workingDirectory"] = working_directory
    session["systemPrompt"] = subagent_system_prompt(prompt, agent_type, parent_session_id, working_directory, fork)
    session["parentSessionId"] = parent_session_id
    session["agentType"] = agent_type
    session["agentHierarchy"] = hierarchy
    definition = resolve_agent_definition(agent_type)
    session["agentDefinition"] = {
        "name": definition["name"],
        "maxTurns": definition["maxTurns"],
        "defaultModel": definition["defaultModel"],
        "readOnly": definition["readOnly"],
    }
    session["agentAllowedTools"] = sorted(definition["allowedTools"]) if definition.get("allowedTools") else None
    session["agentDeniedTools"] = sorted(set(definition.get("deniedTools") or set()).union(GLOBAL_SUBAGENT_DENIED_TOOLS))
    session["updatedAt"] = utc_now()
    return session


def agent_slot_fields(payload: dict[str, Any], parent_session_id: str) -> dict[str, Any]:
    agent_id = str(payload.get("agentId") or payload.get("agent_id") or f"agent-{uuid.uuid4().hex[:8]}")
    agent_type = str(payload.get("subagent_type") or payload.get("agentType") or "general-purpose")
    definition = resolve_agent_definition(agent_type)
    raw_model = str(payload.get("model") or definition.get("defaultModel") or "standard")
    model = raw_model if raw_model == "default" else MODEL_PROVIDERS.resolve_model_alias(raw_model)
    nesting_depth = int(payload.get("nestingDepth") or payload.get("nesting_depth") or 0) + 1
    parent_hierarchy = str(payload.get("agentHierarchy") or payload.get("agent_hierarchy") or "main")
    return {
        "agentId": agent_id,
        "agentType": agent_type,
        "model": model,
        "sessionId": str(payload.get("sessionId") or payload.get("session_id") or parent_session_id),
        "nestingDepth": nesting_depth,
        "agentHierarchy": f"{parent_hierarchy} > subagent-{agent_id}",
        "teamName": str(payload.get("teamName") or payload.get("team_name") or "").strip() or None,
        "fork": bool(payload.get("fork")),
        "isolation": str(payload.get("isolation") or "NONE"),
    }


def find_swarm_by_team(team_name: str) -> dict[str, Any] | None:
    for swarm in STATE.setdefault("swarms", {}).values():
        if swarm.get("teamName") == team_name and swarm.get("phase") != "TERMINATED":
            return swarm
    return None


def dispatch_direct_agent_to_team(payload: dict[str, Any]) -> ToolResult:
    team_name = str(payload.get("teamName") or "").strip()
    agent_id = str(payload.get("agentId") or new_id("agent"))
    session_id = str(payload.get("sessionId") or "default")
    prompt = str(payload.get("prompt") or "")
    try:
        asyncio.get_running_loop()
        return ToolResult(
            "Direct TeamManager dispatch is synchronous; use the async Agent execution path while an event loop is running.",
            isError=True,
            metadata={"status": "team_dispatch_async_context", "agentId": agent_id, "teamName": team_name, "sessionId": session_id},
        )
    except RuntimeError:
        pass
    swarm = find_swarm_by_team(team_name)
    if not swarm:
        return ToolResult(f"Team not found: {team_name}", isError=True, metadata={"status": "team_not_found", "agentId": agent_id, "teamName": team_name, "sessionId": session_id})

    swarm_id = str(swarm.get("swarmId"))
    swarm["workerSequence"] = int(swarm.get("workerSequence") or len(swarm.setdefault("workers", {}))) + 1
    worker_id = str(payload.get("workerId") or f"team-agent-{swarm['workerSequence']}")
    child_session_id = agent_child_session_id(agent_id, False)
    worker = make_worker_state(
        swarm_id,
        worker_id,
        {
            "task": prompt,
            "prompt": prompt,
            "agentType": payload.get("agentType"),
            "model": None if payload.get("model") == "default" else payload.get("model"),
            "toolCalls": payload.get("toolCalls") if isinstance(payload.get("toolCalls"), list) else [],
            "turns": payload.get("turns") if isinstance(payload.get("turns"), list) else [],
        },
    )
    worker["sessionId"] = child_session_id
    worker["parentSessionId"] = session_id
    worker["agentHierarchy"] = payload.get("agentHierarchy")
    swarm.setdefault("workers", {})[worker_id] = worker
    swarm["totalTasks"] = int(swarm.get("totalTasks") or 0) + 1
    append_swarm_event(swarm, "team_dispatch", f"Agent dispatched to team: {team_name}", workerId=worker_id, agentId=agent_id)
    publish_coordinator_event(swarm, "team_dispatch", {"agentId": agent_id, "workerId": worker_id, "teamName": team_name, "prompt": prompt[:500]})
    push_swarm_state(swarm)
    asyncio.run(run_swarm_worker(swarm_id, worker_id))
    result = swarm.setdefault("results", {}).get(worker_id, {})
    content = str(result.get("result") or result.get("error") or "No result from team worker.")
    if len(content) > 100_000:
        content = content[:100_000] + "\n...[truncated]"
    return ToolResult(
        content,
        isError=result.get("status") == "failed",
        metadata={
            "status": "completed" if result.get("status") != "failed" else "failed",
            "agentId": agent_id,
            "teamName": team_name,
            "sessionId": session_id,
            "childSessionId": child_session_id,
            "swarmId": swarm_id,
            "workerId": worker_id,
            "turnCount": result.get("turnCount"),
            "agentType": payload.get("agentType"),
            "model": payload.get("model"),
            "nestingDepth": payload.get("nestingDepth"),
            "agentHierarchy": payload.get("agentHierarchy"),
        },
    )


def dispatch_direct_agent(payload: dict[str, Any]) -> ToolResult:
    # Direct Agent 入口：让 ToolRegistry 直接调用的 Agent 也复用真实子会话和 QueryEngine。
    agent_id = str(payload.get("agentId") or new_id("agent"))
    session_id = str(payload.get("sessionId") or "default")
    prompt = str(payload.get("prompt") or "")
    agent_type = str(payload.get("agentType") or "general-purpose")
    model = payload.get("model")
    hierarchy = str(payload.get("agentHierarchy") or f"main > subagent-{agent_id}")
    fork = bool(payload.get("fork"))
    try:
        asyncio.get_running_loop()
        return ToolResult(
            "Direct Agent dispatch is synchronous; use the async Agent execution path while an event loop is running.",
            isError=True,
            metadata={"status": "agent_dispatch_async_context", "agentId": agent_id, "sessionId": session_id},
        )
    except RuntimeError:
        pass

    worktree_path: Path | None = None
    working_directory = str(payload.get("workingDirectory") or STATE.setdefault("sessions", {}).get(session_id, {}).get("workingDirectory") or ".")
    metadata: dict[str, Any] = {
        "status": "completed",
        "agentId": agent_id,
        "sessionId": session_id,
        "childSessionId": agent_child_session_id(agent_id, fork),
        "agentType": agent_type,
        "model": model,
        "teamName": payload.get("teamName"),
        "nestingDepth": payload.get("nestingDepth"),
        "agentHierarchy": hierarchy,
    }
    try:
        # Worktree 隔离：高风险修改先在独立目录里执行，完成后再合并回主工作区。
        if str(payload.get("isolation") or "").upper() == "WORKTREE":
            worktree_path = create_agent_worktree(agent_id)
            working_directory = str(worktree_path)
            record = STATE.setdefault("agentWorktrees", {}).get(agent_id, {})
            metadata["worktreePath"] = str(worktree_path)
            metadata["worktreeMode"] = record.get("mode")
            metadata["worktreeBranch"] = record.get("branchName")
        # 子 Agent 会话准备：写入父子关系、工作目录、Agent 类型、系统提示和禁用工具。
        prepare_subagent_session(
            agent_id,
            prompt,
            agent_type,
            str(model or STATE["config"].get("defaultModel") or "qwen3.7-plus"),
            session_id,
            working_directory,
            fork,
            hierarchy,
        )
        child_payload = {
            "sessionId": metadata["childSessionId"],
            "prompt": prompt,
            "model": None if model == "default" else model,
            "workingDirectory": working_directory,
            "collapseContext": False,
            "toolCalls": payload.get("toolCalls") if isinstance(payload.get("toolCalls"), list) else [],
        }
        # Direct Agent 本质也是再跑一轮 QueryEngine，但必须套超时防止卡死。
        timeout_ms = resolve_agent_timeout_ms(agent_type, payload)
        try:
            result = asyncio.run(
                wait_for_agent_with_permission_budget(
                    run_query_payload(child_payload),
                    str(metadata["childSessionId"]),
                    timeout_ms,
                )
            )
        except (TimeoutError, asyncio.TimeoutError):
            QUERY_ABORTS.abort(str(metadata["childSessionId"]), "TIMEOUT")
            metadata["status"] = "failed"
            return ToolResult(
                f"Agent timed out after {timeout_ms // 1000} seconds",
                isError=True,
                metadata=metadata,
            )
        answer = str(result.get("answer") or "")
        if len(answer) > 100_000:
            answer = answer[:100_000] + "\n...[truncated]"
        merged_files: list[str] = []
        # Worktree Agent 成功后把隔离目录的改动合并回父工作区。
        if worktree_path is not None:
            merged_files = merge_agent_worktree(agent_id, worktree_path, strategy=str(payload.get("worktreeMergeStrategy") or payload.get("mergeStrategy") or STATE.get("config", {}).get("agentWorktreeMergeStrategy") or "copy").lower())
            record = STATE.setdefault("agentWorktrees", {}).get(agent_id, {})
            metadata["worktreeMergeStrategy"] = record.get("mergeStrategy")
            metadata["worktreeMergeMode"] = record.get("mergeMode")
            if record.get("mergeError"):
                metadata["worktreeMergeError"] = record.get("mergeError")
        merged_file_states = merge_file_state_cache(str(metadata["childSessionId"]), session_id)
        metadata.update(
            {
                "queryLoopId": (result.get("queryLoop") or {}).get("id"),
                "mergedFiles": merged_files,
                "mergedFileStates": merged_file_states,
            }
        )
        return ToolResult(answer, metadata=metadata)
    except Exception as exc:
        metadata["status"] = "failed"
        return ToolResult(f"Agent execution failed: {exc}", isError=True, metadata=metadata)
    finally:
        # 无论执行成功还是失败，都要清理临时 worktree，避免残留目录和分支。
        if worktree_path is not None:
            cleanup_agent_worktree(agent_id, worktree_path)


TOOL_REGISTRY.set_team_dispatcher(dispatch_direct_agent_to_team)
TOOL_REGISTRY.set_agent_dispatcher(dispatch_direct_agent)


def hydrate_background_agent_task(task: dict[str, Any], fields: dict[str, Any], output_file: str, child_session_id: str) -> dict[str, Any]:
    task["agentId"] = fields["agentId"]
    task["agentHierarchy"] = fields["agentHierarchy"]
    task["outputFile"] = output_file
    task["childSessionId"] = child_session_id
    task["agentType"] = fields["agentType"]
    task["model"] = fields["model"]
    task["isolation"] = fields["isolation"]
    task["fork"] = fields["fork"]
    task["teamName"] = fields["teamName"]
    if fields.get("description"):
        task["agentDescription"] = fields["description"]
    return task


async def execute_agent_tool(loop: QueryLoopState, payload: dict[str, Any], live_send: Any | None = None) -> dict[str, Any]:
    # QueryEngine 内部 Agent 入口：把一次 Agent 工具调用变成独立子会话执行。
    prompt = str(payload.get("prompt") or "")
    if not prompt:
        return {"content": "Missing prompt", "isError": True, "metadata": {"status": "error"}}
    description = str(payload.get("description") or "sub-agent task")
    fields = agent_slot_fields(payload, loop.sessionId)
    fields["description"] = description
    agent_id = fields["agentId"]
    parent_session_id = fields["sessionId"]
    nesting_depth = int(fields["nestingDepth"])
    # 启动前先占并发槽位，防止无限 Agent 或单会话抢占所有资源。
    limit_error = TOOL_REGISTRY._acquire_agent_slot(agent_id, parent_session_id, nesting_depth)
    if limit_error:
        return {"content": limit_error, "isError": True, "metadata": {"status": "limit_exceeded", **fields}}

    child_session_id = agent_child_session_id(agent_id, bool(fields["fork"]))
    working_directory = str(payload.get("workingDirectory") or payload.get("working_directory") or STATE.setdefault("sessions", {}).get(loop.sessionId, {}).get("workingDirectory") or ".")
    output_file = str(DATA_DIR / "agents" / f"{agent_id}.txt")
    background = bool(payload.get("run_in_background") or payload.get("runInBackground"))
    metadata = {"agentId": agent_id, "sessionId": parent_session_id, "childSessionId": child_session_id, "description": description, **fields}
    worktree_merge_strategy = str(payload.get("worktreeMergeStrategy") or payload.get("mergeStrategy") or STATE.get("config", {}).get("agentWorktreeMergeStrategy") or "copy").lower()

    async def dispatch_to_team() -> dict[str, Any]:
        # Team 路径：带 teamName 的 Agent 不自己执行，而是动态派发到 Swarm worker。
        team_name = str(fields.get("teamName") or "")
        swarm = find_swarm_by_team(team_name)
        if not swarm:
            # 找不到 team 也必须释放槽位，否则一次失败分发就会泄漏并发额度。
            TOOL_REGISTRY._release_agent_slot(agent_id, parent_session_id)
            return {"content": f"Team not found: {team_name}", "isError": True, "metadata": {**metadata, "status": "team_not_found"}}
        swarm_id = str(swarm.get("swarmId"))
        swarm["workerSequence"] = int(swarm.get("workerSequence") or len(swarm.setdefault("workers", {}))) + 1
        worker_id = str(payload.get("workerId") or f"team-agent-{swarm['workerSequence']}")
        worker = make_worker_state(
            swarm_id,
            worker_id,
            {
                "task": prompt,
                "prompt": prompt,
                "agentType": fields["agentType"],
                "model": None if fields["model"] == "default" else fields["model"],
                "toolCalls": payload.get("toolCalls") if isinstance(payload.get("toolCalls"), list) else [],
                "turns": payload.get("turns") if isinstance(payload.get("turns"), list) else [],
            },
        )
        worker["sessionId"] = child_session_id
        worker["parentSessionId"] = parent_session_id
        worker["agentHierarchy"] = fields["agentHierarchy"]
        swarm.setdefault("workers", {})[worker_id] = worker
        swarm["totalTasks"] = int(swarm.get("totalTasks") or 0) + 1
        append_swarm_event(swarm, "team_dispatch", f"Agent dispatched to team: {team_name}", workerId=worker_id, agentId=agent_id)
        publish_coordinator_event(swarm, "team_dispatch", {"agentId": agent_id, "workerId": worker_id, "teamName": team_name, "prompt": prompt[:500]})
        push_swarm_state(swarm)
        task = await start_swarm_worker_task(swarm_id, worker_id)
        if background:
            # 后台 Team Agent：先返回 async_launched，worker 完成后由 callback 写 task 状态。
            task_record = hydrate_background_agent_task(
                TOOL_REGISTRY._new_task(prompt[:200], session_id=parent_session_id, task_type=f"agent:{fields['agentType']}"),
                fields,
                output_file,
                child_session_id,
            )
            task_record["swarmId"] = swarm_id
            task_record["workerId"] = worker_id
            publish_background_agent_event(agent_id, "agent_started", {"prompt": prompt, "taskId": task_record["taskId"], "childSessionId": child_session_id, "swarmId": swarm_id, "workerId": worker_id})

            def _complete_team_background(done: asyncio.Task[Any]) -> None:
                # 后台 worker 收尾：把 Swarm 结果转成 Agent task 的 COMPLETED/FAILED。
                result = swarm.setdefault("results", {}).get(worker_id, {})
                error = None
                if done.cancelled():
                    error = "Team worker cancelled"
                elif done.exception() is not None:
                    error = str(done.exception())
                elif result.get("status") == "failed":
                    error = str(result.get("error") or "Team worker failed")
                content = str(result.get("result") or result.get("notificationXml") or result.get("error") or error or "")
                if len(content) > 100_000:
                    content = content[:100_000] + "\n...[truncated]"
                try:
                    out = Path(output_file)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(content, encoding="utf-8")
                except OSError as exc:
                    error = error or str(exc)
                task_record["output"] = content
                task_record["updatedAt"] = time.time()
                if error:
                    task_record["status"] = "FAILED"
                    task_record["error"] = error
                    publish_background_agent_event(agent_id, "agent_failed", {"error": error, "childSessionId": child_session_id, "swarmId": swarm_id, "workerId": worker_id})
                else:
                    task_record["status"] = "COMPLETED"
                    publish_background_agent_event(agent_id, "agent_completed", {"resultPreview": content[:500], "childSessionId": child_session_id, "swarmId": swarm_id, "workerId": worker_id})
                # 后台 Team Agent 到这里才真正释放并发槽位。
                TOOL_REGISTRY._release_agent_slot(agent_id, parent_session_id)

            task.add_done_callback(_complete_team_background)
            return {
                "content": (
                    "Agent launched in background.\n"
                    f"Agent ID: {agent_id}\n"
                    f"Output file: {output_file}\n"
                    f"Description: {description}\n"
                    f"Prompt: {prompt}"
                ),
                "isError": False,
                "metadata": {**metadata, "status": "async_launched", "swarmId": swarm_id, "workerId": worker_id, "taskId": task_record["taskId"], "outputFile": output_file},
            }
        try:
            timeout_ms = resolve_agent_timeout_ms(str(fields["agentType"]), payload)
            await wait_for_swarm_worker_with_permission_budget(task, swarm, worker_id, timeout_ms)
            result = swarm.setdefault("results", {}).get(worker_id, {})
            content = str(result.get("result") or result.get("error") or "No result from team worker.")
            return {
                "content": content[:100_000] + ("\n...[truncated]" if len(content) > 100_000 else ""),
                "isError": result.get("status") == "failed",
                "metadata": {**metadata, "status": "completed" if result.get("status") != "failed" else "failed", "swarmId": swarm_id, "workerId": worker_id, "turnCount": result.get("turnCount")},
            }
        except (TimeoutError, asyncio.TimeoutError):
            QUERY_ABORTS.abort(child_session_id, "TIMEOUT")
            timeout_ms = resolve_agent_timeout_ms(str(fields["agentType"]), payload)
            error = f"Agent timed out after {timeout_ms // 1000} seconds"
            result = {"workerId": worker_id, "status": "failed", "error": error}
            swarm.setdefault("results", {})[worker_id] = result
            append_swarm_event(swarm, "team_agent_timeout", f"Agent team dispatch timed out: {worker_id}", workerId=worker_id, agentId=agent_id, error=error)
            push_swarm_state(swarm)
            return {
                "content": error,
                "isError": True,
                "metadata": {**metadata, "status": "failed", "swarmId": swarm_id, "workerId": worker_id},
            }
        finally:
            TOOL_REGISTRY._release_agent_slot(agent_id, parent_session_id)

    if fields.get("teamName"):
        # teamName 是 Agent 从普通子任务切到 Team/Swarm 协作的分流条件。
        await emit_query_event(loop, "agent_team_dispatch", {"agentId": agent_id, "teamName": fields["teamName"], "childSessionId": child_session_id}, live_send)
        return await dispatch_to_team()

    worktree_path: Path | None = None
    # 普通 Agent 的 worktree 隔离：让子 Agent 在临时工作区修改，降低主目录风险。
    if str(fields.get("isolation") or "").upper() == "WORKTREE":
        worktree_path = create_agent_worktree(agent_id)
        working_directory = str(worktree_path)
        record = STATE.setdefault("agentWorktrees", {}).get(agent_id, {})
        metadata["worktreePath"] = str(worktree_path)
        metadata["worktreeMode"] = record.get("mode")
        metadata["worktreeBranch"] = record.get("branchName")

    async def run_child(task_record: dict[str, Any] | None = None) -> dict[str, Any]:
        # run_child 是普通子 Agent 的执行核心；task_record 不为空表示后台模式。
        merged_files: list[str] = []
        merged_file_states: list[str] = []
        try:
            prepare_subagent_session(
                agent_id,
                prompt,
                str(fields["agentType"]),
                str(fields["model"]),
                parent_session_id,
                working_directory,
                bool(fields["fork"]),
                str(fields["agentHierarchy"]),
            )
            # Worktree 元信息写入子 session，便于工具按隔离目录运行和前端展示。
            child_session = STATE.setdefault("sessions", {}).get(child_session_id)
            if child_session is not None and worktree_path is not None:
                record = STATE.setdefault("agentWorktrees", {}).get(agent_id, {})
                child_session["agentWorktreePath"] = str(worktree_path)
                child_session["agentWorktreeMode"] = record.get("mode")
                child_session["agentWorktreeBranch"] = record.get("branchName")
            child_payload = {
                "sessionId": child_session_id,
                "prompt": prompt,
                "model": None if fields["model"] == "default" else fields["model"],
                "workingDirectory": working_directory,
                "collapseContext": False,
                "toolCalls": payload.get("toolCalls") if isinstance(payload.get("toolCalls"), list) else [],
            }
            # 子 Agent 通过 child_payload 再跑一轮 QueryEngine，并用 wait_for 兜住超时。
            result = await wait_for_agent_with_permission_budget(
                run_query_payload(child_payload),
                child_session_id,
                resolve_agent_timeout_ms(str(fields["agentType"]), payload),
            )
            answer = str(result.get("answer") or "")
            if len(answer) > 100_000:
                # 子 Agent 结果会进入父上下文，必须截断避免 token 和响应体爆炸。
                answer = answer[:100_000] + "\n...[truncated]"
            if worktree_path is not None:
                # 子 Agent 完成后合并 worktree 改动，并把合并结果写进 metadata。
                merged_files = merge_agent_worktree(agent_id, worktree_path, strategy=worktree_merge_strategy)
                record = STATE.setdefault("agentWorktrees", {}).get(agent_id, {})
                metadata["worktreeMergeStrategy"] = record.get("mergeStrategy")
                metadata["worktreeMergeMode"] = record.get("mergeMode")
                if record.get("mergeError"):
                    metadata["worktreeMergeError"] = record.get("mergeError")
            merged_file_states = merge_file_state_cache(child_session_id, parent_session_id)
            if task_record is not None:
                # 后台 Agent 完成后要落盘输出并更新 task，否则 await 接口会一直认为它在跑。
                out = Path(output_file)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(answer, encoding="utf-8")
                task_record["status"] = "COMPLETED"
                task_record["output"] = answer
                task_record["mergedFiles"] = merged_files
                task_record["mergedFileStates"] = merged_file_states
                task_record["updatedAt"] = time.time()
                publish_background_agent_event(agent_id, "agent_completed", {"resultPreview": answer[:500], "childSessionId": child_session_id})
            return {
                "content": answer,
                "isError": False,
                "metadata": {**metadata, "status": "completed", "outputFile": output_file, "queryLoopId": (result.get("queryLoop") or {}).get("id"), "mergedFiles": merged_files, "mergedFileStates": merged_file_states},
            }
        except (TimeoutError, asyncio.TimeoutError):
            QUERY_ABORTS.abort(child_session_id, "TIMEOUT")
            timeout_ms = resolve_agent_timeout_ms(str(fields["agentType"]), payload)
            error = f"Agent timed out after {timeout_ms // 1000} seconds"
            if task_record is not None:
                task_record["status"] = "FAILED"
                task_record["error"] = error
                task_record["updatedAt"] = time.time()
                try:
                    out = Path(output_file)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(error, encoding="utf-8")
                except OSError:
                    pass
                publish_background_agent_event(agent_id, "agent_failed", {"error": error, "childSessionId": child_session_id})
            return {"content": error, "isError": True, "metadata": {**metadata, "status": "failed", "outputFile": output_file}}
        except Exception as exc:
            if task_record is not None:
                # 后台 Agent 失败也必须显式标 FAILED，避免后台任务永久 RUNNING。
                task_record["status"] = "FAILED"
                task_record["error"] = str(exc)
                task_record["updatedAt"] = time.time()
                publish_background_agent_event(agent_id, "agent_failed", {"error": str(exc), "childSessionId": child_session_id})
            return {"content": f"Agent execution failed: {exc}", "isError": True, "metadata": {**metadata, "status": "failed", "outputFile": output_file}}
        finally:
            # Agent 所有退出路径都必须清理隔离目录并释放槽位，这是防泄漏的保险丝。
            if worktree_path is not None:
                cleanup_agent_worktree(agent_id, worktree_path)
            TOOL_REGISTRY._release_agent_slot(agent_id, parent_session_id)

    if background:
        # 后台 Agent：创建 task 后立即返回，真实执行交给 asyncio task 收尾。
        task_record = hydrate_background_agent_task(
            TOOL_REGISTRY._new_task(prompt[:200], session_id=parent_session_id, task_type=f"agent:{fields['agentType']}"),
            fields,
            output_file,
            child_session_id,
        )
        asyncio.create_task(run_child(task_record))
        await emit_query_event(loop, "agent_started", {"agentId": agent_id, "taskId": task_record["taskId"], "childSessionId": child_session_id}, live_send)
        publish_background_agent_event(agent_id, "agent_started", {"prompt": prompt, "taskId": task_record["taskId"], "childSessionId": child_session_id})
        return {
            "content": (
                "Agent launched in background.\n"
                f"Agent ID: {agent_id}\n"
                f"Output file: {output_file}\n"
                f"Description: {description}\n"
                f"Prompt: {prompt}"
            ),
            "isError": False,
            "metadata": {**metadata, "status": "async_launched", "taskId": task_record["taskId"], "outputFile": output_file},
        }

    # 同步 Agent：父 QueryEngine 等待子 Agent 结束后再继续生成最终回答。
    await emit_query_event(loop, "agent_started", {"agentId": agent_id, "childSessionId": child_session_id}, live_send)
    result_payload = await run_child(None)
    await emit_query_event(loop, "agent_completed", {"agentId": agent_id, "childSessionId": child_session_id, "status": result_payload["metadata"].get("status")}, live_send)
    return result_payload


async def _execute_query_tools_legacy(loop: QueryLoopState, tool_calls: list[dict[str, Any]], live_send: Any | None = None) -> list[dict[str, Any]]:
    # 工具调度入口：负责排序、权限检查、事件推送、结果回写和连续 Agent 并发执行。
    results = []
    ordered = TOOL_SCHEDULER.ordered_calls(tool_calls)
    index = 0
    while index < len(ordered):
        # 每个工具开始前都检查用户中断，避免取消后继续执行危险操作。
        if QUERY_ABORTS.is_aborted(loop.sessionId):
            loop.abort("USER_INTERRUPT")
            await emit_query_event(loop, "interrupt_ack", {"reason": "USER_INTERRUPT"}, live_send)
            break
        call = ordered[index]
        tool_use_id = str(call["id"])
        tool_name = str(call["name"])
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        arguments = rewrite_tool_arguments_for_session(loop.sessionId, tool_name, arguments)
        # 每次工具调用都写入 QueryLoop，前端才能实时显示工具生命周期。
        loop.transition(QueryPhase.TOOL_RUNNING, f"tool:{tool_name}")
        loop.record_tool_call(tool_use_id, tool_name, arguments)
        await emit_query_event(loop, "tool_use_start", {"toolUseId": tool_use_id, "toolName": tool_name, "input": arguments}, live_send)
        await emit_query_event(loop, "tool_use_input", {"toolUseId": tool_use_id, "toolName": tool_name, "input": arguments}, live_send)
        await emit_query_event(loop, "tool_use_progress", {"toolUseId": tool_use_id, "progress": "started"}, live_send)
        started = time.time()
        session_for_tool = STATE.setdefault("sessions", {}).get(loop.sessionId) or {}
        allowed, deny_reason = agent_session_tool_allowed(session_for_tool, tool_name)
        if not allowed:
            # Agent 类型权限在这里兜底，例如 explore/plan 子 Agent 禁止写文件或再调 Agent。
            result_payload = {"content": deny_reason or "tool denied", "isError": True, "metadata": {"decision": "deny"}}
            loop.update_tool_call(tool_use_id, "error", result_payload, "denied")
            await emit_query_event(
                loop,
                "tool_result",
                {
                    "toolUseId": tool_use_id,
                    "content": result_payload["content"],
                    "isError": True,
                    "result": result_payload,
                    "durationMs": int((time.time() - started) * 1000),
                },
                live_send,
            )
            results.append({"toolUseId": tool_use_id, "toolName": tool_name, **result_payload})
            index += 1
            continue
        if tool_name == "Agent":
            # 连续多个 Agent 工具调用会被分组并发执行，对齐“并行启动多个子 Agent”的语义。
            agent_jobs: list[dict[str, Any]] = [{"toolUseId": tool_use_id, "toolName": tool_name, "arguments": arguments, "started": started}]
            index += 1
            while index < len(ordered) and str(ordered[index].get("name")) == "Agent":
                if QUERY_ABORTS.is_aborted(loop.sessionId):
                    break
                grouped_call = ordered[index]
                grouped_tool_use_id = str(grouped_call["id"])
                grouped_arguments = grouped_call.get("arguments") if isinstance(grouped_call.get("arguments"), dict) else {}
                grouped_arguments = rewrite_tool_arguments_for_session(loop.sessionId, "Agent", grouped_arguments)
                loop.transition(QueryPhase.TOOL_RUNNING, "tool:Agent")
                loop.record_tool_call(grouped_tool_use_id, "Agent", grouped_arguments)
                await emit_query_event(loop, "tool_use_start", {"toolUseId": grouped_tool_use_id, "toolName": "Agent", "input": grouped_arguments}, live_send)
                await emit_query_event(loop, "tool_use_input", {"toolUseId": grouped_tool_use_id, "toolName": "Agent", "input": grouped_arguments}, live_send)
                await emit_query_event(loop, "tool_use_progress", {"toolUseId": grouped_tool_use_id, "progress": "started"}, live_send)
                grouped_started = time.time()
                session_for_tool = STATE.setdefault("sessions", {}).get(loop.sessionId) or {}
                grouped_allowed, grouped_deny_reason = agent_session_tool_allowed(session_for_tool, "Agent")
                if not grouped_allowed:
                    grouped_result_payload = {"content": grouped_deny_reason or "tool denied", "isError": True, "metadata": {"decision": "deny"}}
                    agent_jobs.append({"toolUseId": grouped_tool_use_id, "toolName": "Agent", "arguments": grouped_arguments, "started": grouped_started, "precomputed": grouped_result_payload, "completionReason": "denied"})
                else:
                    agent_jobs.append({"toolUseId": grouped_tool_use_id, "toolName": "Agent", "arguments": grouped_arguments, "started": grouped_started})
                index += 1

            async def run_agent_job(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
                # 被权限提前拒绝的 Agent 不再执行，只把预计算结果回写。
                if "precomputed" in job:
                    return job, job["precomputed"]
                return job, await execute_agent_tool(loop, job["arguments"], live_send)

            # gather 让同一批连续 Agent 真并发执行，而不是一个等一个。
            for job, result_payload in await asyncio.gather(*(run_agent_job(job) for job in agent_jobs)):
                loop.update_tool_call(job["toolUseId"], "error" if result_payload.get("isError") else "completed", result_payload, str(job.get("completionReason") or "completed"))
                await emit_query_event(
                    loop,
                    "tool_result",
                    {
                        "toolUseId": job["toolUseId"],
                        "content": result_payload.get("content"),
                        "isError": bool(result_payload.get("isError")),
                        "result": result_payload,
                        "durationMs": int((time.time() - job["started"]) * 1000),
                        "summary": tool_call_summary(loop, str(job["toolUseId"])),
                    },
                    live_send,
                )
                results.append({"toolUseId": job["toolUseId"], "toolName": job["toolName"], **result_payload})
            continue
        result = await asyncio.to_thread(TOOL_REGISTRY.call, tool_name, arguments)
        result_payload = result.to_dict()
        if (result.metadata or {}).get("decision") == "ask":
            # 需要人工权限时，QueryLoop 切到 WAITING_PERMISSION 并把请求推给前端。
            loop.transition(QueryPhase.WAITING_PERMISSION, f"permission:{tool_name}")
            loop.update_tool_call(tool_use_id, "permission_needed", result_payload, "permission_required")
            await emit_query_event(
                loop,
                "permission_request",
                {
                    "toolUseId": tool_use_id,
                    "toolName": tool_name,
                    "input": arguments,
                    "riskLevel": "high",
                    "reason": result.content,
                },
                live_send,
            )
            await record_termination_decision(loop, live_send=live_send, metadata={"toolUseId": tool_use_id, "toolName": tool_name})
            decision = await await_permission_decision(
                tool_use_id,
                loop.sessionId,
                int(arguments.get("permissionTimeoutMs") or arguments.get("permission_timeout_ms") or 120_000),
            )
            if str(decision.get("decision") or "").lower() in {"allow", "allowed", "approve", "approved"} or decision.get("allowed") is True:
                result = await asyncio.to_thread(execute_approved_tool, tool_name, arguments)
                result_payload = result.to_dict()
                loop.transition(QueryPhase.TOOL_RUNNING, f"permission_approved:{tool_name}")
                loop.update_tool_call(tool_use_id, "error" if result.isError else "completed", result_payload, "permission_approved")
                update_session_file_state(loop.sessionId, tool_name, arguments, result_payload)
                save_tool_file_snapshot(loop.sessionId, tool_use_id, tool_name, result_payload)
                await emit_query_event(
                    loop,
                    "tool_result",
                    {
                        "toolUseId": tool_use_id,
                        "content": result.content,
                        "isError": result.isError,
                        "result": result_payload,
                        "durationMs": int((time.time() - started) * 1000),
                        "permissionDecision": "allow",
                        "summary": tool_call_summary(loop, tool_use_id),
                    },
                    live_send,
                )
            else:
                reason = str(decision.get("reason") or result.content or f"Permission denied for tool: {tool_name}")
                result = ToolResult(reason, isError=True, metadata={"decision": "deny", "permissionDecision": decision})
                result_payload = result.to_dict()
                loop.update_tool_call(tool_use_id, "error", result_payload, "permission_denied")
                await emit_query_event(
                    loop,
                    "tool_permission_denied",
                    {
                        "toolUseId": tool_use_id,
                        "toolName": tool_name,
                        "reason": reason,
                        "decision": decision,
                    },
                    live_send,
                )
        else:
            # 普通工具完成后更新 loop、维护文件状态，并发出 tool_result。
            loop.update_tool_call(tool_use_id, "error" if result.isError else "completed", result_payload, "completed")
            update_session_file_state(loop.sessionId, tool_name, arguments, result_payload)
            save_tool_file_snapshot(loop.sessionId, tool_use_id, tool_name, result_payload)
            await emit_query_event(
                loop,
                "tool_result",
                {
                    "toolUseId": tool_use_id,
                    "content": result.content,
                    "isError": result.isError,
                    "result": result_payload,
                    "durationMs": int((time.time() - started) * 1000),
                    "summary": tool_call_summary(loop, tool_use_id),
                },
                live_send,
            )
        results.append({"toolUseId": tool_use_id, "toolName": tool_name, **result_payload})
        index += 1
    return results


async def execute_query_tools(loop: QueryLoopState, tool_calls: list[dict[str, Any]], live_send: Any | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def tool_retry_max_attempts(arguments: dict[str, Any]) -> int:
        flags = STATE.setdefault("config", {}).setdefault("featureFlags", {})
        if flags.get("TOOL_RETRY", True) is False:
            return 0
        raw = arguments.get("retryMaxAttempts", arguments.get("toolRetryMaxAttempts", flags.get("TOOL_RETRY_MAX_ATTEMPTS", 1)))
        try:
            return max(0, min(int(raw), 3))
        except (TypeError, ValueError):
            return 1

    def is_retryable_tool_result(result: ToolResult) -> bool:
        metadata = result.metadata or {}
        if metadata.get("decision") in {"ask", "deny"}:
            return False
        if "retryable" in metadata:
            return bool(metadata.get("retryable"))
        content = result.content.lower()
        return result.isError and any(token in content for token in ("timeout", "temporar", "rate limit", "connection reset"))

    def is_concurrency_safe(tool_name: str) -> bool:
        tool = TOOL_REGISTRY.get(tool_name)
        return tool_name == "Agent" or bool(tool and (tool.concurrency_safe or tool.read_only))

    async def prepare_job(call: dict[str, Any]) -> dict[str, Any] | None:
        if QUERY_ABORTS.is_aborted(loop.sessionId):
            loop.abort("USER_INTERRUPT")
            await emit_query_event(loop, "interrupt_ack", {"reason": "USER_INTERRUPT"}, live_send)
            return None

        tool_use_id = str(call["id"])
        tool_name = str(call["name"])
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        arguments = rewrite_tool_arguments_for_session(loop.sessionId, tool_name, arguments)
        loop.transition(QueryPhase.TOOL_RUNNING, f"tool:{tool_name}")
        loop.record_tool_call(tool_use_id, tool_name, arguments)
        await emit_query_event(loop, "tool_use_start", {"toolUseId": tool_use_id, "toolName": tool_name, "input": arguments}, live_send)
        await emit_query_event(loop, "tool_use_input", {"toolUseId": tool_use_id, "toolName": tool_name, "input": arguments}, live_send)
        await emit_query_event(loop, "tool_use_progress", {"toolUseId": tool_use_id, "progress": "started"}, live_send)

        job = {
            "toolUseId": tool_use_id,
            "toolName": tool_name,
            "arguments": arguments,
            "started": time.time(),
            "safe": is_concurrency_safe(tool_name),
        }
        session_for_tool = STATE.setdefault("sessions", {}).get(loop.sessionId) or {}
        allowed, deny_reason = agent_session_tool_allowed(session_for_tool, tool_name)
        if not allowed:
            job["precomputed"] = {"content": deny_reason or "tool denied", "isError": True, "metadata": {"decision": "deny"}}
            job["completionReason"] = "denied"
        return job

    async def execute_job(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
        if "precomputed" in job:
            return job, job["precomputed"], False

        tool_name = str(job["toolName"])
        arguments = job["arguments"]
        tool_use_id = str(job["toolUseId"])
        started = float(job["started"])

        if tool_name == "Agent":
            return job, await execute_agent_tool(loop, arguments, live_send), False

        result = await asyncio.to_thread(TOOL_REGISTRY.call, tool_name, arguments)
        for attempt in range(1, tool_retry_max_attempts(arguments) + 1):
            if not result.isError or not is_retryable_tool_result(result):
                break
            retry_event = RecoveryEvent(
                RecoveryEventType.TOOL_RETRY,
                f"Retrying tool {tool_name} after retryable failure.",
                metadata={"toolUseId": tool_use_id, "toolName": tool_name, "attempt": attempt, "content": result.content[:1000]},
            )
            loop.add_recovery(retry_event)
            loop.update_tool_call(tool_use_id, "retrying", result.to_dict(), f"retry:{attempt}")
            await emit_query_event(
                loop,
                "tool_retry",
                {
                    "toolUseId": tool_use_id,
                    "toolName": tool_name,
                    "attempt": attempt,
                    "reason": result.content,
                },
                live_send,
            )
            result = await asyncio.to_thread(TOOL_REGISTRY.call, tool_name, arguments)
        result_payload = result.to_dict()
        if (result.metadata or {}).get("decision") == "ask":
            loop.transition(QueryPhase.WAITING_PERMISSION, f"permission:{tool_name}")
            loop.update_tool_call(tool_use_id, "permission_needed", result_payload, "permission_required")
            await emit_query_event(
                loop,
                "permission_request",
                {
                    "toolUseId": tool_use_id,
                    "toolName": tool_name,
                    "input": arguments,
                    "riskLevel": "high",
                    "reason": result.content,
                },
                live_send,
            )
            await record_termination_decision(loop, live_send=live_send, metadata={"toolUseId": tool_use_id, "toolName": tool_name})
            decision = await await_permission_decision(
                tool_use_id,
                loop.sessionId,
                int(arguments.get("permissionTimeoutMs") or arguments.get("permission_timeout_ms") or 120_000),
            )
            if str(decision.get("decision") or "").lower() in {"allow", "allowed", "approve", "approved"} or decision.get("allowed") is True:
                result = await asyncio.to_thread(execute_approved_tool, tool_name, arguments)
                result_payload = result.to_dict()
                loop.transition(QueryPhase.TOOL_RUNNING, f"permission_approved:{tool_name}")
                loop.update_tool_call(tool_use_id, "error" if result.isError else "completed", result_payload, "permission_approved")
                update_session_file_state(loop.sessionId, tool_name, arguments, result_payload)
                save_tool_file_snapshot(loop.sessionId, tool_use_id, tool_name, result_payload)
            else:
                reason = str(decision.get("reason") or result.content or f"Permission denied for tool: {tool_name}")
                result_payload = ToolResult(reason, isError=True, metadata={"decision": "deny", "permissionDecision": decision}).to_dict()
                loop.update_tool_call(tool_use_id, "error", result_payload, "permission_denied")
                await emit_query_event(
                    loop,
                    "tool_permission_denied",
                    {
                        "toolUseId": tool_use_id,
                        "toolName": tool_name,
                        "reason": reason,
                        "decision": decision,
                    },
                    live_send,
                )
                return job, result_payload, True
        else:
            loop.update_tool_call(tool_use_id, "error" if result.isError else "completed", result_payload, "completed")
            update_session_file_state(loop.sessionId, tool_name, arguments, result_payload)
            save_tool_file_snapshot(loop.sessionId, tool_use_id, tool_name, result_payload)
        return job, result_payload, False

    async def flush_batch(batch: list[dict[str, Any]]) -> None:
        if not batch:
            return
        completed = await asyncio.gather(*(execute_job(job) for job in batch))
        for job, result_payload, emitted_terminal_event in completed:
            if "precomputed" in job:
                loop.update_tool_call(job["toolUseId"], "error" if result_payload.get("isError") else "completed", result_payload, str(job.get("completionReason") or "completed"))
            if not emitted_terminal_event:
                await emit_query_event(
                    loop,
                    "tool_result",
                    {
                        "toolUseId": job["toolUseId"],
                        "content": result_payload.get("content"),
                        "isError": bool(result_payload.get("isError")),
                        "result": result_payload,
                        "durationMs": int((time.time() - float(job["started"])) * 1000),
                        "summary": tool_call_summary(loop, str(job["toolUseId"])),
                    },
                    live_send,
                )
            results.append({"toolUseId": job["toolUseId"], "toolName": job["toolName"], **result_payload})

    current_batch: list[dict[str, Any]] = []
    for call in tool_calls:
        job = await prepare_job(call)
        if job is None:
            break
        if job["safe"]:
            current_batch.append(job)
            continue
        await flush_batch(current_batch)
        current_batch = []
        await flush_batch([job])
    await flush_batch(current_batch)
    return results


async def run_query_payload(payload: dict[str, Any], require_existing_session: bool = False, live_send: Any | None = None) -> dict[str, Any]:
    # 主 QueryEngine 入口：把一次用户请求推进到工具执行、模型回复、事件推送和状态持久化。
    text = str(payload.get("prompt") or payload.get("query") or payload.get("text") or payload.get("message") or "")
    session_id = payload.get("sessionId") or payload.get("session_id")
    if require_existing_session and session_id and session_id not in STATE["sessions"]:
        raise HTTPException(status_code=404, detail="Session not found")
    session = get_or_create_session(str(session_id) if session_id else None)
    session.pop("pendingModelUsage", None)
    if payload.get("model"):
        session["model"] = payload["model"]
    if payload.get("workingDirectory") or payload.get("working_directory"):
        session["workingDirectory"] = payload.get("workingDirectory") or payload.get("working_directory")

    # Hook 在用户 prompt 入队前执行，可用于审计、拦截或改写输入。
    hook_result = execute_hooks("USER_PROMPT_SUBMIT", {"input": text, "sessionId": session["id"], "source": "rest"})
    blocked_answer = None
    if not hook_result.get("proceed", True):
        blocked_answer = str(hook_result.get("message") or "Blocked by hook")
    else:
        text = str((hook_result.get("context") or {}).get("input") or text)

    model_id = str(session.get("model") or STATE["config"].get("defaultModel"))
    capability = MODEL_CAPABILITIES.get_capability(model_id)
    context_budget = await estimate_query_context_budget(session, text, model_id, capability.tokenCharRatio)
    loop = QueryLoopState.start(
        session_id=session["id"],
        user_input=text,
        model=model_id,
        context_window=capability.contextWindow,
        threshold=MODEL_CAPABILITIES.compact_threshold(model_id),
        ratio=capability.tokenCharRatio,
        used_tokens=int(context_budget["usedTokens"]),
    )
    loop.tokenBudgetBreakdown = dict(context_budget["breakdown"])
    memory_context = str(context_budget.get("memoryContext") or "")
    # QueryLoop 先报告 token 预算，前端可以提前提示上下文压力。
    await emit_query_event(
        loop,
        "token_budget_nudge",
        {
            "pct": int(loop.tokenBudget.usage_ratio() * 100),
            "currentTokens": loop.tokenBudget.usedTokens,
            "budgetTokens": loop.tokenBudget.max_input_tokens,
            "breakdown": dict(loop.tokenBudgetBreakdown),
        },
        live_send,
    )
    if loop.tokenBudget.usage_ratio() >= 0.75:
        await emit_query_event(
            loop,
            "token_warning",
            {
                "currentTokens": loop.tokenBudget.usedTokens,
                "maxTokens": loop.tokenBudget.max_input_tokens,
                "usagePercent": int(loop.tokenBudget.usage_ratio() * 100),
                "warningLevel": "critical" if loop.tokenBudget.exceeded else "warning",
                "breakdown": dict(loop.tokenBudgetBreakdown),
            },
            live_send,
        )
    cascade_needed = (
        bool(payload.get("collapseContext"))
        or loop.tokenBudget.exceeded
        or loop.tokenBudget.usage_ratio() >= 0.75
        or len(text) > CONTEXT_CASCADE.max_prompt_chars
    )
    recovery_cause = str(payload.get("recoveryCause") or payload.get("recovery_cause") or "")
    if not recovery_cause and int(payload.get("httpStatus") or payload.get("statusCode") or 0) == 413:
        recovery_cause = "http_413"
    if not recovery_cause and str(payload.get("errorType") or payload.get("error") or "").lower() in {"prompt_too_long", "413"}:
        recovery_cause = "http_413"
    if cascade_needed:
        # 五层 ContextCascade：snip -> micro compact -> auto compact -> drain -> reactive compact。
        loop.transition(QueryPhase.COMPACTING, "context_cascade")
        await emit_query_event(loop, "compact_start", {"sessionId": session["id"], "reason": "context_cascade", "recoveryCause": recovery_cause or None}, live_send)
        cascade = CONTEXT_CASCADE.apply(
            text,
            session.get("messages", []),
            loop.tokenBudget,
            model_id=model_id,
            token_char_ratio=capability.tokenCharRatio,
            protected_tail=int(payload.get("protectedTail") or 6),
            force=bool(payload.get("collapseContext")),
            recovery_cause=recovery_cause or None,
        )
        text = cascade["prompt"]
        session["messages"] = cascade["messages"]
        loop.tokenBudget.usedTokens = int(cascade["usedTokens"])
        loop.contextCascade = {
            "changed": bool(cascade["changed"]),
            "layers": cascade["layers"],
            "usedTokens": int(cascade["usedTokens"]),
            "estimatedCharsFreed": int(cascade["estimatedCharsFreed"]),
        }
        for layer in cascade["layers"]:
            await emit_query_event(
                loop,
                "compact_event",
                {
                    "phase": layer["name"],
                    "changed": layer["changed"],
                    "estimatedCharsFreed": layer["estimatedCharsFreed"],
                    "metadata": layer.get("metadata") or {},
                },
                live_send,
            )
        for compact_event in cascade["events"]:
            loop.add_recovery(compact_event)
            STATE.setdefault("queryEvents", []).append(compact_event.to_dict())
            del STATE["queryEvents"][:-500]
        await emit_query_event(
            loop,
            "compact_complete",
            {
                "sessionId": session["id"],
                "layers": [layer["name"] for layer in cascade["layers"]],
                "removedCount": sum(1 for layer in cascade["layers"] if layer["changed"]),
                "tokensSaved": cascade["estimatedCharsFreed"],
                "summary": "Context cascade complete.",
            },
            live_send,
        )
        await emit_query_event(
            loop,
            "token_budget_nudge",
            {
                "phase": "post_compact",
                "pct": int(loop.tokenBudget.usage_ratio() * 100),
                "currentTokens": loop.tokenBudget.usedTokens,
                "budgetTokens": loop.tokenBudget.max_input_tokens,
                "breakdown": dict(loop.tokenBudgetBreakdown),
            },
            live_send,
        )
    loop.release_withheld()

    now_ms = int(time.time() * 1000)
    user_message = {"type": "user", "uuid": new_id("user"), "timestamp": now_ms, "content": [{"type": "text", "text": text}]}
    session["messages"].append(user_message)
    if not session.get("title"):
        session["title"] = text[:60] if text else "New session"
    session["status"] = "running"
    session["updatedAt"] = utc_now()
    # 工具调用先于最终模型回答执行，工具结果稍后会写回 session 供模型使用。
    tool_results = await execute_query_tools(loop, normalize_tool_calls(payload.get("toolCalls") or payload.get("tools")), live_send)
    if tool_results:
        # 把 tool_use 和 tool_result 都写进消息历史，模型最终回答才能“看见”工具输出。
        assistant_tool_message_history = {
            "type": "assistant",
            "uuid": new_id("assistant"),
            "timestamp": int(time.time() * 1000),
            "content": [
                {"type": "tool_use", "toolUseId": result["toolUseId"], "toolName": result["toolName"], "input": {}}
                for result in tool_results
            ],
            "stopReason": "tool_use",
            "usage": usage(),
        }
        user_tool_message_history = {
            "type": "user",
            "uuid": new_id("user"),
            "timestamp": int(time.time() * 1000),
            "content": [
                {"type": "tool_result", "toolUseId": result["toolUseId"], "content": result["content"], "isError": result["isError"]}
                for result in tool_results
            ],
            "toolUseResult": "\n".join(str(result["content"]) for result in tool_results),
        }
        session["messages"].append(assistant_tool_message_history)
        session["messages"].append(user_tool_message_history)
        tool_tokens = await estimate_query_message_group_tokens([assistant_tool_message_history, user_tool_message_history], model_id, capability.tokenCharRatio)
        loop.tokenBudget.usedTokens += tool_tokens
        loop.tokenBudgetBreakdown["tool"] = int(loop.tokenBudgetBreakdown.get("tool") or 0) + tool_tokens
        await emit_query_event(
            loop,
            "token_budget_nudge",
            {
                "phase": "post_tools",
                "pct": int(loop.tokenBudget.usage_ratio() * 100),
                "currentTokens": loop.tokenBudget.usedTokens,
                "budgetTokens": loop.tokenBudget.max_input_tokens,
                "breakdown": dict(loop.tokenBudgetBreakdown),
            },
            live_send,
        )
    model_input_text = text
    flags = STATE.setdefault("config", {}).setdefault("featureFlags", {})
    if flags.get("BACKGROUND_AGENT_WAIT", True) is not False:
        # 后台 Agent 默认会被等待并汇总，避免父回答在子 Agent 结果回来前就结束。
        launched_background_ids = [
            str((result.get("metadata") or {}).get("agentId"))
            for result in tool_results
            if result.get("toolName") == "Agent"
            and (result.get("metadata") or {}).get("status") == "async_launched"
            and (result.get("metadata") or {}).get("agentId")
        ]
        running_agents = active_background_agent_ids(session["id"])
        if running_agents or launched_background_ids:
            # 只等待当前目标 Agent，避免旧后台结果反复污染新一轮回答。
            target_agent_ids = list(dict.fromkeys([*running_agents, *launched_background_ids]))
            await emit_query_event(loop, "waiting_for_background_agents", {"activeAgentIds": running_agents, "launchedAgentIds": launched_background_ids, "agentIds": target_agent_ids}, live_send)
            wait_result = await await_background_agents(session["id"], timeout_ms=int(payload.get("backgroundAgentWaitTimeoutMs") or 900_000), agent_ids=target_agent_ids)
            await emit_query_event(loop, "background_agents_wait_complete", wait_result, live_send)
            if wait_result.get("completed"):
                summary = format_background_agent_results(wait_result.get("agents") or [])
                if summary:
                    # 后台 Agent 结果作为 systemInjected 消息注入模型输入。
                    system_text = "[System] Background agents completed:\n" + summary
                    background_message = {
                        "type": "user",
                        "uuid": new_id("user"),
                        "timestamp": int(time.time() * 1000),
                        "content": [{"type": "text", "text": system_text}],
                        "systemInjected": True,
                    }
                    session["messages"].append(background_message)
                    background_tokens = await estimate_query_message_group_tokens([background_message], model_id, capability.tokenCharRatio)
                    loop.tokenBudget.usedTokens += background_tokens
                    loop.tokenBudgetBreakdown["tool"] = int(loop.tokenBudgetBreakdown.get("tool") or 0) + background_tokens
                    model_input_text = f"{text}\n\n{system_text}"
    if loop.phase != QueryPhase.WAITING_PERMISSION and loop.phase != QueryPhase.ABORTED:
        # 工具和后台 Agent 处理完后，才进入最终 LLM 调用阶段。
        loop.transition(QueryPhase.MODEL_CALL, "llm_request")
    answer = blocked_answer or await generate_llm_reply(session, model_input_text, memory_context)
    if not blocked_answer:
        answer = await run_query_self_correction(session, loop, model_input_text, answer, memory_context, live_send)
    model_usage = consume_session_model_usage(session) if not blocked_answer else usage()
    if QUERY_ABORTS.is_aborted(session["id"]):
        # 模型生成期间也可能被用户中断，必须落盘 aborted 状态并返回空回答。
        loop.abort("USER_INTERRUPT")
        session["status"] = "idle"
        session["updatedAt"] = utc_now()
        await emit_query_event(loop, "interrupt_ack", {"reason": "USER_INTERRUPT"}, live_send)
        await record_termination_decision(loop, requested_stop_reason="aborted", error="USER_INTERRUPT", live_send=live_send)
        persist_query_loop(loop)
        session["lastQueryLoopId"] = loop.id
        save_state()
        return {"sessionId": session["id"], "answer": "", "usage": model_usage, "costUsd": 0.0, "toolCalls": tool_results, "stopReason": "aborted", "error": "USER_INTERRUPT", "message": None, "queryLoop": loop.to_dict(), "events": [event.to_dict() for event in loop.events]}
    loop.transition(QueryPhase.STREAMING, "assistant_stream")
    message_id = new_id("assistant")
    # 将最终回答拆成 stream_delta 事件，供 SSE/WebSocket 实时显示。
    for chunk in chunk_text(answer):
        await emit_query_event(loop, "stream_delta", {"delta": chunk, "messageId": message_id}, live_send)
    assistant_message = {
        "type": "assistant",
        "uuid": message_id,
        "timestamp": int(time.time() * 1000),
        "content": [{"type": "text", "text": answer}],
        "stopReason": "end_turn",
        "usage": model_usage,
    }
    session["messages"].append(assistant_message)
    session["status"] = "idle"
    session["updatedAt"] = utc_now()
    await record_termination_decision(loop, requested_stop_reason="end_turn", live_send=live_send)
    loop.finish(stop_reason="end_turn")
    await emit_query_event(loop, "message_complete", {"messageId": assistant_message["uuid"], "usage": model_usage, "stopReason": "end_turn"}, live_send)
    await emit_query_event(loop, "cost_update", {"sessionCost": 0.0, "totalCost": 0.0, "usage": model_usage}, live_send)
    # 一轮 query 的最后一步：持久化 QueryLoop 和 session 状态。
    persist_query_loop(loop)
    session["lastQueryLoopId"] = loop.id
    save_state()
    return {
        "sessionId": session["id"],
        "answer": answer,
        "usage": model_usage,
        "costUsd": 0.0,
        "toolCalls": tool_results,
        "stopReason": "end_turn",
        "error": None,
        "message": assistant_message,
        "queryLoop": loop.to_dict(),
        "events": [event.to_dict() for event in loop.events],
    }


@app.post("/api/query")
async def query(request: Request) -> dict[str, Any]:
    payload = await request.json()
    return await run_query_payload(payload)


@app.post("/api/query/conversation")
async def query_conversation(request: Request) -> dict[str, Any]:
    payload = await request.json()
    return await run_query_payload(payload, require_existing_session=bool(payload.get("sessionId") or payload.get("session_id")))


@app.post("/api/query/stream")
async def query_stream(request: Request) -> StreamingResponse:
    payload = await request.json()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def live_send(event: dict[str, Any]) -> None:
        await queue.put(event)

    async def events():
        try:
            task = asyncio.create_task(run_query_payload(payload, live_send=live_send))
            while not task.done() or not queue.empty():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.05)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    continue
            result = await task
            yield f"data: {json.dumps({'type': 'query_complete', 'sessionId': result['sessionId'], 'loopId': result['queryLoop']['id'], 'status': result['queryLoop']['status']}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/query/loops")
async def query_loops(sessionId: str | None = None, limit: int = 50) -> dict[str, Any]:
    loops = list(STATE.setdefault("queryLoops", {}).values())
    if sessionId:
        loops = [loop for loop in loops if loop.get("sessionId") == sessionId]
    loops = sorted(loops, key=lambda item: item.get("startedAt", 0), reverse=True)[: max(1, min(limit, 200))]
    return {"loops": loops, "total": len(loops)}


@app.get("/api/query/session/{session_id}/loop")
async def query_session_loop(session_id: str) -> dict[str, Any]:
    session = STATE.setdefault("sessions", {}).get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    loop_id = session.get("lastQueryLoopId")
    loop = STATE.setdefault("queryLoops", {}).get(loop_id) if loop_id else None
    return {"sessionId": session_id, "loop": loop}


@app.get("/api/query/loops/{loop_id}")
async def query_loop_detail(loop_id: str) -> dict[str, Any]:
    loop = STATE.setdefault("queryLoops", {}).get(loop_id)
    if not loop:
        raise HTTPException(status_code=404, detail="Query loop not found")
    return loop


@app.get("/api/query/loops/{loop_id}/events")
async def query_loop_events(loop_id: str, limit: int = 200) -> dict[str, Any]:
    loop = STATE.setdefault("queryLoops", {}).get(loop_id)
    if not loop:
        raise HTTPException(status_code=404, detail="Query loop not found")
    events = loop.get("events", [])[-max(1, min(limit, 1000)) :]
    return {"loopId": loop_id, "events": events, "total": len(events)}


@app.get("/api/query/session/{session_id}/events")
async def query_session_events(session_id: str, limit: int = 200) -> dict[str, Any]:
    events = [
        event
        for event in STATE.setdefault("queryEvents", [])
        if event.get("sessionId") == session_id
    ][-max(1, min(limit, 1000)) :]
    return {"sessionId": session_id, "events": events, "total": len(events)}


@app.post("/api/query/tools/schedule")
async def query_tool_schedule(request: Request) -> dict[str, Any]:
    payload = await request.json()
    calls = payload.get("toolCalls") or payload.get("tools") or []
    if not isinstance(calls, list):
        raise HTTPException(status_code=400, detail="toolCalls must be a list")
    ordered = TOOL_SCHEDULER.ordered_calls(calls)
    names = [str(item.get("name") or (item.get("function") or {}).get("name") or "") for item in calls if isinstance(item, dict)]
    paths = [
        str(item.get("path") or item.get("filePath") or ((item.get("arguments") or {}).get("path") if isinstance(item.get("arguments"), dict) else "") or "")
        if isinstance(item, dict)
        else ""
        for item in calls
    ]
    return {"toolCalls": ordered, "hasConflict": TOOL_SCHEDULER.has_conflict(names, paths)}


@app.post("/api/query/session/{session_id}/collapse")
async def query_collapse_session(session_id: str, request: Request) -> dict[str, Any]:
    session = STATE.setdefault("sessions", {}).get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    payload = await request.json()
    result = CONTEXT_COLLAPSE.collapse_messages(session.get("messages", []), int(payload.get("protectedTail") or 6))
    session["messages"] = result["messages"]
    session["updatedAt"] = utc_now()
    save_state()
    return {"sessionId": session_id, **result}


@app.post("/api/query/side")
async def query_side(request: Request) -> dict[str, Any]:
    payload = await request.json()
    result = SIDE_QUERY_SERVICE.query(
        str(payload.get("systemPrompt") or payload.get("system_prompt") or "Summarize relevant context."),
        str(payload.get("content") or payload.get("query") or ""),
        max_tokens=int(payload.get("maxTokens") or payload.get("max_tokens") or 512),
        timeout_ms=int(payload.get("timeoutMs") or payload.get("timeout_ms") or 3000),
    )
    return result


@app.post("/api/query/micro-compact")
async def query_micro_compact(request: Request) -> dict[str, Any]:
    payload = await request.json()
    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    return MICRO_COMPACT.compact_tool_results(messages)


@app.post("/api/query/tool-result/summary")
async def query_tool_result_summary(request: Request) -> dict[str, Any]:
    payload = await request.json()
    return TOOL_RESULT_SUMMARIZER.summarize(str(payload.get("toolName") or payload.get("tool_name") or "tool"), str(payload.get("content") or ""), int(payload.get("maxChars") or payload.get("max_chars") or 500))


@app.post("/api/query/session/{session_id}/abort")
async def query_abort_session(session_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    record = QUERY_ABORTS.abort(session_id, str(payload.get("reason") or "USER_INTERRUPT"))
    session = STATE.setdefault("sessions", {}).get(session_id)
    if session:
        session["status"] = "idle"
        session["updatedAt"] = utc_now()
        session["abortedAt"] = utc_now()
    WS_SESSION_MANAGER.publish_event(session_id, "interrupt_ack", {"reason": record["reason"]})
    save_state()
    return {"success": True, **record}


@app.get("/api/query/session/{session_id}/abort")
async def query_abort_status(session_id: str) -> dict[str, Any]:
    return {"sessionId": session_id, "aborted": QUERY_ABORTS.is_aborted(session_id), "record": QUERY_ABORTS.aborted.get(session_id)}


def swarm_worker_snapshot(worker: dict[str, Any]) -> dict[str, Any]:
    return {
        "workerId": worker.get("workerId"),
        "status": worker.get("status", "IDLE"),
        "currentTask": worker.get("currentTask", ""),
        "agentType": worker.get("agentType"),
        "model": worker.get("model"),
        "parentSessionId": worker.get("parentSessionId"),
        "agentHierarchy": worker.get("agentHierarchy"),
        "toolCallCount": int(worker.get("toolCallCount") or 0),
        "tokenConsumed": int(worker.get("tokenConsumed") or 0),
        "recentToolCalls": list(worker.get("recentToolCalls") or []),
        "recentToolCallRecords": list(worker.get("recentToolCallRecords") or []),
        "progressPercent": worker.get("progressPercent"),
        "totalSteps": worker.get("totalSteps"),
        "completedSteps": worker.get("completedSteps"),
        "errorMessage": worker.get("errorMessage"),
        "currentStepDescription": worker.get("currentStepDescription"),
        "terminationReason": worker.get("terminationReason"),
        "sessionId": worker.get("sessionId"),
    }


def update_swarm_counts(swarm: dict[str, Any]) -> None:
    workers = swarm.setdefault("workers", {})
    swarm["activeWorkers"] = sum(1 for item in workers.values() if item.get("status") in {"STARTING", "WORKING", "WAITING_PERMISSION"})
    swarm["completedTasks"] = sum(1 for item in workers.values() if item.get("status") in {"IDLE", "TERMINATED"} and item.get("terminationReason") in {None, "completed"})
    swarm["totalWorkers"] = max(int(swarm.get("totalWorkers") or 0), len(workers))
    swarm["totalTasks"] = max(int(swarm.get("totalTasks") or 0), len(workers))
    if workers and swarm["activeWorkers"] == 0 and swarm.get("phase") == "RUNNING":
        swarm["phase"] = "IDLE"
    swarm["updatedAt"] = utc_now()


def push_swarm_state(swarm: dict[str, Any]) -> None:
    update_swarm_counts(swarm)
    update_team_runtime(swarm)
    session_id = str(swarm.get("sessionId") or "default")
    workers = {worker_id: swarm_worker_snapshot(worker) for worker_id, worker in swarm.setdefault("workers", {}).items()}
    WS_SESSION_MANAGER.publish_event(
        session_id,
        "swarm_state_update",
        {
            "swarmId": swarm.get("swarmId"),
            "phase": swarm.get("phase"),
            "activeWorkers": swarm.get("activeWorkers", 0),
            "totalWorkers": swarm.get("totalWorkers", 0),
            "completedTasks": swarm.get("completedTasks", 0),
            "totalTasks": swarm.get("totalTasks", 0),
            "workers": workers,
        },
    )


def push_worker_progress(swarm: dict[str, Any], worker: dict[str, Any]) -> None:
    session_id = str(swarm.get("sessionId") or "default")
    WS_SESSION_MANAGER.publish_event(
        session_id,
        "worker_progress",
        {
            "swarmId": swarm.get("swarmId"),
            **swarm_worker_snapshot(worker),
        },
    )


def append_swarm_event(swarm: dict[str, Any], event_type: str, message: str, **payload: Any) -> dict[str, Any]:
    event = {"id": new_id("swarm-event"), "type": event_type, "timestamp": utc_now(), "message": message, **payload}
    swarm.setdefault("events", []).append(event)
    del swarm["events"][:-500]
    return event


def publish_coordinator_event(swarm: dict[str, Any], event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    session_id = str(swarm.get("sessionId") or "default")
    swarm_id = str(swarm.get("swarmId") or swarm.get("workflowId") or session_id)
    coordinator_workflow_id = str(swarm.get("workflowId") or swarm_id)
    event_payload = dict(payload or {})
    event_payload.setdefault("swarmId", swarm_id)
    event_payload.setdefault("workflowId", swarm_id)
    event_payload.setdefault("coordinatorWorkflowId", coordinator_workflow_id)
    if swarm.get("teamName"):
        event_payload.setdefault("teamName", swarm.get("teamName"))
    event_payload.setdefault("teamPrefix", str(event_payload.get("teamPrefix") or swarm_id))
    envelope = {
        "type": "coordinator_event",
        "ts": int(time.time() * 1000),
        "uuid": new_id("coord"),
        "sessionId": session_id,
        "workflowId": swarm_id,
        "coordinatorWorkflowId": coordinator_workflow_id,
        "swarmId": swarm_id,
        "teamName": swarm.get("teamName"),
        "teamPrefix": str(event_payload.get("teamPrefix") or swarm_id),
        "eventType": event_type,
        "payload": event_payload,
    }
    swarm.setdefault("coordinatorEvents", []).append(envelope)
    del swarm["coordinatorEvents"][:-500]
    WS_SESSION_MANAGER.publish_event(session_id, "coordinator_event", envelope)
    return envelope


def write_mailbox(
    recipient_id: str,
    sender_id: str,
    content: str,
    swarm_id: str | None = None,
    phase: str | None = None,
    task_id: str | None = None,
    channel: str | None = None,
) -> dict[str, Any]:
    normalized_phase = normalize_swarm_workflow_phase(phase) if phase else None
    message_id = new_id("mail")
    created_at = utc_now()
    message = {
        "id": message_id,
        "messageId": message_id,
        "swarmId": swarm_id,
        "teamPrefix": swarm_id,
        "senderId": sender_id,
        "recipientId": recipient_id,
        "content": content,
        "contentLength": len(content or ""),
        "phase": normalized_phase,
        "phaseIndex": swarm_workflow_phase_index(normalized_phase),
        "taskId": task_id,
        "channel": channel or "default",
        "timestamp": created_at,
        "createdAt": created_at,
        "read": False,
    }
    SWARM_MAILBOXES.setdefault(recipient_id, []).append(message)
    del SWARM_MAILBOXES[recipient_id][:-200]
    if swarm_id and swarm_id in STATE.setdefault("swarms", {}):
        journal = STATE["swarms"][swarm_id].setdefault("mailboxJournal", [])
        journal.append(dict(message))
        del journal[:-1000]
    return message


def mailbox_message_matches(message: dict[str, Any], phase: str | None = None, channel: str | None = None) -> bool:
    if phase and message.get("phase") != normalize_swarm_workflow_phase(phase):
        return False
    if channel and str(message.get("channel") or "default") != channel:
        return False
    return True


def read_mailbox(worker_id: str, phase: str | None = None, channel: str | None = None) -> list[dict[str, Any]]:
    return [message for message in SWARM_MAILBOXES.get(worker_id, []) if mailbox_message_matches(message, phase, channel)]


def drain_mailbox(worker_id: str, phase: str | None = None, channel: str | None = None) -> list[dict[str, Any]]:
    messages = SWARM_MAILBOXES.get(worker_id, [])
    if not messages:
        return []
    unread = [message for message in messages if mailbox_message_matches(message, phase, channel)]
    for message in unread:
        message["read"] = True
        message["readAt"] = utc_now()
    drained_ids = {message["id"] for message in unread}
    SWARM_MAILBOXES[worker_id] = [message for message in messages if message.get("id") not in drained_ids]
    return unread


def replay_mailbox_journal(swarm: dict[str, Any], worker_id: str, phase: str | None = None, channel: str | None = None, include_acked: bool = False) -> list[dict[str, Any]]:
    messages = []
    for message in swarm.setdefault("mailboxJournal", []):
        if message.get("recipientId") != worker_id:
            continue
        if not include_acked and message.get("acked"):
            continue
        if mailbox_message_matches(message, phase, channel):
            messages.append(message)
    return messages


def ack_mailbox_messages(swarm: dict[str, Any], worker_id: str, message_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(message_ids)
    acked: list[dict[str, Any]] = []
    for message in swarm.setdefault("mailboxJournal", []):
        if message.get("recipientId") == worker_id and message.get("id") in wanted and not message.get("acked"):
            message["acked"] = True
            message["ackedAt"] = utc_now()
            acked.append(message)
    for message in SWARM_MAILBOXES.get(worker_id, []):
        if message.get("id") in wanted:
            message["acked"] = True
            message["ackedAt"] = utc_now()
    return acked


def recover_mailbox_messages(swarm: dict[str, Any], worker_id: str, phase: str | None = None, channel: str | None = None) -> list[dict[str, Any]]:
    live_ids = {message.get("id") for message in SWARM_MAILBOXES.get(worker_id, [])}
    recovered = []
    for message in replay_mailbox_journal(swarm, worker_id, phase=phase, channel=channel, include_acked=False):
        if message.get("id") in live_ids:
            continue
        restored = dict(message)
        restored["recoveredAt"] = utc_now()
        SWARM_MAILBOXES.setdefault(worker_id, []).append(restored)
        recovered.append(restored)
    del SWARM_MAILBOXES.setdefault(worker_id, [])[:-200]
    return recovered


def params_hash(params: dict[str, Any] | None) -> str:
    if not params:
        return "empty"
    try:
        encoded = json.dumps({key: value for key, value in sorted(params.items()) if value is not None}, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        encoded = str(params)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()


def record_worker_tool_call(worker: dict[str, Any], tool_name: str, params: dict[str, Any] | None, status: str, error: str | None = None, duration_ms: int | None = None) -> dict[str, Any]:
    record = {
        "toolName": tool_name,
        "paramsHash": params_hash(params),
        "status": status,
        "timestamp": int(time.time() * 1000),
        "errorDetail": error,
        "durationMs": duration_ms,
    }
    records = worker.setdefault("recentToolCallRecords", [])
    records.append(record)
    del records[:-10]
    worker["toolCallCount"] = int(worker.get("toolCallCount") or 0) + 1
    worker["recentToolCalls"] = [item["toolName"] for item in records[-5:]]
    fingerprints: dict[str, int] = {}
    repetition = False
    for item in records:
        key = f"{item['toolName']}:{item['paramsHash']}"
        fingerprints[key] = fingerprints.get(key, 0) + 1
        if fingerprints[key] >= int(worker.get("repetitionThreshold") or 3):
            repetition = True
    worker["repetitionDetected"] = repetition
    return record


def check_worker_timeout(swarm: dict[str, Any], worker: dict[str, Any], started: float) -> None:
    timeout_ms = int(swarm.get("workerTimeoutMs") or 30 * 60 * 1000)
    effective_ms = int((time.time() - started) * 1000) - int(worker.get("permissionWaitMs") or 0)
    worker["effectiveRuntimeMs"] = max(0, effective_ms)
    if timeout_ms > 0 and effective_ms > timeout_ms:
        raise TimeoutError(f"Worker timed out after {effective_ms}ms excluding permission wait")


def format_task_notification(worker_id: str, result: dict[str, Any]) -> str:
    status = str(result.get("status") or "completed")
    full = str(result.get("result") or result.get("error") or "")
    summary = full[:200] if full else "No output"
    duration_ms = int(result.get("durationMs") or 0)
    return (
        "<task-notification>\n"
        f"<task-id>{escape(worker_id)}</task-id>\n"
        f"<status>{escape(status)}</status>\n"
        f"<summary>{escape(summary)}</summary>\n"
        f"<result>{escape(full)}</result>\n"
        "<usage>\n"
        f"  <duration_ms>{duration_ms}</duration_ms>\n"
        "</usage>\n"
        "</task-notification>\n"
    )


def worker_tool_allowed(swarm: dict[str, Any], tool_name: str) -> tuple[bool, str | None]:
    allow_list = {str(item) for item in swarm.get("workerToolAllowList") or []}
    deny_list = {str(item) for item in swarm.get("workerToolDenyList") or []}
    forbidden = {"Agent", "TeamCreate", "TeamDelete"}
    if tool_name in forbidden:
        return False, f"Tool {tool_name} is disabled for swarm workers"
    if allow_list and tool_name not in allow_list:
        return False, f"Tool {tool_name} is not in worker allow list"
    if tool_name in deny_list:
        return False, f"Tool {tool_name} is denied for swarm workers"
    return True, None


def worker_session_id(swarm_id: str, worker_id: str) -> str:
    return f"{swarm_id}-{worker_id}-session"


def register_team_runtime(swarm: dict[str, Any]) -> dict[str, Any]:
    team_id = str(swarm.get("teamId") or swarm.get("swarmId"))
    team = {
        "teamId": team_id,
        "name": swarm.get("teamName"),
        "teamName": swarm.get("teamName"),
        "swarmId": swarm.get("swarmId"),
        "sessionId": swarm.get("sessionId"),
        "status": "RUNNING" if swarm.get("phase") in {"INITIALIZING", "RUNNING"} else str(swarm.get("phase") or "IDLE"),
        "maxWorkers": swarm.get("maxConcurrentWorkers") or swarm.get("totalWorkers"),
        "workerCount": swarm.get("maxConcurrentWorkers") or swarm.get("totalWorkers"),
        "activeWorkers": swarm.get("activeWorkers", 0),
        "idleWorkers": sum(1 for item in (swarm.get("workers") or {}).values() if item.get("status") == "IDLE"),
        "completedTasks": swarm.get("completedTasks", 0),
        "totalTasks": swarm.get("totalTasks", 0),
        "totalTokenConsumed": sum(int(item.get("tokenConsumed") or 0) for item in (swarm.get("workers") or {}).values()),
        "workerIds": sorted(list((swarm.get("workers") or {}).keys())),
        "queuedTasks": len(swarm.get("queuedTasks") or []),
        "createdAt": swarm.get("createdAt") or utc_now(),
        "updatedAt": utc_now(),
        "executionBackend": swarm.get("executionBackend", "python-asyncio"),
    }
    STATE.setdefault("teams", {})[team_id] = team
    return team


def update_team_runtime(swarm: dict[str, Any], status: str | None = None) -> dict[str, Any] | None:
    team_id = str(swarm.get("teamId") or swarm.get("swarmId") or "")
    if not team_id:
        return None
    team = STATE.setdefault("teams", {}).setdefault(team_id, {"teamId": team_id, "createdAt": swarm.get("createdAt") or utc_now()})
    team.update(
        {
            "teamName": swarm.get("teamName"),
            "name": swarm.get("teamName"),
            "swarmId": swarm.get("swarmId"),
            "sessionId": swarm.get("sessionId"),
            "status": status or ("RUNNING" if swarm.get("phase") == "RUNNING" else str(swarm.get("phase") or team.get("status") or "IDLE")),
            "maxWorkers": swarm.get("maxConcurrentWorkers") or swarm.get("totalWorkers"),
            "workerCount": swarm.get("maxConcurrentWorkers") or swarm.get("totalWorkers"),
            "activeWorkers": swarm.get("activeWorkers", 0),
            "idleWorkers": sum(1 for item in (swarm.get("workers") or {}).values() if item.get("status") == "IDLE"),
            "completedTasks": swarm.get("completedTasks", 0),
            "totalTasks": swarm.get("totalTasks", 0),
            "totalTokenConsumed": sum(int(item.get("tokenConsumed") or 0) for item in (swarm.get("workers") or {}).values()),
            "workerIds": sorted(list((swarm.get("workers") or {}).keys())),
            "queuedTasks": len(swarm.get("queuedTasks") or []),
            "updatedAt": utc_now(),
            "executionBackend": swarm.get("executionBackend", "python-asyncio"),
        }
    )
    return team


def cleanup_swarm_runtime(swarm_id: str, swarm: dict[str, Any]) -> None:
    for request in swarm.setdefault("pendingPermissions", []):
        request["status"] = "cancelled"
        request["updatedAt"] = utc_now()
        request_id = request.get("requestId")
        if request_id:
            future = SWARM_PERMISSION_WAITERS.pop(str(request_id), None)
            if future and not future.done():
                future.set_result({"requestId": request_id, "decision": "deny", "approved": False, "reason": "swarm shutdown", "updatedAt": utc_now()})
    for key in [str(worker_id) for worker_id in swarm.setdefault("workers", {})]:
        SWARM_MAILBOXES.pop(key, None)
    SWARM_MAILBOXES.pop(f"{swarm_id}-leader", None)
    SWARM_TASKS.pop(swarm_id, None)


def pending_swarm_permission_records(swarm_id: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for current_swarm_id, swarm in STATE.setdefault("swarms", {}).items():
        if swarm_id and str(current_swarm_id) != str(swarm_id):
            continue
        for item in swarm.get("pendingPermissions", []):
            if item.get("status") == "pending":
                records.append(decorate_swarm_permission_deadline(item))
    return records


def clear_pending_swarm_permissions(swarm_id: str | None = None, reason: str = "cleared") -> int:
    cleared = 0
    for current_swarm_id, swarm in STATE.setdefault("swarms", {}).items():
        if swarm_id and str(current_swarm_id) != str(swarm_id):
            continue
        for item in swarm.get("pendingPermissions", []):
            if item.get("status") != "pending":
                continue
            item["status"] = "cancelled"
            item["reason"] = reason
            item["updatedAt"] = utc_now()
            request_id = str(item.get("requestId") or "")
            future = SWARM_PERMISSION_WAITERS.pop(request_id, None)
            if future and not future.done():
                future.set_result({"requestId": request_id, "decision": "deny", "approved": False, "reason": reason, "updatedAt": utc_now()})
            cleared += 1
        swarm["pendingPermissions"] = [item for item in swarm.get("pendingPermissions", []) if item.get("status") == "pending"]
    return cleared


def parse_timestamp_ms(value: Any, fallback_ms: int | None = None) -> int:
    if value is None:
        return int(fallback_ms if fallback_ms is not None else time.time() * 1000)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return int(numeric if numeric > 10_000_000_000 else numeric * 1000)
    text = str(value).strip()
    if not text:
        return int(fallback_ms if fallback_ms is not None else time.time() * 1000)
    try:
        return int(float(text))
    except ValueError:
        pass
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return int(fallback_ms if fallback_ms is not None else time.time() * 1000)


def timestamp_ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def decorate_swarm_permission_deadline(record: dict[str, Any], now_ms: int | None = None) -> dict[str, Any]:
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    timeout_ms = max(1, int(record.get("timeoutMs") or record.get("timeout_ms") or 60_000))
    if not record.get("createdAt"):
        record["createdAt"] = timestamp_ms_to_iso(now)
    created_ms = parse_timestamp_ms(record.get("createdAt"), now)
    expires_ms = parse_timestamp_ms(record.get("expiresAt"), created_ms + timeout_ms) if record.get("expiresAt") else created_ms + timeout_ms
    elapsed_ms = max(0, now - created_ms)
    remaining_ms = max(0, expires_ms - now)
    record["timeoutMs"] = timeout_ms
    record["expiresAt"] = timestamp_ms_to_iso(expires_ms)
    record["elapsedMs"] = elapsed_ms
    record["remainingMs"] = remaining_ms
    record["deadlineStatus"] = "expired" if remaining_ms <= 0 else "pending"
    return record


def expire_swarm_permission_timeouts(swarm_id: str | None = None, now_ms: int | None = None) -> dict[str, Any]:
    expired: list[dict[str, Any]] = []
    for current_swarm_id, swarm in STATE.setdefault("swarms", {}).items():
        if swarm_id and str(current_swarm_id) != str(swarm_id):
            continue
        for item in list(swarm.get("pendingPermissions", [])):
            if item.get("status") != "pending":
                continue
            decorated = decorate_swarm_permission_deadline(item, now_ms)
            if decorated.get("remainingMs", 1) > 0:
                continue
            request_id = str(item.get("requestId") or "")
            if not request_id:
                continue
            resolution = {
                "requestId": request_id,
                "decision": "deny",
                "approved": False,
                "reason": "permission timeout",
                "expired": True,
                "updatedAt": utc_now(),
            }
            result = resolve_swarm_permission_request(request_id, resolution)
            expired.append({"swarmId": current_swarm_id, **result})
            append_swarm_event(swarm, "permission_timeout", f"Permission request timed out: {request_id}", requestId=request_id, workerId=item.get("workerId"), toolName=item.get("toolName"), severity="warning")
            publish_coordinator_event(
                swarm,
                "permission_timeout",
                {"requestId": request_id, "workerId": item.get("workerId"), "toolName": item.get("toolName"), "timeoutMs": item.get("timeoutMs")},
            )
    if expired:
        save_state()
    return {"success": True, "expired": expired, "expiredCount": len(expired)}


def ensure_swarm_enabled() -> None:
    flags = STATE.setdefault("config", {}).setdefault("featureFlags", {})
    if flags.get("ENABLE_AGENT_SWARMS", True) is False:
        raise HTTPException(status_code=409, detail="Agent Swarms feature is disabled. Enable 'ENABLE_AGENT_SWARMS' flag to use.")


def validate_team_name(team_name: str, explicit: bool = True) -> None:
    if not team_name and explicit:
        raise HTTPException(status_code=400, detail="Invalid teamName: must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", team_name):
        raise HTTPException(status_code=400, detail="Invalid teamName: use 1-64 letters, digits, underscores, or dashes")


def assert_swarm_owner(swarm: dict[str, Any], payload: dict[str, Any], request: Request) -> None:
    owner_session = swarm.get("sessionId")
    if not owner_session:
        return
    supplied_session = payload.get("sessionId") or payload.get("session_id")
    if supplied_session:
        if str(supplied_session) != str(owner_session):
            raise HTTPException(status_code=403, detail="Session is not allowed to control this swarm")
        return
    principal = str(payload.get("principal") or request.headers.get("X-Principal") or request.headers.get("Authorization") or request.headers.get("login") or "")
    if principal and WS_SESSION_MANAGER.get_session_for_principal(principal) == owner_session:
        return
    raise HTTPException(status_code=403, detail="Swarm control requires matching sessionId or bound principal")


def parse_task_spec(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        prompt = str(raw.get("prompt") or raw.get("task") or raw.get("content") or "")
        return {
            "task": prompt,
            "prompt": prompt,
            "agentType": raw.get("agentType") or raw.get("agent_type"),
            "model": raw.get("model"),
            "toolCalls": raw.get("toolCalls") if isinstance(raw.get("toolCalls"), list) else [],
            "turns": raw.get("turns") if isinstance(raw.get("turns"), list) else [],
        }
    prompt = str(raw)
    return {"task": prompt, "prompt": prompt, "agentType": None, "model": None, "toolCalls": [], "turns": []}


def make_worker_state(swarm_id: str, worker_id: str, task_spec: dict[str, Any], worker_tool_calls: dict[str, Any] | None = None, worker_turns: dict[str, Any] | None = None) -> dict[str, Any]:
    task_prompt = str(task_spec.get("task") or task_spec.get("prompt") or "")
    return {
        "workerId": worker_id,
        "sessionId": worker_session_id(swarm_id, worker_id),
        "status": "STARTING",
        "currentTask": task_prompt,
        "agentType": task_spec.get("agentType"),
        "model": task_spec.get("model"),
        "parentSessionId": None,
        "agentHierarchy": None,
        "toolCallCount": 0,
        "tokenConsumed": 0,
        "recentToolCalls": [],
        "progressPercent": 0,
        "totalSteps": 5,
        "completedSteps": 0,
        "errorMessage": None,
        "currentStepDescription": "queued",
        "terminationReason": None,
        "toolCalls": (worker_tool_calls or {}).get(worker_id, task_spec.get("toolCalls") or []),
        "turns": (worker_turns or {}).get(worker_id, task_spec.get("turns") or []),
        "startedAt": utc_now(),
    }


def build_worker_system_prompt(swarm: dict[str, Any], worker: dict[str, Any], task_prompt: str) -> str:
    scratchpad = str(swarm.get("scratchpadDir") or "N/A")
    team_name = str(swarm.get("teamName") or "")
    agent_type = str(worker.get("agentType") or "general")
    return (
        "You are a worker agent in a Swarm team. Your job is to complete the assigned task efficiently.\n\n"
        "## Constraints\n"
        f"- You are part of team '{team_name}'\n"
        f"- Your agent type is '{agent_type}'\n"
        "- Complete your assigned task and return a clear, concise result\n"
        "- You do NOT have access to: Agent, TeamCreate, TeamDelete tools\n"
        "- If you modify files, list all modified file paths in your final response\n"
        "- Focus on your specific task, do not attempt unrelated work\n"
        "- If you need information from other workers, check the scratchpad directory\n\n"
        "## Scratchpad\n"
        f"- Shared directory: {scratchpad}\n"
        "- Use this for inter-worker file exchange if needed\n\n"
        "## Your Task\n"
        f"{task_prompt}"
    )


def worker_llm_definitions(swarm: dict[str, Any]) -> list[dict[str, Any]]:
    sync_mcp_tools()
    definitions = []
    for definition in TOOL_REGISTRY.llm_definitions():
        name = str(((definition.get("function") or {}).get("name")) or "")
        if not name:
            continue
        allowed, _reason = worker_tool_allowed(swarm, name)
        if allowed:
            definitions.append(definition)
    return definitions


def append_session_message(session: dict[str, Any], message_type: str, content: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    message = {
        "type": message_type,
        "uuid": new_id(message_type),
        "timestamp": int(time.time() * 1000),
        "content": content,
        **extra,
    }
    session.setdefault("messages", []).append(message)
    session["updatedAt"] = utc_now()
    return message


def cap_worker_session_messages(session: dict[str, Any], cap: int = WORKER_MESSAGE_CAP) -> None:
    messages = session.setdefault("messages", [])
    if len(messages) > cap:
        del messages[:-cap]


def tool_messages_from_worker_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for result in results:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(result.get("toolUseId") or new_id("tool")),
                "name": str(result.get("toolName") or ""),
                "content": json.dumps(
                    {
                        "content": result.get("content"),
                        "isError": bool(result.get("isError")),
                        "metadata": result.get("metadata") or {},
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return messages


async def generate_worker_llm_reply(
    swarm: dict[str, Any],
    worker: dict[str, Any],
    session: dict[str, Any],
    loop: QueryLoopState,
    text: str,
) -> tuple[str, list[dict[str, Any]]]:
    settings = llm_settings()
    if swarm.get("workerUseLlm") is False:
        return fallback_answer(text), []
    if not settings["apiKey"] or not settings["baseUrl"]:
        return fallback_answer(text), []
    endpoint = f"{settings['baseUrl']}/chat/completions"
    messages: list[dict[str, Any]] = list(llm_messages(session, text))
    tools = worker_llm_definitions(swarm)
    executed_results: list[dict[str, Any]] = []
    headers = {"Authorization": f"Bearer {settings['apiKey']}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for iteration in range(3):
                loop.transition(QueryPhase.MODEL_CALL, f"worker_llm_request:{iteration + 1}")
                loop.event("worker_model_request", {"workerId": worker.get("workerId"), "iteration": iteration + 1})
                payload: dict[str, Any] = {
                    "model": session.get("model") or swarm.get("workerModel") or settings["model"],
                    "messages": messages,
                    "temperature": 0.2,
                    "stream": False,
                }
                if tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = "auto"
                response = await client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    break
                message = choices[0].get("message") or {}
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    messages.append(assistant_tool_message(message))
                    worker_results = await execute_worker_tool_calls(swarm, worker, loop, normalize_tool_calls(tool_calls))
                    executed_results.extend(worker_results)
                    messages.extend(tool_messages_from_worker_results(worker_results))
                    if worker.get("status") == "TERMINATED":
                        break
                    continue
                content = message.get("content")
                if content:
                    return str(content), executed_results
        return "The worker model returned an empty response.", executed_results
    except Exception as exc:
        return f"LLM request failed in the Python worker: {exc}", executed_results


async def execute_worker_query_turn(
    swarm: dict[str, Any],
    worker: dict[str, Any],
    loop: QueryLoopState,
    turn_prompt: str,
    planned_tool_results: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    swarm_id = str(swarm.get("swarmId") or "swarm")
    worker_id = str(worker.get("workerId") or "worker")
    session_id = str(worker.get("sessionId") or worker_session_id(swarm_id, worker_id))
    worker["sessionId"] = session_id
    session = get_or_create_session(session_id)
    session["title"] = session.get("title") or f"{swarm.get('teamName')} / {worker_id}"
    session["model"] = worker.get("model") or swarm.get("workerModel") or session.get("model") or STATE["config"].get("defaultModel", "qwen3.7-max")
    session["workingDirectory"] = swarm.get("workingDirectory") or session.get("workingDirectory") or "."
    session["systemPrompt"] = build_worker_system_prompt(swarm, worker, turn_prompt)
    session["parentSessionId"] = swarm.get("sessionId")
    session["agentHierarchy"] = worker.get("agentHierarchy")
    session["agentType"] = worker.get("agentType") or "general"
    session["status"] = "running"
    loop.sessionId = session_id
    user_content = turn_prompt
    if planned_tool_results:
        tool_context = "\n".join(f"{item.get('toolName')}: {item.get('content')}" for item in planned_tool_results)
        user_content = f"{turn_prompt}\n\nTool context:\n{tool_context}"
    append_session_message(session, "user", [{"type": "text", "text": user_content}], swarmId=swarm_id, workerId=worker_id)
    cap_worker_session_messages(session)
    loop.transition(QueryPhase.MODEL_CALL, "worker_query_engine")
    loop.event("worker_query_start", {"workerId": worker_id, "sessionId": session_id, "prompt": turn_prompt})
    answer, llm_tool_results = await generate_worker_llm_reply(swarm, worker, session, loop, user_content)
    loop.transition(QueryPhase.STREAMING, "worker_assistant_stream")
    message_id = new_id("assistant")
    for chunk in chunk_text(answer):
        loop.event("stream_delta", {"delta": chunk, "messageId": message_id})
        WS_SESSION_MANAGER.publish_event(session_id, "stream_delta", loop.events[-1].to_dict())
    assistant_message = append_session_message(
        session,
        "assistant",
        [{"type": "text", "text": answer}],
        uuid=message_id,
        stopReason="end_turn",
        usage=usage(),
        swarmId=swarm_id,
        workerId=worker_id,
    )
    cap_worker_session_messages(session)
    session["status"] = "idle"
    session["lastQueryLoopId"] = loop.id
    loop.event("message_complete", {"messageId": assistant_message["uuid"], "usage": usage(), "stopReason": "end_turn"})
    persist_query_loop(loop)
    WS_SESSION_MANAGER.publish_event(str(swarm.get("sessionId") or "default"), "worker_query_complete", {"swarmId": swarm_id, "workerId": worker_id, "sessionId": session_id, "loopId": loop.id})
    return answer, llm_tool_results, assistant_message


async def request_swarm_permission(
    swarm: dict[str, Any],
    worker: dict[str, Any],
    tool_name: str,
    input_payload: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    if not swarm.get("permissionBubbleEnabled", True):
        return {"requestId": None, "decision": "deny", "approved": False, "reason": "permission bubble disabled", "updatedAt": utc_now()}
    request_id = new_id("perm")
    session_id = str(swarm.get("sessionId") or "default")
    worker_id = str(worker.get("workerId"))
    timeout_ms = int(swarm.get("permissionTimeoutMs") or 60_000)
    request_record = {
        "requestId": request_id,
        "swarmId": swarm.get("swarmId"),
        "workerId": worker_id,
        "toolName": tool_name,
        "input": input_payload,
        "riskLevel": "high",
        "reason": reason,
        "status": "pending",
        "createdAt": utc_now(),
        "timeoutMs": timeout_ms,
    }
    request_record = decorate_swarm_permission_deadline(request_record)
    future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
    SWARM_PERMISSION_WAITERS[request_id] = future
    swarm.setdefault("pendingPermissions", []).append(request_record)
    worker["status"] = "WAITING_PERMISSION"
    worker["currentStepDescription"] = f"waiting permission for {tool_name}"
    pending_count = len([item for item in swarm.setdefault("pendingPermissions", []) if item.get("status") == "pending"])
    permission_payload = {
        "requestId": request_id,
        "swarmId": swarm.get("swarmId"),
        "workerId": worker_id,
        "toolName": tool_name,
        "riskLevel": "high",
        "reason": reason,
        "status": "pending",
        "timeoutMs": request_record.get("timeoutMs"),
        "expiresAt": request_record.get("expiresAt"),
        "elapsedMs": request_record.get("elapsedMs"),
        "remainingMs": request_record.get("remainingMs"),
        "deadlineStatus": request_record.get("deadlineStatus"),
        "createdAt": request_record.get("createdAt"),
        "pendingRequestCount": pending_count,
        "leaderSessionId": session_id,
    }
    append_swarm_event(
        swarm,
        "permission_bubble",
        f"Permission requested: {tool_name}",
        requestId=request_id,
        workerId=worker_id,
        toolName=tool_name,
        riskLevel="high",
        timeoutMs=request_record.get("timeoutMs"),
        expiresAt=request_record.get("expiresAt"),
        remainingMs=request_record.get("remainingMs"),
    )
    publish_coordinator_event(swarm, "permission_bubble", permission_payload)
    WS_SESSION_MANAGER.publish_event(
        session_id,
        "permission_bubble",
        permission_payload,
    )
    push_worker_progress(swarm, worker)
    push_swarm_state(swarm)
    wait_started = time.time()
    deadline = time.time() + timeout_ms / 1000
    try:
        while time.time() < deadline:
            if future.done():
                decision = future.result()
                worker["permissionWaitMs"] = int(worker.get("permissionWaitMs") or 0) + int((time.time() - wait_started) * 1000)
                return decision
            stored = STATE.setdefault("permissionResponses", {}).get(request_id)
            if stored:
                worker["permissionWaitMs"] = int(worker.get("permissionWaitMs") or 0) + int((time.time() - wait_started) * 1000)
                return stored
            await asyncio.sleep(0.05)
        worker["permissionWaitMs"] = int(worker.get("permissionWaitMs") or 0) + int((time.time() - wait_started) * 1000)
        return {"requestId": request_id, "decision": "deny", "approved": False, "reason": "permission timeout", "updatedAt": utc_now()}
    finally:
        SWARM_PERMISSION_WAITERS.pop(request_id, None)
        for item in swarm.setdefault("pendingPermissions", []):
            if item.get("requestId") == request_id and item.get("status") == "pending":
                item["status"] = "expired"
                item["updatedAt"] = utc_now()


async def execute_worker_tool_calls(
    swarm: dict[str, Any],
    worker: dict[str, Any],
    loop: QueryLoopState,
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for call in TOOL_SCHEDULER.ordered_calls(tool_calls):
        tool_use_id = str(call["id"])
        tool_name = str(call["name"])
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        allowed, reason = worker_tool_allowed(swarm, tool_name)
        if not allowed:
            result_payload = {"content": reason or "tool denied", "isError": True, "metadata": {"decision": "deny"}}
            loop.record_tool_call(tool_use_id, tool_name, arguments, status="error")
            loop.update_tool_call(tool_use_id, "error", result_payload, "denied")
            record_worker_tool_call(worker, tool_name, arguments, "error", result_payload["content"])
            results.append({"toolUseId": tool_use_id, "toolName": tool_name, **result_payload})
            continue
        loop.transition(QueryPhase.TOOL_RUNNING, f"worker_tool:{tool_name}")
        loop.record_tool_call(tool_use_id, tool_name, arguments)
        push_worker_progress(swarm, worker)
        tool = TOOL_REGISTRY.get(tool_name)
        if not tool:
            result_payload = {"content": f"Unknown tool: {tool_name}", "isError": True, "metadata": {}}
            loop.update_tool_call(tool_use_id, "error", result_payload, "unknown_tool")
            record_worker_tool_call(worker, tool_name, arguments, "error", result_payload["content"])
            results.append({"toolUseId": tool_use_id, "toolName": tool_name, **result_payload})
            continue
        tool_started = time.time()
        result = TOOL_REGISTRY.call(tool_name, arguments)
        if (result.metadata or {}).get("decision") == "ask":
            decision = await request_swarm_permission(swarm, worker, tool_name, arguments, result.content)
            if decision.get("decision") not in {"allow", "ALLOW"} and not decision.get("approved"):
                result_payload = {"content": f"Permission denied for tool: {tool_name}", "isError": True, "metadata": {"decision": "deny", "requestId": decision.get("requestId")}}
                loop.update_tool_call(tool_use_id, "error", result_payload, "permission_denied")
                record_worker_tool_call(worker, tool_name, arguments, "error", result_payload["content"], int((time.time() - tool_started) * 1000))
                results.append({"toolUseId": tool_use_id, "toolName": tool_name, **result_payload})
                worker["status"] = "TERMINATED"
                worker["terminationReason"] = "permission_denied"
                worker["errorMessage"] = result_payload["content"]
                push_worker_progress(swarm, worker)
                break
            try:
                result = tool.handler(arguments)
            except Exception as exc:
                result_payload = {"content": str(exc), "isError": True, "metadata": {"requestId": decision.get("requestId")}}
                loop.update_tool_call(tool_use_id, "error", result_payload, "exception")
                record_worker_tool_call(worker, tool_name, arguments, "error", result_payload["content"], int((time.time() - tool_started) * 1000))
                results.append({"toolUseId": tool_use_id, "toolName": tool_name, **result_payload})
                continue
        result_payload = result.to_dict()
        loop.update_tool_call(tool_use_id, "error" if result.isError else "completed", result_payload, "completed")
        record_worker_tool_call(
            worker,
            tool_name,
            arguments,
            "error" if result.isError else "success",
            result.content if result.isError else None,
            int((time.time() - tool_started) * 1000),
        )
        if worker.get("repetitionDetected"):
            event = append_swarm_event(
                swarm,
                "worker_repetition",
                f"Repeated tool call pattern detected for {worker.get('workerId')}",
                workerId=worker.get("workerId"),
                toolName=tool_name,
                severity="warning",
            )
            WS_SESSION_MANAGER.publish_event(str(swarm.get("sessionId") or "default"), "worker_progress", {"swarmId": swarm.get("swarmId"), **swarm_worker_snapshot(worker), "anomaly": event})
        results.append({"toolUseId": tool_use_id, "toolName": tool_name, **result_payload})
    return results


def complete_permission_tool_from_endpoint(swarm: dict[str, Any], request_record: dict[str, Any], resolution: dict[str, Any]) -> None:
    worker_id = str(request_record.get("workerId") or "")
    worker = swarm.setdefault("workers", {}).get(worker_id)
    recoverable_cancel = worker and worker.get("status") == "TERMINATED" and "permission" in str(worker.get("currentStepDescription") or "")
    if not worker or (worker.get("status") not in {"WAITING_PERMISSION", "WORKING"} and not recoverable_cancel):
        return
    tool_name = str(request_record.get("toolName") or "")
    input_payload = request_record.get("input") if isinstance(request_record.get("input"), dict) else {}
    task_prompt = str(worker.get("currentTask") or "")
    started = time.time()
    if not worker.get("permissionWaitMs") and request_record.get("createdAt"):
        try:
            created_at = datetime.fromisoformat(str(request_record["createdAt"]).replace("Z", "+00:00")).timestamp()
            worker["permissionWaitMs"] = max(0, int((time.time() - created_at) * 1000))
        except Exception:
            worker["permissionWaitMs"] = 0
    result_payload: dict[str, Any]
    if resolution.get("decision") == "allow" or resolution.get("approved"):
        tool = TOOL_REGISTRY.get(tool_name)
        if tool:
            try:
                result_payload = tool.handler(input_payload).to_dict()
            except Exception as exc:
                result_payload = {"content": str(exc), "isError": True, "metadata": {"requestId": request_record.get("requestId")}}
        else:
            result_payload = {"content": f"Unknown tool: {tool_name}", "isError": True, "metadata": {"requestId": request_record.get("requestId")}}
    else:
        result_payload = {"content": f"Permission denied for tool: {tool_name}", "isError": True, "metadata": {"requestId": request_record.get("requestId")}}
    record_worker_tool_call(
        worker,
        tool_name,
        input_payload,
        "error" if result_payload.get("isError") else "success",
        str(result_payload.get("content") or "") if result_payload.get("isError") else None,
        int((time.time() - started) * 1000),
    )
    worker["progressPercent"] = 100
    worker["completedSteps"] = worker.get("totalSteps") or 4
    worker["currentStepDescription"] = "completed after permission decision"
    worker["terminationReason"] = "completed" if not result_payload.get("isError") else "permission_denied"
    worker["status"] = "IDLE" if not result_payload.get("isError") else "TERMINATED"
    worker["errorMessage"] = None if not result_payload.get("isError") else worker.get("errorMessage")
    if result_payload.get("isError"):
        worker["errorMessage"] = result_payload["content"]
    result = {
        "workerId": worker_id,
        "task": task_prompt,
        "status": "completed" if worker["status"] == "IDLE" else "failed",
        "result": f"Worker {worker_id} completed task: {task_prompt}\n{tool_name}: {result_payload['content']}",
        "turnCount": 1,
        "tokensConsumed": max(
            1,
            TOKEN_COUNTER.estimate_text_for_model(
                task_prompt,
                str(worker.get("model") or swarm.get("workerModel") or STATE["config"].get("defaultModel")),
                MODEL_CAPABILITIES.get_capability(str(worker.get("model") or swarm.get("workerModel") or STATE["config"].get("defaultModel"))).tokenCharRatio,
            ),
        ),
        "durationMs": int((time.time() - started) * 1000),
        "toolResults": [{"toolUseId": request_record.get("requestId"), "toolName": tool_name, **result_payload}],
        "turns": [],
        "mailbox": [],
        "completedAt": utc_now(),
    }
    worker["tokenConsumed"] = int(worker.get("tokenConsumed") or 0) + result["tokensConsumed"]
    result["notificationXml"] = format_task_notification(worker_id, result)
    swarm.setdefault("results", {})[worker_id] = result
    append_swarm_event(swarm, "worker_complete", f"Worker completed after permission: {worker_id}", workerId=worker_id, result=result)
    WS_SESSION_MANAGER.publish_event(str(swarm.get("sessionId") or "default"), "task_update", {"taskId": worker_id, "status": result["status"], "result": result["notificationXml"]})
    aggregate_swarm_results(swarm)
    push_worker_progress(swarm, worker)
    push_swarm_state(swarm)


def aggregate_swarm_results(swarm: dict[str, Any]) -> str:
    results = list(swarm.setdefault("results", {}).values())
    if not results:
        return f"No results to aggregate for team: {swarm.get('teamName')}"
    success = sum(1 for item in results if item.get("status") == "completed")
    partial = len(results) - success
    sections = [f"## Team Results: {swarm.get('teamName')}", ""]
    for index, item in enumerate(sorted(results, key=lambda row: row.get("workerId", "")), start=1):
        status = "SUCCESS" if item.get("status") == "completed" else "PARTIAL"
        task = str(item.get("task") or "")[:200]
        result = str(item.get("result") or item.get("error") or "(no output)")
        if len(result) > 50_000:
            result = result[:50_000] + "...[truncated]"
        sections.extend([f"### Worker {index} [{status}]", f"**Task**: {task}", "**Result**:", result, ""])
    sections.extend(["---", f"**Summary**: {len(results)} workers completed ({success} success, {partial} partial/failed)"])
    aggregate = "\n".join(sections)
    if len(aggregate) > 200_000:
        aggregate = aggregate[:200_000] + "\n...[aggregation truncated]"
    swarm["aggregateResult"] = aggregate
    return aggregate


async def run_swarm_worker(swarm_id: str, worker_id: str) -> None:
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        return
    worker = swarm.setdefault("workers", {}).get(worker_id)
    if not worker:
        return
    started = time.time()
    worker["status"] = "WORKING"
    worker["progressPercent"] = 5
    worker["completedSteps"] = 0
    worker["totalSteps"] = 5
    worker["currentStepDescription"] = "starting isolated worker context"
    worker["terminationReason"] = None
    worker["parentSessionId"] = swarm.get("sessionId")
    worker["agentHierarchy"] = worker.get("agentHierarchy") or f"main/{worker_id}"
    swarm["phase"] = "RUNNING"
    append_swarm_event(swarm, "worker_start", f"Worker started: {worker_id}", workerId=worker_id)
    push_worker_progress(swarm, worker)
    push_swarm_state(swarm)
    try:
        task_prompt = str(worker.get("currentTask") or "")
        await asyncio.sleep(float(swarm.get("stepDelayMs") or 0) / 1000)
        inbox = drain_mailbox(worker_id)
        worker["completedSteps"] = 1
        worker["progressPercent"] = 25
        worker["currentStepDescription"] = "read team mailbox"
        push_worker_progress(swarm, worker)

        record_worker_tool_call(worker, "TaskContext", {"task": task_prompt}, "success", duration_ms=0)
        worker["completedSteps"] = 2
        worker["progressPercent"] = 50
        worker["currentStepDescription"] = "executing task"
        push_worker_progress(swarm, worker)
        check_worker_timeout(swarm, worker, started)

        raw_turns = worker.get("turns") if isinstance(worker.get("turns"), list) else []
        if not raw_turns:
            raw_turns = [{"prompt": task_prompt, "toolCalls": worker.get("toolCalls") or []}]
        worker["turns"] = raw_turns
        turn_results: list[dict[str, Any]] = []
        for turn_index, turn in enumerate(raw_turns, start=1):
            if worker.get("status") == "TERMINATED":
                break
            turn_prompt = str((turn or {}).get("prompt") or task_prompt)
            tool_calls = normalize_tool_calls((turn or {}).get("toolCalls") or [])
            worker_model = str(worker.get("model") or swarm.get("workerModel") or STATE["config"].get("defaultModel"))
            worker_capability = MODEL_CAPABILITIES.get_capability(worker_model)
            worker_used_tokens = await estimate_query_input_tokens(turn_prompt, worker_model, worker_capability.tokenCharRatio)
            loop = QueryLoopState.start(
                session_id=str(worker.get("sessionId") or worker_session_id(swarm_id, worker_id)),
                user_input=turn_prompt,
                model=worker_model,
                context_window=worker_capability.contextWindow,
                threshold=MODEL_CAPABILITIES.compact_threshold(worker_model),
                ratio=worker_capability.tokenCharRatio,
                used_tokens=worker_used_tokens,
            )
            worker["status"] = "WORKING"
            worker["currentStepDescription"] = f"executing turn {turn_index}/{len(raw_turns)}"
            worker["progressPercent"] = min(75, 50 + int((turn_index - 1) / max(1, len(raw_turns)) * 25))
            push_worker_progress(swarm, worker)
            check_worker_timeout(swarm, worker, started)
            tool_results = await execute_worker_tool_calls(swarm, worker, loop, tool_calls)
            check_worker_timeout(swarm, worker, started)
            answer, llm_tool_results, assistant_message = await execute_worker_query_turn(swarm, worker, loop, turn_prompt, tool_results)
            tool_results.extend(llm_tool_results)
            check_worker_timeout(swarm, worker, started)
            loop.finish("end_turn", error=worker.get("errorMessage") if worker.get("status") == "TERMINATED" else None)
            persist_query_loop(loop)
            turn_results.append(
                {
                    "turn": turn_index,
                    "prompt": turn_prompt,
                    "answer": answer,
                    "assistantMessageId": assistant_message.get("uuid"),
                    "sessionId": worker.get("sessionId"),
                    "toolResults": tool_results,
                    "queryLoop": loop.to_dict(),
                }
            )
        tool_results = [item for turn in turn_results for item in turn.get("toolResults", [])]
        if worker.get("status") == "TERMINATED":
            swarm.setdefault("results", {})[worker_id] = {
                "workerId": worker_id,
                "task": task_prompt,
                "status": "failed",
                "error": worker.get("errorMessage") or worker.get("terminationReason") or "terminated",
                "turns": turn_results,
                "durationMs": int((time.time() - started) * 1000),
                "completedAt": utc_now(),
            }
            aggregate_swarm_results(swarm)
            push_swarm_state(swarm)
            await start_next_queued_worker(swarm_id)
            save_state()
            return
        await asyncio.sleep(float(swarm.get("stepDelayMs") or 0) / 1000)

        mailbox_note = "\n".join(f"Mail from {item['senderId']}: {item['content']}" for item in inbox)
        tool_note = "\n".join(f"{item['toolName']}: {item['content']}" for item in tool_results)
        answer_note = "\n".join(f"Turn {item['turn']}: {item.get('answer')}" for item in turn_results if item.get("answer"))
        result_text = "\n".join(
            part
            for part in [
                f"Worker {worker_id} completed task: {task_prompt}",
                mailbox_note,
                tool_note,
                answer_note,
            ]
            if part
        )
        worker["completedSteps"] = 3
        worker["progressPercent"] = 80
        worker["currentStepDescription"] = "publishing result"
        push_worker_progress(swarm, worker)

        result = {
            "workerId": worker_id,
            "task": task_prompt,
            "status": "completed",
            "result": result_text,
            "turnCount": len(turn_results),
            "tokensConsumed": max(
                1,
                TOKEN_COUNTER.estimate_text_for_model(
                    f"{task_prompt}\n{result_text}",
                    str(worker.get("model") or swarm.get("workerModel") or STATE["config"].get("defaultModel")),
                    MODEL_CAPABILITIES.get_capability(str(worker.get("model") or swarm.get("workerModel") or STATE["config"].get("defaultModel"))).tokenCharRatio,
                ),
            ),
            "durationMs": int((time.time() - started) * 1000),
            "toolResults": tool_results,
            "turns": turn_results,
            "mailbox": inbox,
            "completedAt": utc_now(),
        }
        result["notificationXml"] = format_task_notification(worker_id, result)
        swarm.setdefault("results", {})[worker_id] = result
        worker["status"] = "IDLE"
        worker["tokenConsumed"] = int(worker.get("tokenConsumed") or 0) + result["tokensConsumed"]
        worker["progressPercent"] = 100
        worker["completedSteps"] = 5
        worker["currentStepDescription"] = "completed"
        worker["terminationReason"] = "completed"
        append_swarm_event(swarm, "worker_complete", f"Worker completed: {worker_id}", workerId=worker_id, result=result)
        WS_SESSION_MANAGER.publish_event(str(swarm.get("sessionId") or "default"), "task_update", {"taskId": worker_id, "status": "completed", "result": result["notificationXml"]})
        aggregate_swarm_results(swarm)
        push_worker_progress(swarm, worker)
        push_swarm_state(swarm)
        await start_next_queued_worker(swarm_id)
        save_state()
    except asyncio.CancelledError:
        worker["status"] = "TERMINATED"
        worker["terminationReason"] = "aborted"
        worker["errorMessage"] = "Worker task cancelled."
        worker["progressPercent"] = worker.get("progressPercent") or 0
        append_swarm_event(swarm, "worker_abort", f"Worker aborted: {worker_id}", workerId=worker_id)
        push_worker_progress(swarm, worker)
        push_swarm_state(swarm)
        save_state()
        raise
    except TimeoutError as exc:
        worker["status"] = "TERMINATED"
        worker["terminationReason"] = "timeout"
        worker["errorMessage"] = str(exc)
        swarm.setdefault("results", {})[worker_id] = {
            "workerId": worker_id,
            "task": worker.get("currentTask"),
            "status": "failed",
            "error": str(exc),
            "durationMs": int((time.time() - started) * 1000),
            "effectiveRuntimeMs": worker.get("effectiveRuntimeMs"),
            "permissionWaitMs": worker.get("permissionWaitMs", 0),
            "completedAt": utc_now(),
        }
        swarm["results"][worker_id]["notificationXml"] = format_task_notification(worker_id, swarm["results"][worker_id])
        append_swarm_event(swarm, "worker_timeout", f"Worker timed out: {worker_id}", workerId=worker_id, error=str(exc), severity="high")
        aggregate_swarm_results(swarm)
        push_worker_progress(swarm, worker)
        push_swarm_state(swarm)
        await start_next_queued_worker(swarm_id)
        save_state()
    except Exception as exc:
        worker["status"] = "TERMINATED"
        worker["terminationReason"] = "error"
        worker["errorMessage"] = str(exc)
        swarm.setdefault("results", {})[worker_id] = {
            "workerId": worker_id,
            "task": worker.get("currentTask"),
            "status": "failed",
            "error": str(exc),
            "durationMs": int((time.time() - started) * 1000),
            "completedAt": utc_now(),
        }
        swarm["results"][worker_id]["notificationXml"] = format_task_notification(worker_id, swarm["results"][worker_id])
        append_swarm_event(swarm, "worker_error", f"Worker failed: {worker_id}", workerId=worker_id, error=str(exc))
        aggregate_swarm_results(swarm)
        push_worker_progress(swarm, worker)
        push_swarm_state(swarm)
        await start_next_queued_worker(swarm_id)
        save_state()


async def start_swarm_worker_task(swarm_id: str, worker_id: str) -> asyncio.Task[Any]:
    task = asyncio.create_task(run_swarm_worker(swarm_id, worker_id))
    SWARM_TASKS.setdefault(swarm_id, {})[worker_id] = task
    def _cleanup(done: asyncio.Task[Any]) -> None:
        SWARM_TASKS.get(swarm_id, {}).pop(worker_id, None)
    task.add_done_callback(_cleanup)
    return task


async def start_next_queued_worker(swarm_id: str) -> str | None:
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        return None
    queued = swarm.setdefault("queuedTasks", [])
    if not queued:
        return None
    max_concurrent = max(1, int(swarm.get("maxConcurrentWorkers") or swarm.get("totalWorkers") or 1))
    active = sum(1 for item in swarm.setdefault("workers", {}).values() if item.get("status") in {"STARTING", "WORKING", "WAITING_PERMISSION"})
    if active >= max_concurrent:
        return None
    task_spec = queued.pop(0)
    swarm["workerSequence"] = int(swarm.get("workerSequence") or len(swarm.setdefault("workers", {}))) + 1
    worker_id = str(task_spec.get("workerId") or f"worker-{swarm['workerSequence']}")
    swarm.setdefault("workers", {})[worker_id] = make_worker_state(swarm_id, worker_id, task_spec)
    append_swarm_event(swarm, "worker_dequeued", f"Queued worker started: {worker_id}", workerId=worker_id)
    update_team_runtime(swarm)
    await start_swarm_worker_task(swarm_id, worker_id)
    push_swarm_state(swarm)
    return worker_id


def approved_from_permission_payload(payload: dict[str, Any]) -> bool:
    decision = payload.get("decision")
    return bool(payload.get("approved", decision in {"allow", "ALLOW", True}))


def resolve_swarm_permission_request(request_id: str, resolution: dict[str, Any]) -> dict[str, Any]:
    STATE.setdefault("permissionResponses", {})[request_id] = resolution
    resolved = False
    resolved_swarm_id: str | None = None
    for swarm_id, swarm in STATE.setdefault("swarms", {}).items():
        for item in swarm.get("pendingPermissions", []):
            if item.get("requestId") != request_id:
                continue
            resolved = True
            resolved_swarm_id = str(swarm_id)
            item.update(resolution)
            item["status"] = "resolved"
            worker = swarm.setdefault("workers", {}).get(item.get("workerId"))
            if worker and worker.get("status") == "WAITING_PERMISSION":
                worker["status"] = "WORKING"
                worker["currentStepDescription"] = f"permission {resolution['decision']} for {item.get('toolName')}"
            complete_permission_tool_from_endpoint(swarm, item, resolution)
            WS_SESSION_MANAGER.publish_event(
                str(swarm.get("sessionId") or "default"),
                "permission_bubble",
                {
                    "requestId": request_id,
                    "swarmId": swarm.get("swarmId"),
                    "workerId": item.get("workerId"),
                    "toolName": item.get("toolName"),
                    "riskLevel": item.get("riskLevel", "high"),
                    "reason": item.get("reason", ""),
                    "decision": resolution["decision"],
                    "resolved": True,
                    "updatedAt": resolution.get("updatedAt"),
                },
            )
            publish_coordinator_event(
                swarm,
                "permission_resolved",
                {
                    "requestId": request_id,
                    "workerId": item.get("workerId"),
                    "toolName": item.get("toolName"),
                    "riskLevel": item.get("riskLevel", "high"),
                    "reason": item.get("reason", ""),
                    "decision": resolution["decision"],
                    "approved": resolution.get("approved"),
                    "resolved": True,
                    "updatedAt": resolution.get("updatedAt"),
                    "leaderSessionId": str(swarm.get("sessionId") or "default"),
                    "pendingRequestCount": max(0, len([p for p in swarm.get("pendingPermissions", []) if p.get("requestId") != request_id and p.get("status") == "pending"])),
                },
            )
        swarm["pendingPermissions"] = [item for item in swarm.get("pendingPermissions", []) if item.get("requestId") != request_id]
    waiter = SWARM_PERMISSION_WAITERS.pop(request_id, None)
    if waiter and not waiter.done():
        waiter.set_result(resolution)
    return {"success": resolved, "swarmId": resolved_swarm_id, **resolution}


def resolve_swarm_permission_batch(payload: dict[str, Any], swarm_id: str | None = None) -> dict[str, Any]:
    approved = approved_from_permission_payload(payload)
    reason = str(payload.get("reason") or ("batch allow" if approved else "batch deny"))
    explicit_request_ids = bool(payload.get("requestIds") or payload.get("request_ids"))
    request_ids = [str(item) for item in (payload.get("requestIds") or payload.get("request_ids") or []) if item]
    if payload.get("all") or not request_ids:
        request_ids = [str(item.get("requestId")) for item in pending_swarm_permission_records(swarm_id) if item.get("requestId")]
    processed = []
    missing = []
    for request_id in dict.fromkeys(request_ids):
        resolution = {
            "requestId": request_id,
            "decision": "allow" if approved else "deny",
            "approved": approved,
            "reason": reason,
            "updatedAt": utc_now(),
        }
        result = resolve_swarm_permission_request(request_id, resolution)
        if result.get("success"):
            processed.append(result)
        else:
            missing.append(request_id)
    pending_records = pending_swarm_permission_records(swarm_id)
    if explicit_request_ids:
        pending_request_count = sum(1 for item in pending_records if str(item.get("requestId")) in set(request_ids))
    else:
        pending_request_count = len(pending_records)
    return {
        "success": True,
        "swarmId": swarm_id,
        "decision": "allow" if approved else "deny",
        "processed": processed,
        "processedCount": len(processed),
        "missingRequestIds": missing,
        "pendingRequestCount": pending_request_count,
    }


def swarm_task_description(task: dict[str, Any]) -> str:
    return str(task.get("description") or task.get("task") or task.get("prompt") or task.get("currentTask") or "")


def build_swarm_shared_task_list(swarm: dict[str, Any]) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for index, task in enumerate(swarm.get("queuedTasks") or [], start=1):
        tasks.append(
            {
                "taskId": str(task.get("taskId") or task.get("workerId") or f"queued-{index}"),
                "workerId": task.get("workerId"),
                "description": swarm_task_description(task),
                "status": "queued",
                "phase": "queued",
                "source": "queuedTasks",
                "createdAt": task.get("createdAt"),
                "updatedAt": task.get("updatedAt"),
            }
        )
    for worker_id, worker in sorted((swarm.get("workers") or {}).items()):
        status = str(worker.get("status") or "UNKNOWN").lower()
        if status in {"idle"} and worker_id in swarm.get("results", {}):
            continue
        tasks.append(
            {
                "taskId": str(worker.get("taskId") or worker_id),
                "workerId": worker_id,
                "description": swarm_task_description(worker),
                "status": status,
                "phase": status,
                "source": "workers",
                "progressPercent": worker.get("progressPercent", 0),
                "createdAt": worker.get("startedAt"),
                "updatedAt": worker.get("updatedAt"),
            }
        )
    for worker_id, result in sorted((swarm.get("results") or {}).items()):
        tasks.append(
            {
                "taskId": str(result.get("taskId") or worker_id),
                "workerId": worker_id,
                "description": swarm_task_description(result),
                "status": str(result.get("status") or "completed"),
                "phase": "completed" if result.get("status") == "completed" else "failed",
                "source": "results",
                "createdAt": result.get("createdAt"),
                "updatedAt": result.get("completedAt") or result.get("updatedAt"),
            }
        )
    counts = {
        "queued": sum(1 for item in tasks if item["status"] == "queued"),
        "running": sum(1 for item in tasks if item["status"] in {"starting", "working", "waiting_permission"}),
        "completed": sum(1 for item in tasks if item["status"] == "completed"),
        "failed": sum(1 for item in tasks if item["status"] in {"failed", "terminated", "error"}),
        "total": len(tasks),
    }
    return {"swarmId": swarm.get("swarmId"), "tasks": tasks, "counts": counts}


def live_swarm_worker_task(swarm_id: str, worker_id: str) -> asyncio.Task[Any] | None:
    task = SWARM_TASKS.get(swarm_id, {}).get(worker_id)
    if task and not task.done() and not task.cancelled():
        return task
    return None


def recover_orphaned_swarm_workers(swarm_id: str, swarm: dict[str, Any], reason: str) -> dict[str, Any]:
    recoverable_statuses = {"STARTING", "WORKING", "WAITING_PERMISSION"}
    queued_tasks = swarm.setdefault("queuedTasks", [])
    recovered_workers: list[dict[str, Any]] = []
    skipped_workers: list[dict[str, Any]] = []
    now = utc_now()
    for worker_id, worker in sorted((swarm.get("workers") or {}).items()):
        status = str(worker.get("status") or "").upper()
        if status not in recoverable_statuses:
            skipped_workers.append({"workerId": worker_id, "status": status or "UNKNOWN", "reason": "not_recoverable_status"})
            continue
        if live_swarm_worker_task(swarm_id, worker_id):
            skipped_workers.append({"workerId": worker_id, "status": status, "reason": "live_task_exists"})
            continue
        if worker.get("requeuedTaskId"):
            skipped_workers.append({"workerId": worker_id, "status": status, "reason": "already_requeued", "taskId": worker.get("requeuedTaskId")})
            continue
        description = swarm_task_description(worker)
        if not description.strip():
            skipped_workers.append({"workerId": worker_id, "status": status, "reason": "empty_task"})
            continue
        task_id = new_id("recovered-task")
        queued_task = {
            "taskId": task_id,
            "task": description,
            "prompt": description,
            "description": description,
            "agentType": worker.get("agentType"),
            "model": worker.get("model"),
            "toolCalls": copy.deepcopy(worker.get("toolCalls") or []),
            "turns": copy.deepcopy(worker.get("turns") or []),
            "recoveredFromWorkerId": worker_id,
            "recoveredReason": reason,
            "recoveredAt": now,
            "createdAt": now,
            "updatedAt": now,
        }
        queued_tasks.append(queued_task)
        worker["status"] = "TERMINATED"
        worker["terminationReason"] = "requeued_after_restart"
        worker["errorMessage"] = f"Worker lost its live runtime task and was requeued: {reason}"
        worker["recoveredAt"] = now
        worker["requeuedTaskId"] = task_id
        worker["updatedAt"] = now
        recovered_workers.append({"workerId": worker_id, "status": status, "queuedTask": queued_task})
        append_swarm_event(swarm, "worker_requeued_after_recovery", f"Worker requeued after recovery: {worker_id}", workerId=worker_id, taskId=task_id, reason=reason)
        publish_coordinator_event(
            swarm,
            "worker_requeued_after_recovery",
            {"workerId": worker_id, "taskId": task_id, "reason": reason, "previousStatus": status, "description": description[:500]},
        )
    update_swarm_counts(swarm)
    swarm["updatedAt"] = utc_now()
    return {
        "success": True,
        "swarmId": swarm_id,
        "recoveredWorkers": recovered_workers,
        "recoveredCount": len(recovered_workers),
        "skippedWorkers": skipped_workers,
        "skippedCount": len(skipped_workers),
        "queuedTaskCount": len(queued_tasks),
    }


def recover_all_orphaned_swarm_workers(
    reason: str = "startup recovery",
    active_only: bool = True,
    persist: bool = True,
    swarm_ids: list[str] | None = None,
) -> dict[str, Any]:
    swarms: list[dict[str, Any]] = []
    skipped_swarms: list[dict[str, Any]] = []
    recovered_worker_count = 0
    scanned = 0
    selected_ids = {str(item) for item in swarm_ids} if swarm_ids else None
    for swarm_id, swarm in sorted(STATE.setdefault("swarms", {}).items()):
        if selected_ids is not None and str(swarm_id) not in selected_ids:
            continue
        if active_only and str(swarm.get("phase") or "").upper() == "TERMINATED":
            skipped_swarms.append({"swarmId": swarm_id, "reason": "terminated"})
            continue
        scanned += 1
        result = recover_orphaned_swarm_workers(str(swarm_id), swarm, reason)
        if result.get("recoveredCount"):
            recovered_worker_count += int(result.get("recoveredCount") or 0)
            swarms.append(result)
            update_team_runtime(swarm)
            push_swarm_state(swarm)
    if recovered_worker_count and persist:
        save_state()
    return {
        "success": True,
        "reason": reason,
        "scannedSwarmCount": scanned,
        "skippedSwarmCount": len(skipped_swarms),
        "skippedSwarms": skipped_swarms,
        "recoveredSwarmCount": len(swarms),
        "recoveredWorkerCount": recovered_worker_count,
        "requestedSwarmIds": sorted(selected_ids) if selected_ids is not None else None,
        "swarms": swarms,
    }


SWARM_WORKFLOW_PHASES = ["Research", "Synthesis", "Implementation", "Verification"]


def normalize_swarm_workflow_phase(value: Any) -> str:
    raw = str(value or "").strip()
    for phase in SWARM_WORKFLOW_PHASES:
        if phase.lower() == raw.lower():
            return phase
    raise HTTPException(status_code=400, detail=f"Invalid phase: {raw or '(empty)'}")


def swarm_workflow_phase_index(phase: str | None) -> int:
    if not phase:
        return -1
    try:
        return SWARM_WORKFLOW_PHASES.index(normalize_swarm_workflow_phase(phase))
    except HTTPException:
        return -1


def get_or_restore_swarm_workflow(swarm: dict[str, Any]):
    session_id = str(swarm.get("sessionId") or "default")
    workflow = COORDINATOR_ENGINE.active.get(session_id)
    if workflow:
        return workflow
    stored = swarm.get("workflow")
    if isinstance(stored, dict):
        workflow = COORDINATOR_ENGINE.restore_workflow(stored)
        swarm["workflowId"] = workflow.workflowId
        swarm["workflow"] = workflow.to_dict()
        return workflow
    workflow = COORDINATOR_ENGINE.start_workflow(session_id, str(swarm.get("objective") or swarm.get("teamName") or swarm.get("swarmId") or "swarm"))
    swarm["workflowId"] = workflow.workflowId
    swarm["workflow"] = workflow.to_dict()
    return workflow


def sync_swarm_workflow(swarm: dict[str, Any], workflow: Any) -> dict[str, Any]:
    workflow_payload = workflow.to_dict()
    swarm["workflowId"] = workflow_payload.get("workflowId")
    swarm["workflow"] = workflow_payload
    swarm["updatedAt"] = utc_now()
    WS_SESSION_MANAGER.publish_event(str(swarm.get("sessionId") or "default"), "workflow_phase_update", {"workflow": workflow_payload})
    return workflow_payload


def advance_swarm_workflow_after_barrier(swarm: dict[str, Any], barrier: dict[str, Any]) -> dict[str, Any] | None:
    if barrier.get("status") != "released":
        return None
    phase = normalize_swarm_workflow_phase(barrier.get("phase"))
    phase_index = swarm_workflow_phase_index(phase)
    if barrier.get("workflowAdvanced"):
        return swarm.get("workflow") if isinstance(swarm.get("workflow"), dict) else None

    workflow = get_or_restore_swarm_workflow(swarm)
    current_phase = workflow.currentPhase.name.value if workflow.currentPhase else None
    current_index = swarm_workflow_phase_index(current_phase)
    if current_index < 0:
        return None
    if current_index > phase_index:
        barrier["workflowAdvanced"] = True
        barrier["workflowAdvancedAt"] = utc_now()
        barrier["workflowAdvanceReason"] = "already_past_phase"
        return sync_swarm_workflow(swarm, workflow)
    if current_phase != phase:
        return None

    previous_phase = current_phase
    summary = f"{phase} barrier released for {len(barrier.get('reachedWorkerIds') or [])} workers"
    advanced = COORDINATOR_ENGINE.advance_workflow(str(swarm.get("sessionId") or "default"), summary)
    if not advanced:
        return None
    workflow_payload = sync_swarm_workflow(swarm, advanced)
    next_phase = workflow_payload.get("currentPhase", {}).get("name") if workflow_payload.get("currentPhase") else None
    barrier["workflowAdvanced"] = True
    barrier["workflowAdvancedAt"] = utc_now()
    barrier["workflowId"] = workflow_payload.get("workflowId")
    append_swarm_event(swarm, "workflow_phase_advanced", f"Workflow advanced after {phase} barrier", previousPhase=previous_phase, nextPhase=next_phase, workflowStatus=workflow_payload.get("status"))
    publish_coordinator_event(
        swarm,
        "workflow_phase_advanced",
        {
            "barrierId": barrier.get("barrierId"),
            "phase": phase,
            "previousPhase": previous_phase,
            "nextPhase": next_phase,
            "workflowStatus": workflow_payload.get("status"),
            "workflowId": workflow_payload.get("workflowId"),
        },
    )
    return workflow_payload


def evaluate_swarm_phase_barrier(swarm: dict[str, Any], phase: str, worker_ids: list[str] | None = None) -> dict[str, Any]:
    normalized_phase = normalize_swarm_workflow_phase(phase)
    target_index = swarm_workflow_phase_index(normalized_phase)
    workers = swarm.setdefault("workers", {})
    selected_worker_ids = worker_ids or sorted(str(worker_id) for worker_id in workers)
    reached: list[str] = []
    missing: list[str] = []
    for worker_id in selected_worker_ids:
        worker = workers.get(worker_id)
        worker_phase_index = int(worker.get("workflowPhaseIndex", swarm_workflow_phase_index(worker.get("currentWorkflowPhase")))) if worker else -1
        if worker and worker_phase_index >= target_index:
            reached.append(worker_id)
        else:
            missing.append(worker_id)
    barrier_id = normalized_phase.lower()
    previous = swarm.setdefault("phaseBarriers", {}).get(barrier_id, {})
    status = "released" if not missing else "waiting"
    barrier = {
        **previous,
        "barrierId": barrier_id,
        "phase": normalized_phase,
        "phaseIndex": target_index,
        "workerIds": selected_worker_ids,
        "reachedWorkerIds": reached,
        "missingWorkerIds": missing,
        "status": status,
        "updatedAt": utc_now(),
    }
    if status == "released" and previous.get("status") != "released":
        barrier["releasedAt"] = utc_now()
        append_swarm_event(swarm, "phase_barrier_released", f"Phase barrier released: {normalized_phase}", phase=normalized_phase, workerIds=selected_worker_ids)
    swarm.setdefault("phaseBarriers", {})[barrier_id] = barrier
    if status == "released":
        advance_swarm_workflow_after_barrier(swarm, barrier)
    return barrier


@app.post("/api/swarm")
async def create_swarm(request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    payload = await request.json()
    swarm_id = new_id("swarm")
    raw_team_name = payload.get("teamName")
    team_name = str(raw_team_name if raw_team_name is not None else f"swarm-team-{swarm_id[-6:]}")
    validate_team_name(team_name, explicit=raw_team_name is not None)
    if find_swarm_by_team(team_name):
        raise HTTPException(status_code=409, detail=f"Team already exists: {team_name}")
    raw_max_workers = payload.get("maxWorkers", 5)
    if raw_max_workers is None:
        raw_max_workers = 5
    try:
        max_workers = int(raw_max_workers)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="maxWorkers must be an integer") from exc
    if max_workers < 1 or max_workers > 20:
        raise HTTPException(status_code=400, detail=f"Worker count must be between 1 and 20, got: {max_workers}")
    all_task_specs = [parse_task_spec(item) for item in (payload.get("tasks") or [])]
    task_queue_size = max(0, int(payload.get("taskQueueSize") or 50))
    tasks = all_task_specs[:max_workers]
    queued_tasks = all_task_specs[max_workers : max_workers + task_queue_size]
    worker_tool_calls = payload.get("workerToolCalls") if isinstance(payload.get("workerToolCalls"), dict) else {}
    worker_turns = payload.get("workerTurns") if isinstance(payload.get("workerTurns"), dict) else {}
    workers = {}
    for index, task_spec in enumerate(tasks):
        worker_id = f"worker-{index + 1}"
        workers[worker_id] = make_worker_state(swarm_id, worker_id, task_spec, worker_tool_calls, worker_turns)
    requested_scratchpad = payload.get("scratchpadDir") or payload.get("scratchpad_dir")
    scratchpad_dir = safe_workspace_path(str(requested_scratchpad), default=ROOT) if requested_scratchpad else DATA_DIR / "swarms" / swarm_id
    scratchpad_dir.mkdir(parents=True, exist_ok=True)
    swarm = {
        "swarmId": swarm_id,
        "teamId": swarm_id,
        "teamName": team_name,
        "sessionId": payload.get("sessionId") or "default",
        "phase": "INITIALIZING" if tasks else "IDLE",
        "activeWorkers": len(tasks),
        "totalWorkers": max_workers,
        "completedTasks": 0,
        "totalTasks": len(tasks) + len(queued_tasks),
        "maxConcurrentWorkers": max_workers,
        "taskQueueSize": task_queue_size,
        "queuedTasks": [
            {
                **task,
                "toolCalls": worker_tool_calls.get(f"worker-{max_workers + index + 1}", task.get("toolCalls") or []),
                "turns": worker_turns.get(f"worker-{max_workers + index + 1}", task.get("turns") or []),
            }
            for index, task in enumerate(queued_tasks)
        ],
        "workerSequence": len(workers),
        "workers": workers,
        "events": [{"type": "created", "timestamp": utc_now(), "message": f"Swarm {team_name} created"}],
        "pendingPermissions": [],
        "results": {},
        "aggregateResult": "",
        "workerModel": payload.get("workerModel"),
        "workerUseLlm": payload.get("workerUseLlm", True),
        "workerToolAllowList": payload.get("workerToolAllowList") or [],
        "workerToolDenyList": payload.get("workerToolDenyList") or [],
        "permissionTimeoutMs": int(payload.get("permissionTimeoutMs") or 60_000),
        "workerTimeoutMs": int(payload.get("workerTimeoutMs") or 30 * 60 * 1000),
        "workerIdleTimeoutMs": int(payload.get("workerIdleTimeoutMs") or 300_000),
        "scratchpadDir": str(scratchpad_dir),
        "workingDirectory": str(scratchpad_dir),
        "stepDelayMs": int(payload.get("stepDelayMs") or 0),
        "permissionBubbleEnabled": bool(payload.get("permissionBubbleEnabled", True)),
        "executionBackend": payload.get("backend") or "python-asyncio",
        "createdAt": utc_now(),
        "updatedAt": utc_now(),
        "request": payload,
    }
    workflow = COORDINATOR_ENGINE.start_workflow(str(swarm["sessionId"]), str(payload.get("objective") or payload.get("goal") or team_name))
    swarm["workflowId"] = workflow.workflowId
    swarm["workflow"] = workflow.to_dict()
    WS_SESSION_MANAGER.publish_event(str(swarm["sessionId"]), "workflow_phase_update", {"workflow": workflow.to_dict()})
    STATE["swarms"][swarm_id] = swarm
    register_team_runtime(swarm)
    publish_coordinator_event(swarm, "phase_transition", {"previousPhase": None, "nextPhase": (workflow.currentPhase.name.value if workflow.currentPhase else None)})
    push_swarm_state(swarm)
    started_tasks = []
    for worker_id in workers:
        started_tasks.append(await start_swarm_worker_task(swarm_id, worker_id))
    if started_tasks:
        swarm["phase"] = "RUNNING"
        push_swarm_state(swarm)
    if payload.get("awaitCompletion"):
        while True:
            running = list(SWARM_TASKS.get(swarm_id, {}).values())
            if running:
                await asyncio.gather(*running, return_exceptions=True)
                continue
            if not swarm.get("queuedTasks"):
                break
            await start_next_queued_worker(swarm_id)
        aggregate_swarm_results(swarm)
        push_swarm_state(swarm)
    save_state()
    return swarm


@app.get("/api/swarm")
async def list_swarms(activeOnly: bool = True) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarms = list(STATE["swarms"].values())
    if activeOnly:
        swarms = [swarm for swarm in swarms if swarm.get("phase") != "TERMINATED"]
    return {"swarms": swarms}


@app.get("/api/teams")
async def list_teams(activeOnly: bool = True) -> dict[str, Any]:
    ensure_swarm_enabled()
    for swarm in STATE.setdefault("swarms", {}).values():
        update_team_runtime(swarm)
    teams = list(STATE.setdefault("teams", {}).values())
    if activeOnly:
        teams = [team for team in teams if team.get("status") != "TERMINATED"]
    return {"teams": teams}


@app.post("/api/swarm/permissions/batch")
async def batch_swarm_permissions(request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    payload = await request.json()
    result = resolve_swarm_permission_batch(payload)
    save_state()
    return result


@app.post("/api/swarm/{swarm_id}/permissions/batch")
async def batch_swarm_permissions_for_swarm(swarm_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    payload = await request.json()
    result = resolve_swarm_permission_batch(payload, swarm_id=swarm_id)
    save_state()
    return result


@app.get("/api/swarm/{swarm_id}")
async def get_swarm(swarm_id: str) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE["swarms"].get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    expire_swarm_permission_timeouts(swarm_id)
    for item in swarm.setdefault("pendingPermissions", []):
        if item.get("status") == "pending":
            decorate_swarm_permission_deadline(item)
    update_swarm_counts(swarm)
    return swarm


@app.get("/api/swarm/{swarm_id}/runtime")
async def get_swarm_runtime(swarm_id: str) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    expire_swarm_permission_timeouts(swarm_id)
    update_swarm_counts(swarm)
    team = update_team_runtime(swarm)
    tasks = SWARM_TASKS.get(swarm_id, {})
    mailboxes = {worker_id: len(SWARM_MAILBOXES.get(worker_id, [])) for worker_id in swarm.setdefault("workers", {})}
    pending = [decorate_swarm_permission_deadline(item) for item in swarm.setdefault("pendingPermissions", []) if item.get("status") == "pending"]
    return {
        "swarmId": swarm_id,
        "team": team,
        "runningTasks": sorted(tasks.keys()),
        "queuedTasks": list(swarm.get("queuedTasks") or []),
        "mailboxes": mailboxes,
        "pendingPermissions": pending,
        "pendingPermissionCount": len(pending),
        "workers": {worker_id: swarm_worker_snapshot(worker) for worker_id, worker in swarm.setdefault("workers", {}).items()},
    }


@app.post("/api/swarm/{swarm_id}/recover")
async def recover_swarm_runtime(swarm_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    payload = await request.json()
    reason = str(payload.get("reason") or "runtime recovery")
    result = recover_orphaned_swarm_workers(swarm_id, swarm, reason)
    started_worker_ids: list[str] = []
    if payload.get("autoStart") is True:
        for _ in range(result["recoveredCount"]):
            started = await start_next_queued_worker(swarm_id)
            if not started:
                break
            started_worker_ids.append(started)
    result["startedWorkerIds"] = started_worker_ids
    result["queuedTaskCount"] = len(swarm.get("queuedTasks") or [])
    push_swarm_state(swarm)
    save_state()
    return result


async def startup_recover_swarm_runtime() -> None:
    result = recover_all_orphaned_swarm_workers(reason="startup scan")
    if result.get("recoveredWorkerCount"):
        STATE.setdefault("notifications", []).append(
            {
                "id": new_id("notification"),
                "type": "swarm_recovery",
                "severity": "info",
                "message": f"Recovered {result['recoveredWorkerCount']} orphaned swarm worker(s) during startup",
                "payload": result,
                "createdAt": utc_now(),
            }
        )
        save_state()


app.add_event_handler("startup", startup_recover_swarm_runtime)


@app.get("/api/swarm/{swarm_id}/tasks")
async def get_swarm_tasks(swarm_id: str) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    return build_swarm_shared_task_list(swarm)


@app.post("/api/swarm/{swarm_id}/worker/{worker_id}/phase")
async def update_swarm_worker_phase(swarm_id: str, worker_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    worker = swarm.setdefault("workers", {}).get(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    payload = await request.json()
    phase = normalize_swarm_workflow_phase(payload.get("phase"))
    worker["currentWorkflowPhase"] = phase
    worker["workflowPhaseIndex"] = swarm_workflow_phase_index(phase)
    worker["phaseReachedAt"] = utc_now()
    worker["updatedAt"] = utc_now()
    swarm["updatedAt"] = utc_now()
    append_swarm_event(swarm, "worker_phase_update", f"{worker_id} reached {phase}", workerId=worker_id, phase=phase)
    barriers = [
        evaluate_swarm_phase_barrier(swarm, item.get("phase") or phase, list(item.get("workerIds") or []))
        for item in list(swarm.setdefault("phaseBarriers", {}).values())
    ]
    push_swarm_state(swarm)
    save_state()
    return {"success": True, "swarmId": swarm_id, "worker": worker, "barriers": barriers}


@app.post("/api/swarm/{swarm_id}/phase-barrier")
async def swarm_phase_barrier(swarm_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    payload = await request.json()
    phase = normalize_swarm_workflow_phase(payload.get("phase"))
    worker_ids = [str(item) for item in (payload.get("workerIds") or payload.get("worker_ids") or []) if item]
    barrier = evaluate_swarm_phase_barrier(swarm, phase, worker_ids or None)
    push_swarm_state(swarm)
    save_state()
    return {"success": True, "swarmId": swarm_id, "released": barrier["status"] == "released", "barrier": barrier, "missingWorkerIds": barrier["missingWorkerIds"]}


@app.post("/api/swarm/{swarm_id}/tasks")
async def add_swarm_task(swarm_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    payload = await request.json()
    raw_task = payload.get("task", payload)
    task_spec = parse_task_spec(raw_task)
    description = swarm_task_description(task_spec)
    if not description.strip():
        raise HTTPException(status_code=400, detail="task must not be empty")
    task_id = str(payload.get("taskId") or payload.get("task_id") or new_id("swarm-task"))
    creator_id = str(payload.get("creatorId") or payload.get("creator_id") or f"{swarm_id}-leader")
    now = utc_now()
    queued_task = {
        **task_spec,
        "taskId": task_id,
        "teamName": swarm.get("teamName"),
        "description": description,
        "creatorId": creator_id,
        "status": "PENDING",
        "assigneeId": None,
        "result": None,
        "createdAt": now,
        "updatedAt": now,
        "completedAt": None,
    }
    swarm.setdefault("queuedTasks", []).append(queued_task)
    swarm["totalTasks"] = int(swarm.get("totalTasks") or 0) + 1
    swarm["updatedAt"] = now
    pending_count = sum(1 for item in swarm.setdefault("queuedTasks", []) if item.get("status") in {None, "PENDING", "queued"})
    append_swarm_event(
        swarm,
        "task_queued",
        f"Task queued: {description[:120]}",
        taskId=task_id,
        teamName=swarm.get("teamName"),
        creatorId=creator_id,
        status="PENDING",
        pendingCount=pending_count,
    )
    publish_coordinator_event(
        swarm,
        "shared_task_queued",
        {
            "taskId": task_id,
            "teamName": swarm.get("teamName"),
            "description": description,
            "creatorId": creator_id,
            "status": "PENDING",
            "assigneeId": None,
            "result": None,
            "createdAt": now,
            "updatedAt": now,
            "completedAt": None,
            "pendingCount": pending_count,
            "totalTaskCount": len(swarm.setdefault("queuedTasks", [])),
        },
    )
    started_worker_id = None
    if payload.get("autoStart", True) is not False:
        started_worker_id = await start_next_queued_worker(swarm_id)
    push_swarm_state(swarm)
    save_state()
    return {"success": True, "swarmId": swarm_id, "queuedTask": queued_task, "startedWorkerId": started_worker_id, **build_swarm_shared_task_list(swarm)}


@app.post("/api/swarm/{swarm_id}/workers")
async def add_swarm_worker(swarm_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    payload = await request.json()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    max_workers = int(swarm.get("totalWorkers") or 1)
    workers = swarm.setdefault("workers", {})
    if len(workers) >= max_workers:
        raise HTTPException(status_code=409, detail=f"Max workers reached: {max_workers}")
    worker_id = str(payload.get("workerId") or f"worker-{len(workers) + 1}")
    task_prompt = str(payload.get("task") or payload.get("prompt") or "")
    if not task_prompt:
        raise HTTPException(status_code=400, detail="task is required")
    workers[worker_id] = make_worker_state(swarm_id, worker_id, parse_task_spec(payload))
    swarm["totalTasks"] = int(swarm.get("totalTasks") or 0) + 1
    append_swarm_event(swarm, "worker_added", f"Worker added: {worker_id}", workerId=worker_id)
    update_team_runtime(swarm)
    task = await start_swarm_worker_task(swarm_id, worker_id)
    if payload.get("awaitCompletion"):
        await asyncio.gather(task, return_exceptions=True)
    push_swarm_state(swarm)
    save_state()
    return {"success": True, "workerId": worker_id, "swarm": swarm}


@app.get("/api/swarm/{swarm_id}/results")
async def swarm_results(swarm_id: str) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    aggregate = aggregate_swarm_results(swarm)
    return {"swarmId": swarm_id, "results": list(swarm.setdefault("results", {}).values()), "aggregateResult": aggregate}


@app.get("/api/swarm/{swarm_id}/coordinator-events")
async def swarm_coordinator_events(swarm_id: str, limit: int = 200) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    events = swarm.setdefault("coordinatorEvents", [])[-max(1, min(limit, 1000)) :]
    return {"swarmId": swarm_id, "events": events, "total": len(events)}


@app.post("/api/swarm/{swarm_id}/worker/{worker_id}/mail")
async def swarm_send_worker_mail(swarm_id: str, worker_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    payload = await request.json()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    if worker_id not in swarm.setdefault("workers", {}):
        raise HTTPException(status_code=404, detail="Worker not found")
    sender = str(payload.get("senderId") or f"{swarm_id}-leader")
    content = str(payload.get("content") or payload.get("message") or "")
    phase = payload.get("phase")
    channel = str(payload.get("channel") or "default")
    task_id = str(payload.get("taskId") or payload.get("task_id") or "") or None
    mail = write_mailbox(worker_id, sender, content, swarm_id, phase=phase, task_id=task_id, channel=channel)
    mailbox_depth = len(read_mailbox(worker_id))
    append_swarm_event(
        swarm,
        "mailbox_write",
        f"Mail sent to {worker_id}",
        messageId=mail.get("messageId"),
        senderId=sender,
        recipientId=worker_id,
        phase=mail.get("phase"),
        phaseIndex=mail.get("phaseIndex"),
        channel=mail.get("channel"),
        taskId=task_id,
    )
    publish_coordinator_event(
        swarm,
        "mailbox_write",
        {
            "messageId": mail.get("messageId"),
            "senderId": sender,
            "recipientId": worker_id,
            "content": content[:500],
            "contentLength": len(content),
            "phase": mail.get("phase"),
            "phaseIndex": mail.get("phaseIndex"),
            "channel": mail.get("channel"),
            "taskId": task_id,
            "createdAt": mail.get("createdAt"),
            "timestamp": mail.get("timestamp"),
            "mailboxDepth": mailbox_depth,
        },
    )
    WS_SESSION_MANAGER.publish_event(str(swarm.get("sessionId") or "default"), "teammate_message", {"fromId": sender, "content": content, "recipientId": worker_id, "phase": mail.get("phase"), "channel": mail.get("channel"), "taskId": task_id})
    save_state()
    return {"success": True, "message": mail}


@app.post("/api/swarm/{swarm_id}/broadcast")
async def swarm_broadcast_mail(swarm_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    payload = await request.json()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    sender = str(payload.get("senderId") or f"{swarm_id}-leader")
    content = str(payload.get("content") or payload.get("message") or "")
    phase = payload.get("phase")
    channel = str(payload.get("channel") or "default")
    task_id = str(payload.get("taskId") or payload.get("task_id") or "") or None
    messages = [write_mailbox(worker_id, sender, content, swarm_id, phase=phase, task_id=task_id, channel=channel) for worker_id in swarm.setdefault("workers", {}) if worker_id != sender]
    recipient_ids = [str(message.get("recipientId")) for message in messages]
    message_ids = [str(message.get("messageId") or message.get("id")) for message in messages]
    append_swarm_event(
        swarm,
        "mailbox_broadcast",
        f"Broadcast sent to {len(messages)} workers",
        senderId=sender,
        recipientIds=recipient_ids,
        messageIds=message_ids,
        phase=messages[0].get("phase") if messages else None,
        phaseIndex=messages[0].get("phaseIndex") if messages else -1,
        channel=channel,
        taskId=task_id,
    )
    publish_coordinator_event(
        swarm,
        "mailbox_broadcast",
        {
            "teamPrefix": swarm_id,
            "senderId": sender,
            "recipientIds": recipient_ids,
            "recipientCount": len(recipient_ids),
            "messageIds": message_ids,
            "content": content[:500],
            "contentLength": len(content),
            "phase": messages[0].get("phase") if messages else None,
            "phaseIndex": messages[0].get("phaseIndex") if messages else -1,
            "channel": channel,
            "taskId": task_id,
            "createdAt": messages[0].get("createdAt") if messages else None,
        },
    )
    WS_SESSION_MANAGER.publish_event(str(swarm.get("sessionId") or "default"), "teammate_message", {"fromId": sender, "content": content, "broadcast": True, "phase": messages[0].get("phase") if messages else None, "channel": channel, "taskId": task_id})
    save_state()
    return {"success": True, "count": len(messages), "messages": messages}


@app.get("/api/swarm/{swarm_id}/mailbox/{worker_id}")
async def swarm_read_mailbox(swarm_id: str, worker_id: str, drain: bool = True, phase: str | None = None, channel: str | None = None) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    if worker_id not in swarm.setdefault("workers", {}):
        raise HTTPException(status_code=404, detail="Worker not found")
    if drain:
        messages = drain_mailbox(worker_id, phase=phase, channel=channel)
    else:
        messages = read_mailbox(worker_id, phase=phase, channel=channel)
    return {"workerId": worker_id, "messages": messages, "count": len(messages), "phase": normalize_swarm_workflow_phase(phase) if phase else None, "channel": channel}


@app.post("/api/swarm/{swarm_id}/mailbox/{worker_id}/ack")
async def swarm_ack_mailbox(swarm_id: str, worker_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    if worker_id not in swarm.setdefault("workers", {}):
        raise HTTPException(status_code=404, detail="Worker not found")
    payload = await request.json()
    message_ids = [str(item) for item in (payload.get("messageIds") or payload.get("message_ids") or []) if item]
    if payload.get("all") or not message_ids:
        phase = payload.get("phase")
        channel = payload.get("channel")
        message_ids = [str(message.get("id")) for message in replay_mailbox_journal(swarm, worker_id, phase=phase, channel=channel, include_acked=False)]
    acked = ack_mailbox_messages(swarm, worker_id, message_ids)
    append_swarm_event(swarm, "mailbox_ack", f"Mailbox acked {len(acked)} messages for {worker_id}", workerId=worker_id, messageIds=[message.get("id") for message in acked])
    save_state()
    return {"success": True, "swarmId": swarm_id, "workerId": worker_id, "acked": acked, "ackedCount": len(acked)}


@app.get("/api/swarm/{swarm_id}/mailbox/{worker_id}/replay")
async def swarm_replay_mailbox(swarm_id: str, worker_id: str, phase: str | None = None, channel: str | None = None, includeAcked: bool = False) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    if worker_id not in swarm.setdefault("workers", {}):
        raise HTTPException(status_code=404, detail="Worker not found")
    messages = replay_mailbox_journal(swarm, worker_id, phase=phase, channel=channel, include_acked=includeAcked)
    return {"success": True, "swarmId": swarm_id, "workerId": worker_id, "messages": messages, "count": len(messages), "includeAcked": includeAcked}


@app.post("/api/swarm/{swarm_id}/mailbox/{worker_id}/recover")
async def swarm_recover_mailbox(swarm_id: str, worker_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE.setdefault("swarms", {}).get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    if worker_id not in swarm.setdefault("workers", {}):
        raise HTTPException(status_code=404, detail="Worker not found")
    payload = await request.json()
    recovered = recover_mailbox_messages(swarm, worker_id, phase=payload.get("phase"), channel=payload.get("channel"))
    append_swarm_event(swarm, "mailbox_recover", f"Mailbox recovered {len(recovered)} messages for {worker_id}", workerId=worker_id, messageIds=[message.get("id") for message in recovered])
    save_state()
    return {"success": True, "swarmId": swarm_id, "workerId": worker_id, "messages": recovered, "recoveredCount": len(recovered)}


@app.post("/api/coordinator/workflows")
async def coordinator_start_workflow(request: Request) -> dict[str, Any]:
    payload = await request.json()
    session_id = str(payload.get("sessionId") or payload.get("session_id") or "default")
    objective = str(payload.get("objective") or payload.get("goal") or "")
    if not objective:
        raise HTTPException(status_code=400, detail="objective is required")
    workflow = COORDINATOR_ENGINE.start_workflow(session_id, objective)
    WS_SESSION_MANAGER.publish_event(session_id, "workflow_phase_update", {"workflow": workflow.to_dict()})
    return workflow.to_dict()


@app.get("/api/coordinator/workflows/{session_id}")
async def coordinator_get_workflow(session_id: str) -> dict[str, Any]:
    workflow = COORDINATOR_ENGINE.active.get(session_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow.to_dict()


@app.post("/api/coordinator/workflows/{session_id}/advance")
async def coordinator_advance_workflow(session_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    workflow = COORDINATOR_ENGINE.advance_workflow(session_id, str(payload.get("summary") or payload.get("resultSummary") or ""))
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    WS_SESSION_MANAGER.publish_event(session_id, "workflow_phase_update", {"workflow": workflow.to_dict()})
    return workflow.to_dict()


@app.post("/api/coordinator/workflows/{session_id}/scratchpad")
async def coordinator_scratchpad(session_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    item = COORDINATOR_ENGINE.add_scratchpad(session_id, str(payload.get("author") or "agent"), str(payload.get("content") or ""), payload.get("phase"))
    WS_SESSION_MANAGER.publish_event(session_id, "scratchpad_added", item)
    return {"success": True, "item": item}


@app.post("/api/coordinator/detect-phase")
async def coordinator_detect_phase(request: Request) -> dict[str, Any]:
    payload = await request.json()
    session_id = str(payload.get("sessionId") or payload.get("session_id") or "")
    if session_id:
        return COORDINATOR_ENGINE.detect_and_validate_phase(session_id, str(payload.get("output") or payload.get("text") or ""))
    return {"detected": COORDINATOR_ENGINE.detect_phase(str(payload.get("output") or payload.get("text") or "")).to_dict(), "warning": None}


@app.post("/api/coordinator/validate-delegation")
async def coordinator_validate_delegation(request: Request) -> dict[str, Any]:
    payload = await request.json()
    result = COORDINATOR_ENGINE.validate_delegation(str(payload.get("phase") or "Implementation"), str(payload.get("prompt") or ""))
    return result.to_dict()


@app.post("/api/swarm/{swarm_id}/shutdown")
async def shutdown_swarm(swarm_id: str) -> dict[str, Any]:
    ensure_swarm_enabled()
    swarm = STATE["swarms"].setdefault(swarm_id, {"swarmId": swarm_id})
    swarm["phase"] = "SHUTTING_DOWN"
    swarm["activeWorkers"] = 0
    for task in list(SWARM_TASKS.get(swarm_id, {}).values()):
        task.cancel()
    for worker in swarm.setdefault("workers", {}).values():
        if worker.get("status") not in {"IDLE", "TERMINATED"}:
            worker["status"] = "TERMINATED"
            worker["terminationReason"] = "shutdown"
            worker["errorMessage"] = "Swarm shutdown requested."
        else:
            worker["terminationReason"] = worker.get("terminationReason") or "completed"
    swarm["phase"] = "TERMINATED"
    swarm.setdefault("events", []).append({"type": "shutdown", "timestamp": utc_now(), "message": "Swarm shutdown requested"})
    swarm["updatedAt"] = utc_now()
    cleanup_swarm_runtime(swarm_id, swarm)
    update_team_runtime(swarm, "TERMINATED")
    push_swarm_state(swarm)
    save_state()
    return {"status": "shutdown_initiated", "success": True, "swarmId": swarm_id}


@app.post("/api/swarm/{swarm_id}/force-stop")
async def force_stop_swarm(swarm_id: str) -> dict[str, Any]:
    result = await shutdown_swarm(swarm_id)
    result["status"] = "force_stopped"
    return result


@app.post("/api/swarm/{swarm_id}/worker/{worker_id}/abort")
async def abort_worker(swarm_id: str, worker_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    payload = await request.json()
    swarm = STATE["swarms"].get(swarm_id)
    if not swarm:
        raise HTTPException(status_code=404, detail="Swarm not found")
    assert_swarm_owner(swarm, payload, request)
    worker = swarm.setdefault("workers", {}).get(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    reason = payload.get("reason") or "aborted"
    task = SWARM_TASKS.get(swarm_id, {}).get(worker_id)
    if task:
        task.cancel()
    worker["status"] = "TERMINATED"
    worker["terminationReason"] = reason
    worker["errorMessage"] = reason
    worker["progressPercent"] = worker.get("progressPercent") or 0
    update_swarm_counts(swarm)
    event = {
        "id": new_id("anomaly"),
        "type": "worker_abort",
        "workerId": worker_id,
        "severity": "high",
        "message": f"Worker aborted: {reason}",
        "timestamp": utc_now(),
    }
    swarm.setdefault("events", []).append(event)
    swarm["updatedAt"] = utc_now()
    push_worker_progress(swarm, worker)
    push_swarm_state(swarm)
    save_state()
    return {"workerId": worker_id, "status": "aborted", "message": event["message"], "success": True, "swarmId": swarm_id}


@app.get("/api/swarm/permissions/pending-count")
async def swarm_pending_permission_count(swarmId: str | None = None) -> dict[str, Any]:
    ensure_swarm_enabled()
    expired = expire_swarm_permission_timeouts(swarmId)
    records = pending_swarm_permission_records(swarmId)
    return {"pendingRequestCount": len(records), "requests": records, **expired}


@app.post("/api/swarm/permissions/clear")
async def clear_all_swarm_permissions(request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    payload = await request.json()
    reason = str(payload.get("reason") or "cleared")
    cleared = clear_pending_swarm_permissions(reason=reason)
    save_state()
    return {"success": True, "cleared": cleared, "pendingRequestCount": len(pending_swarm_permission_records())}


@app.post("/api/swarm/{swarm_id}/permissions/clear")
async def clear_swarm_permissions(swarm_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    payload = await request.json()
    reason = str(payload.get("reason") or "swarm permissions cleared")
    cleared = clear_pending_swarm_permissions(swarm_id, reason=reason)
    save_state()
    return {"success": True, "swarmId": swarm_id, "cleared": cleared, "pendingRequestCount": len(pending_swarm_permission_records(swarm_id))}


@app.post("/api/swarm/permission/{request_id}")
async def swarm_permission(request_id: str, request: Request) -> dict[str, Any]:
    ensure_swarm_enabled()
    payload = await request.json()
    approved = approved_from_permission_payload(payload)
    resolution = {
        "requestId": request_id,
        "decision": "allow" if approved else "deny",
        "approved": approved,
        "reason": str(payload.get("reason") or ""),
        "updatedAt": utc_now(),
    }
    result = resolve_swarm_permission_request(request_id, resolution)
    save_state()
    return result


DANGEROUS_COMMAND_TOKENS = {
    "rm",
    "del",
    "erase",
    "format",
    "shutdown",
    "reboot",
    "reg",
    "diskpart",
    "mkfs",
}


def normalize_command(command: Any) -> list[str]:
    if isinstance(command, list):
        return [str(part) for part in command if str(part)]
    return shlex.split(str(command or ""), posix=False)


def command_is_blocked(parts: list[str]) -> str | None:
    lowered = [part.lower() for part in parts]
    if not lowered:
        return "empty command"
    executable = Path(lowered[0]).name
    if executable in DANGEROUS_COMMAND_TOKENS:
        return f"blocked dangerous command: {executable}"
    joined = " ".join(lowered)
    if re.search(r"\b(remove-item|rd|rmdir)\b.*\b(-recurse|/s)\b", joined):
        return "blocked recursive delete"
    return None


@app.post("/api/tasks")
async def create_task(request: Request) -> dict[str, Any]:
    payload = await request.json()
    task_id = str(payload.get("id") or new_id("task"))
    now = utc_now()
    task = {
        "id": task_id,
        "title": payload.get("title") or payload.get("summary") or "Task",
        "description": payload.get("description") or "",
        "type": payload.get("type") or "general",
        "status": payload.get("status") or "pending",
        "priority": payload.get("priority") or "normal",
        "assignee": payload.get("assignee"),
        "output": [],
        "createdAt": now,
        "updatedAt": now,
    }
    STATE.setdefault("tasks", {})[task_id] = task
    save_state()
    return task


@app.get("/api/tasks")
async def list_tasks(status: str | None = None) -> dict[str, Any]:
    tasks = list(STATE.setdefault("tasks", {}).values())
    if status:
        tasks = [task for task in tasks if task.get("status") == status]
    return {"tasks": sorted(tasks, key=lambda item: item.get("updatedAt", ""), reverse=True)}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    task = STATE.setdefault("tasks", {}).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.patch("/api/tasks/{task_id}")
async def update_task(task_id: str, request: Request) -> dict[str, Any]:
    task = STATE.setdefault("tasks", {}).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    payload = await request.json()
    for key in ["title", "description", "type", "status", "priority", "assignee", "progress"]:
        if key in payload:
            task[key] = payload[key]
    task["updatedAt"] = utc_now()
    save_state()
    return task


@app.post("/api/tasks/{task_id}/output")
async def append_task_output(task_id: str, request: Request) -> dict[str, Any]:
    task = STATE.setdefault("tasks", {}).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    payload = await request.json()
    entry = {"id": new_id("task-output"), "content": payload.get("content") or "", "kind": payload.get("kind") or "log", "createdAt": utc_now()}
    task.setdefault("output", []).append(entry)
    task["updatedAt"] = utc_now()
    save_state()
    return {"success": True, "entry": entry, "task": task}


@app.post("/api/tasks/{task_id}/stop")
async def stop_task(task_id: str) -> dict[str, Any]:
    task = STATE.setdefault("tasks", {}).get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task["status"] = "stopped"
    task["updatedAt"] = utc_now()
    save_state()
    return {"success": True, "task": task}


def agent_status_from_task(task: dict[str, Any]) -> dict[str, Any]:
    status = str(task.get("status") or "").lower()
    normalized = "running" if status == "running" else ("completed" if status == "completed" else ("failed" if status == "failed" else status))
    return {
        "agentId": task.get("agentId"),
        "taskId": task.get("taskId"),
        "sessionId": task.get("sessionId"),
        "agentType": task.get("agentType") or str(task.get("taskType") or "agent:").split(":", 1)[-1],
        "model": task.get("model"),
        "isolation": task.get("isolation"),
        "fork": task.get("fork"),
        "teamName": task.get("teamName"),
        "description": task.get("agentDescription"),
        "prompt": task.get("prompt") or task.get("description"),
        "outputFile": task.get("outputFile"),
        "status": normalized,
        "startedAt": task.get("createdAt"),
        "completedAt": task.get("updatedAt") if normalized in {"completed", "failed"} else None,
        "error": task.get("error"),
        "childSessionId": task.get("childSessionId"),
        "agentHierarchy": task.get("agentHierarchy"),
    }


def background_agent_tasks(session_id: str | None = None, agent_ids: list[str] | None = None) -> list[dict[str, Any]]:
    id_filter = {str(agent_id) for agent_id in agent_ids or [] if str(agent_id)}
    tasks = [
        task
        for task in TOOL_REGISTRY._tasks.values()
        if str(task.get("taskType") or "").startswith("agent:")
        and (not session_id or task.get("sessionId") == session_id)
        and (not id_filter or str(task.get("agentId") or "") in id_filter)
    ]
    tasks.sort(key=lambda item: float(item.get("createdAt") or 0))
    return tasks


def active_background_agent_ids(session_id: str, agent_ids: list[str] | None = None) -> list[str]:
    return [
        str(task.get("agentId"))
        for task in background_agent_tasks(session_id, agent_ids)
        if task.get("agentId") and str(task.get("status") or "").upper() == "RUNNING"
    ]


def publish_background_agent_event(agent_id: str, event_type: str, data: dict[str, Any] | None = None) -> dict[str, Any] | None:
    task = next((item for item in TOOL_REGISTRY._tasks.values() if item.get("agentId") == agent_id), None)
    if not task:
        return None
    session_id = str(task.get("sessionId") or "default")
    return WS_SESSION_MANAGER.publish_event(
        session_id,
        "task_update",
        {"agentId": agent_id, "eventType": event_type, "data": {"agentId": agent_id, **(data or {})}},
    )


TOOL_REGISTRY.set_background_agent_event_publisher(publish_background_agent_event)


async def await_background_agents(session_id: str, timeout_ms: int = 900_000, poll_ms: int = 50, agent_ids: list[str] | None = None) -> dict[str, Any]:
    target_ids = list(dict.fromkeys(str(agent_id) for agent_id in agent_ids or [] if str(agent_id)))
    deadline = time.time() + max(0, timeout_ms) / 1000
    while True:
        running = active_background_agent_ids(session_id, target_ids or None)
        if not running:
            agents = [agent_status_from_task(task) for task in background_agent_tasks(session_id, target_ids or None)]
            return {"completed": True, "activeAgentIds": [], "agentIds": target_ids, "agents": agents}
        if QUERY_ABORTS.is_aborted(session_id) or time.time() >= deadline:
            agents = [agent_status_from_task(task) for task in background_agent_tasks(session_id, target_ids or None)]
            return {"completed": False, "activeAgentIds": running, "agentIds": target_ids, "agents": agents}
        await asyncio.sleep(max(10, poll_ms) / 1000)


def remove_background_agent_session(session_id: str, delete_output_files: bool = False) -> int:
    removed = 0
    for task_id, task in list(TOOL_REGISTRY._tasks.items()):
        if not str(task.get("taskType") or "").startswith("agent:") or task.get("sessionId") != session_id:
            continue
        if delete_output_files:
            delete_background_agent_output(task)
        TOOL_REGISTRY._tasks.pop(task_id, None)
        removed += 1
    return removed


def delete_background_agent_output(task: dict[str, Any]) -> bool:
    output_file = str(task.get("outputFile") or "")
    if not output_file:
        return False
    try:
        path = Path(output_file)
        if not path.exists():
            return False
        path.unlink()
        return True
    except OSError:
        return False


def cleanup_background_agents(max_age_minutes: int = 30, delete_output_files: bool = True) -> dict[str, Any]:
    cutoff = time.time() - max(0, max_age_minutes) * 60
    removed: list[str] = []
    deleted_files = 0
    for task_id, task in list(TOOL_REGISTRY._tasks.items()):
        if not str(task.get("taskType") or "").startswith("agent:"):
            continue
        if str(task.get("status") or "").upper() == "RUNNING":
            continue
        if float(task.get("updatedAt") or task.get("createdAt") or 0) > cutoff:
            continue
        if delete_output_files and delete_background_agent_output(task):
            deleted_files += 1
        TOOL_REGISTRY._tasks.pop(task_id, None)
        removed.append(str(task.get("agentId") or task_id))
    return {"removed": len(removed), "removedAgentIds": removed, "deletedOutputFiles": deleted_files}


def format_background_agent_results(agents: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for agent in agents:
        output = ""
        output_file = str(agent.get("outputFile") or "")
        if output_file:
            try:
                output = Path(output_file).read_text(encoding="utf-8")
            except OSError:
                output = ""
        if not output:
            task = next((item for item in TOOL_REGISTRY._tasks.values() if item.get("agentId") == agent.get("agentId")), {})
            output = str(task.get("output") or task.get("error") or "")
        if len(output) > 2000:
            output = output[:2000] + "\n...[truncated]"
        lines.append(
            "\n".join(
                [
                    f"Agent: {agent.get('agentId')}",
                    f"Status: {agent.get('status')}",
                    f"Output file: {agent.get('outputFile') or ''}",
                    f"Result: {output}",
                ]
            )
        )
    return "\n\n".join(lines)


def agent_snapshot_path(agent_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", agent_id).strip(".-") or "agent"
    return AGENT_SNAPSHOT_DIR / f"{safe_id}.json"


def save_agent_snapshot_record(snapshot: dict[str, Any]) -> dict[str, Any]:
    agent_id = str(snapshot.get("agentId") or snapshot.get("agent_id") or "")
    if not agent_id:
        raise HTTPException(status_code=400, detail="Missing agentId")
    parent_session_id = str(snapshot.get("parentSessionId") or snapshot.get("parent_session_id") or snapshot.get("sessionId") or "")
    messages = snapshot.get("messages")
    if not isinstance(messages, list) and parent_session_id in STATE.setdefault("sessions", {}):
        messages = STATE["sessions"][parent_session_id].get("messages", [])
    if not isinstance(messages, list):
        messages = []
    record = {
        "agentId": agent_id,
        "taskDescription": str(snapshot.get("taskDescription") or snapshot.get("task_description") or snapshot.get("prompt") or ""),
        "messages": messages,
        "createdAt": snapshot.get("createdAt") or utc_now(),
        "parentSessionId": parent_session_id,
        "nestingDepth": int(snapshot.get("nestingDepth") or snapshot.get("nesting_depth") or 0),
        "workingDirectory": str(snapshot.get("workingDirectory") or snapshot.get("working_directory") or "."),
        "model": snapshot.get("model") or STATE["config"].get("defaultModel"),
    }
    encoded = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
    if len(encoded) > MAX_AGENT_SNAPSHOT_SIZE:
        raise HTTPException(status_code=413, detail="Agent snapshot exceeds 10MB limit")
    AGENT_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    target = agent_snapshot_path(agent_id)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_bytes(encoded)
    tmp.replace(target)
    return {**record, "path": str(target), "sizeBytes": len(encoded)}


def load_agent_snapshot_record(agent_id: str) -> dict[str, Any] | None:
    path = agent_snapshot_path(agent_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_agent_snapshot_ids() -> list[str]:
    if not AGENT_SNAPSHOT_DIR.is_dir():
        return []
    return sorted(path.stem for path in AGENT_SNAPSHOT_DIR.glob("*.json"))


def delete_agent_snapshot_record(agent_id: str) -> bool:
    path = agent_snapshot_path(agent_id)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def purge_expired_agent_snapshots(max_age_hours: int = 24) -> dict[str, Any]:
    if not AGENT_SNAPSHOT_DIR.is_dir():
        return {"purged": 0, "agentIds": []}
    cutoff = time.time() - max(0, max_age_hours) * 3600
    purged: list[str] = []
    for path in AGENT_SNAPSHOT_DIR.glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                purged.append(path.stem)
        except OSError:
            continue
    return {"purged": len(purged), "agentIds": purged}


@app.get("/api/agents/background")
async def list_background_agents(sessionId: str | None = None, activeOnly: bool = True) -> dict[str, Any]:
    tasks = background_agent_tasks(sessionId)
    if activeOnly:
        tasks = [task for task in tasks if str(task.get("status") or "").upper() == "RUNNING"]
    agents = [agent_status_from_task(task) for task in tasks]
    return {"agents": agents, "count": len(agents), "activeAgentIds": [item["agentId"] for item in agents if item.get("status") == "running"]}


@app.get("/api/agents/background/active-ids")
async def get_active_background_agent_ids(sessionId: str) -> dict[str, Any]:
    active_ids = active_background_agent_ids(sessionId)
    return {"sessionId": sessionId, "activeAgentIds": active_ids, "count": len(active_ids)}


@app.post("/api/agents/background/await")
async def await_background_agent_endpoint(request: Request) -> dict[str, Any]:
    payload = await request.json()
    session_id = str(payload.get("sessionId") or payload.get("session_id") or "")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing sessionId")
    timeout_ms = int(payload.get("timeoutMs") or payload.get("timeout_ms") or 900_000)
    agent_ids = payload.get("agentIds") or payload.get("agent_ids")
    return await await_background_agents(session_id, timeout_ms=timeout_ms, agent_ids=agent_ids if isinstance(agent_ids, list) else None)


@app.delete("/api/agents/background/session/{session_id}")
async def remove_background_agent_session_endpoint(session_id: str, deleteOutputFiles: bool = False) -> dict[str, Any]:
    removed = remove_background_agent_session(session_id, delete_output_files=deleteOutputFiles)
    return {"sessionId": session_id, "removed": removed, "success": True}


@app.post("/api/agents/background/cleanup")
async def cleanup_background_agent_endpoint(request: Request) -> dict[str, Any]:
    payload = await request.json()
    result = cleanup_background_agents(
        max_age_minutes=int(payload.get("maxAgeMinutes") or payload.get("max_age_minutes") or 30),
        delete_output_files=bool(payload.get("deleteOutputFiles", payload.get("delete_output_files", True))),
    )
    return {"success": True, **result}


@app.get("/api/agents/snapshots")
async def list_agent_snapshots() -> dict[str, Any]:
    snapshots = []
    for agent_id in list_agent_snapshot_ids():
        record = load_agent_snapshot_record(agent_id)
        if record:
            snapshots.append(
                {
                    "agentId": record.get("agentId") or agent_id,
                    "taskDescription": record.get("taskDescription"),
                    "parentSessionId": record.get("parentSessionId"),
                    "nestingDepth": record.get("nestingDepth"),
                    "workingDirectory": record.get("workingDirectory"),
                    "model": record.get("model"),
                    "messageCount": len(record.get("messages") or []),
                    "createdAt": record.get("createdAt"),
                }
            )
    return {"snapshots": snapshots, "count": len(snapshots)}


@app.post("/api/agents/snapshots")
async def save_agent_snapshot(request: Request) -> dict[str, Any]:
    snapshot = save_agent_snapshot_record(await request.json())
    return {"success": True, "snapshot": {k: v for k, v in snapshot.items() if k != "messages"}, "messageCount": len(snapshot.get("messages") or [])}


@app.post("/api/agents/snapshots/purge")
async def purge_agent_snapshots(request: Request) -> dict[str, Any]:
    payload = await request.json()
    result = purge_expired_agent_snapshots(int(payload.get("maxAgeHours") or payload.get("max_age_hours") or 24))
    return {"success": True, **result}


@app.get("/api/agents/snapshots/{agent_id}")
async def get_agent_snapshot(agent_id: str) -> dict[str, Any]:
    snapshot = load_agent_snapshot_record(agent_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Agent snapshot not found")
    return {"snapshot": snapshot}


@app.delete("/api/agents/snapshots/{agent_id}")
async def delete_agent_snapshot(agent_id: str) -> dict[str, Any]:
    return {"success": delete_agent_snapshot_record(agent_id), "agentId": agent_id}


@app.post("/api/agents/{agent_id}/resume")
async def resume_agent(agent_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    snapshot = load_agent_snapshot_record(agent_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Agent snapshot not found")
    resume_session_id = f"resumed-agent-{agent_id}"
    session = get_or_create_session(resume_session_id)
    session["title"] = session.get("title") or f"Resumed agent {agent_id}"
    session["messages"] = [json.loads(json.dumps(message, ensure_ascii=False)) for message in snapshot.get("messages") or []]
    session["parentSessionId"] = snapshot.get("parentSessionId")
    session["agentType"] = "resumed"
    session["agentHierarchy"] = f"resumed-agent-{agent_id}"
    session["workingDirectory"] = snapshot.get("workingDirectory") or "."
    if snapshot.get("model"):
        session["model"] = snapshot.get("model")
    additional = str(payload.get("additionalContext") or payload.get("additional_context") or "")
    prompt = f"[Resumed] Continuing previous task: {snapshot.get('taskDescription') or ''}"
    if additional.strip():
        prompt += "\n\nAdditional context: " + additional
    result = await run_query_payload(
        {
            "sessionId": resume_session_id,
            "prompt": prompt,
            "model": session.get("model"),
            "workingDirectory": session.get("workingDirectory"),
            "collapseContext": False,
            "toolCalls": payload.get("toolCalls") if isinstance(payload.get("toolCalls"), list) else [],
        },
        require_existing_session=True,
    )
    delete_agent_snapshot_record(agent_id)
    return {"success": True, "agentId": agent_id, "sessionId": resume_session_id, "result": result}


@app.get("/api/agents/background/{agent_id}")
async def get_background_agent(agent_id: str) -> dict[str, Any]:
    for task in TOOL_REGISTRY._tasks.values():
        if task.get("agentId") == agent_id:
            return {"agent": agent_status_from_task(task)}
    raise HTTPException(status_code=404, detail="Agent not found")


@app.post("/api/cron/tasks")
async def create_cron_task(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        task = CRON_SERVICE.add_task(
            str(payload.get("cron") or ""),
            str(payload.get("prompt") or ""),
            recurring=bool(payload.get("recurring", True)),
            durable=bool(payload.get("durable", False)),
            agent_id=str(payload.get("agentId") or payload.get("agent_id") or "") or None,
        )
    except CronValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "task": task.to_dict(), "total": CRON_SERVICE.task_count()}


@app.get("/api/cron/tasks")
async def list_cron_tasks() -> dict[str, Any]:
    tasks = [task.to_dict() for task in CRON_SERVICE.list_all()]
    return {"success": True, "tasks": tasks, "total": len(tasks)}


@app.delete("/api/cron/tasks/{task_id}")
async def delete_cron_task(task_id: str) -> dict[str, Any]:
    removed = CRON_SERVICE.remove(task_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Scheduled task not found")
    return {"success": True, "deleted": removed.to_dict(), "remaining": CRON_SERVICE.task_count()}


@app.post("/api/sandbox/execute")
async def sandbox_execute(request: Request) -> dict[str, Any]:
    payload = await request.json()
    parts = normalize_command(payload.get("command") or payload.get("args"))
    blocked = command_is_blocked(parts)
    run_id = new_id("sandbox")
    cwd = safe_workspace_path(payload.get("cwd") or payload.get("workingDirectory") or ".")
    timeout = min(max(int(payload.get("timeoutMs") or payload.get("timeout") or 30_000), 100), 120_000) / 1000
    record = {
        "id": run_id,
        "command": parts,
        "cwd": snapshot_rel_path(cwd),
        "startedAt": utc_now(),
        "status": "blocked" if blocked else "running",
        "blockedReason": blocked,
    }
    if blocked:
        STATE.setdefault("sandboxRuns", {})[run_id] = record
        save_state()
        return {"success": False, "blocked": True, **record}
    try:
        completed = subprocess.run(parts, cwd=cwd, text=True, capture_output=True, timeout=timeout)
        record.update(
            {
                "status": "completed" if completed.returncode == 0 else "failed",
                "exitCode": completed.returncode,
                "stdout": completed.stdout[-100_000:],
                "stderr": completed.stderr[-100_000:],
                "finishedAt": utc_now(),
            }
        )
    except subprocess.TimeoutExpired as exc:
        record.update(
            {
                "status": "timeout",
                "exitCode": 124,
                "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                "stderr": "Command timed out",
                "finishedAt": utc_now(),
            }
        )
    except OSError as exc:
        record.update({"status": "failed", "exitCode": 127, "stdout": "", "stderr": str(exc), "finishedAt": utc_now()})
    STATE.setdefault("sandboxRuns", {})[run_id] = record
    save_state()
    return {"success": record["status"] == "completed", "blocked": False, **record}


@app.get("/api/sandbox/runs")
async def list_sandbox_runs() -> dict[str, Any]:
    return {"runs": sorted(STATE.setdefault("sandboxRuns", {}).values(), key=lambda item: item.get("startedAt", ""), reverse=True)}


@app.post("/api/cost/record")
async def record_cost(request: Request) -> dict[str, Any]:
    payload = await request.json()
    event = {
        "id": new_id("cost"),
        "sessionId": payload.get("sessionId"),
        "model": payload.get("model") or STATE.get("config", {}).get("defaultModel"),
        "inputTokens": int(payload.get("inputTokens") or 0),
        "outputTokens": int(payload.get("outputTokens") or 0),
        "costUsd": float(payload.get("costUsd") or 0),
        "createdAt": utc_now(),
    }
    STATE.setdefault("costEvents", []).append(event)
    del STATE["costEvents"][:-1000]
    save_state()
    return {"success": True, "event": event}


@app.get("/api/cost")
async def cost_summary(sessionId: str | None = None) -> dict[str, Any]:
    events = STATE.setdefault("costEvents", [])
    if sessionId:
        events = [event for event in events if event.get("sessionId") == sessionId]
    return {
        "events": events,
        "totalCostUsd": round(sum(float(event.get("costUsd") or 0) for event in events), 8),
        "inputTokens": sum(int(event.get("inputTokens") or 0) for event in events),
        "outputTokens": sum(int(event.get("outputTokens") or 0) for event in events),
    }


@app.post("/api/anomalies")
async def create_anomaly(request: Request) -> dict[str, Any]:
    payload = await request.json()
    anomaly = {
        "id": payload.get("id") or new_id("anomaly"),
        "type": payload.get("type") or "runtime",
        "severity": payload.get("severity") or "medium",
        "message": payload.get("message") or "",
        "source": payload.get("source") or "python",
        "status": payload.get("status") or "open",
        "createdAt": utc_now(),
        "metadata": payload.get("metadata") or {},
    }
    STATE.setdefault("anomalies", []).append(anomaly)
    save_state()
    return anomaly


@app.get("/api/anomalies")
async def list_anomalies(status: str | None = None) -> dict[str, Any]:
    anomalies = STATE.setdefault("anomalies", [])
    if status:
        anomalies = [item for item in anomalies if item.get("status") == status]
    return {"anomalies": sorted(anomalies, key=lambda item: item.get("createdAt", ""), reverse=True)}


@app.patch("/api/anomalies/{anomaly_id}")
async def update_anomaly(anomaly_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    for item in STATE.setdefault("anomalies", []):
        if item.get("id") == anomaly_id:
            item.update({key: value for key, value in payload.items() if key in {"status", "severity", "message"}})
            item["updatedAt"] = utc_now()
            save_state()
            return item
    raise HTTPException(status_code=404, detail="Anomaly not found")


@app.get("/api/bridge/status")
async def bridge_status() -> dict[str, Any]:
    bridge = STATE.setdefault("bridge", {"devices": [], "messages": []})
    return {"enabled": True, "deviceCount": len(bridge.get("devices", [])), "messageCount": len(bridge.get("messages", []))}


@app.get("/api/bridge/devices")
async def bridge_devices() -> dict[str, Any]:
    return {"devices": STATE.setdefault("bridge", {"devices": [], "messages": []}).setdefault("devices", [])}


@app.post("/api/bridge/devices")
async def pair_bridge_device(request: Request) -> dict[str, Any]:
    payload = await request.json()
    device = {
        "id": payload.get("id") or new_id("device"),
        "name": payload.get("name") or "Trusted device",
        "trusted": bool(payload.get("trusted", True)),
        "pairedAt": utc_now(),
        "lastSeenAt": utc_now(),
    }
    STATE.setdefault("bridge", {"devices": [], "messages": []}).setdefault("devices", []).append(device)
    save_state()
    return device


@app.post("/api/bridge/messages")
async def bridge_message(request: Request) -> dict[str, Any]:
    payload = await request.json()
    message = {"id": new_id("bridge-msg"), "deviceId": payload.get("deviceId"), "type": payload.get("type") or "message", "payload": payload.get("payload") or {}, "createdAt": utc_now()}
    STATE.setdefault("bridge", {"devices": [], "messages": []}).setdefault("messages", []).append(message)
    save_state()
    return {"success": True, "message": message}


@app.get("/api/keybindings")
async def list_keybindings() -> dict[str, Any]:
    bindings = STATE.setdefault("keybindings", {})
    if not bindings:
        bindings.update({"ctrl+k": "command_palette", "ctrl+enter": "send_message", "esc": "cancel"})
    return {"keybindings": bindings}


@app.put("/api/keybindings")
async def replace_keybindings(request: Request) -> dict[str, Any]:
    payload = await request.json()
    STATE["keybindings"] = {str(key).lower(): str(value) for key, value in (payload.get("keybindings") or payload).items()}
    save_state()
    return {"keybindings": STATE["keybindings"]}


@app.post("/api/keybindings/resolve")
async def resolve_keybinding(request: Request) -> dict[str, Any]:
    payload = await request.json()
    key = str(payload.get("key") or payload.get("keystroke") or "").lower()
    bindings = (await list_keybindings())["keybindings"]
    action = bindings.get(key)
    return {"key": key, "action": action, "handled": action is not None}


def python_symbols(path: Path, root: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text)
    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append({"name": node.name, "kind": "class" if isinstance(node, ast.ClassDef) else "function", "filePath": path.relative_to(root).as_posix(), "line": node.lineno, "column": node.col_offset})
    return sorted(symbols, key=lambda item: (item["line"], item["name"]))


@app.get("/api/lsp/symbols")
async def lsp_symbols(path: str) -> dict[str, Any]:
    file_path = safe_workspace_path(path)
    server = LSP_MANAGER.get_server_for_file(str(file_path))
    return {"symbols": document_symbols(ROOT, file_path), "server": server.config.name if server else None}


@app.get("/api/lsp/references")
async def lsp_references(symbol: str, path: str | None = None, limit: int = 100) -> dict[str, Any]:
    return {"references": references(ROOT, symbol, path, limit)}


@app.get("/api/lsp/servers")
async def lsp_servers() -> dict[str, Any]:
    return LSP_MANAGER.status()


@app.post("/api/lsp/open")
async def lsp_open(request: Request) -> dict[str, Any]:
    payload = await request.json()
    path = str(payload.get("path") or payload.get("filePath") or "")
    if not path:
        raise HTTPException(status_code=400, detail="Missing path")
    server = LSP_MANAGER.get_server_for_file(path)
    if not server:
        return {"success": False, "error": "No LSP server available", "path": path}
    LSP_MANAGER.open_file(path)
    return {"success": True, "path": path, "server": server.to_dict(), "open": True}


@app.post("/api/lsp/close")
async def lsp_close(request: Request) -> dict[str, Any]:
    payload = await request.json()
    path = str(payload.get("path") or payload.get("filePath") or "")
    LSP_MANAGER.close_file(path)
    return {"success": True, "path": path, "open": False}


@app.get("/api/lsp/workspace-symbols")
async def lsp_workspace_symbols(query: str, limit: int = 100) -> dict[str, Any]:
    return {"symbols": workspace_symbols(ROOT, query, limit)}


@app.get("/api/lsp/hover")
async def lsp_hover(path: str, line: int, character: int = 0) -> dict[str, Any]:
    return {"hover": hover(ROOT, path, line, character)}


@app.get("/api/lsp/definition")
async def lsp_definition(path: str, line: int, character: int = 0) -> dict[str, Any]:
    return go_to_definition(ROOT, path, line, character)


@app.get("/api/lsp/call-hierarchy")
async def lsp_call_hierarchy(symbol: str, path: str | None = None) -> dict[str, Any]:
    return call_hierarchy(ROOT, symbol, path)


@app.post("/api/verify/journey")
async def verify_journey(request: Request) -> dict[str, Any]:
    payload = await request.json()
    started = time.time()
    gate = CapabilityGate(payload.get("featureFlags"), payload.get("capabilities"))
    if not gate.verify_enabled():
        result = {
            "success": False,
            "verdict": "unavailable",
            "results": [],
            "stepResults": [],
            "errorMessage": "Runtime verification capability unavailable",
            "durationMs": round((time.time() - started) * 1000),
            "timestamp": utc_now(),
        }
    else:
        journey = verifier_for(payload, gate).verify(payload)
        detail = journey.to_dict()
        result = {
            "success": journey.passed,
            "verdict": "passed" if journey.passed else journey.verdict,
            "result": detail,
            "stepResults": detail["stepResults"],
            "results": [
                {
                    "index": item["stepIndex"] + 1,
                    "name": item["action"],
                    "status": "passed" if item["ok"] else "failed",
                    "evidence": item["evidence"],
                    "durationMs": item["durationMs"],
                    "error": item["error"],
                }
                for item in detail["stepResults"]
            ],
            "errorMessage": detail["errorMessage"],
            "durationMs": round((time.time() - started) * 1000),
            "timestamp": utc_now(),
        }
    result_id = new_id("journey")
    STATE.setdefault("journeyResults", {})[result_id] = result
    save_state()
    return {"id": result_id, **result}


@app.post("/api/correction/parse")
async def correction_parse(request: Request) -> dict[str, Any]:
    payload = await request.json()
    output = str(payload.get("output") or payload.get("log") or payload.get("stderr") or "")
    report = build_correction_instruction(output)
    report_id = new_id("correction")
    record = {**report, "id": report_id, "createdAt": utc_now()}
    STATE.setdefault("correctionReports", {})[report_id] = record
    save_state()
    return record


@app.post("/api/correction/detect")
async def correction_detect(request: Request) -> dict[str, Any]:
    payload = await request.json()
    output = str(payload.get("output") or payload.get("log") or payload.get("stderr") or "")
    previous_attempts = int(payload.get("previousAttempts") or payload.get("previous_attempts") or 0)
    max_attempts = int(payload.get("maxAttempts") or payload.get("max_attempts") or 3)
    loop = SelfCorrectionLoop(max_attempts=max_attempts)
    instruction = loop.detect_and_prepare(output, previous_attempts=previous_attempts)
    return {"instruction": instruction.to_dict() if instruction else None}


@app.post("/api/correction/should-abort")
async def correction_should_abort(request: Request) -> dict[str, Any]:
    payload = await request.json()
    loop = SelfCorrectionLoop()
    return {
        "abort": loop.should_abort(
            str(payload.get("newOutput") or payload.get("new_output") or ""),
            str(payload.get("previousOutput") or payload.get("previous_output") or ""),
        )
    }


@app.get("/api/correction/reports/{report_id}")
async def correction_report(report_id: str) -> dict[str, Any]:
    report = STATE.setdefault("correctionReports", {}).get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Correction report not found")
    return report


async def proxy_python_service(request: Request, path: str) -> Response:
    url = f"{PYTHON_SERVICE_URL}/api/{path}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            proxied = await client.request(
                request.method,
                url,
                params=request.query_params,
                content=await request.body(),
                headers={k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}},
            )
        return Response(
            content=proxied.content,
            status_code=proxied.status_code,
            media_type=proxied.headers.get("content-type"),
        )
    except Exception as exc:
        return JSONResponse(
            {
                "success": False,
                "proxied": True,
                "target": url,
                "error": str(exc),
            },
            status_code=502,
        )


@app.api_route("/api/git/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_git(request: Request, path: str) -> Response:
    return await proxy_python_service(request, f"git/{path}")


@app.api_route("/api/code-quality/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_code_quality(request: Request, path: str) -> Response:
    return await proxy_python_service(request, f"code-quality/{path}")


@app.api_route("/api/analysis/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_analysis(request: Request, path: str) -> Response:
    return await proxy_python_service(request, f"analysis/{path}")


@app.api_route("/api/files/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_files(request: Request, path: str) -> Response:
    return await proxy_python_service(request, f"files/{path}")


@app.api_route("/api/browser/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_browser(request: Request, path: str) -> Response:
    return await proxy_python_service(request, f"browser/{path}")


@app.api_route("/api/tokenizer/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_tokenizer(request: Request, path: str) -> Response:
    return await proxy_python_service(request, f"tokenizer/{path}")


@app.api_route("/api/v1/tokens/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_token_estimator(request: Request, path: str) -> Response:
    return await proxy_python_service(request, f"v1/tokens/{path}")


@app.api_route("/api/http/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_http_api(request: Request, path: str) -> Response:
    return await proxy_python_service(request, f"http/{path}")


@app.post("/mcp")
async def mcp_entrypoint(request: Request) -> dict[str, Any]:
    payload = await request.json()
    method = payload.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "codeagent-python-backend", "version": app.version}, "capabilities": {}}
    elif method == "tools/list":
        result = {"tools": []}
    else:
        result = {}
    return {"jsonrpc": "2.0", "id": payload.get("id"), "result": result}


@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def api_fallback(path: str, request: Request) -> dict[str, Any]:
    body: Any = None
    if request.method in {"POST", "PUT", "PATCH"}:
        try:
            body = await request.json()
        except Exception:
            body = None
    return {
        "success": True,
        "status": "not_implemented",
        "path": f"/api/{path}",
        "method": request.method,
        "data": [] if request.method == "GET" else body,
    }


def sockjs_frame(payload: str) -> str:
    return "a" + json.dumps([payload])


def parse_sockjs_messages(raw: str) -> list[str]:
    if raw == "h":
        return []
    data = raw
    if data.startswith("a["):
        data = data[1:]
    try:
        parsed = json.loads(data)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [raw]


def parse_stomp(frame: str) -> tuple[str, dict[str, str], str]:
    frame = frame.rstrip("\x00")
    head, _, body = frame.partition("\n\n")
    lines = head.splitlines()
    if not lines:
        return "", {}, body
    command = lines[0].strip()
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
    return command, headers, body


def build_stomp(command: str, headers: dict[str, str] | None = None, body: Any = "") -> str:
    if isinstance(body, (dict, list)):
        body_text = json.dumps(body, ensure_ascii=False)
    else:
        body_text = str(body)
    header_lines = [command]
    for key, value in (headers or {}).items():
        header_lines.append(f"{key}:{value}")
    return "\n".join(header_lines) + "\n\n" + body_text + "\x00"


async def send_user_message(ws: WebSocket, subscription_id: str, payload: dict[str, Any]) -> None:
    headers = {
        "subscription": subscription_id,
        "message-id": new_id("msg"),
        "destination": "/user/queue/messages",
        "content-type": "application/json",
    }
    await ws.send_text(sockjs_frame(build_stomp("MESSAGE", headers, {**payload, "ts": int(time.time() * 1000)})))


async def send_queued_user_message(ws: WebSocket, subscription_id: str, queued: dict[str, Any]) -> None:
    payload = queued.get("payload") if isinstance(queued.get("payload"), dict) else {}
    destination = str(queued.get("destination") or "/user/queue/messages")
    headers = {
        "subscription": subscription_id,
        "message-id": str(queued.get("id") or new_id("msg")),
        "destination": destination,
        "content-type": "application/json",
    }
    await ws.send_text(sockjs_frame(build_stomp("MESSAGE", headers, {**payload, "ts": int(time.time() * 1000)})))


async def handle_app_send(ws: WebSocket, destination: str, body: str, state: dict[str, Any]) -> None:
    try:
        payload = json.loads(body) if body else {}
    except Exception:
        payload = {}
    subscription_id = state.get("subscription_id", "sub-0")
    session_id = payload.get("sessionId") or state.get("session_id") or new_id("session")
    session = get_or_create_session(session_id)
    state["session_id"] = session_id
    WS_SESSION_MANAGER.refresh_activity(session_id)

    if destination == "/app/bind-session":
        principal = str(payload.get("principal") or state.get("principal") or f"ws-{session_id}")
        state["principal"] = principal
        WS_SESSION_MANAGER.bind_session(principal, session_id)
        await send_user_message(
            ws,
            subscription_id,
            {
                "type": "session_restored",
                "messages": session.get("messages", []),
                "activities": STATE["activities"].get(session_id, [])[-50:],
                "metadata": {"sessionId": session_id, "model": session.get("model"), "status": "idle"},
                "hasMore": False,
            },
        )
        return

    if destination == "/app/ping":
        await send_user_message(ws, subscription_id, {"type": "pong", "timestamp": int(time.time() * 1000)})
        return

    if destination == "/app/model":
        session["model"] = payload.get("model") or session.get("model")
        session["updatedAt"] = utc_now()
        save_state()
        await send_user_message(ws, subscription_id, {"type": "model_changed", "model": session["model"]})
        return

    if destination == "/app/permission-mode":
        session["permissionMode"] = payload.get("mode", "default")
        session["updatedAt"] = utc_now()
        save_state()
        await send_user_message(ws, subscription_id, {"type": "permission_mode_changed", "mode": session["permissionMode"]})
        return

    if destination == "/app/permission":
        tool_use_id = payload.get("toolUseId") or payload.get("requestId") or new_id("perm")
        decision = payload.get("decision") or "deny"
        STATE.setdefault("permissionResponses", {})[tool_use_id] = {**payload, "decision": decision, "updatedAt": utc_now()}
        if payload.get("remember") and decision in {"allow", "allow_always", "deny"}:
            rule_decision = "allow" if decision in {"allow", "allow_always"} else "deny"
            STATE["permissions"].append(
                {
                    "id": new_id("rule"),
                    "tool": payload.get("toolName") or payload.get("tool") or "*",
                    "decision": rule_decision,
                    "scope": payload.get("scope") or "session",
                    "createdAt": utc_now(),
                }
            )
            rebuild_runtime_registries()
        save_state()
        await send_user_message(ws, subscription_id, {"type": "permission_processed", "toolUseId": tool_use_id, "decision": decision})
        return

    if destination == "/app/interrupt":
        session["status"] = "idle"
        QUERY_ABORTS.abort(session_id, "USER_INTERRUPT")
        save_state()
        await send_user_message(ws, subscription_id, {"type": "interrupt_ack", "reason": "USER_INTERRUPT"})
        return

    if destination == "/app/command":
        command = (payload.get("command") or "").lstrip("/")
        args = payload.get("args") or ""
        result = COMMAND_REGISTRY.execute(
            command,
            args,
            {"sessionId": session_id, "model": session.get("model"), "session": session},
        )
        if result.type == ResultType.TEXT and result.data and result.data.get("setModel"):
            session["model"] = str(result.data["setModel"])
            session["updatedAt"] = utc_now()
            save_state()
        await send_user_message(ws, subscription_id, result.to_ws_payload(command))
        return

    if destination == "/app/mcp":
        operation = payload.get("operation") or "list"
        server_id = payload.get("serverId") or payload.get("server") or payload.get("name")
        servers = STATE.setdefault("mcpServers", [])
        if operation == "connect":
            resolved_server_id = server_id or new_id("mcp")
            server = {"id": resolved_server_id, "name": resolved_server_id, "config": payload.get("config") or {}, "status": "connected", "updatedAt": utc_now()}
            servers[:] = [item for item in servers if item.get("id") != server["id"] and item.get("name") != server["name"]]
            servers.append(server)
            save_state()
            await send_user_message(ws, subscription_id, {"type": "notification", "key": "mcp-connect", "level": "info", "message": f"MCP server connected: {server['name']}", "timeout": 3000})
        elif operation == "disconnect":
            servers[:] = [item for item in servers if item.get("id") != server_id and item.get("name") != server_id]
            save_state()
            await send_user_message(ws, subscription_id, {"type": "notification", "key": "mcp-disconnect", "level": "info", "message": f"MCP server disconnected: {server_id}", "timeout": 3000})
        elif operation == "refresh":
            for server in servers:
                if server.get("id") == server_id or server.get("name") == server_id:
                    server["status"] = "refreshed"
                    server["updatedAt"] = utc_now()
            save_state()
            await send_user_message(ws, subscription_id, {"type": "notification", "key": "mcp-refresh", "level": "info", "message": f"MCP server refreshed: {server_id}", "timeout": 3000})
        elif operation == "list":
            await send_user_message(ws, subscription_id, {"type": "mcp_status", "servers": servers})
        else:
            await send_user_message(ws, subscription_id, {"type": "error", "code": "INVALID_MCP_OP", "message": f"Unknown MCP operation: {operation}", "retryable": False})
        return

    if destination == "/app/rewind":
        file_paths = payload.get("filePaths") or []
        result = rewind_files(session_id, str(payload.get("messageId") or ""), file_paths)
        save_state()
        await send_user_message(
            ws,
            subscription_id,
            {
                "type": "rewind_complete",
                "messageId": payload.get("messageId"),
                "success": result["success"],
                "restoredFiles": result["restoredFiles"],
                "skippedFiles": result["skippedFiles"],
                "errors": result["errors"],
                "files": file_paths,
            },
        )
        return

    if destination == "/app/elicitation":
        request_id = payload.get("requestId") or new_id("elicitation")
        STATE.setdefault("elicitations", {})[request_id] = {**payload, "sessionId": session_id, "updatedAt": utc_now()}
        save_state()
        await send_user_message(ws, subscription_id, {"type": "elicitation_resolved", "requestId": request_id})
        return

    if destination == "/app/activity-save":
        activity = {**payload}
        activity.setdefault("id", new_id("activity"))
        activity.setdefault("sessionId", session_id)
        activity.setdefault("createdAt", utc_now())
        STATE.setdefault("activities", {}).setdefault(session_id, []).append(activity)
        save_state()
        await send_user_message(ws, subscription_id, {"type": "activity_saved", "activity": activity})
        return

    if destination == "/app/activity-update":
        activity_id = payload.get("id")
        updated = None
        for activity in STATE.setdefault("activities", {}).setdefault(session_id, []):
            if activity.get("id") == activity_id:
                activity.update(payload)
                activity["updatedAt"] = utc_now()
                updated = activity
                break
        save_state()
        await send_user_message(ws, subscription_id, {"type": "activity_updated", "activity": updated or payload})
        return

    if destination == "/app/chat":
        async def live_send(event: dict[str, Any]) -> None:
            await send_user_message(ws, subscription_id, event)

        await run_query_payload(
            {
                **payload,
                "sessionId": session_id,
                "prompt": payload.get("text") or payload.get("prompt") or "",
            },
            live_send=live_send,
        )
        await send_user_message(ws, subscription_id, {"type": "session_list_updated"})
        return

    await send_user_message(ws, subscription_id, {"type": "notification", "key": "python-backend", "level": "info", "message": f"Handled {destination}", "timeout": 3000})


@app.get("/ws/info")
async def sockjs_info() -> dict[str, Any]:
    return {"websocket": True, "origins": ["*:*"], "cookie_needed": False, "entropy": random.randint(0, 2**31)}


@app.websocket("/ws/websocket")
async def ws_direct(websocket: WebSocket) -> None:
    await sockjs_websocket(websocket, "0", "direct")


@app.websocket("/ws/{server_id}/{session_id}/websocket")
async def sockjs_websocket(websocket: WebSocket, server_id: str, session_id: str) -> None:
    await websocket.accept()
    await websocket.send_text("o")
    state: dict[str, Any] = {"server_id": server_id, "sockjs_session_id": session_id, "subscription_id": "sub-0"}
    try:
        while True:
            raw = await websocket.receive_text()
            for stomp_frame in parse_sockjs_messages(raw):
                command, headers, body = parse_stomp(stomp_frame)
                if command in {"CONNECT", "STOMP"}:
                    state["session_id"] = headers.get("X-Session-Id") or state.get("session_id")
                    principal = headers.get("Authorization") or headers.get("login") or f"ws-{state.get('session_id') or session_id}"
                    state["principal"] = principal
                    WS_SESSION_MANAGER.connect(principal, f"{server_id}:{session_id}", state.get("session_id"))
                    connected = build_stomp("CONNECTED", {"version": "1.2", "heart-beat": "10000,10000"})
                    await websocket.send_text(sockjs_frame(connected))
                elif command == "SUBSCRIBE":
                    state["subscription_id"] = headers.get("id", "sub-0")
                    destination = headers.get("destination") or "/user/queue/messages"
                    state["ack_mode"] = headers.get("ack", "auto")
                    last_message_id = headers.get("last-message-id") or headers.get("lastMessageId") or headers.get("x-last-message-id")
                    bound_session_id = state.get("session_id")
                    if bound_session_id:
                        if last_message_id:
                            resume = WS_SESSION_MANAGER.resume_subscription(
                                str(bound_session_id),
                                state["subscription_id"],
                                destination,
                                ack_mode=state["ack_mode"],
                                since_id=last_message_id,
                            )
                            delivered = resume["messages"]
                        else:
                            WS_SESSION_MANAGER.register_subscription(str(bound_session_id), state["subscription_id"], destination, ack_mode=state["ack_mode"])
                            delivered = WS_SESSION_MANAGER.deliver_messages(str(bound_session_id), state["subscription_id"], ack_mode=state["ack_mode"])
                        for queued in delivered:
                            payload = queued.get("payload")
                            if isinstance(payload, dict):
                                await send_queued_user_message(websocket, state["subscription_id"], queued)
                elif command == "UNSUBSCRIBE":
                    subscription_id = headers.get("id") or state.get("subscription_id") or "sub-0"
                    if state.get("session_id"):
                        WS_SESSION_MANAGER.unsubscribe(str(state["session_id"]), str(subscription_id))
                elif command == "SEND":
                    await handle_app_send(websocket, headers.get("destination", ""), body, state)
                elif command == "ACK":
                    message_id = headers.get("message-id") or headers.get("id")
                    if state.get("session_id") and message_id:
                        WS_SESSION_MANAGER.ack_messages(str(state["session_id"]), [message_id])
                elif command == "NACK":
                    message_id = headers.get("message-id") or headers.get("id")
                    if state.get("session_id") and message_id:
                        WS_SESSION_MANAGER.nack_messages(str(state["session_id"]), [message_id], reason=headers.get("reason") or "nack")
                elif command == "DISCONNECT":
                    await websocket.close()
                    return
    except WebSocketDisconnect:
        return


@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str) -> Response:
    candidate = FRONTEND_DIST_DIR / path
    if FRONTEND_DIST_DIR.exists() and candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    if react_frontend_available():
        return FileResponse(react_index_file())
    return JSONResponse({"service": "codeagent-python-backend", "status": "ok"})
