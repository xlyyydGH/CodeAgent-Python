import json
import os
import subprocess
import sys
import threading
import time
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

import app as app_module  # noqa: E402
from app import QUERY_ABORTS, STATE, SWARM_PERMISSION_WAITERS, TOOL_REGISTRY, WORKER_MESSAGE_CAP, WS_SESSION_MANAGER, QueryLoopState, app, execute_agent_tool, execute_query_tools, merge_agent_worktree  # noqa: E402
from zhikun_py.mcp_runtime import JsonRpcMessage  # noqa: E402
from zhikun_py.permissions import PermissionDecision, PermissionRule  # noqa: E402
from zhikun_py.tools import MAX_AGENT_NESTING_DEPTH, MAX_CONCURRENT_AGENTS, MAX_CONCURRENT_AGENTS_PER_SESSION, Tool, ToolResult  # noqa: E402


client = TestClient(app)


class HttpMcpEchoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[str] = []

    def log_message(self, _format: str, *args) -> None:  # noqa: ANN002
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        message = JsonRpcMessage.parse(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(str(message.method))
        if message.method == "initialize":
            self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": message.id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "http-api-echo", "version": "0.1"},
                        "capabilities": {"tools": {}},
                    },
                }
            )
            return
        if message.method == "notifications/initialized":
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            return
        if message.method == "tools/list":
            self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": message.id,
                    "result": {
                        "tools": [
                            {
                                "name": "remote_echo",
                                "description": "Echo through HTTP MCP API.",
                                "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}},
                            }
                        ]
                    },
                }
            )
            return
        if message.method == "tools/call":
            args = (message.params or {}).get("arguments", {}) if isinstance(message.params, dict) else {}
            self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": message.id,
                    "result": {"content": [{"type": "text", "text": f"http-api:{args.get('message', '')}"}], "isError": False},
                }
            )
            return
        self._send_json({"jsonrpc": "2.0", "id": message.id, "error": {"code": -32601, "message": "unknown method"}})

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def run_http_mcp_echo_server() -> tuple[HTTPServer, str]:
    HttpMcpEchoHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), HttpMcpEchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/mcp"


def workspace() -> Path:
    root = BACKEND_DIR / ".test-workspace" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.05):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    return last


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["service"] == "zhikuncode-python-backend"


def test_react_frontend_root_when_built(tmp_path, monkeypatch) -> None:
    dist = tmp_path / "frontend-dist"
    dist.mkdir()
    (dist / "index.html").write_text('<!doctype html><div id="root"></div>', encoding="utf-8")
    monkeypatch.setattr(app_module, "FRONTEND_DIST_DIR", dist)

    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<div id="root">' in response.text


def test_python_ui_fallback_page() -> None:
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ZhikunCode Python" in response.text


def test_python_ui_chat_roundtrip() -> None:
    response = client.post("/ui/chat", data={"text": "hello from form"}, follow_redirects=False)
    assert response.status_code == 303
    follow = client.get(response.headers["location"])
    assert "hello from form" in follow.text
    assert "Assistant" in follow.text


def test_python_ui_page_opens_latest_chat_session(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER_XFYUN_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_ZHIPU_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_ZENMUX_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    text = f"root latest chat {uuid4().hex}"
    response = client.post("/ui/chat", data={"text": text}, follow_redirects=False)
    assert response.status_code == 303

    ui = client.get("/ui")
    assert text in ui.text
    assert "Assistant" in ui.text
    assert "Python backend is ready" in ui.text


def test_python_ui_chat_uses_quick_fallback_when_model_is_slow(monkeypatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9")

    async def slow_reply(_session: dict, _text: str) -> str:
        await asyncio.sleep(2)
        return "late model answer"

    monkeypatch.setattr("app.generate_llm_reply", slow_reply)
    started = time.time()
    response = client.post("/ui/chat", data={"text": "quick fallback please"}, follow_redirects=False)
    elapsed = time.time() - started

    assert response.status_code == 303
    assert elapsed < 1.0
    follow = client.get(response.headers["location"])
    assert "quick fallback please" in follow.text
    assert "late model answer" not in follow.text


def test_python_ui_replacement_pages() -> None:
    paths = ["/ui/realtime", "/ui/dashboard", "/ui/sessions", "/ui/tools", "/ui/tasks", "/ui/settings", "/ui/files", "/ui/activity", "/ui/verify", "/ui/mcp", "/ui/memory"]
    for path in paths:
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Python-native UI" in response.text
    assert "Realtime Workspace" in client.get("/ui/realtime").text

    created = client.post("/ui/memory", data={"title": "UI memory", "content": "saved from python ui"}, follow_redirects=False)
    assert created.status_code == 303
    memory_page = client.get(created.headers["location"])
    assert "saved from python ui" in memory_page.text
    memory_search = client.get("/ui/memory?q=saved")
    assert "Search Results" in memory_search.text


def test_config_roundtrip() -> None:
    response = client.put("/api/config", json={"locale": "en-US"})
    assert response.status_code == 200
    assert response.json()["locale"] == "en-US"
    assert client.get("/api/config").json()["locale"] == "en-US"


def test_models_expose_capabilities_aliases_and_retry_contract() -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert data["aliases"]["premium"] == "qwen3.7-max"
    assert "capabilities" in data

    cap = client.get("/api/models/qwen3.7-max/capability")
    assert cap.status_code == 200
    payload = cap.json()
    assert payload["capability"]["tokenCharRatio"] == 2.5
    assert payload["compactThreshold"] == 0.85
    assert payload["retry"]["maxRetries"] == 8

    resolved = client.get("/api/models/resolve/light")
    assert resolved.status_code == 200
    assert resolved.json()["model"] == "qwen3.7-plus"


def test_xfyun_maas_provider_settings(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER_DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.delenv("LLM_PROVIDER_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER_XFYUN_API_KEY", "xfyun-test-key")

    settings = app_module.llm_settings()

    assert settings["apiKey"] == "xfyun-test-key"
    assert settings["baseUrl"] == "https://maas-api.cn-huabei-1.xf-yun.com/v2"
    assert settings["model"] == "xopqwen36v35b"


def test_xfyun_key_overrides_persisted_default_model(monkeypatch, tmp_path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"config": {"defaultModel": "qwen3.7-max"}}), encoding="utf-8")
    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path / "attachments")
    monkeypatch.setattr(app_module, "STATE_FILE", state_file)
    monkeypatch.delenv("LLM_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER_XFYUN_API_KEY", "xfyun-test-key")

    state = app_module.load_state()

    assert state["config"]["defaultModel"] == "xopqwen36v35b"


def test_xfyun_request_model_ignores_legacy_session_model() -> None:
    model = app_module.request_model_for_settings(
        {"model": "qwen3.7-max"},
        {
            "apiKey": "xfyun-test-key",
            "baseUrl": "https://maas-api.cn-huabei-1.xf-yun.com/v2",
            "model": "xopqwen36v35b",
        },
    )

    assert model == "xopqwen36v35b"


def test_extended_openai_compatible_provider_settings(monkeypatch) -> None:
    provider_envs = [
        "LLM_PROVIDER_XFYUN_API_KEY",
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_PROVIDER_ZHIPU_API_KEY",
        "LLM_PROVIDER_MINIMAX_API_KEY",
        "LLM_PROVIDER_ZENMUX_API_KEY",
        "LLM_API_KEY",
    ]
    for name in provider_envs:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("LLM_PROVIDER_ZHIPU_API_KEY", "zhipu-test-key")
    monkeypatch.setenv("LLM_PROVIDER_ZHIPU_MODEL", "glm-5.1")
    settings = app_module.llm_settings()
    assert settings == {
        "apiKey": "zhipu-test-key",
        "baseUrl": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-5.1",
    }

    monkeypatch.delenv("LLM_PROVIDER_ZHIPU_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER_MINIMAX_API_KEY", "minimax-test-key")
    monkeypatch.setenv("LLM_PROVIDER_MINIMAX_MODEL", "MiniMax-M3")
    settings = app_module.llm_settings()
    assert settings == {
        "apiKey": "minimax-test-key",
        "baseUrl": "https://api.minimax.chat/v1",
        "model": "MiniMax-M3",
    }

    monkeypatch.delenv("LLM_PROVIDER_MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER_ZENMUX_API_KEY", "zenmux-test-key")
    monkeypatch.setenv("LLM_PROVIDER_ZENMUX_BASE_URL", "https://zenmux.example/v1")
    monkeypatch.setenv("LLM_PROVIDER_ZENMUX_MODEL", "anthropic/claude-opus-4.8")
    settings = app_module.llm_settings()
    assert settings == {
        "apiKey": "zenmux-test-key",
        "baseUrl": "https://zenmux.example/v1",
        "model": "anthropic/claude-opus-4.8",
    }


def test_llm_error_classifier_marks_retry_and_fallback_behavior() -> None:
    assert app_module.classify_llm_error(status_code=401)["type"] == "auth"
    assert app_module.classify_llm_error(status_code=401)["retryable"] is False
    assert app_module.classify_llm_error(status_code=429)["type"] == "rate_limit"
    assert app_module.classify_llm_error(status_code=429)["fallbackAllowed"] is True
    assert app_module.classify_llm_error(status_code=503)["type"] == "overloaded"
    assert app_module.classify_llm_error(status_code=413)["fallbackAllowed"] is False
    assert app_module.classify_llm_error(status_code=400, body="input is too long")["type"] == "prompt_too_long"
    assert app_module.classify_llm_error(error=app_module.httpx.TimeoutException("slow"))["type"] == "timeout"


def test_llm_model_fallback_chain_uses_source_degradation_order() -> None:
    settings = {"apiKey": "key", "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen3.7-max"}

    chain = app_module.llm_model_fallback_chain({"id": "s1", "model": "qwen3.7-max"}, settings)

    assert chain[:3] == ["qwen3.7-max", "qwen3.7-plus", "deepseek-v4-flash"]


def test_llm_retry_after_is_applied_through_model_policy(monkeypatch) -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    class FakeResponse:
        def __init__(self, status_code: int, body: dict[str, Any] | None = None, text: str = "", headers: dict[str, str] | None = None) -> None:
            self.status_code = status_code
            self._body = body or {}
            self.text = text
            self.headers = headers or {}

        def json(self):
            return self._body

    class FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, headers=None, json=None):
            attempts.append(json["model"])
            if len(attempts) == 1:
                return FakeResponse(429, text="rate limit", headers={"Retry-After": "45"})
            return FakeResponse(200, {"choices": [{"message": {"content": "retry-after ok"}}]})

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(app_module, "llm_settings", lambda: {"apiKey": "key", "baseUrl": "https://llm.test/v1", "model": "deepseek-v4-pro"})
    monkeypatch.setattr(app_module, "RETRY_POLICY", app_module.ModelAwareRetryPolicy(jitter_factor=0.0))
    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(app_module.httpx, "AsyncClient", FakeAsyncClient)

    answer = asyncio.run(app_module.generate_llm_reply({"id": "s1", "model": "deepseek-v4-pro", "messages": []}, "hello"))

    assert answer == "retry-after ok"
    assert attempts == ["deepseek-v4-pro", "deepseek-v4-pro"]
    assert sleeps == [45.0]


def test_llm_reply_falls_back_to_secondary_model_after_retryable_error(monkeypatch) -> None:
    calls: list[str] = []

    class NoRetryPolicy:
        def get_retry_config(self, model_id):
            return type("RetryConfig", (), {"maxRetries": 0, "baseDelayMs": 0})()

        def calculate_delay_ms(self, model_id, attempt):
            return 0

    class FakeResponse:
        def __init__(self, status_code: int, body: dict[str, Any] | None = None, text: str = "") -> None:
            self.status_code = status_code
            self._body = body or {}
            self.text = text

        def json(self):
            return self._body

    class FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, headers=None, json=None):
            model = json["model"]
            calls.append(model)
            if model == "broken-model":
                return FakeResponse(503, text="temporarily down")
            return FakeResponse(200, {"choices": [{"message": {"content": "fallback ok"}}]})

    monkeypatch.setattr(app_module, "llm_settings", lambda: {"apiKey": "key", "baseUrl": "https://llm.test/v1", "model": "broken-model"})
    monkeypatch.setattr(app_module, "RETRY_POLICY", NoRetryPolicy())
    monkeypatch.setattr(app_module.httpx, "AsyncClient", FakeAsyncClient)

    answer = asyncio.run(app_module.generate_llm_reply({"id": "s1", "model": "broken-model", "messages": []}, "hello"))

    assert answer == "fallback ok"
    assert calls[0] == "broken-model"
    assert len(calls) >= 2
    assert calls[1] != "broken-model"


def test_llm_reply_records_provider_usage_on_session(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = ""
        headers: dict[str, str] = {}

        def json(self):
            return {
                "choices": [{"message": {"content": "usage ok"}}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            }

    class FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, headers=None, json=None):
            return FakeResponse()

    session = {"id": "usage-session", "model": "qwen3.7-max", "messages": []}
    monkeypatch.setattr(app_module, "llm_settings", lambda: {"apiKey": "key", "baseUrl": "https://llm.test/v1", "model": "qwen3.7-max"})
    monkeypatch.setattr(app_module.httpx, "AsyncClient", FakeAsyncClient)

    answer = asyncio.run(app_module.generate_llm_reply(session, "hello"))

    assert answer == "usage ok"
    assert session["pendingModelUsage"]["inputTokens"] == 12
    assert session["pendingModelUsage"]["outputTokens"] == 5
    assert session["pendingModelUsage"]["cacheReadInputTokens"] == 3


def test_rest_query_returns_real_model_usage_in_message_and_events(monkeypatch) -> None:
    async def fake_reply(session: dict[str, Any], _text: str, _memory_context: str | None = None) -> str:
        session["pendingModelUsage"] = {
            "inputTokens": 21,
            "outputTokens": 8,
            "cacheReadInputTokens": 4,
            "cacheCreationInputTokens": 2,
        }
        return "usage surfaced"

    monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)

    response = asyncio.run(app_module.run_query_payload({"prompt": "surface usage"}))

    assert response["answer"] == "usage surfaced"
    assert response["usage"]["inputTokens"] == 21
    assert response["usage"]["outputTokens"] == 8
    assert response["message"]["usage"]["cacheReadInputTokens"] == 4
    complete_events = [event for event in response["events"] if event["type"] == "message_complete"]
    assert complete_events[-1]["usage"]["cacheCreationInputTokens"] == 2


def test_query_input_token_estimator_uses_precise_tokenizer_and_fallback(monkeypatch) -> None:
    previous_flags = dict(STATE.setdefault("config", {}).setdefault("featureFlags", {}))

    async def precise_count(text: str, model: str) -> int:
        return 7

    try:
        STATE["config"]["featureFlags"] = {**previous_flags, "PRECISE_TOKENIZER": True}
        monkeypatch.setattr(app_module, "count_exact_tokens", precise_count)
        assert asyncio.run(app_module.estimate_query_input_tokens("hello world", "qwen", 3.5)) == 7

        async def failed_count(text: str, model: str) -> int:
            return -1

        monkeypatch.setattr(app_module, "count_exact_tokens", failed_count)
        mixed_chinese = "你好智能体" * 8
        assert asyncio.run(app_module.estimate_query_input_tokens(mixed_chinese, "qwen", 3.5)) > int(len(mixed_chinese) / 3.5)
    finally:
        STATE["config"]["featureFlags"] = previous_flags


def test_query_context_budget_counts_full_conversation_memory_and_tools(monkeypatch) -> None:
    previous_flags = dict(STATE.setdefault("config", {}).setdefault("featureFlags", {}))
    previous_memories = list(STATE.setdefault("memories", []))
    session = {
        "id": "budget-helper-session",
        "systemPrompt": "Use the project conventions.",
        "messages": [
            {"type": "user", "content": [{"type": "text", "text": "first request"}]},
            {"type": "assistant", "content": [{"type": "text", "text": "first answer"}]},
        ],
    }
    tool_message = {
        "type": "user",
        "content": [{"type": "tool_result", "toolUseId": "tool-1", "content": "tool output from README.md"}],
        "toolUseResult": "tool output from README.md",
    }

    try:
        STATE["config"]["featureFlags"] = {**previous_flags, "PRECISE_TOKENIZER": False}
        STATE["memories"] = [{"id": "mem-1", "title": "Project rule", "content": "Always explain token budget changes.", "category": "semantic"}]

        budget = asyncio.run(
            app_module.estimate_query_context_budget(
                session,
                "current prompt",
                "qwen3.7-max",
                2.5,
                extra_messages=[tool_message],
            )
        )
        current_only = asyncio.run(app_module.estimate_query_input_tokens("current prompt", "qwen3.7-max", 2.5))

        assert set(budget["breakdown"]) >= {"system", "history", "user", "memory", "tool"}
        assert budget["breakdown"]["system"] > 0
        assert budget["breakdown"]["history"] > 0
        assert budget["breakdown"]["user"] > 0
        assert budget["breakdown"]["memory"] > 0
        assert budget["breakdown"]["tool"] > 0
        assert budget["usedTokens"] > current_only
        assert "Project rule" in budget["memoryContext"]
    finally:
        STATE["config"]["featureFlags"] = previous_flags
        STATE["memories"] = previous_memories


def test_auth_admin_and_remote_control(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_MODE", "lan_token")
    monkeypatch.setenv("AUTH_LAN_TOKEN", "test-token")
    unauthenticated = client.get("/api/auth/status").json()
    assert unauthenticated["authenticated"] is False
    authenticated = client.get("/api/auth/status", headers={"Authorization": "Bearer test-token"}).json()
    assert authenticated["authenticated"] is True
    assert client.get("/api/auth/token").json()["token"] == "test-token"

    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    unconfigured = client.post("/api/admin/login", json={"password": "secret"})
    assert unconfigured.status_code == 503

    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    assert client.get("/api/admin/status").json()["configured"] is True
    assert client.post("/api/admin/login", json={"password": "bad"}).status_code == 401
    login = client.post("/api/admin/login", json={"password": "secret"})
    assert login.status_code == 200
    assert login.json()["success"] is True
    assert client.get("/api/admin/status").json()["authenticated"] is True
    assert client.post("/api/admin/logout").json()["success"] is True

    session_id = client.post("/api/sessions", json={"title": "remote", "model": "test-model"}).json()["sessionId"]
    STATE["sessions"][session_id]["status"] = "streaming"
    STATE["sessions"][session_id]["online"] = True
    status = client.get("/api/remote/status").json()
    assert status["activeSessions"] >= 1
    assert any(item["sessionId"] == session_id for item in status["sessions"])
    bound = client.post("/api/ws/sessions/bind", json={"principal": "tester", "sessionId": session_id}).json()
    assert bound["success"] is True
    ws_status = client.get("/api/ws/sessions").json()
    assert any(item["sessionId"] == session_id and item["principal"] == "tester" for item in ws_status["sessions"])
    queued = client.post(f"/api/ws/sessions/{session_id}/push", json={"type": "notification", "payload": {"message": "hello"}}).json()
    assert queued["success"] is True
    peeked = client.get(f"/api/ws/sessions/{session_id}/messages/peek").json()
    assert peeked["count"] == 1
    drained = client.get(f"/api/ws/sessions/{session_id}/messages").json()
    assert drained["count"] == 1
    assert drained["messages"][0]["payload"]["message"] == "hello"
    broadcast = client.post("/api/ws/events/broadcast", json={"type": "system_notice", "payload": {"message": "all"}}).json()
    assert broadcast["count"] >= 1
    notice = client.post("/api/notifications", json={"sessionId": session_id, "key": "n1", "level": "info", "message": "toast"}).json()
    assert notice["success"] is True
    notices = client.get(f"/api/notifications?sessionId={session_id}").json()["notifications"]
    assert any(item["key"] == "n1" for item in notices)
    assert client.delete("/api/notifications/n1").json()["notification"]["dismissed"] is True
    interrupted = client.post("/api/remote/interrupt").json()
    assert interrupted["interrupted"] is True
    assert session_id in interrupted["sessions"]
    assert STATE["sessions"][session_id]["status"] == "idle"
    ack_messages = client.get(f"/api/ws/sessions/{session_id}/messages").json()["messages"]
    assert any(item["type"] == "interrupt_ack" for item in ack_messages)


def test_session_create_and_list() -> None:
    create = client.post("/api/sessions", json={"dir": ".", "model": "test-model"})
    assert create.status_code == 200
    session_id = create.json()["sessionId"]

    listed = client.get("/api/sessions")
    assert listed.status_code == 200
    assert any(item["id"] == session_id for item in listed.json()["sessions"])


def test_rest_query_persists_conversation_messages(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER_DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER_DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER_MOONSHOT_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER_XFYUN_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")

    first = client.post("/api/query", json={"prompt": "hello query"}).json()
    assert "hello query" in first["answer"]
    session_id = first["sessionId"]
    assert first["queryLoop"]["status"] == "completed"
    assert first["queryLoop"]["phase"] == "completed"
    assert first["queryLoop"]["tokenBudget"]["usedTokens"] >= 1
    assert first["queryLoop"]["terminationDecision"]["action"] == "stop"
    assert first["queryLoop"]["terminationDecision"]["stopReason"] == "end_turn"
    assert any(event["type"] == "stream_delta" for event in first["events"])
    assert any(event["type"] == "termination_decision" for event in first["events"])

    second = client.post("/api/query/conversation", json={"sessionId": session_id, "prompt": "follow up"}).json()
    assert second["sessionId"] == session_id
    messages = client.get(f"/api/sessions/{session_id}/messages").json()["messages"]
    assert [message["type"] for message in messages[-4:]] == ["user", "assistant", "user", "assistant"]

    db_status = client.get("/api/database/status").json()
    assert db_status["status"] == "ok"
    assert db_status["sessions"] >= 1
    db_session = client.get(f"/api/database/sessions/{session_id}").json()
    assert len(db_session["messages"]) >= 4
    rollback = client.post(f"/api/database/sessions/{session_id}/messages/delete-after/2").json()
    assert rollback["deleted"] >= 2
    migrations = client.get("/api/database/migrations").json()["migrations"]
    assert all(item["applied"] for item in migrations)

    loops = client.get(f"/api/query/loops?sessionId={session_id}").json()
    assert loops["total"] >= 1
    latest = client.get(f"/api/query/session/{session_id}/loop").json()
    assert latest["loop"]["sessionId"] == session_id
    loop_detail = client.get(f"/api/query/loops/{latest['loop']['id']}").json()
    assert loop_detail["transitions"]
    loop_events = client.get(f"/api/query/loops/{latest['loop']['id']}/events").json()
    assert any(event["type"] == "message_complete" for event in loop_events["events"])
    session_events = client.get(f"/api/query/session/{session_id}/events").json()
    assert session_events["total"] >= 1


def test_rest_query_loop_reports_full_context_token_budget(monkeypatch) -> None:
    previous_flags = dict(STATE.setdefault("config", {}).setdefault("featureFlags", {}))
    previous_memories = list(STATE.setdefault("memories", []))
    session_id = f"budget-query-{uuid4().hex}"

    async def fake_reply(session, text, memory_context=None):
        return "budget ok"

    try:
        monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)
        STATE["config"]["featureFlags"] = {**previous_flags, "PRECISE_TOKENIZER": False}
        STATE["memories"] = [{"id": "mem-2", "title": "Budget memory", "content": "Memory context must be counted.", "category": "semantic"}]
        now = app_module.utc_now()
        STATE["sessions"][session_id] = {
            "id": session_id,
            "title": "budget query",
            "model": "qwen3.7-max",
            "systemPrompt": "Count all context.",
            "workingDirectory": ".",
            "messages": [
                {"type": "user", "uuid": "u1", "timestamp": 1, "content": [{"type": "text", "text": "older user"}]},
                {"type": "assistant", "uuid": "a1", "timestamp": 2, "content": [{"type": "text", "text": "older assistant"}]},
                {
                    "type": "user",
                    "uuid": "u2",
                    "timestamp": 3,
                    "content": [{"type": "tool_result", "toolUseId": "tool-old", "content": "old tool result"}],
                    "toolUseResult": "old tool result",
                },
            ],
            "costUsd": 0,
            "createdAt": now,
            "updatedAt": now,
            "status": "idle",
        }
        STATE["activities"].setdefault(session_id, [])

        result = client.post("/api/query/conversation", json={"sessionId": session_id, "prompt": "new prompt"}).json()
        loop = result["queryLoop"]
        breakdown = loop["tokenBudgetBreakdown"]
        current_only = asyncio.run(app_module.estimate_query_input_tokens("new prompt", "qwen3.7-max", 2.5))

        assert result["answer"] == "budget ok"
        assert set(breakdown) >= {"system", "history", "user", "memory", "tool"}
        assert breakdown["system"] > 0
        assert breakdown["history"] > 0
        assert breakdown["user"] > 0
        assert breakdown["memory"] > 0
        assert breakdown["tool"] > 0
        assert loop["tokenBudget"]["usedTokens"] > current_only
        assert any(event["type"] == "token_budget_nudge" and event.get("breakdown") for event in result["events"])
    finally:
        STATE["config"]["featureFlags"] = previous_flags
        STATE["memories"] = previous_memories
        STATE["sessions"].pop(session_id, None)


def test_rest_query_tool_result_event_includes_large_output_summary(monkeypatch) -> None:
    session_id = f"tool-summary-{uuid4().hex}"
    root = workspace()
    target = root / "large-output.txt"
    target.write_text("\n".join(f"line {index}" for index in range(300)), encoding="utf-8")
    relative_path = target.relative_to(ROOT).as_posix()

    async def fake_reply(session, text, memory_context=None):
        return "done"

    monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)
    payload = client.post(
        "/api/query",
        json={
            "sessionId": session_id,
            "prompt": "run large tool",
            "toolCalls": [
                {
                    "id": "call-large",
                    "function": {"name": "read_file", "arguments": {"path": relative_path}},
                }
            ],
        },
    ).json()

    tool_events = [event for event in payload["queryLoop"]["events"] if event["type"] == "tool_result"]
    assert tool_events
    summary = tool_events[-1]["summary"]
    assert summary["toolName"] == "read_file"
    assert summary["originalChars"] > 500
    assert summary["truncated"] is True


def test_rest_query_uses_context_cascade_under_budget_pressure(monkeypatch) -> None:
    for name in [
        "LLM_PROVIDER_XFYUN_API_KEY",
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_PROVIDER_ZHIPU_API_KEY",
        "LLM_PROVIDER_MINIMAX_API_KEY",
        "LLM_PROVIDER_ZENMUX_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)

    async def fake_reply(session, text, memory_context=None):
        return "cascade ok"

    monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)
    session_id = client.post("/api/sessions", json={"dir": ".", "model": "qwen3.7-max"}).json()["sessionId"]
    session = STATE["sessions"][session_id]
    session["messages"] = [
        {"type": "assistant", "content": [{"type": "text", "text": "old context " + ("a" * 4_000)}]},
        {"type": "user", "toolUseResult": "tool result " + ("b" * 5_000)},
        {"type": "assistant", "content": [{"type": "text", "text": "recent context"}]},
    ]

    response = client.post(
        "/api/query/conversation",
        json={"sessionId": session_id, "prompt": "长任务\n" + ("p" * 30_000), "collapseContext": True, "protectedTail": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "cascade ok"
    compact_complete = [event for event in payload["events"] if event["type"] == "compact_complete"][-1]
    assert compact_complete["layers"] == [
        "snip_selection",
        "micro_compact",
        "auto_compact",
        "collapse_drain",
        "reactive_compact",
    ]
    assert payload["queryLoop"]["contextCascade"]["changed"] is True
    STATE.setdefault("activities", {}).pop(session_id, None)


def test_rest_query_context_cascade_marks_http_413_media_strip(monkeypatch) -> None:
    for name in [
        "LLM_PROVIDER_XFYUN_API_KEY",
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_PROVIDER_ZHIPU_API_KEY",
        "LLM_PROVIDER_MINIMAX_API_KEY",
        "LLM_PROVIDER_ZENMUX_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)

    async def fake_reply(session, text, memory_context=None):
        return "413 recovered"

    monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)
    session_id = client.post("/api/sessions", json={"dir": ".", "model": "qwen3.7-max"}).json()["sessionId"]
    STATE["sessions"][session_id]["messages"] = [
        {
            "type": "user",
            "content": [
                {"type": "text", "text": "image context"},
                {"type": "image", "width": 2048, "height": 2048, "source": "data:image/png;base64," + ("x" * 8_000)},
            ],
        }
    ]

    response = client.post(
        "/api/query/conversation",
        json={"sessionId": session_id, "prompt": "recover 413", "collapseContext": True, "protectedTail": 1, "recoveryCause": "http_413"},
    )

    payload = response.json()
    assert response.status_code == 200
    reactive = [layer for layer in payload["queryLoop"]["contextCascade"]["layers"] if layer["name"] == "reactive_compact"][0]
    assert reactive["metadata"]["recoveryCause"] == "http_413"
    assert reactive["metadata"]["mediaStrippedCount"] == 1
    assert any(event.get("metadata", {}).get("mediaStrippedCount") == 1 for event in payload["queryLoop"]["recoveryEvents"])


def test_rest_query_self_correction_retries_model_on_parsed_failure(monkeypatch) -> None:
    previous_flags = dict(STATE.setdefault("config", {}).setdefault("featureFlags", {}))
    calls: list[str] = []

    async def fake_reply(session, text, memory_context=None):
        calls.append(text)
        if len(calls) == 1:
            return "FAILED tests/test_calc.py::test_add - AssertionError: expected 2 but got 3"
        return "Fixed the add implementation and reran the failing check."

    try:
        STATE["config"]["featureFlags"] = {**previous_flags, "SELF_CORRECTION_LOOP": True}
        monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)

        result = client.post("/api/query", json={"prompt": "fix failing tests"}).json()
        events = result["events"]

        assert result["answer"] == "Fixed the add implementation and reran the failing check."
        assert len(calls) == 2
        assert "Fix the failing pytest test: test_add" in calls[1]
        assert result["queryLoop"]["correctionAttempts"] == 1
        assert any(event["type"] == "self_correction_start" for event in events)
        assert any(transition["toPhase"] == "self_correcting" for transition in result["queryLoop"]["transitions"])
    finally:
        STATE["config"]["featureFlags"] = previous_flags


def test_rest_query_self_correction_stops_at_attempt_limit(monkeypatch) -> None:
    previous_flags = dict(STATE.setdefault("config", {}).setdefault("featureFlags", {}))
    calls: list[str] = []

    async def fake_reply(session, text, memory_context=None):
        calls.append(text)
        return "FAILED tests/test_calc.py::test_add - AssertionError: still failing"

    try:
        STATE["config"]["featureFlags"] = {**previous_flags, "SELF_CORRECTION_LOOP": True, "SELF_CORRECTION_MAX_ATTEMPTS": 3}
        monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)

        result = client.post("/api/query", json={"prompt": "fix persistent failure"}).json()

        assert len(calls) == 4
        assert result["queryLoop"]["correctionAttempts"] == 3
        assert result["answer"] == "FAILED tests/test_calc.py::test_add - AssertionError: still failing"
        assert sum(1 for event in result["events"] if event["type"] == "self_correction_start") == 3
    finally:
        STATE["config"]["featureFlags"] = previous_flags


def test_query_tool_events_match_frontend_realtime_contract(monkeypatch) -> None:
    async def fake_reply(session, text, memory_context=None):
        return "tool event contract ok"

    monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)
    response = client.post(
        "/api/query",
        json={
            "prompt": "inspect files",
            "toolCalls": [{"id": "tool-read-1", "name": "list_files", "arguments": {"pattern": "backend-python/*.py", "limit": 3}}],
        },
    ).json()
    events = response["events"]
    assert response["toolCalls"][0]["toolUseId"] == "tool-read-1"
    assert [event["type"] for event in events if event["type"].startswith("tool_")][:3] == ["tool_use_start", "tool_use_input", "tool_use_progress"]
    assert any(event["type"] == "tool_result" and event["toolUseId"] == "tool-read-1" for event in events)
    assert any(call["status"] in {"completed", "error"} for call in response["queryLoop"]["toolCalls"])


def test_query_write_tool_records_history_snapshot_for_rewind(monkeypatch) -> None:
    root = workspace()
    target = root / "tool-history.py"
    target.write_text("before\n", encoding="utf-8")
    rel = target.relative_to(ROOT).as_posix()
    session_id = f"tool-history-{uuid4().hex}"

    async def fake_reply(session, text, memory_context=None):
        return "history snapshot recorded"

    rule = PermissionRule("write_file", PermissionDecision.ALLOW)
    TOOL_REGISTRY.policy.rules.insert(0, rule)
    try:
        monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)
        response = client.post(
            "/api/query",
            json={
                "sessionId": session_id,
                "prompt": "overwrite file",
                "toolCalls": [
                    {
                        "id": "write-history-1",
                        "name": "write_file",
                        "arguments": {"path": rel, "content": "after\n", "agentId": "tool-agent"},
                    }
                ],
            },
        ).json()

        assert target.read_text(encoding="utf-8") == "after\n"
        metadata = response["toolCalls"][0]["metadata"]
        assert metadata["snapshotBeforeWrite"]["content"] == "before\n"

        snapshots = client.get(f"/api/sessions/{session_id}/history/snapshots").json()
        assert any(item["messageId"] == "write-history-1" and rel in item["trackedFiles"] for item in snapshots["snapshots"])

        diff = client.get(f"/api/sessions/{session_id}/history/diff?fromMessageId=write-history-1&toMessageId=current").json()
        assert "-before" in diff["diff"]
        assert "+after" in diff["diff"]

        rewind = client.post(f"/api/sessions/{session_id}/history/rewind", json={"messageId": "write-history-1", "filePaths": [rel]}).json()
        assert rewind["success"] is True
        assert target.read_text(encoding="utf-8") == "before\n"
    finally:
        try:
            TOOL_REGISTRY.policy.rules.remove(rule)
        except ValueError:
            pass


def test_query_tool_retry_reexecutes_retryable_failure(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_reply(session, text, memory_context=None):
        return "tool retry checked"

    def flaky_tool(payload: dict[str, Any]) -> ToolResult:
        calls.append(payload)
        if len(calls) == 1:
            return ToolResult("temporary tool failure", isError=True, metadata={"retryable": True})
        return ToolResult("retry success", isError=False, metadata={"attempts": len(calls)})

    TOOL_REGISTRY.register(
        Tool(
            name="flaky_retry_tool",
            description="Test-only retryable tool.",
            input_schema={"type": "object", "properties": {"retryMaxAttempts": {"type": "integer"}}},
            handler=flaky_tool,
            read_only=True,
            concurrency_safe=False,
        )
    )
    try:
        monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)
        response = client.post(
            "/api/query",
            json={
                "prompt": "run flaky tool",
                "toolCalls": [{"id": "tool-retry-1", "name": "flaky_retry_tool", "arguments": {"retryMaxAttempts": 1}}],
            },
        ).json()

        assert len(calls) == 2
        assert response["toolCalls"][0]["content"] == "retry success"
        assert any(event["type"] == "tool_retry" and event["toolUseId"] == "tool-retry-1" for event in response["events"])
        assert any(event["type"] == "tool_retry" for event in response["queryLoop"]["recoveryEvents"])
        assert response["queryLoop"]["toolCalls"][0]["status"] == "completed"
    finally:
        TOOL_REGISTRY._tools.pop("flaky_retry_tool", None)


def test_query_permission_wait_emits_termination_decision(monkeypatch) -> None:
    previous_rules = list(TOOL_REGISTRY.policy.rules)

    async def fake_reply(session, text, memory_context=None):
        return "permission handled"

    async def fake_permission(tool_use_id: str, session_id: str, timeout_ms: int = 120_000):
        return {"toolUseId": tool_use_id, "decision": "deny", "reason": "test deny"}

    try:
        TOOL_REGISTRY.policy.rules = []
        monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)
        monkeypatch.setattr(app_module, "await_permission_decision", fake_permission)

        response = client.post(
            "/api/query",
            json={
                "prompt": "write with permission",
                "toolCalls": [{"id": "perm-tool-1", "name": "write_file", "arguments": {"path": "backend-python/.test-workspace/blocked.txt", "content": "blocked"}}],
            },
        ).json()

        wait_decisions = [
            event["decision"]
            for event in response["events"]
            if event["type"] == "termination_decision" and event["decision"]["action"] == "wait"
        ]
        assert wait_decisions
        assert wait_decisions[0]["reason"] == "permission_wait"
        assert response["queryLoop"]["terminationDecision"]["action"] == "stop"
        assert response["toolCalls"][0]["isError"] is True
    finally:
        TOOL_REGISTRY.policy.rules = previous_rules


def test_query_runtime_compacts_prompt_that_exceeds_budget(monkeypatch) -> None:
    async def fake_reply(session, text, memory_context=None):
        return "compact ok"

    monkeypatch.setattr(app_module, "generate_llm_reply", fake_reply)
    long_prompt = "x" * 140_000
    response = client.post("/api/query", json={"prompt": long_prompt}).json()
    loop = response["queryLoop"]
    assert loop["promptTooLongWithheld"] is False
    assert any(event["type"] == "prompt_too_long" for event in loop["recoveryEvents"])
    assert any(event["type"] == "compact_event" for event in response["events"])
    messages = client.get(f"/api/sessions/{response['sessionId']}/messages").json()["messages"]
    assert "prompt compacted" in messages[-2]["content"][0]["text"]


def test_session_compact_and_export_downloads() -> None:
    session_id = client.post("/api/query", json={"prompt": "export me"}).json()["sessionId"]
    for index in range(3):
        client.post("/api/query/conversation", json={"sessionId": session_id, "prompt": f"turn {index}"})

    compacted = client.post(f"/api/sessions/{session_id}/compact").json()
    assert compacted["success"] is True
    assert compacted["beforeTokens"] >= compacted["afterTokens"]

    exported_json = client.post(f"/api/sessions/{session_id}/export")
    assert exported_json.status_code == 200
    assert "attachment" in exported_json.headers["content-disposition"]
    assert exported_json.json()["sessionId"] == session_id

    exported_md = client.post(f"/api/sessions/{session_id}/export?format=md")
    assert exported_md.status_code == 200
    assert exported_md.text.startswith(f"# Session {session_id}")


def test_session_snapshot_save_resume_delete() -> None:
    create = client.post("/api/sessions", json={"title": "snap", "model": "snapshot-model"})
    session_id = create.json()["sessionId"]

    saved = client.post(f"/api/sessions/{session_id}/snapshot")
    assert saved.status_code == 201
    assert saved.json()["sessionId"] == session_id

    listed = client.get("/api/sessions/snapshots").json()
    assert any(item["sessionId"] == session_id for item in listed["snapshots"])

    resumed = client.post(f"/api/sessions/{session_id}/snapshot/resume").json()
    assert resumed["model"] == "snapshot-model"

    deleted = client.delete(f"/api/sessions/snapshots/{session_id}").json()
    assert deleted["success"] is True


def test_files_tree_endpoint_returns_frontend_contract() -> None:
    root = workspace()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('hello')", encoding="utf-8")

    response = client.post("/api/files/tree", json={"root_path": str(root.relative_to(ROOT))})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["type"] == "dir"
    assert any(child["name"] == "src" for child in body["data"]["children"])


def test_attachment_upload_accepts_multipart_and_downloads_by_uuid() -> None:
    uploaded = client.post(
        "/api/attachments/upload",
        files={"file": ("note.txt", b"hello attachment", "text/plain")},
    )
    assert uploaded.status_code == 201
    body = uploaded.json()
    assert body["fileName"] == "note.txt"
    assert body["size"] == len(b"hello attachment")
    assert body["error"] is None

    downloaded = client.get(f"/api/attachments/{body['fileUuid']}")
    assert downloaded.status_code == 200
    assert downloaded.content == b"hello attachment"


def test_git_log_diff_and_blame_endpoints() -> None:
    repo = workspace()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / "app.py").write_text("print('one')\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "app.py").write_text("print('one')\nprint('two')\n", encoding="utf-8")

    repo_path = str(repo.relative_to(ROOT))
    log = client.post("/api/git/log", json={"repo_path": repo_path, "max_count": 5}).json()
    diff = client.post("/api/git/diff", json={"repo_path": repo_path, "ref1": "HEAD"}).json()
    blame = client.post("/api/git/blame", json={"repo_path": repo_path, "file_path": "app.py"}).json()

    assert log["success"] is True
    assert log["data"]["commits"][0]["message"] == "initial"
    assert diff["success"] is True
    assert "print('two')" in diff["data"]["detailed"]
    assert blame["success"] is True
    assert blame["data"]["total_lines"] >= 1


def test_complexity_endpoint_analyzes_python_files() -> None:
    root = workspace()
    (root / "pkg").mkdir()
    (root / "pkg" / "module.py").write_text("def f(x):\n    if x:\n        return 1\n    return 0\n", encoding="utf-8")

    version = client.get(f"/api/files/version?path={(root / 'pkg' / 'module.py').relative_to(ROOT)}").json()
    assert version["contentHash"]
    recovery = client.post("/api/files/recovery", json={"toolName": "edit_file", "error": "old_string not found in file"}).json()
    assert recovery["action"] == "report_to_llm"
    assert "Content mismatch" in recovery["message"]

    response = client.post("/api/code-quality/complexity", json={"project_root": str(root.relative_to(ROOT)), "languages": ["python"]})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["stats"]["total_files"] == 1
    assert body["data"]["root"]["children"][0]["file_path"] == "pkg/module.py"


def test_code_path_endpoints_trace_and_diagram_are_scanned() -> None:
    endpoints = client.post("/api/code-path/endpoints", json={"project_root": ".", "languages": ["python"]}).json()
    assert endpoints["success"] is True
    assert any(item["path"] == "/api/health" for item in endpoints["endpoints"])

    trace = client.post("/api/code-path/trace", json={"project_root": ".", "path": "/api/health", "languages": ["python"]}).json()
    assert trace["success"] is True
    assert trace["nodes"][0]["file_path"].endswith("app.py")

    diagram = client.post("/api/code-diagrams/generate", json={"project_root": ".", "apiPath": "/api/health", "type": "sequence"}).json()
    assert diagram["success"] is True
    assert "sequenceDiagram" in diagram["diagram"]
    assert diagram["matchedEndpoint"]["path"] == "/api/health"


def test_change_impact_endpoint_returns_graph_contract() -> None:
    root = workspace()
    (root / "service.py").write_text("def target():\n    return 1\n", encoding="utf-8")
    (root / "api.py").write_text("from service import target\n@app.get('/x')\ndef route():\n    return target()\n", encoding="utf-8")

    response = client.post(
        "/api/analysis/change-impact",
        json={"project_root": str(root.relative_to(ROOT)), "file_path": "service.py", "changed_lines": [1], "depth": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["changed_file"] == "service.py"
    assert body["data"]["summary"]["indirect_count"] == 1
    assert body["data"]["impact_edges"][0]["source"] == "changed-file"


def test_file_history_snapshot_diff_and_rewind() -> None:
    root = workspace()
    target = root / "note.txt"
    target.write_text("before\n", encoding="utf-8")
    rel = str(target.relative_to(ROOT))
    session_id = "history-session"

    first = client.post(f"/api/sessions/{session_id}/history/snapshot", json={"messageId": "m1", "filePaths": [rel]}).json()
    assert first["success"] is True
    target.write_text("after\n", encoding="utf-8")
    second = client.post(f"/api/sessions/{session_id}/history/snapshot", json={"messageId": "m2", "filePaths": [rel]}).json()
    assert second["success"] is True

    diff = client.get(f"/api/sessions/{session_id}/history/diff?fromMessageId=m1&toMessageId=m2").json()
    assert "-before" in diff["diff"]
    assert "+after" in diff["diff"]

    rewind = client.post(f"/api/sessions/{session_id}/history/rewind", json={"messageId": "m1", "filePaths": [rel]}).json()
    assert rewind["success"] is True
    assert target.read_text(encoding="utf-8") == "before\n"


def test_verify_creates_evidence_bundle_and_blob() -> None:
    session_id = "evidence-session"
    created = client.post("/api/verify/run-checks", json={"sessionId": session_id, "claim": "checks pass"}).json()
    assert created["success"] is True
    bundle_id = created["bundleId"]

    bundle = client.get(f"/api/evidence/{bundle_id}").json()
    assert bundle["sessionId"] == session_id
    assert bundle["verdict"] == "verified"
    blob_sha = bundle["items"][0]["blobSha256"]
    blob = client.get(f"/api/evidence/blob/{blob_sha}")
    assert blob.status_code == 200
    assert b"checks pass" in blob.content

    session_bundles = client.get(f"/api/evidence/session/{session_id}").json()
    assert any(item["bundleId"] == bundle_id for item in session_bundles)


def test_verify_runs_python_file_checks() -> None:
    root = workspace()
    target = root / "module.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    response = client.post(
        "/api/verify/run-checks",
        json={
            "sessionId": "python-verify",
            "workingDirectory": str(root.relative_to(ROOT)),
            "filePaths": ["module.py"],
            "checks": ["python"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["python"]["status"] == "pass"
    assert body["overallStatus"] == "pass"
    assert body["signal"] in {"auto_approve", "review_recommended"}


def test_browser_replay_stateful_timeline() -> None:
    session_id = "browser-session"
    created = client.post(f"/api/browser/replay/{session_id}", json={"url": "http://localhost", "title": "Home"}).json()
    assert created["success"] is True

    timeline = client.get(f"/api/browser/replay/{session_id}").json()
    assert timeline["snapshots"][0]["url"] == "http://localhost"

    deleted = client.delete(f"/api/browser/replay/{session_id}").json()
    assert deleted["success"] is True
    assert client.get(f"/api/browser/replay/{session_id}").json()["snapshots"] == []
    assert client.get("/api/browser/replay/bad.session").status_code == 400


def test_skills_plugins_and_memory_contracts() -> None:
    skills = client.get("/api/skills").json()
    assert any(skill["name"] == "review" for skill in skills)
    review = client.get("/api/skills/review").json()
    assert review["name"] == "review"
    assert "content" in review

    plugin_id = f"plugin-{uuid4().hex}"
    installed = client.post("/api/plugins/install", json={"id": plugin_id, "name": plugin_id, "version": "1.0.0"}).json()
    assert installed["id"] == plugin_id
    listed = client.get("/api/plugins").json()
    assert any(plugin["name"] == plugin_id for plugin in listed["plugins"])
    reloaded = client.post("/api/plugins/reload").json()
    assert reloaded["loaded"] >= 1
    assert client.delete(f"/api/plugins/{plugin_id}").json()["success"] is True
    assert client.delete("/api/plugins/missing-plugin").status_code == 404
    assert client.post("/api/plugins/install", json={"id": "bad/plugin", "name": "bad/plugin"}).status_code == 400
    builtin_id = f"builtin-{uuid4().hex}"
    client.post("/api/plugins/install", json={"id": builtin_id, "name": builtin_id, "isBuiltin": True}).json()
    assert client.delete(f"/api/plugins/{builtin_id}").status_code == 400
    STATE["plugins"] = [plugin for plugin in STATE["plugins"] if plugin.get("id") != builtin_id]

    allowed_skill = client.post("/api/skills/validate-tool", json={"skillName": "review", "allowedTools": ["Bash"], "toolName": "bash"}).json()
    assert allowed_skill["allowed"] is True
    denied_skill = client.post("/api/skills/validate-tool", json={"skillName": "review", "allowedTools": ["FileRead"], "toolName": "Bash"}).json()
    assert denied_skill["allowed"] is False
    dangerous_args = client.post("/api/skills/validate-tool", json={"skillName": "review", "toolName": "Bash", "args": {"cmd": "echo $(whoami)"}}).json()
    assert dangerous_args["allowed"] is False
    fork_denied = client.post("/api/skills/validate-tool", json={"skill": {"name": "forked", "context": "fork"}, "toolName": "Bash", "nestingDepth": 3}).json()
    assert fork_denied["allowed"] is False

    memory_id = f"memory-{uuid4().hex}"
    created = client.post("/api/memory", json={"id": memory_id, "title": "Fact", "content": "Python rewrite memory", "category": "semantic"}).json()
    assert created["success"] is True
    memories = client.get("/api/memory").json()
    assert any(item["id"] == memory_id for item in memories["entries"])
    categories = client.get("/api/memory/categories").json()
    assert any(item["tag"] == "team" for item in categories["categories"])
    search = client.get("/api/memory/search?q=Python%20rewrite&limit=3").json()
    assert any(item["id"] == memory_id for item in search["results"])
    by_category = client.get("/api/memory/category/semantic").json()
    assert any(item["id"] == memory_id for item in by_category["entries"])
    prompt = client.get("/api/memory/prompt").json()
    assert "Memory categories" in prompt["prompt"]
    all_memories = client.get("/api/memory/all").json()
    assert "sqlite" in all_memories
    assert client.delete(f"/api/memory/{memory_id}").json()["success"] is True


def test_mcp_resources_prompts_and_local_capability_invoke() -> None:
    server_name = f"mcp-{uuid4().hex}"
    server = {
        "name": server_name,
        "type": "sse",
        "status": "connected",
        "resources": [{"uri": "memory://demo", "name": "Demo", "content": "resource body"}],
        "prompts": [{"name": "hello", "template": "Hello {{name}}", "arguments": [{"name": "name", "required": True}]}],
        "tools": [{"name": "remote_echo", "result": {"content": "ok", "isError": False}}],
    }
    created = client.post("/api/mcp/servers", json=server).json()
    assert created["success"] is True
    assert created["connection"]["status"] == "connected"

    status = client.get("/api/mcp/status").json()
    assert status["connectionCount"] >= 1
    wrapped = client.get("/api/mcp/tools/wrapped").json()
    assert any(tool["name"] == f"mcp__{server_name}__remote_echo" for tool in wrapped["tools"])
    rpc = client.post("/api/mcp/json-rpc", json={"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}}).json()
    assert rpc["message"]["method"] == "tools/list"

    resources = client.get(f"/api/mcp/resources?server={server_name}").json()
    assert resources["resources"][server_name][0]["uri"] == "memory://demo"
    content = client.get(f"/api/mcp/resources/read?server={server_name}&uri=memory%3A%2F%2Fdemo").json()
    assert content["content"] == "resource body"

    prompts = client.get(f"/api/mcp/prompts?server={server_name}").json()
    assert prompts["prompts"][server_name][0]["name"] == "hello"
    prompt_result = client.post("/api/mcp/prompts/execute", json={"server": server_name, "promptName": "hello", "arguments": {"name": "Python"}}).json()
    assert prompt_result["success"] is True
    assert prompt_result["messages"][0]["content"] == "Hello Python"

    cap_id = f"cap-{uuid4().hex}"
    cap = client.post("/api/mcp/capabilities", json={"id": cap_id, "toolName": "list_files", "domain": "workspace"}).json()
    assert cap["enabled"] is True
    assert client.post(f"/api/mcp/capabilities/{cap_id}/test").json()["status"] == "reachable"
    invoked = client.post(f"/api/mcp/capabilities/{cap_id}/invoke", json={"arguments": {"pattern": "backend-python/*.py", "limit": 5}}).json()
    assert invoked["status"] == "success"
    assert invoked["connectionType"] == "local"

    failure = client.post(f"/api/mcp/auth-failures/{server_name}").json()
    assert failure["cached"] is True
    assert client.delete(f"/api/mcp/auth-failures/{server_name}").json()["cached"] is False
    approval = client.post(f"/api/mcp/approvals/{server_name}", json={"decision": "allow", "reason": "test"}).json()
    assert approval["trusted"] is True
    token = client.post("/api/mcp/tokens/encrypt", json={"token": "secret-token"}).json()
    assert token["roundTrip"] is True
    assert token["encrypted"] != "secret-token"


def test_mcp_reconnect_respects_auth_cache_and_status_reports_schema_validation() -> None:
    server_name = f"mcp-auth-{uuid4().hex}"
    server = {
        "name": server_name,
        "type": "streamable-http",
        "url": "http://localhost:9/mcp",
        "status": "failed",
        "tools": [
            {"name": "safe_tool", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}},
            {"name": "bad_tool", "inputSchema": {"type": "array"}},
        ],
    }
    client.post("/api/mcp/servers", json=server)
    try:
        status = client.get("/api/mcp/status").json()
        assert status["schemaValidation"]["invalidCount"] >= 1
        assert any(item["toolName"] == "bad_tool" for item in status["schemaValidation"]["invalidTools"])

        client.post(f"/api/mcp/auth-failures/{server_name}")
        blocked = client.post(f"/api/mcp/reconnect?server={server_name}").json()
        assert blocked["success"] is False
        assert blocked["status"] == "blocked"
        assert blocked["reason"] == "auth_failure_cached"
        assert blocked["backoffMs"] > 0

        client.delete(f"/api/mcp/auth-failures/{server_name}")
        allowed = client.post(f"/api/mcp/reconnect?server={server_name}").json()
        assert allowed["success"] is True
        assert allowed["status"] == "CONNECTED"
        assert allowed["connection"]["nextRetryAt"] is None
    finally:
        client.delete(f"/api/mcp/servers/{server_name}")


def test_mcp_capability_invokes_remote_tool_via_manager_call_semantics() -> None:
    server_name = f"mcp-call-{uuid4().hex}"
    cap_id = f"cap-call-{uuid4().hex}"
    server = {
        "name": server_name,
        "type": "sse",
        "status": "connected",
        "tools": [
            {
                "name": "remote_echo",
                "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}},
                "result": {"content": [{"type": "text", "text": "remote reply"}], "isError": False},
            }
        ],
    }
    client.post("/api/mcp/servers", json=server)
    client.post("/api/mcp/capabilities", json={"id": cap_id, "serverName": server_name, "toolName": "remote_echo"})
    try:
        wrapped_name = f"mcp__{server_name}__remote_echo"
        tools = client.get("/api/tools").json()
        assert any(tool["name"] == wrapped_name and tool["group"] == "mcp" for tool in tools)
        detail = client.get(f"/api/tools/{wrapped_name}").json()
        assert detail["inputSchema"]["properties"]["message"]["type"] == "string"

        invoked = client.post(f"/api/mcp/capabilities/{cap_id}/invoke", json={"arguments": {"message": "hi"}}).json()
        assert invoked["status"] == "success"
        assert invoked["connectionType"] == "sse"
        assert invoked["result"]["content"] == "remote reply"
        assert invoked["result"]["request"]["method"] == "tools/call"
        assert invoked["result"]["request"]["params"] == {"name": "remote_echo", "arguments": {"message": "hi"}}
    finally:
        client.delete(f"/api/mcp/capabilities/{cap_id}")
        client.delete(f"/api/mcp/servers/{server_name}")


def test_mcp_stdio_server_restart_discovers_and_invokes_real_transport() -> None:
    server_name = f"mcp-stdio-{uuid4().hex}"
    cap_id = f"cap-stdio-{uuid4().hex}"
    server_script = BACKEND_DIR / "tests" / "fixtures" / "mcp_stdio_echo_server.py"
    created = client.post(
        "/api/mcp/servers",
        json={
            "name": server_name,
            "type": "stdio",
            "command": sys.executable,
            "args": [str(server_script)],
            "status": "pending",
        },
    ).json()
    assert created["success"] is True
    try:
        restarted = client.post(f"/api/mcp/servers/{server_name}/restart").json()
        assert restarted["success"] is True
        assert restarted["status"] == "connected"
        assert restarted["connection"]["tools"][0]["name"] == "remote_echo"

        wrapped_name = f"mcp__{server_name}__remote_echo"
        tools = client.get("/api/tools").json()
        assert any(tool["name"] == wrapped_name for tool in tools)

        client.post("/api/mcp/capabilities", json={"id": cap_id, "serverName": server_name, "toolName": "remote_echo"})
        invoked = client.post(f"/api/mcp/capabilities/{cap_id}/invoke", json={"arguments": {"message": "api"}}).json()
        assert invoked["status"] == "success"
        assert invoked["connectionType"] == "stdio"
        assert invoked["result"]["content"] == "stdio:api"
    finally:
        client.delete(f"/api/mcp/capabilities/{cap_id}")
        client.delete(f"/api/mcp/servers/{server_name}")


def test_mcp_streamable_http_server_restart_discovers_and_invokes_real_transport() -> None:
    http_server, url = run_http_mcp_echo_server()
    server_name = f"mcp-http-{uuid4().hex}"
    cap_id = f"cap-http-{uuid4().hex}"
    created = client.post(
        "/api/mcp/servers",
        json={
            "name": server_name,
            "type": "streamable_http",
            "url": url,
            "status": "pending",
        },
    ).json()
    assert created["success"] is True
    try:
        restarted = client.post(f"/api/mcp/servers/{server_name}/restart").json()
        assert restarted["success"] is True
        assert restarted["status"] == "connected"
        assert restarted["connection"]["tools"][0]["name"] == "remote_echo"
        assert "initialize" in HttpMcpEchoHandler.requests
        assert "tools/list" in HttpMcpEchoHandler.requests

        wrapped_name = f"mcp__{server_name}__remote_echo"
        tools = client.get("/api/tools").json()
        assert any(tool["name"] == wrapped_name for tool in tools)

        client.post("/api/mcp/capabilities", json={"id": cap_id, "serverName": server_name, "toolName": "remote_echo"})
        invoked = client.post(f"/api/mcp/capabilities/{cap_id}/invoke", json={"arguments": {"message": "api"}}).json()
        assert invoked["status"] == "success"
        assert invoked["connectionType"] == "streamable_http"
        assert invoked["result"]["content"] == "http-api:api"
        assert HttpMcpEchoHandler.requests[-1] == "tools/call"
    finally:
        client.delete(f"/api/mcp/capabilities/{cap_id}")
        client.delete(f"/api/mcp/servers/{server_name}")
        http_server.shutdown()


def test_dialog_decisions_are_persisted() -> None:
    snapshot_id = f"snapshot-{uuid4().hex}"
    snapshot = client.post(f"/api/dialogs/snapshot-update/{snapshot_id}/decision", json={"action": "replace"}).json()
    assert snapshot["action"] == "replace"
    assert STATE["dialogDecisions"][snapshot_id]["action"] == "replace"

    permission_id = f"plugin-{uuid4().hex}"
    permission = client.post(f"/api/dialogs/plugin-permission/{permission_id}/decision", json={"allowed": True}).json()
    assert permission["decision"]["decision"] == "allow"
    assert STATE["permissionResponses"][permission_id]["allowed"] is True


def test_swarm_worker_abort_and_permission_resolution() -> None:
    team_name = f"py-team-{uuid4().hex[:8]}"
    session_id = f"swarm-session-{uuid4().hex[:8]}"
    created = client.post(
        "/api/swarm",
        json={"teamName": team_name, "maxWorkers": 2, "sessionId": session_id, "tasks": ["inspect", "fix"], "workerUseLlm": False},
    ).json()
    swarm_id = created["swarmId"]
    assert created["phase"] == "RUNNING"
    assert created["totalWorkers"] == 2
    assert "worker-1" in created["workers"]
    assert created["workflow"]["currentPhase"]["name"] == "Research"
    workflow = client.get(f"/api/coordinator/workflows/{session_id}").json()
    assert workflow["workflowId"] == created["workflowId"]
    advanced = client.post(f"/api/coordinator/workflows/{session_id}/advance", json={"summary": "research done"}).json()
    assert advanced["currentPhase"]["name"] == "Synthesis"
    scratchpad = client.post(f"/api/coordinator/workflows/{session_id}/scratchpad", json={"author": "worker-1", "content": "found route"}).json()
    assert scratchpad["item"]["author"] == "worker-1"
    detected = client.post("/api/coordinator/detect-phase", json={"sessionId": session_id, "output": "FileEdit will modify app.py"}).json()
    assert detected["detected"]["name"] == "Implementation"
    delegation = client.post("/api/coordinator/validate-delegation", json={"phase": "Implementation", "prompt": "fix the bug"}).json()
    assert delegation["valid"] is False

    aborted = client.post(f"/api/swarm/{swarm_id}/worker/worker-1/abort", json={"reason": "test abort", "sessionId": session_id}).json()
    assert aborted["status"] == "aborted"
    swarm = client.get(f"/api/swarm/{swarm_id}").json()
    assert swarm["workers"]["worker-1"]["status"] == "TERMINATED"

    permission = client.post("/api/swarm/permission/perm-1", json={"approved": True}).json()
    assert permission["decision"] == "allow"

    shutdown = client.post(f"/api/swarm/{swarm_id}/shutdown").json()
    assert shutdown["status"] == "shutdown_initiated"


def test_swarm_runs_workers_concurrently_mailbox_and_aggregates_results() -> None:
    team_name = f"concurrent-team-{uuid4().hex[:8]}"
    created = client.post(
        "/api/swarm",
        json={
            "teamName": team_name,
            "maxWorkers": 4,
            "sessionId": f"swarm-concurrent-{uuid4().hex[:8]}",
            "tasks": ["research api", "inspect files", "write summary"],
            "awaitCompletion": True,
            "stepDelayMs": 5,
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    assert created["executionBackend"] == "python-asyncio"
    assert created["phase"] == "IDLE"
    assert created["completedTasks"] == 3
    assert len(created["results"]) == 3
    assert "Team Results" in created["aggregateResult"]
    assert all(worker["status"] == "IDLE" for worker in created["workers"].values())

    mailed = client.post(f"/api/swarm/{swarm_id}/worker/worker-1/mail", json={"senderId": "leader", "content": "use cached findings"}).json()
    assert mailed["success"] is True
    mailbox = client.get(f"/api/swarm/{swarm_id}/mailbox/worker-1?drain=false").json()
    assert mailbox["count"] >= 1
    drained = client.get(f"/api/swarm/{swarm_id}/mailbox/worker-1").json()
    assert any(item["content"] == "use cached findings" for item in drained["messages"])

    broadcast = client.post(f"/api/swarm/{swarm_id}/broadcast", json={"content": "sync point"}).json()
    assert broadcast["count"] == 3
    coordinator_events = client.get(f"/api/swarm/{swarm_id}/coordinator-events").json()["events"]
    assert any(event["type"] == "coordinator_event" and event["eventType"] == "mailbox_broadcast" for event in coordinator_events)
    results = client.get(f"/api/swarm/{swarm_id}/results").json()
    assert len(results["results"]) == 3
    assert "workers completed" in results["aggregateResult"]

    added = client.post(f"/api/swarm/{swarm_id}/workers", json={"task": "late verification", "awaitCompletion": True}).json()
    assert added["success"] is True
    assert added["swarm"]["workers"][added["workerId"]]["status"] == "IDLE"


def test_swarm_mailbox_coordinator_events_include_source_compatible_fields() -> None:
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"mailbox-schema-team-{uuid4().hex[:8]}",
            "maxWorkers": 2,
            "sessionId": f"swarm-mailbox-schema-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    STATE["swarms"][swarm_id]["workers"] = {
        "worker-1": {"workerId": "worker-1", "status": "WORKING", "currentTask": "read"},
        "worker-2": {"workerId": "worker-2", "status": "WORKING", "currentTask": "review"},
    }

    mailed = client.post(
        f"/api/swarm/{swarm_id}/worker/worker-1/mail",
        json={"senderId": f"{swarm_id}-leader", "content": "handoff note", "phase": "Research", "channel": "handoff", "taskId": "task-a"},
    ).json()

    message = mailed["message"]
    assert message["messageId"] == message["id"]
    assert message["createdAt"] == message["timestamp"]
    assert message["contentLength"] == len("handoff note")

    coordinator_events = client.get(f"/api/swarm/{swarm_id}/coordinator-events").json()["events"]
    mailbox_event = next(event for event in coordinator_events if event["eventType"] == "mailbox_write")
    assert mailbox_event["type"] == "coordinator_event"
    assert mailbox_event["workflowId"] == swarm_id
    assert mailbox_event["swarmId"] == swarm_id
    assert mailbox_event["teamPrefix"] == swarm_id
    assert isinstance(mailbox_event["ts"], int)
    assert mailbox_event["uuid"]

    payload = mailbox_event["payload"]
    assert payload["messageId"] == message["id"]
    assert payload["senderId"] == f"{swarm_id}-leader"
    assert payload["recipientId"] == "worker-1"
    assert payload["content"] == "handoff note"
    assert payload["contentLength"] == len("handoff note")
    assert payload["phase"] == "Research"
    assert payload["phaseIndex"] == 0
    assert payload["channel"] == "handoff"
    assert payload["taskId"] == "task-a"
    assert payload["mailboxDepth"] >= 1


def test_swarm_worker_uses_isolated_query_session_and_team_runtime(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"isolated-query-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-query-session-{uuid4().hex[:8]}",
            "tasks": ["explain isolated worker execution"],
            "awaitCompletion": True,
            "stepDelayMs": 1,
        },
    ).json()
    swarm_id = created["swarmId"]
    worker = created["workers"]["worker-1"]
    result = created["results"]["worker-1"]
    worker_session_id = worker["sessionId"]

    assert result["turns"][0]["sessionId"] == worker_session_id
    assert "explain isolated worker execution" in result["turns"][0]["answer"]
    messages = client.get(f"/api/sessions/{worker_session_id}/messages").json()["messages"]
    assert [message["type"] for message in messages[-2:]] == ["user", "assistant"]
    loops = client.get(f"/api/query/loops?sessionId={worker_session_id}").json()["loops"]
    assert any(event["type"] == "worker_query_start" for loop in loops for event in loop["events"])

    runtime = client.get(f"/api/swarm/{swarm_id}/runtime").json()
    assert runtime["team"]["teamName"].startswith("isolated-query-team-")
    assert runtime["workers"]["worker-1"]["sessionId"] == worker_session_id
    teams = client.get("/api/teams").json()["teams"]
    assert any(team["swarmId"] == swarm_id and team["workerIds"] == ["worker-1"] for team in teams)

    client.post(f"/api/swarm/{swarm_id}/worker/worker-1/mail", json={"content": "cleanup this"})
    shutdown = client.post(f"/api/swarm/{swarm_id}/shutdown").json()
    assert shutdown["success"] is True
    runtime_after = client.get(f"/api/swarm/{swarm_id}/runtime").json()
    assert runtime_after["team"]["status"] == "TERMINATED"
    assert runtime_after["mailboxes"]["worker-1"] == 0


def test_swarm_worker_session_messages_are_capped_like_original_runner() -> None:
    turns = [{"prompt": f"worker turn {index}", "toolCalls": []} for index in range(30)]
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"message-cap-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-message-cap-{uuid4().hex[:8]}",
            "tasks": ["long worker context"],
            "workerTurns": {"worker-1": turns},
            "awaitCompletion": True,
            "workerUseLlm": False,
        },
    ).json()
    worker_session_id = created["workers"]["worker-1"]["sessionId"]
    messages = client.get(f"/api/sessions/{worker_session_id}/messages").json()["messages"]
    assert len(messages) == WORKER_MESSAGE_CAP
    assert [message["type"] for message in messages[-2:]] == ["user", "assistant"]


def test_swarm_task_spec_scratchpad_mailbox_drain_and_feature_gate() -> None:
    scratchpad = f"backend-python/.test-workspace/{uuid4().hex}/scratch"
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"spec-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-spec-{uuid4().hex}",
            "scratchpadDir": scratchpad,
            "tasks": [
                {
                    "prompt": "inspect task spec",
                    "agentType": "reviewer",
                    "model": "worker-special",
                    "turns": [{"prompt": "turn from spec"}],
                }
            ],
            "awaitCompletion": True,
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    worker = created["workers"]["worker-1"]
    assert worker["agentType"] == "reviewer"
    assert worker["model"] == "worker-special"
    assert Path(created["scratchpadDir"]).exists()
    session = client.get(f"/api/sessions/{worker['sessionId']}").json()["session"]
    assert session["workingDirectory"] == created["scratchpadDir"]
    assert session["model"] == "worker-special"
    assert "worker agent in a Swarm team" in STATE["sessions"][worker["sessionId"]]["systemPrompt"]
    assert "reviewer" in STATE["sessions"][worker["sessionId"]]["systemPrompt"]
    assert created["results"]["worker-1"]["turns"][0]["prompt"] == "turn from spec"

    client.post(f"/api/swarm/{swarm_id}/worker/worker-1/mail", json={"content": "drain once"})
    assert client.get(f"/api/swarm/{swarm_id}/mailbox/worker-1?drain=false").json()["count"] == 1
    assert client.get(f"/api/swarm/{swarm_id}/mailbox/worker-1").json()["count"] == 1
    assert client.get(f"/api/swarm/{swarm_id}/mailbox/worker-1?drain=false").json()["count"] == 0

    previous_flags = dict(STATE["config"].get("featureFlags") or {})
    try:
        STATE["config"]["featureFlags"] = {**previous_flags, "ENABLE_AGENT_SWARMS": False}
        disabled = client.post("/api/swarm", json={"teamName": "disabled", "tasks": []})
        assert disabled.status_code == 409
    finally:
        STATE["config"]["featureFlags"] = previous_flags


def test_swarm_scratchpad_path_must_stay_inside_workspace() -> None:
    escaped = client.post(
        "/api/swarm",
        json={"teamName": f"scratch-escape-{uuid4().hex[:8]}", "maxWorkers": 1, "scratchpadDir": "../outside-scratch", "tasks": []},
    )
    assert escaped.status_code == 400
    assert "escapes workspace" in escaped.json()["detail"]


def test_swarm_team_creation_enforces_original_team_manager_constraints() -> None:
    team_name = f"unique-team-{uuid4().hex[:8]}"
    first = client.post("/api/swarm", json={"teamName": team_name, "maxWorkers": 1, "sessionId": f"team-constraints-{uuid4().hex}", "tasks": []})
    assert first.status_code == 200

    duplicate = client.post("/api/swarm", json={"teamName": team_name, "maxWorkers": 1, "sessionId": f"team-constraints-{uuid4().hex}", "tasks": []})
    assert duplicate.status_code == 409
    assert "Team already exists" in duplicate.json()["detail"]

    too_few = client.post("/api/swarm", json={"teamName": f"too-few-{uuid4().hex[:8]}", "maxWorkers": 0, "tasks": []})
    assert too_few.status_code == 400
    assert "between 1 and 20" in too_few.json()["detail"]

    too_many = client.post("/api/swarm", json={"teamName": f"too-many-{uuid4().hex[:8]}", "maxWorkers": 21, "tasks": []})
    assert too_many.status_code == 400
    assert "between 1 and 20" in too_many.json()["detail"]


def test_swarm_team_name_validation_matches_controller_security_contract() -> None:
    invalid_names = ["../../../tmp/pwned", "ok/bad", "ok\\bad", "team name", "", "a" * 65]
    for name in invalid_names:
        response = client.post("/api/swarm", json={"teamName": name, "maxWorkers": 2, "tasks": []})
        assert response.status_code == 400
        assert "Invalid teamName" in response.json()["detail"]

    valid = client.post("/api/swarm", json={"teamName": f"team-alpha_01-{uuid4().hex[:8]}", "maxWorkers": 2, "tasks": []})
    assert valid.status_code == 200
    defaulted = client.post("/api/swarm", json={"maxWorkers": 1, "tasks": []})
    assert defaulted.status_code == 200


def test_swarm_abort_worker_requires_owner_session_or_bound_principal() -> None:
    session_id = f"swarm-owner-{uuid4().hex[:8]}"
    created = client.post(
        "/api/swarm",
        json={"teamName": f"auth-team-{uuid4().hex[:8]}", "maxWorkers": 1, "sessionId": session_id, "tasks": ["auth gated"], "stepDelayMs": 200, "workerUseLlm": False},
    ).json()
    swarm_id = created["swarmId"]

    wrong = client.post(f"/api/swarm/{swarm_id}/worker/worker-1/abort", json={"reason": "wrong", "sessionId": "wrong-session"})
    assert wrong.status_code == 403

    missing = client.post(f"/api/swarm/{swarm_id}/worker/worker-1/abort", json={"reason": "missing session"})
    assert missing.status_code == 403

    principal = f"user-{uuid4().hex[:8]}"
    assert client.post("/api/ws/sessions/bind", json={"principal": principal, "sessionId": session_id}).json()["success"] is True
    allowed = client.post(f"/api/swarm/{swarm_id}/worker/worker-1/abort", json={"reason": "principal abort"}, headers={"X-Principal": principal})
    assert allowed.status_code == 200
    assert allowed.json()["success"] is True


def test_swarm_list_defaults_to_active_only_but_can_include_terminated() -> None:
    session_id = f"swarm-list-{uuid4().hex[:8]}"
    created = client.post(
        "/api/swarm",
        json={"teamName": f"list-team-{uuid4().hex[:8]}", "maxWorkers": 1, "sessionId": session_id, "tasks": []},
    ).json()
    swarm_id = created["swarmId"]
    assert any(item["swarmId"] == swarm_id for item in client.get("/api/swarm").json()["swarms"])
    assert any(item["swarmId"] == swarm_id for item in client.get("/api/teams").json()["teams"])

    client.post(f"/api/swarm/{swarm_id}/shutdown")
    active = client.get("/api/swarm").json()["swarms"]
    assert all(item["swarmId"] != swarm_id for item in active)
    all_swarms = client.get("/api/swarm?activeOnly=false").json()["swarms"]
    assert any(item["swarmId"] == swarm_id and item["phase"] == "TERMINATED" for item in all_swarms)
    active_teams = client.get("/api/teams").json()["teams"]
    assert all(item["swarmId"] != swarm_id for item in active_teams)
    all_teams = client.get("/api/teams?activeOnly=false").json()["teams"]
    assert any(item["swarmId"] == swarm_id and item["status"] == "TERMINATED" for item in all_teams)


def test_swarm_abort_cancels_running_worker() -> None:
    session_id = f"swarm-abort-{uuid4().hex[:8]}"
    created = client.post(
        "/api/swarm",
        json={"teamName": f"abort-team-{uuid4().hex[:8]}", "maxWorkers": 1, "sessionId": session_id, "tasks": ["long task"], "stepDelayMs": 200, "workerUseLlm": False},
    ).json()
    swarm_id = created["swarmId"]
    aborted = client.post(f"/api/swarm/{swarm_id}/worker/worker-1/abort", json={"reason": "user stop", "sessionId": session_id}).json()
    assert aborted["success"] is True
    fetched = client.get(f"/api/swarm/{swarm_id}").json()
    assert fetched["workers"]["worker-1"]["status"] == "TERMINATED"
    assert fetched["workers"]["worker-1"]["terminationReason"] == "user stop"


def test_swarm_worker_multi_turn_tool_plan_and_queue_completion() -> None:
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"queued-team-{uuid4().hex[:8]}",
            "maxWorkers": 2,
            "sessionId": f"swarm-queue-{uuid4().hex[:8]}",
            "tasks": ["task 1", "task 2", "task 3", "task 4"],
            "workerTurns": {
                "worker-1": [
                    {"prompt": "list first", "toolCalls": [{"id": "t1", "name": "list_files", "arguments": {"pattern": "backend-python/*.py", "limit": 2}}]},
                    {"prompt": "search second", "toolCalls": [{"id": "t2", "name": "search_files", "arguments": {"query": "FastAPI", "limit": 2}}]},
                ]
            },
            "awaitCompletion": True,
            "stepDelayMs": 1,
            "workerUseLlm": False,
        },
    ).json()
    assert created["phase"] == "IDLE"
    assert created["totalTasks"] == 4
    assert len(created["results"]) == 4
    assert created["queuedTasks"] == []
    first = created["results"]["worker-1"]
    assert len(first["turns"]) == 2
    assert first["turns"][0]["queryLoop"]["toolCalls"][0]["toolName"] == "list_files"
    assert "<task-notification>" in first["notificationXml"]


def test_swarm_permission_bubble_allows_worker_to_continue() -> None:
    rel = f"backend-python/.test-workspace/{uuid4().hex}/permission-write.txt"
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"permission-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-permission-{uuid4().hex[:8]}",
            "tasks": ["write guarded file"],
            "workerToolCalls": {
                "worker-1": [{"id": "write-1", "name": "write_file", "arguments": {"path": rel, "content": "approved write"}}]
            },
            "permissionTimeoutMs": 5000,
            "stepDelayMs": 1,
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]

    pending = wait_until(lambda: client.get(f"/api/swarm/{swarm_id}").json().get("pendingPermissions"), timeout=3)
    assert pending
    request_id = pending[0]["requestId"]
    assert pending[0]["toolName"] == "write_file"
    assert client.post(f"/api/swarm/permission/{request_id}", json={"approved": True}).json()["decision"] == "allow"

    completed = wait_until(
        lambda: client.get(f"/api/swarm/{swarm_id}").json()
        if client.get(f"/api/swarm/{swarm_id}").json()["workers"]["worker-1"]["status"] == "IDLE"
        else None,
        timeout=5,
    )
    assert completed
    assert completed["results"]["worker-1"]["status"] == "completed"
    worker = completed["workers"]["worker-1"]
    assert worker["toolCallCount"] >= 1
    assert worker["tokenConsumed"] >= 1
    assert worker["recentToolCalls"][-1] == "write_file"
    recent_record = worker["recentToolCallRecords"][-1]
    assert recent_record["toolName"] == "write_file"
    assert recent_record["status"] == "success"
    assert recent_record["paramsHash"] != "empty"
    assert (ROOT / rel).read_text(encoding="utf-8") == "approved write"


def test_swarm_permission_bubble_publishes_leader_bridge_coordinator_event() -> None:
    rel = f"backend-python/.test-workspace/{uuid4().hex}/permission-bridge-write.txt"
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"permission-bridge-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-permission-bridge-{uuid4().hex[:8]}",
            "tasks": ["write guarded file"],
            "workerToolCalls": {
                "worker-1": [{"id": "write-bridge-1", "name": "write_file", "arguments": {"path": rel, "content": "approval needed"}}]
            },
            "permissionTimeoutMs": 5000,
            "stepDelayMs": 1,
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]

    pending = wait_until(lambda: client.get(f"/api/swarm/{swarm_id}").json().get("pendingPermissions"), timeout=3)
    assert pending
    request_id = pending[0]["requestId"]

    coordinator_events = client.get(f"/api/swarm/{swarm_id}/coordinator-events").json()["events"]
    bubble_event = next(event for event in coordinator_events if event["eventType"] == "permission_bubble")
    payload = bubble_event["payload"]

    assert bubble_event["workflowId"] == swarm_id
    assert bubble_event["swarmId"] == swarm_id
    assert bubble_event["sessionId"] == created["sessionId"]
    assert payload["requestId"] == request_id
    assert payload["workerId"] == "worker-1"
    assert payload["toolName"] == "write_file"
    assert payload["riskLevel"] == "high"
    assert payload["reason"]
    assert payload["timeoutMs"] == 5000
    assert payload["expiresAt"] == pending[0]["expiresAt"]
    assert payload["remainingMs"] <= 5000
    assert payload["pendingRequestCount"] == 1
    assert payload["leaderSessionId"] == created["sessionId"]

    assert client.post(f"/api/swarm/permission/{request_id}", json={"approved": False}).json()["decision"] == "deny"


def test_swarm_permission_pending_count_and_clear_release_waiters() -> None:
    class FakeWaiter:
        def __init__(self) -> None:
            self.value = None

        def done(self) -> bool:
            return self.value is not None

        def set_result(self, value) -> None:
            self.value = value

    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"permission-clear-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-permission-clear-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    request_id = f"perm-{uuid4().hex}"
    STATE["swarms"][swarm_id]["pendingPermissions"].append(
        {
            "requestId": request_id,
            "swarmId": swarm_id,
            "workerId": "worker-1",
            "toolName": "write_file",
            "input": {"path": "blocked.txt", "content": "blocked"},
            "riskLevel": "high",
            "reason": "test pending",
            "status": "pending",
            "createdAt": app_module.utc_now(),
            "timeoutMs": 5000,
        }
    )
    waiter = FakeWaiter()
    SWARM_PERMISSION_WAITERS[request_id] = waiter

    counted = client.get(f"/api/swarm/permissions/pending-count?swarmId={swarm_id}").json()
    assert counted["pendingRequestCount"] == 1

    cleared = client.post(f"/api/swarm/{swarm_id}/permissions/clear", json={"reason": "test clear"}).json()
    assert cleared["cleared"] == 1
    assert cleared["pendingRequestCount"] == 0
    assert waiter.done()
    assert waiter.value["decision"] == "deny"
    assert request_id not in SWARM_PERMISSION_WAITERS


def test_swarm_permission_pending_count_includes_deadline_metadata() -> None:
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"permission-deadline-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-permission-deadline-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    request_id = f"perm-deadline-{uuid4().hex}"
    STATE["swarms"][swarm_id]["pendingPermissions"].append(
        {
            "requestId": request_id,
            "swarmId": swarm_id,
            "workerId": "worker-1",
            "toolName": "write_file",
            "input": {"path": "blocked.txt", "content": "blocked"},
            "riskLevel": "high",
            "reason": "test pending",
            "status": "pending",
            "createdAt": app_module.utc_now(),
            "timeoutMs": 5000,
        }
    )

    counted = client.get(f"/api/swarm/permissions/pending-count?swarmId={swarm_id}").json()
    request = counted["requests"][0]

    assert counted["pendingRequestCount"] == 1
    assert request["requestId"] == request_id
    assert request["expiresAt"]
    assert 0 <= request["elapsedMs"] <= request["timeoutMs"]
    assert 0 < request["remainingMs"] <= request["timeoutMs"]
    assert request["deadlineStatus"] == "pending"


def test_swarm_permission_timeout_auto_denies_and_releases_waiter() -> None:
    class FakeWaiter:
        def __init__(self) -> None:
            self.value = None

        def done(self) -> bool:
            return self.value is not None

        def set_result(self, value) -> None:
            self.value = value

    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"permission-expire-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-permission-expire-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    request_id = f"perm-expired-{uuid4().hex}"
    STATE["swarms"][swarm_id]["workers"] = {
        "worker-1": {"workerId": "worker-1", "status": "WAITING_PERMISSION", "currentTask": "guarded write"}
    }
    STATE["swarms"][swarm_id]["pendingPermissions"].append(
        {
            "requestId": request_id,
            "swarmId": swarm_id,
            "workerId": "worker-1",
            "toolName": "write_file",
            "input": {"path": "blocked.txt", "content": "blocked"},
            "riskLevel": "high",
            "reason": "test pending",
            "status": "pending",
            "createdAt": "2000-01-01T00:00:00Z",
            "timeoutMs": 1,
        }
    )
    waiter = FakeWaiter()
    SWARM_PERMISSION_WAITERS[request_id] = waiter

    counted = client.get(f"/api/swarm/permissions/pending-count?swarmId={swarm_id}").json()
    swarm = client.get(f"/api/swarm/{swarm_id}").json()

    assert counted["pendingRequestCount"] == 0
    assert counted["expiredCount"] == 1
    assert counted["expired"][0]["requestId"] == request_id
    assert waiter.done()
    assert waiter.value["decision"] == "deny"
    assert waiter.value["reason"] == "permission timeout"
    assert request_id not in SWARM_PERMISSION_WAITERS
    assert STATE["permissionResponses"][request_id]["expired"] is True
    assert swarm["workers"]["worker-1"]["status"] == "TERMINATED"
    assert swarm["workers"]["worker-1"]["terminationReason"] == "permission_denied"
    assert swarm["results"]["worker-1"]["status"] == "failed"


def test_swarm_permission_batch_resolution_releases_waiters() -> None:
    class FakeWaiter:
        def __init__(self) -> None:
            self.value = None

        def done(self) -> bool:
            return self.value is not None

        def set_result(self, value) -> None:
            self.value = value

    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"permission-batch-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-permission-batch-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    request_ids = [f"perm-batch-{uuid4().hex}", f"perm-batch-{uuid4().hex}"]
    waiters = []
    for request_id in request_ids:
        STATE["swarms"][swarm_id]["pendingPermissions"].append(
            {
                "requestId": request_id,
                "swarmId": swarm_id,
                "workerId": "worker-1",
                "toolName": "write_file",
                "input": {"path": "blocked.txt", "content": "blocked"},
                "riskLevel": "high",
                "reason": "test pending",
                "status": "pending",
                    "createdAt": app_module.utc_now(),
                "timeoutMs": 5000,
            }
        )
        waiter = FakeWaiter()
        SWARM_PERMISSION_WAITERS[request_id] = waiter
        waiters.append(waiter)

    resolved = client.post(
        "/api/swarm/permissions/batch",
        json={"requestIds": request_ids, "decision": "deny", "reason": "batch deny"},
    ).json()

    assert resolved["success"] is True
    assert resolved["processedCount"] == 2
    assert resolved["pendingRequestCount"] == 0
    assert all(item["decision"] == "deny" for item in resolved["processed"])
    assert all(waiter.done() and waiter.value["decision"] == "deny" for waiter in waiters)
    assert all(request_id not in SWARM_PERMISSION_WAITERS for request_id in request_ids)


def test_swarm_shared_task_list_adds_durable_queued_tasks() -> None:
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"task-list-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-task-list-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]

    added = client.post(f"/api/swarm/{swarm_id}/tasks", json={"task": "queued durable work", "autoStart": False}).json()
    listed = client.get(f"/api/swarm/{swarm_id}/tasks").json()

    assert added["success"] is True
    assert added["queuedTask"]["description"] == "queued durable work"
    assert listed["counts"]["queued"] == 1
    assert listed["tasks"][0]["status"] == "queued"
    assert listed["tasks"][0]["description"] == "queued durable work"
    assert STATE["swarms"][swarm_id]["queuedTasks"][0]["description"] == "queued durable work"


def test_swarm_shared_task_list_emits_java_compatible_task_event() -> None:
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"task-event-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-task-event-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    team_name = created["teamName"]

    added = client.post(
        f"/api/swarm/{swarm_id}/tasks",
        json={"task": "queued shared task", "creatorId": "leader", "autoStart": False},
    ).json()
    queued = added["queuedTask"]

    assert queued["teamName"] == team_name
    assert queued["creatorId"] == "leader"
    assert queued["status"] == "PENDING"
    assert queued["assigneeId"] is None
    assert queued["result"] is None
    assert queued["completedAt"] is None

    coordinator_events = client.get(f"/api/swarm/{swarm_id}/coordinator-events").json()["events"]
    task_event = next(event for event in coordinator_events if event["eventType"] == "shared_task_queued")
    payload = task_event["payload"]

    assert task_event["workflowId"] == swarm_id
    assert task_event["swarmId"] == swarm_id
    assert payload["taskId"] == queued["taskId"]
    assert payload["teamName"] == team_name
    assert payload["description"] == "queued shared task"
    assert payload["creatorId"] == "leader"
    assert payload["status"] == "PENDING"
    assert payload["pendingCount"] == 1
    assert payload["totalTaskCount"] == 1


def test_swarm_recover_requeues_orphaned_running_worker_after_restart() -> None:
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"recover-team-{uuid4().hex[:8]}",
            "maxWorkers": 2,
            "sessionId": f"swarm-recover-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    STATE["swarms"][swarm_id]["workers"] = {
        "worker-1": {
            "workerId": "worker-1",
            "status": "WORKING",
            "currentTask": "resume lost research",
            "turns": [{"prompt": "resume lost research"}],
            "toolCalls": [],
            "progressPercent": 40,
            "startedAt": "2026-06-30T00:00:00Z",
        },
        "worker-2": {
            "workerId": "worker-2",
            "status": "IDLE",
            "currentTask": "already finished",
            "progressPercent": 100,
        },
    }
    app_module.SWARM_TASKS.pop(swarm_id, None)

    recovered = client.post(f"/api/swarm/{swarm_id}/recover", json={"autoStart": False, "reason": "process restart"}).json()
    fetched = client.get(f"/api/swarm/{swarm_id}").json()
    task_list = client.get(f"/api/swarm/{swarm_id}/tasks").json()
    coordinator_events = client.get(f"/api/swarm/{swarm_id}/coordinator-events").json()["events"]

    assert recovered["success"] is True
    assert recovered["recoveredCount"] == 1
    assert recovered["recoveredWorkers"][0]["workerId"] == "worker-1"
    assert recovered["queuedTaskCount"] == 1
    assert fetched["workers"]["worker-1"]["status"] == "TERMINATED"
    assert fetched["workers"]["worker-1"]["terminationReason"] == "requeued_after_restart"
    assert fetched["queuedTasks"][0]["recoveredFromWorkerId"] == "worker-1"
    assert fetched["queuedTasks"][0]["prompt"] == "resume lost research"
    assert task_list["counts"]["queued"] == 1
    assert any(event["eventType"] == "worker_requeued_after_recovery" and event["payload"]["workerId"] == "worker-1" for event in coordinator_events)


def test_startup_recovery_scan_requeues_orphaned_workers_across_swarms() -> None:
    first = client.post(
        "/api/swarm",
        json={
            "teamName": f"startup-recover-a-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"startup-recover-a-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    second = client.post(
        "/api/swarm",
        json={
            "teamName": f"startup-recover-b-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"startup-recover-b-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    first_id = first["swarmId"]
    second_id = second["swarmId"]
    STATE["swarms"][first_id]["workers"] = {
        "worker-1": {"workerId": "worker-1", "status": "WORKING", "currentTask": "recover first worker"}
    }
    STATE["swarms"][second_id]["workers"] = {
        "worker-1": {"workerId": "worker-1", "status": "WAITING_PERMISSION", "currentTask": "recover second worker"}
    }
    app_module.SWARM_TASKS.pop(first_id, None)
    app_module.SWARM_TASKS.pop(second_id, None)

    recovery_scan = getattr(app_module, "recover_all_orphaned_swarm_workers", None)
    assert callable(recovery_scan)
    result = recovery_scan(reason="startup scan", swarm_ids=[first_id, second_id])

    assert result["success"] is True
    assert result["recoveredSwarmCount"] == 2
    assert result["recoveredWorkerCount"] == 2
    assert {item["swarmId"] for item in result["swarms"]} >= {first_id, second_id}
    assert STATE["swarms"][first_id]["queuedTasks"][0]["prompt"] == "recover first worker"
    assert STATE["swarms"][second_id]["queuedTasks"][0]["prompt"] == "recover second worker"
    assert STATE["swarms"][first_id]["workers"]["worker-1"]["terminationReason"] == "requeued_after_restart"
    assert STATE["swarms"][second_id]["workers"]["worker-1"]["terminationReason"] == "requeued_after_restart"


def test_swarm_worker_phase_barrier_releases_when_all_workers_arrive() -> None:
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"phase-barrier-team-{uuid4().hex[:8]}",
            "maxWorkers": 2,
            "sessionId": f"swarm-phase-barrier-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    STATE["swarms"][swarm_id]["workers"] = {
        "worker-1": {"workerId": "worker-1", "status": "WORKING", "currentTask": "research A"},
        "worker-2": {"workerId": "worker-2", "status": "WORKING", "currentTask": "research B"},
    }

    first = client.post(f"/api/swarm/{swarm_id}/worker/worker-1/phase", json={"phase": "Research"}).json()
    waiting = client.post(
        f"/api/swarm/{swarm_id}/phase-barrier",
        json={"phase": "Research", "workerIds": ["worker-1", "worker-2"]},
    ).json()
    second = client.post(f"/api/swarm/{swarm_id}/worker/worker-2/phase", json={"phase": "Research"}).json()
    released = client.post(
        f"/api/swarm/{swarm_id}/phase-barrier",
        json={"phase": "Research", "workerIds": ["worker-1", "worker-2"]},
    ).json()

    assert first["worker"]["currentWorkflowPhase"] == "Research"
    assert waiting["released"] is False
    assert waiting["missingWorkerIds"] == ["worker-2"]
    assert second["worker"]["currentWorkflowPhase"] == "Research"
    assert released["released"] is True
    assert released["barrier"]["status"] == "released"
    assert set(released["barrier"]["reachedWorkerIds"]) == {"worker-1", "worker-2"}


def test_swarm_phase_barrier_release_advances_coordinator_workflow() -> None:
    session_id = f"swarm-workflow-advance-{uuid4().hex[:8]}"
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"workflow-advance-team-{uuid4().hex[:8]}",
            "maxWorkers": 2,
            "sessionId": session_id,
            "objective": "coordinate staged research and synthesis",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    STATE["swarms"][swarm_id]["workers"] = {
        "worker-1": {"workerId": "worker-1", "status": "WORKING", "currentTask": "research A"},
        "worker-2": {"workerId": "worker-2", "status": "WORKING", "currentTask": "research B"},
    }

    client.post(f"/api/swarm/{swarm_id}/worker/worker-1/phase", json={"phase": "Research"})
    waiting = client.post(
        f"/api/swarm/{swarm_id}/phase-barrier",
        json={"phase": "Research", "workerIds": ["worker-1", "worker-2"]},
    ).json()
    assert waiting["released"] is False
    assert client.get(f"/api/coordinator/workflows/{session_id}").json()["currentPhase"]["name"] == "Research"

    client.post(f"/api/swarm/{swarm_id}/worker/worker-2/phase", json={"phase": "Research"})
    released = client.post(
        f"/api/swarm/{swarm_id}/phase-barrier",
        json={"phase": "Research", "workerIds": ["worker-1", "worker-2"]},
    ).json()
    fetched = client.get(f"/api/swarm/{swarm_id}").json()
    workflow = client.get(f"/api/coordinator/workflows/{session_id}").json()
    coordinator_events = client.get(f"/api/swarm/{swarm_id}/coordinator-events").json()["events"]

    assert released["released"] is True
    assert released["barrier"]["workflowAdvanced"] is True
    assert fetched["workflow"]["currentPhase"]["name"] == "Synthesis"
    assert workflow["currentPhase"]["name"] == "Synthesis"
    assert workflow["history"][-1]["name"] == "Synthesis"
    assert any(event["eventType"] == "workflow_phase_advanced" and event["payload"]["nextPhase"] == "Synthesis" for event in coordinator_events)


def test_swarm_mailbox_supports_phase_filter_and_partial_drain() -> None:
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"phase-mailbox-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-phase-mailbox-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    STATE["swarms"][swarm_id]["workers"] = {"worker-1": {"workerId": "worker-1", "status": "WORKING", "currentTask": "work"}}

    client.post(
        f"/api/swarm/{swarm_id}/worker/worker-1/mail",
        json={"senderId": "leader", "content": "research note", "phase": "Research", "channel": "handoff", "taskId": "task-r"},
    )
    client.post(
        f"/api/swarm/{swarm_id}/worker/worker-1/mail",
        json={"senderId": "leader", "content": "implementation note", "phase": "Implementation", "channel": "handoff", "taskId": "task-i"},
    )

    peek_research = client.get(f"/api/swarm/{swarm_id}/mailbox/worker-1?drain=false&phase=Research").json()
    drained_research = client.get(f"/api/swarm/{swarm_id}/mailbox/worker-1?phase=Research").json()
    remaining = client.get(f"/api/swarm/{swarm_id}/mailbox/worker-1?drain=false").json()

    assert peek_research["count"] == 1
    assert peek_research["messages"][0]["content"] == "research note"
    assert peek_research["messages"][0]["phase"] == "Research"
    assert peek_research["messages"][0]["channel"] == "handoff"
    assert peek_research["messages"][0]["taskId"] == "task-r"
    assert drained_research["count"] == 1
    assert remaining["count"] == 1
    assert remaining["messages"][0]["content"] == "implementation note"


def test_swarm_mailbox_ack_replay_and_recover_unacked_messages() -> None:
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"mailbox-replay-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-mailbox-replay-{uuid4().hex[:8]}",
            "tasks": [],
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    STATE["swarms"][swarm_id]["workers"] = {"worker-1": {"workerId": "worker-1", "status": "WORKING", "currentTask": "work"}}

    first = client.post(f"/api/swarm/{swarm_id}/worker/worker-1/mail", json={"content": "acked", "phase": "Research"}).json()["message"]
    second = client.post(f"/api/swarm/{swarm_id}/worker/worker-1/mail", json={"content": "recover me", "phase": "Research"}).json()["message"]

    acked = client.post(f"/api/swarm/{swarm_id}/mailbox/worker-1/ack", json={"messageIds": [first["id"]]}).json()
    replay = client.get(f"/api/swarm/{swarm_id}/mailbox/worker-1/replay?includeAcked=false&phase=Research").json()

    app_module.SWARM_MAILBOXES["worker-1"] = []
    recovered = client.post(f"/api/swarm/{swarm_id}/mailbox/worker-1/recover", json={"phase": "Research"}).json()
    mailbox = client.get(f"/api/swarm/{swarm_id}/mailbox/worker-1?drain=false").json()

    assert acked["ackedCount"] == 1
    assert first["id"] not in [message["id"] for message in replay["messages"]]
    assert second["id"] in [message["id"] for message in replay["messages"]]
    assert recovered["recoveredCount"] == 1
    assert mailbox["count"] == 1
    assert mailbox["messages"][0]["id"] == second["id"]


def test_swarm_permission_wait_is_excluded_from_worker_timeout() -> None:
    rel = f"backend-python/.test-workspace/{uuid4().hex}/permission-timeout-write.txt"
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"permission-timeout-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-permission-timeout-{uuid4().hex[:8]}",
            "tasks": ["write after wait"],
            "workerToolCalls": {
                "worker-1": [{"id": "write-timeout-1", "name": "write_file", "arguments": {"path": rel, "content": "wait excluded"}}]
            },
            "permissionTimeoutMs": 5000,
            "workerTimeoutMs": 25,
            "stepDelayMs": 1,
            "workerUseLlm": False,
        },
    ).json()
    swarm_id = created["swarmId"]
    pending = wait_until(lambda: client.get(f"/api/swarm/{swarm_id}").json().get("pendingPermissions"), timeout=3)
    assert pending
    time.sleep(0.08)
    request_id = pending[0]["requestId"]
    assert client.post(f"/api/swarm/permission/{request_id}", json={"approved": True}).json()["decision"] == "allow"
    completed = wait_until(
        lambda: client.get(f"/api/swarm/{swarm_id}").json()
        if client.get(f"/api/swarm/{swarm_id}").json()["workers"]["worker-1"]["status"] == "IDLE"
        else None,
        timeout=5,
    )
    assert completed
    worker = completed["workers"]["worker-1"]
    assert worker["permissionWaitMs"] >= 25
    assert completed["results"]["worker-1"]["status"] == "completed"


def test_swarm_worker_tool_allow_deny_lists_are_enforced() -> None:
    denied = client.post(
        "/api/swarm",
        json={
            "teamName": f"deny-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-deny-{uuid4().hex[:8]}",
            "tasks": ["attempt search"],
            "workerToolAllowList": ["list_files"],
            "workerToolCalls": {
                "worker-1": [{"id": "deny-1", "name": "search_files", "arguments": {"query": "FastAPI"}}]
            },
            "awaitCompletion": True,
            "workerUseLlm": False,
        },
    ).json()
    result = denied["results"]["worker-1"]["toolResults"][0]
    assert result["isError"] is True
    assert "allow list" in result["content"]


def test_swarm_worker_repetition_detection_and_timeout() -> None:
    repeated = client.post(
        "/api/swarm",
        json={
            "teamName": f"repeat-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-repeat-{uuid4().hex[:8]}",
            "tasks": ["repeat same tool"],
            "workerTurns": {
                "worker-1": [
                    {"prompt": "a", "toolCalls": [{"id": "r1", "name": "list_files", "arguments": {"pattern": "backend-python/*.py", "limit": 1}}]},
                    {"prompt": "b", "toolCalls": [{"id": "r2", "name": "list_files", "arguments": {"pattern": "backend-python/*.py", "limit": 1}}]},
                    {"prompt": "c", "toolCalls": [{"id": "r3", "name": "list_files", "arguments": {"pattern": "backend-python/*.py", "limit": 1}}]},
                ]
            },
            "awaitCompletion": True,
            "workerUseLlm": False,
        },
    ).json()
    worker = repeated["workers"]["worker-1"]
    assert worker["repetitionDetected"] is True
    assert any(event["type"] == "worker_repetition" for event in repeated["events"])

    timed_out = client.post(
        "/api/swarm",
        json={
            "teamName": f"timeout-team-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-timeout-{uuid4().hex[:8]}",
            "tasks": ["too slow"],
            "workerTimeoutMs": 1,
            "stepDelayMs": 20,
            "awaitCompletion": True,
            "workerUseLlm": False,
        },
    ).json()
    timeout_worker = timed_out["workers"]["worker-1"]
    assert timeout_worker["terminationReason"] == "timeout"
    assert timed_out["results"]["worker-1"]["status"] == "failed"
    assert "timed out" in timed_out["results"]["worker-1"]["error"]


def test_swarm_worker_tool_count_tracks_total_while_recent_records_are_capped() -> None:
    turns = [
        {"prompt": f"turn {index}", "toolCalls": [{"id": f"many-{index}", "name": "list_files", "arguments": {"pattern": "backend-python/*.py", "limit": 1 + (index % 2)}}]}
        for index in range(12)
    ]
    created = client.post(
        "/api/swarm",
        json={
            "teamName": f"many-tools-{uuid4().hex[:8]}",
            "maxWorkers": 1,
            "sessionId": f"swarm-many-tools-{uuid4().hex[:8]}",
            "tasks": ["many tool calls"],
            "workerTurns": {"worker-1": turns},
            "awaitCompletion": True,
            "workerUseLlm": False,
        },
    ).json()
    worker = created["workers"]["worker-1"]
    assert worker["toolCallCount"] == 13
    assert len(worker["recentToolCallRecords"]) == 10
    assert len(worker["recentToolCalls"]) == 5
    assert worker["recentToolCallRecords"][0]["toolName"] == "list_files"
    assert worker["recentToolCallRecords"][-1]["toolName"] == "list_files"
    assert worker["recentToolCalls"] == ["list_files"] * 5


def test_agent_tool_concurrency_context_background_and_team_route() -> None:
    agent_tool = TOOL_REGISTRY.get("Agent")
    assert agent_tool is not None
    agent_api = agent_tool.api_dict()
    assert agent_api["name"] == "Agent"
    assert agent_api["group"] == "agent"
    assert agent_api["readOnly"] is False
    assert agent_api["concurrencySafe"] is True
    schema = agent_tool.input_schema
    assert "description" in schema["properties"]
    assert schema["properties"]["subagent_type"]["enum"] == ["explore", "verification", "plan", "general-purpose", "guide"]
    assert schema["properties"]["isolation"]["enum"] == ["none", "worktree"]
    assert "light" in schema["properties"]["model"]["enum"]
    nested = agent_tool.handler({"prompt": "too deep", "sessionId": "agent-session", "nestingDepth": 3}).to_dict()
    assert nested["isError"] is True
    assert nested["metadata"]["status"] == "limit_exceeded"
    assert "nesting depth" in nested["content"]

    team_name = f"agent-tool-team-{uuid4().hex[:8]}"
    swarm = client.post(
        "/api/swarm",
        json={"teamName": team_name, "maxWorkers": 2, "sessionId": f"agent-tool-swarm-{uuid4().hex[:8]}", "tasks": [], "workerUseLlm": False},
    ).json()
    routed = agent_tool.handler(
        {
            "prompt": "delegate to team",
            "teamName": team_name,
            "subagent_type": "explore",
            "model": "light",
            "sessionId": "agent-session",
            "agentHierarchy": "main",
            "fork": True,
        },
    ).to_dict()
    assert routed["isError"] is False
    assert routed["metadata"]["teamName"] == team_name
    assert routed["metadata"]["swarmId"] == swarm["swarmId"]
    assert routed["metadata"]["workerId"].startswith("team-agent-")
    assert "delegate to team" in routed["content"]
    fetched_swarm = client.get(f"/api/swarm/{swarm['swarmId']}").json()
    assert routed["metadata"]["workerId"] in fetched_swarm["workers"]
    assert fetched_swarm["results"][routed["metadata"]["workerId"]]["status"] == "completed"

    old_rules = list(TOOL_REGISTRY.policy.rules)
    try:
        TOOL_REGISTRY.policy.rules = []
        direct_call = TOOL_REGISTRY.call("Agent", {"prompt": "direct permissionless call", "sessionId": "agent-session"}).to_dict()
        assert direct_call["isError"] is False
        assert direct_call["metadata"]["status"] == "completed"
        assert direct_call["metadata"]["model"] == "qwen3.7-plus"
        assert direct_call["metadata"]["childSessionId"].startswith("subagent-")
        assert direct_call["metadata"]["queryLoopId"]
        assert "direct permissionless call" in direct_call["content"]
    finally:
        TOOL_REGISTRY.policy.rules = old_rules

    background = agent_tool.handler(
        {
            "prompt": "background work",
            "description": "background direct work",
            "run_in_background": True,
            "sessionId": "agent-session",
            "agentId": f"agent-bg-{uuid4().hex[:8]}",
        },
    ).to_dict()
    assert background["metadata"]["status"] == "async_launched"
    assert "Description: background direct work" in background["content"]
    assert "Prompt: background work" in background["content"]
    task_id = background["metadata"]["taskId"]
    completed = wait_until(lambda: TOOL_REGISTRY._tasks.get(task_id) if TOOL_REGISTRY._tasks.get(task_id, {}).get("status") == "COMPLETED" else None, timeout=10)
    assert completed
    assert completed["metadata"]["queryLoopId"]
    assert "background work" in Path(background["metadata"]["outputFile"]).read_text(encoding="utf-8")

    bg_team_name = f"agent-tool-bg-team-{uuid4().hex[:8]}"
    bg_swarm = client.post(
        "/api/swarm",
        json={"teamName": bg_team_name, "maxWorkers": 1, "sessionId": f"agent-tool-bg-swarm-{uuid4().hex[:8]}", "tasks": [], "workerUseLlm": False},
    ).json()
    background_team = agent_tool.handler(
        {
            "prompt": "background delegate to team",
            "description": "background team direct",
            "teamName": bg_team_name,
            "run_in_background": True,
            "sessionId": "agent-session",
            "agentId": f"agent-bg-team-direct-{uuid4().hex[:8]}",
        },
    ).to_dict()
    assert background_team["metadata"]["status"] == "async_launched"
    bg_team_task_id = background_team["metadata"]["taskId"]
    bg_team_completed = wait_until(lambda: TOOL_REGISTRY._tasks.get(bg_team_task_id) if TOOL_REGISTRY._tasks.get(bg_team_task_id, {}).get("status") == "COMPLETED" else None, timeout=10)
    assert bg_team_completed
    assert bg_team_completed["swarmId"] == bg_swarm["swarmId"]
    assert bg_team_completed["workerId"].startswith("team-agent-")
    assert "background delegate to team" in Path(background_team["metadata"]["outputFile"]).read_text(encoding="utf-8")
    fetched_bg_swarm = client.get(f"/api/swarm/{bg_swarm['swarmId']}").json()
    assert fetched_bg_swarm["results"][bg_team_completed["workerId"]]["status"] == "completed"

    worktree_agent_id = f"agent-direct-worktree-{uuid4().hex[:8]}"
    worktree_rel = f"backend-python/.test-workspace/{uuid4().hex}/direct-worktree.txt"
    old_rules = list(TOOL_REGISTRY.policy.rules)
    try:
        TOOL_REGISTRY.policy.rules.insert(0, PermissionRule("write_file", PermissionDecision.ALLOW))
        direct_worktree = agent_tool.handler(
            {
                "prompt": "direct worktree write",
                "agentId": worktree_agent_id,
                "sessionId": "agent-session",
                "isolation": "WORKTREE",
                "toolCalls": [
                    {
                        "id": "direct-worktree-write",
                        "name": "write_file",
                        "arguments": {"path": worktree_rel, "content": "direct merged from worktree"},
                    }
                ],
            },
        ).to_dict()
    finally:
        TOOL_REGISTRY.policy.rules = old_rules
    assert direct_worktree["isError"] is False
    assert direct_worktree["metadata"]["status"] == "completed"
    assert direct_worktree["metadata"]["worktreeMode"] in {"git", "lightweight"}
    assert direct_worktree["metadata"]["mergedFiles"] == [worktree_rel]
    assert (ROOT / worktree_rel).read_text(encoding="utf-8") == "direct merged from worktree"
    assert not Path(direct_worktree["metadata"]["worktreePath"]).exists()
    assert STATE["agentWorktrees"][worktree_agent_id]["status"] == "cleaned"

    bg_worktree_agent_id = f"agent-bg-worktree-{uuid4().hex[:8]}"
    bg_worktree_rel = f"backend-python/.test-workspace/{uuid4().hex}/background-direct-worktree.txt"
    old_rules = list(TOOL_REGISTRY.policy.rules)
    try:
        TOOL_REGISTRY.policy.rules.insert(0, PermissionRule("write_file", PermissionDecision.ALLOW))
        background_worktree = agent_tool.handler(
            {
                "prompt": "background direct worktree write",
                "description": "bg worktree direct",
                "agentId": bg_worktree_agent_id,
                "sessionId": "agent-session",
                "run_in_background": True,
                "isolation": "WORKTREE",
                "toolCalls": [
                    {
                        "id": "background-direct-worktree-write",
                        "name": "write_file",
                        "arguments": {"path": bg_worktree_rel, "content": "background direct merged from worktree"},
                    }
                ],
            },
        ).to_dict()
        assert background_worktree["metadata"]["status"] == "async_launched"
        bg_worktree_task_id = background_worktree["metadata"]["taskId"]
        bg_worktree_completed = wait_until(lambda: TOOL_REGISTRY._tasks.get(bg_worktree_task_id) if TOOL_REGISTRY._tasks.get(bg_worktree_task_id, {}).get("status") == "COMPLETED" else None, timeout=10)
    finally:
        TOOL_REGISTRY.policy.rules = old_rules
    assert bg_worktree_completed
    assert bg_worktree_completed["metadata"]["worktreeMode"] in {"git", "lightweight"}
    assert bg_worktree_completed["metadata"]["mergedFiles"] == [bg_worktree_rel]
    assert (ROOT / bg_worktree_rel).read_text(encoding="utf-8") == "background direct merged from worktree"
    assert not Path(bg_worktree_completed["metadata"]["worktreePath"]).exists()
    assert STATE["agentWorktrees"][bg_worktree_agent_id]["status"] == "cleaned"

    agent_id = f"agent-held-{uuid4().hex[:8]}"
    assert TOOL_REGISTRY._acquire_agent_slot(agent_id, "agent-session", 1) is None
    assert agent_id in TOOL_REGISTRY._active_agents
    TOOL_REGISTRY._release_agent_slot(agent_id, "agent-session")
    assert agent_id not in TOOL_REGISTRY._active_agents


def test_agent_concurrency_limits_match_original_contract() -> None:
    assert MAX_CONCURRENT_AGENTS == 30
    assert MAX_CONCURRENT_AGENTS_PER_SESSION == 10
    assert MAX_AGENT_NESTING_DEPTH == 3

    global_slots: list[tuple[str, str]] = []
    try:
        for index in range(MAX_CONCURRENT_AGENTS):
            agent_id = f"global-slot-{uuid4().hex[:8]}-{index}"
            session_id = f"global-session-{uuid4().hex[:8]}-{index}"
            assert TOOL_REGISTRY._acquire_agent_slot(agent_id, session_id, 1) is None
            global_slots.append((agent_id, session_id))
        rejected = TOOL_REGISTRY._acquire_agent_slot(f"global-reject-{uuid4().hex[:8]}", "global-reject-session", 1)
        assert rejected is not None
        assert "Concurrent agent limit reached" in rejected
    finally:
        for agent_id, session_id in global_slots:
            TOOL_REGISTRY._release_agent_slot(agent_id, session_id)

    session_id = f"session-limit-{uuid4().hex[:8]}"
    session_slots: list[str] = []
    try:
        for index in range(MAX_CONCURRENT_AGENTS_PER_SESSION):
            agent_id = f"session-slot-{uuid4().hex[:8]}-{index}"
            assert TOOL_REGISTRY._acquire_agent_slot(agent_id, session_id, 1) is None
            session_slots.append(agent_id)
        rejected = TOOL_REGISTRY._acquire_agent_slot(f"session-reject-{uuid4().hex[:8]}", session_id, 1)
        assert rejected is not None
        assert "concurrent agent limit reached" in rejected
    finally:
        for agent_id in session_slots:
            TOOL_REGISTRY._release_agent_slot(agent_id, session_id)

    duplicate_session = f"duplicate-session-{uuid4().hex[:8]}"
    duplicate_agent = f"duplicate-agent-{uuid4().hex[:8]}"
    try:
        assert TOOL_REGISTRY._acquire_agent_slot(duplicate_agent, duplicate_session, 1) is None
        assert TOOL_REGISTRY._acquire_agent_slot(duplicate_agent, duplicate_session, 1) is None
        assert TOOL_REGISTRY._active_agent_counts[duplicate_agent] == 2
        assert TOOL_REGISTRY._session_agent_counts[duplicate_session][duplicate_agent] == 2
        TOOL_REGISTRY._release_agent_slot(duplicate_agent, duplicate_session)
        assert duplicate_agent in TOOL_REGISTRY._active_agents
        assert TOOL_REGISTRY._active_agent_counts[duplicate_agent] == 1
    finally:
        TOOL_REGISTRY._release_agent_slot(duplicate_agent, duplicate_session)
    assert duplicate_agent not in TOOL_REGISTRY._active_agents
    assert duplicate_session not in TOOL_REGISTRY._session_agents

    depth_agent_id = f"depth-ok-{uuid4().hex[:8]}"
    assert TOOL_REGISTRY._acquire_agent_slot(depth_agent_id, "depth-session", MAX_AGENT_NESTING_DEPTH) is None
    TOOL_REGISTRY._release_agent_slot(depth_agent_id, "depth-session")
    too_deep = TOOL_REGISTRY._acquire_agent_slot(f"depth-reject-{uuid4().hex[:8]}", "depth-session", MAX_AGENT_NESTING_DEPTH + 1)
    assert too_deep is not None
    assert "nesting depth" in too_deep


def test_agent_handler_background_concurrency_slots_release_after_parallel_run() -> None:
    agent_tool = TOOL_REGISTRY.get("Agent")
    session_id = f"parallel-session-{uuid4().hex[:8]}"
    entered: list[str] = []
    entered_lock = threading.Lock()
    release = threading.Event()
    old_dispatcher = TOOL_REGISTRY.agent_dispatcher

    def fake_dispatcher(payload: dict) -> ToolResult:
        agent_id = str(payload["agentId"])
        with entered_lock:
            entered.append(agent_id)
        release.wait(timeout=5)
        return ToolResult(
            f"parallel done: {payload.get('prompt')}",
            metadata={"status": "completed", "agentId": agent_id, "sessionId": payload.get("sessionId"), "childSessionId": f"subagent-{agent_id}"},
        )

    task_ids: list[str] = []
    agent_ids: list[str] = []
    completed = False
    try:
        TOOL_REGISTRY.set_agent_dispatcher(fake_dispatcher)
        for index in range(MAX_CONCURRENT_AGENTS_PER_SESSION):
            agent_id = f"parallel-agent-{uuid4().hex[:8]}-{index}"
            launched = agent_tool.handler(
                {
                    "prompt": f"parallel background {index}",
                    "agentId": agent_id,
                    "sessionId": session_id,
                    "run_in_background": True,
                },
            ).to_dict()
            assert launched["metadata"]["status"] == "async_launched"
            agent_ids.append(agent_id)
            task_ids.append(launched["metadata"]["taskId"])

        active = wait_until(lambda: len(entered) if len(entered) == MAX_CONCURRENT_AGENTS_PER_SESSION else None, timeout=2)
        assert active == MAX_CONCURRENT_AGENTS_PER_SESSION
        assert sum(TOOL_REGISTRY._session_agent_counts[session_id].values()) == MAX_CONCURRENT_AGENTS_PER_SESSION

        rejected = agent_tool.handler(
            {
                "prompt": "parallel overflow",
                "agentId": f"parallel-agent-overflow-{uuid4().hex[:8]}",
                "sessionId": session_id,
                "run_in_background": True,
            },
        ).to_dict()
        assert rejected["isError"] is True
        assert rejected["metadata"]["status"] == "limit_exceeded"
        assert "Session" in rejected["content"]
    finally:
        release.set()
        completed = bool(
            wait_until(
                lambda: all((TOOL_REGISTRY._tasks.get(task_id) or {}).get("status") == "COMPLETED" for task_id in task_ids),
                timeout=5,
            )
        )
        TOOL_REGISTRY.set_agent_dispatcher(old_dispatcher)

    assert completed
    assert session_id not in TOOL_REGISTRY._session_agent_counts
    assert session_id not in TOOL_REGISTRY._session_agents
    assert not any(agent_id in TOOL_REGISTRY._active_agents for agent_id in agent_ids)


def test_agent_handler_direct_dispatch_times_out_and_releases_slot(monkeypatch) -> None:
    async def slow_query(payload: dict, require_existing_session: bool = False, live_send=None) -> dict:
        await asyncio.sleep(0.2)
        return {"answer": "too late", "queryLoop": {"id": "slow-loop"}}

    monkeypatch.setattr("app.run_query_payload", slow_query)
    agent_tool = TOOL_REGISTRY.get("Agent")
    agent_id = f"timeout-agent-{uuid4().hex[:8]}"

    result = agent_tool.handler(
        {
            "prompt": "slow direct dispatch",
            "agentId": agent_id,
            "sessionId": "timeout-session",
            "timeoutMs": 20,
        },
    ).to_dict()

    assert result["isError"] is True
    assert result["metadata"]["status"] == "failed"
    assert "timed out" in result["content"]
    assert "too late" not in result["content"]
    assert QUERY_ABORTS.aborted[f"subagent-{agent_id}"]["reason"] == "TIMEOUT"
    assert agent_id not in TOOL_REGISTRY._active_agents


def test_agent_handler_background_direct_dispatch_timeout_marks_task_failed(monkeypatch) -> None:
    async def slow_query(payload: dict, require_existing_session: bool = False, live_send=None) -> dict:
        await asyncio.sleep(0.2)
        return {"answer": "too late", "queryLoop": {"id": "slow-loop"}}

    monkeypatch.setattr("app.run_query_payload", slow_query)
    agent_tool = TOOL_REGISTRY.get("Agent")
    agent_id = f"timeout-bg-agent-{uuid4().hex[:8]}"
    launched = agent_tool.handler(
        {
            "prompt": "slow background dispatch",
            "description": "timeout background",
            "agentId": agent_id,
            "sessionId": "timeout-bg-session",
            "run_in_background": True,
            "timeoutMs": 20,
        },
    ).to_dict()

    assert launched["metadata"]["status"] == "async_launched"
    task_id = launched["metadata"]["taskId"]
    failed = wait_until(lambda: TOOL_REGISTRY._tasks.get(task_id) if TOOL_REGISTRY._tasks.get(task_id, {}).get("status") == "FAILED" else None, timeout=5)
    assert failed
    assert "timed out" in failed["error"]
    assert Path(launched["metadata"]["outputFile"]).read_text(encoding="utf-8") == failed["error"]
    assert QUERY_ABORTS.aborted[f"subagent-{agent_id}"]["reason"] == "TIMEOUT"
    assert agent_id not in TOOL_REGISTRY._active_agents


def test_query_agent_permission_wait_is_excluded_from_agent_timeout(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    old_rules = list(TOOL_REGISTRY.policy.rules)
    TOOL_REGISTRY.policy.rules = []
    parent_session = f"agent-timeout-permission-parent-{uuid4().hex[:8]}"
    loop = QueryLoopState.start(
        session_id=parent_session,
        user_input="agent waits on bubbled permission",
        model="test-model",
        context_window=128_000,
        threshold=0.85,
        ratio=2.5,
    )
    agent_id = f"agent-permission-timeout-{uuid4().hex[:8]}"
    rel = f"backend-python/.test-workspace/{uuid4().hex}/agent-permission-timeout.txt"

    async def run_case() -> dict:
        task = asyncio.create_task(
            execute_agent_tool(
                loop,
                {
                    "prompt": "write after parent approval",
                    "agentId": agent_id,
                    "sessionId": parent_session,
                    "model": "test-model",
                    "timeoutMs": 50,
                    "toolCalls": [
                        {
                            "id": "child-permission-timeout",
                            "name": "write_file",
                            "arguments": {"path": rel, "content": "approved despite wait", "permissionTimeoutMs": 1000},
                        }
                    ],
                },
            )
        )
        await asyncio.sleep(0.12)
        STATE.setdefault("permissionResponses", {})["child-permission-timeout"] = {
            "toolUseId": "child-permission-timeout",
            "decision": "allow",
            "allowed": True,
            "updatedAt": time.time(),
        }
        return await asyncio.wait_for(task, timeout=30)

    try:
        response = asyncio.run(run_case())
    finally:
        TOOL_REGISTRY.policy.rules = old_rules

    assert response["isError"] is False
    assert response["metadata"]["status"] == "completed"
    assert (ROOT / rel).read_text(encoding="utf-8") == "approved despite wait"
    assert f"subagent-{agent_id}" not in QUERY_ABORTS.aborted
    assert agent_id not in TOOL_REGISTRY._active_agents


def test_query_agent_timeout_aborts_child_session_and_releases_slot(monkeypatch) -> None:
    async def slow_query(payload: dict, require_existing_session: bool = False, live_send=None) -> dict:
        await asyncio.sleep(0.2)
        return {"answer": "too late", "queryLoop": {"id": "slow-loop"}}

    monkeypatch.setattr("app.run_query_payload", slow_query)
    parent_session = f"query-agent-timeout-parent-{uuid4().hex[:8]}"
    loop = QueryLoopState.start(
        session_id=parent_session,
        user_input="timeout child agent",
        model="test-model",
        context_window=128_000,
        threshold=0.85,
        ratio=2.5,
    )
    agent_id = f"query-timeout-agent-{uuid4().hex[:8]}"

    response = asyncio.run(
        execute_agent_tool(
            loop,
            {
                "prompt": "slow query agent",
                "agentId": agent_id,
                "sessionId": parent_session,
                "model": "test-model",
                "timeoutMs": 20,
            },
        )
    )

    child_session_id = f"subagent-{agent_id}"
    assert response["isError"] is True
    assert response["metadata"]["status"] == "failed"
    assert "timed out" in response["content"]
    assert QUERY_ABORTS.aborted[child_session_id]["reason"] == "TIMEOUT"
    assert agent_id not in TOOL_REGISTRY._active_agents


def test_agent_handler_background_direct_dispatch_publishes_tracker_events() -> None:
    session_id = f"direct-bg-events-{uuid4().hex[:8]}"
    WS_SESSION_MANAGER.drain_messages(session_id)
    agent_tool = TOOL_REGISTRY.get("Agent")
    agent_id = f"direct-bg-event-agent-{uuid4().hex[:8]}"
    old_dispatcher = TOOL_REGISTRY.agent_dispatcher

    def fake_dispatcher(payload: dict) -> ToolResult:
        return ToolResult(
            f"event result: {payload.get('prompt')}",
            metadata={"status": "completed", "agentId": payload.get("agentId"), "sessionId": payload.get("sessionId"), "childSessionId": f"subagent-{payload.get('agentId')}"},
        )

    try:
        TOOL_REGISTRY.set_agent_dispatcher(fake_dispatcher)
        launched = agent_tool.handler(
            {
                "prompt": "background event dispatch",
                "description": "event dispatch",
                "agentId": agent_id,
                "sessionId": session_id,
                "run_in_background": True,
            },
        ).to_dict()
        task_id = launched["metadata"]["taskId"]
        completed = wait_until(lambda: TOOL_REGISTRY._tasks.get(task_id) if TOOL_REGISTRY._tasks.get(task_id, {}).get("status") == "COMPLETED" else None, timeout=5)
    finally:
        TOOL_REGISTRY.set_agent_dispatcher(old_dispatcher)

    assert completed
    messages = WS_SESSION_MANAGER.peek_messages(session_id)
    agent_events = [message["payload"] for message in messages if message["type"] == "task_update" and message["payload"].get("agentId") == agent_id]
    assert [event["eventType"] for event in agent_events] == ["agent_started", "agent_completed"]
    assert agent_events[0]["data"]["prompt"] == "background event dispatch"
    assert "event result" in agent_events[1]["data"]["resultPreview"]


def test_query_multiple_agent_tool_calls_run_concurrently(monkeypatch) -> None:
    starts: list[tuple[str, float]] = []

    async def fake_execute_agent_tool(loop: QueryLoopState, payload: dict, live_send=None) -> dict:
        agent_id = str(payload["agentId"])
        starts.append((agent_id, time.time()))
        await asyncio.sleep(0.12)
        return {
            "content": f"done {agent_id}",
            "isError": False,
            "metadata": {"status": "completed", "agentId": agent_id, "sessionId": payload.get("sessionId")},
        }

    monkeypatch.setattr("app.execute_agent_tool", fake_execute_agent_tool)
    loop = QueryLoopState.start(
        session_id=f"parallel-query-{uuid4().hex[:8]}",
        user_input="run agents in parallel",
        model="test-model",
        context_window=128_000,
        threshold=0.85,
        ratio=2.5,
    )

    started = time.time()
    results = asyncio.run(
        execute_query_tools(
            loop,
            [
                {"id": "parallel-agent-1", "name": "Agent", "arguments": {"prompt": "first", "agentId": "parallel-a"}},
                {"id": "parallel-agent-2", "name": "Agent", "arguments": {"prompt": "second", "agentId": "parallel-b"}},
            ],
        )
    )
    elapsed = time.time() - started

    assert [result["toolUseId"] for result in results] == ["parallel-agent-1", "parallel-agent-2"]
    assert [agent_id for agent_id, _ in starts] == ["parallel-a", "parallel-b"]
    assert abs(starts[1][1] - starts[0][1]) < 0.06
    assert elapsed < 0.2


def test_query_concurrency_safe_tools_do_not_block_agent_parallelism(monkeypatch) -> None:
    starts: list[tuple[str, float]] = []
    rel = f"backend-python/.test-workspace/{uuid4().hex}/safe-read.txt"
    target = ROOT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("safe read", encoding="utf-8")

    async def fake_execute_agent_tool(loop: QueryLoopState, payload: dict, live_send=None) -> dict:
        agent_id = str(payload["agentId"])
        starts.append((agent_id, time.time()))
        await asyncio.sleep(0.12)
        return {
            "content": f"done {agent_id}",
            "isError": False,
            "metadata": {"status": "completed", "agentId": agent_id, "sessionId": payload.get("sessionId")},
        }

    monkeypatch.setattr("app.execute_agent_tool", fake_execute_agent_tool)
    loop = QueryLoopState.start(
        session_id=f"parallel-safe-query-{uuid4().hex[:8]}",
        user_input="run agents around safe read",
        model="test-model",
        context_window=128_000,
        threshold=0.85,
        ratio=2.5,
    )

    started = time.time()
    results = asyncio.run(
        execute_query_tools(
            loop,
            [
                {"id": "parallel-safe-agent-1", "name": "Agent", "arguments": {"prompt": "first", "agentId": "parallel-safe-a"}},
                {"id": "parallel-safe-read", "name": "read_file", "arguments": {"path": rel}},
                {"id": "parallel-safe-agent-2", "name": "Agent", "arguments": {"prompt": "second", "agentId": "parallel-safe-b"}},
            ],
        )
    )
    elapsed = time.time() - started

    assert [result["toolUseId"] for result in results] == ["parallel-safe-agent-1", "parallel-safe-read", "parallel-safe-agent-2"]
    assert [agent_id for agent_id, _ in starts] == ["parallel-safe-a", "parallel-safe-b"]
    assert abs(starts[1][1] - starts[0][1]) < 0.06
    assert elapsed < 0.2


def test_query_blocking_safe_tool_does_not_delay_agent_start(monkeypatch) -> None:
    starts: dict[str, float] = {}

    def slow_read(_payload: dict) -> ToolResult:
        starts["slow_read"] = time.time()
        time.sleep(0.15)
        return ToolResult("slow read complete")

    async def fake_execute_agent_tool(loop: QueryLoopState, payload: dict, live_send=None) -> dict:
        starts["agent"] = time.time()
        await asyncio.sleep(0.01)
        return {
            "content": "agent complete",
            "isError": False,
            "metadata": {"status": "completed", "agentId": payload["agentId"], "sessionId": payload.get("sessionId")},
        }

    TOOL_REGISTRY.register(
        Tool(
            name="slow_read_safe",
            description="Slow concurrency-safe read test tool.",
            input_schema={"type": "object", "properties": {}},
            handler=slow_read,
            group="read",
            read_only=True,
        )
    )
    monkeypatch.setattr("app.execute_agent_tool", fake_execute_agent_tool)
    loop = QueryLoopState.start(
        session_id=f"parallel-blocking-safe-{uuid4().hex[:8]}",
        user_input="run safe blocking tool with agent",
        model="test-model",
        context_window=128_000,
        threshold=0.85,
        ratio=2.5,
    )

    started = time.time()
    results = asyncio.run(
        execute_query_tools(
            loop,
            [
                {"id": "slow-read-safe-1", "name": "slow_read_safe", "arguments": {}},
                {"id": "agent-after-slow-read", "name": "Agent", "arguments": {"prompt": "agent should start immediately", "agentId": "agent-after-safe-read"}},
            ],
        )
    )
    elapsed = time.time() - started

    assert [result["toolUseId"] for result in results] == ["slow-read-safe-1", "agent-after-slow-read"]
    assert starts["agent"] - starts["slow_read"] < 0.06
    assert elapsed < 0.22


def test_query_agent_tool_executes_isolated_subagent_session(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    parent = client.post("/api/sessions", json={"title": "agent parent", "model": "test-model"}).json()["sessionId"]
    agent_id = f"agent-sync-{uuid4().hex[:8]}"
    response = client.post(
        "/api/query",
        json={
            "sessionId": parent,
            "prompt": "parent delegates",
            "toolCalls": [
                {
                    "id": "agent-call-1",
                    "name": "Agent",
                    "arguments": {
                        "prompt": "child should inspect isolated execution",
                        "agentId": agent_id,
                        "subagent_type": "explore",
                        "sessionId": parent,
                        "fork": True,
                        "model": "test-model",
                    },
                }
            ],
        },
    ).json()
    agent_result = response["toolCalls"][0]
    child_session_id = f"fork-{agent_id}"
    assert agent_result["toolName"] == "Agent"
    assert agent_result["metadata"]["status"] == "completed"
    assert agent_result["metadata"]["childSessionId"] == child_session_id
    assert "child should inspect isolated execution" in agent_result["content"]
    child = client.get(f"/api/sessions/{child_session_id}").json()
    assert child["session"]["id"] == child_session_id
    assert STATE["sessions"][child_session_id]["parentSessionId"] == parent
    assert STATE["sessions"][child_session_id]["agentType"] == "explore"
    loops = client.get(f"/api/query/loops?sessionId={child_session_id}").json()["loops"]
    assert loops and loops[-1]["status"] == "completed"
    assert any(event["type"] == "agent_completed" for event in response["events"])


def test_query_agent_model_alias_is_resolved_for_child_session(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    parent = client.post("/api/sessions", json={"title": "agent model alias", "model": "test-model"}).json()["sessionId"]
    agent_id = f"agent-alias-{uuid4().hex[:8]}"
    response = client.post(
        "/api/query",
        json={
            "sessionId": parent,
            "prompt": "launch default explore child",
            "toolCalls": [
                {
                    "id": "agent-alias-1",
                    "name": "Agent",
                    "arguments": {
                        "prompt": "use explore default model",
                        "agentId": agent_id,
                        "subagent_type": "explore",
                        "sessionId": parent,
                    },
                }
            ],
        },
    ).json()
    metadata = response["toolCalls"][0]["metadata"]
    assert metadata["model"] == "qwen3.7-plus"
    child = STATE["sessions"][f"subagent-{agent_id}"]
    assert child["model"] == "qwen3.7-plus"
    assert "你是一个搜索和探索专家" in child["systemPrompt"]
    assert "ExitPlanMode" in child["agentDeniedTools"]

    general_agent_id = f"agent-standard-{uuid4().hex[:8]}"
    general = client.post(
        "/api/query",
        json={
            "sessionId": parent,
            "prompt": "launch standard default child",
            "toolCalls": [
                {
                    "id": "agent-standard-1",
                    "name": "Agent",
                    "arguments": {
                        "prompt": "use general default model",
                        "agentId": general_agent_id,
                        "sessionId": parent,
                    },
                }
            ],
        },
    ).json()
    general_meta = general["toolCalls"][0]["metadata"]
    assert general_meta["model"] == "qwen3.7-plus"
    general_child = STATE["sessions"][f"subagent-{general_agent_id}"]
    assert general_child["model"] == "qwen3.7-plus"
    assert "你是一个通用 worker 代理" in general_child["systemPrompt"]


def test_query_agent_tool_background_runs_query_and_writes_output(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    parent = client.post("/api/sessions", json={"title": "agent background", "model": "test-model"}).json()["sessionId"]
    agent_id = f"agent-bg-query-{uuid4().hex[:8]}"
    response = client.post(
        "/api/query",
        json={
            "sessionId": parent,
            "prompt": "launch background child",
            "toolCalls": [
                {
                    "id": "agent-bg-1",
                    "name": "Agent",
                    "arguments": {
                        "prompt": "background child query",
                        "description": "background child",
                        "agentId": agent_id,
                        "sessionId": parent,
                        "run_in_background": True,
                        "model": "test-model",
                    },
                }
            ],
        },
    ).json()
    metadata = response["toolCalls"][0]["metadata"]
    assert metadata["status"] == "async_launched"
    assert metadata["description"] == "background child"
    assert "Description: background child" in response["toolCalls"][0]["content"]
    task_id = metadata["taskId"]
    completed = wait_until(lambda: TOOL_REGISTRY._tasks.get(task_id) if TOOL_REGISTRY._tasks.get(task_id, {}).get("status") == "COMPLETED" else None)
    assert completed
    output = Path(metadata["outputFile"]).read_text(encoding="utf-8")
    assert "background child query" in output
    assert f"subagent-{agent_id}" in STATE["sessions"]
    assert agent_id not in TOOL_REGISTRY._active_agents
    status = client.get(f"/api/agents/background/{agent_id}").json()["agent"]
    assert status["agentId"] == agent_id
    assert status["status"] == "completed"
    assert status["childSessionId"] == f"subagent-{agent_id}"
    assert status["agentType"] == "general-purpose"
    assert status["model"] == "test-model"
    assert status["description"] == "background child"
    listed = client.get(f"/api/agents/background?sessionId={parent}&activeOnly=false").json()
    assert any(agent["agentId"] == agent_id for agent in listed["agents"])
    assert any(event["type"] == "waiting_for_background_agents" for event in response["events"])
    assert any(event["type"] == "background_agents_wait_complete" for event in response["events"])
    assert "Background agents completed" in response["answer"]


def test_query_agent_background_wait_only_injects_current_agent_results(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    parent = client.post("/api/sessions", json={"title": "agent precise wait", "model": "test-model"}).json()["sessionId"]
    old_agent_id = f"old-bg-{uuid4().hex[:8]}"
    old_output = workspace() / "old-bg-output.txt"
    old_output.write_text("old background result should not reappear", encoding="utf-8")
    old_task = TOOL_REGISTRY._new_task("old background", session_id=parent, task_type="agent:general-purpose")
    old_task["agentId"] = old_agent_id
    old_task["outputFile"] = str(old_output)
    old_task["status"] = "COMPLETED"
    old_task["output"] = "old background result should not reappear"
    old_task["updatedAt"] = time.time()

    new_agent_id = f"new-bg-{uuid4().hex[:8]}"
    response = client.post(
        "/api/query",
        json={
            "sessionId": parent,
            "prompt": "launch only current background child",
            "toolCalls": [
                {
                    "id": "agent-bg-current",
                    "name": "Agent",
                    "arguments": {
                        "prompt": "new background result should appear",
                        "agentId": new_agent_id,
                        "sessionId": parent,
                        "run_in_background": True,
                        "model": "test-model",
                    },
                }
            ],
        },
    ).json()

    assert "new background result should appear" in response["answer"]
    assert "old background result should not reappear" not in response["answer"]
    wait_event = next(event for event in response["events"] if event["type"] == "background_agents_wait_complete")
    assert wait_event["agentIds"] == [new_agent_id]


def test_background_agent_tracker_lifecycle_endpoints() -> None:
    parent = client.post("/api/sessions", json={"title": "background tracker lifecycle"}).json()["sessionId"]
    agent_id = f"manual-bg-{uuid4().hex[:8]}"
    task = TOOL_REGISTRY._new_task("manual background", session_id=parent, task_type="agent:manual")
    task["agentId"] = agent_id
    task["outputFile"] = str(workspace() / "manual-bg-output.txt")

    active = client.get(f"/api/agents/background/active-ids?sessionId={parent}").json()
    assert active["activeAgentIds"] == [agent_id]

    timed_out = client.post("/api/agents/background/await", json={"sessionId": parent, "timeoutMs": 20}).json()
    assert timed_out["completed"] is False
    assert agent_id in timed_out["activeAgentIds"]

    output_path = Path(task["outputFile"])
    output_path.write_text("manual background result", encoding="utf-8")
    task["status"] = "COMPLETED"
    task["output"] = "manual background result"
    task["updatedAt"] = time.time()
    completed = client.post("/api/agents/background/await", json={"sessionId": parent, "timeoutMs": 500, "agentIds": [agent_id]}).json()
    assert completed["completed"] is True
    assert completed["agentIds"] == [agent_id]
    assert any(agent["agentId"] == agent_id and agent["status"] == "completed" for agent in completed["agents"])

    removed = client.delete(f"/api/agents/background/session/{parent}?deleteOutputFiles=true").json()
    assert removed["removed"] == 1
    assert not output_path.exists()
    assert client.get(f"/api/agents/background/{agent_id}").status_code == 404


def test_background_agent_cleanup_removes_stale_completed_outputs() -> None:
    parent = client.post("/api/sessions", json={"title": "background cleanup"}).json()["sessionId"]
    agent_id = f"stale-bg-{uuid4().hex[:8]}"
    output_path = workspace() / "stale-bg-output.txt"
    output_path.write_text("old result", encoding="utf-8")
    task = TOOL_REGISTRY._new_task("stale background", session_id=parent, task_type="agent:cleanup")
    task["agentId"] = agent_id
    task["outputFile"] = str(output_path)
    task["status"] = "COMPLETED"
    task["updatedAt"] = time.time() - 3600

    cleanup = client.post("/api/agents/background/cleanup", json={"maxAgeMinutes": 30, "deleteOutputFiles": True}).json()
    assert cleanup["success"] is True
    assert agent_id in cleanup["removedAgentIds"]
    assert not output_path.exists()
    assert client.get(f"/api/agents/background/{agent_id}").status_code == 404


def test_agent_snapshot_resume_lifecycle(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    parent = client.post("/api/sessions", json={"title": "agent snapshot parent", "model": "test-model"}).json()["sessionId"]
    agent_id = f"resume-agent-{uuid4().hex[:8]}"
    snapshot = client.post(
        "/api/agents/snapshots",
        json={
            "agentId": agent_id,
            "taskDescription": "resume saved task",
            "parentSessionId": parent,
            "nestingDepth": 2,
            "workingDirectory": ".",
            "model": "test-model",
            "messages": [{"type": "user", "uuid": "m1", "timestamp": int(time.time() * 1000), "content": [{"type": "text", "text": "saved context"}]}],
        },
    ).json()
    assert snapshot["success"] is True
    assert snapshot["messageCount"] == 1
    listed = client.get("/api/agents/snapshots").json()
    assert any(item["agentId"] == agent_id for item in listed["snapshots"])
    loaded = client.get(f"/api/agents/snapshots/{agent_id}").json()["snapshot"]
    assert loaded["taskDescription"] == "resume saved task"

    resumed = client.post(f"/api/agents/{agent_id}/resume", json={"additionalContext": "resume now"}).json()
    assert resumed["success"] is True
    assert resumed["sessionId"] == f"resumed-agent-{agent_id}"
    assert "resume saved task" in resumed["result"]["answer"]
    assert "resume now" in resumed["result"]["answer"]
    assert client.get(f"/api/agents/snapshots/{agent_id}").status_code == 404
    resumed_session = client.get(f"/api/sessions/resumed-agent-{agent_id}").json()["session"]
    assert resumed_session["id"] == f"resumed-agent-{agent_id}"


def test_agent_snapshot_purge_removes_expired_records() -> None:
    agent_id = f"expired-agent-{uuid4().hex[:8]}"
    created = client.post("/api/agents/snapshots", json={"agentId": agent_id, "taskDescription": "old", "messages": []}).json()
    assert created["success"] is True
    snapshot_file = Path(created["snapshot"]["path"])
    old_time = time.time() - 48 * 3600
    os.utime(snapshot_file, (old_time, old_time))
    purged = client.post("/api/agents/snapshots/purge", json={"maxAgeHours": 24}).json()
    assert purged["success"] is True
    assert agent_id in purged["agentIds"]
    assert not snapshot_file.exists()


def test_query_agent_type_filters_denied_tools(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    parent = client.post("/api/sessions", json={"title": "agent tool filter", "model": "test-model"}).json()["sessionId"]
    agent_id = f"agent-filter-{uuid4().hex[:8]}"
    rel = f"backend-python/.test-workspace/{uuid4().hex}/denied-write.txt"
    response = client.post(
        "/api/query",
        json={
            "sessionId": parent,
            "prompt": "launch read-only child",
            "toolCalls": [
                {
                    "id": "agent-filter-1",
                    "name": "Agent",
                    "arguments": {
                        "prompt": "attempt write in explore agent",
                        "agentId": agent_id,
                        "subagent_type": "explore",
                        "sessionId": parent,
                        "model": "test-model",
                        "toolCalls": [
                            {
                                "id": "denied-write",
                                "name": "write_file",
                                "arguments": {"path": rel, "content": "should not write"},
                            }
                        ],
                    },
                }
            ],
        },
    ).json()
    child_session_id = f"subagent-{agent_id}"
    assert response["toolCalls"][0]["metadata"]["status"] == "completed"
    assert not (ROOT / rel).exists()
    child_session = STATE["sessions"][child_session_id]
    assert "write_file" in child_session["agentDeniedTools"]
    loops = client.get(f"/api/query/loops?sessionId={child_session_id}").json()["loops"]
    denied = loops[-1]["toolCalls"][0]
    assert denied["toolName"] == "write_file"
    assert denied["status"] == "error"
    assert "not available" in denied["result"]["content"]


def test_query_agent_permission_request_bubbles_to_parent_session(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    parent = client.post("/api/sessions", json={"title": "agent permission bubble", "model": "test-model"}).json()["sessionId"]
    WS_SESSION_MANAGER.drain_messages(parent)
    old_rules = list(TOOL_REGISTRY.policy.rules)
    TOOL_REGISTRY.policy.rules = []
    try:
        agent_id = f"agent-permission-{uuid4().hex[:8]}"
        rel = f"backend-python/.test-workspace/{uuid4().hex}/needs-permission.txt"
        response_holder: dict[str, dict] = {}

        def run_query() -> None:
            response_holder["response"] = client.post(
                "/api/query",
                json={
                    "sessionId": parent,
                    "prompt": "launch child needing permission",
                    "toolCalls": [
                        {
                            "id": "agent-permission-1",
                            "name": "Agent",
                            "arguments": {
                                "prompt": "write requiring parent approval",
                                "agentId": agent_id,
                                "sessionId": parent,
                                "model": "test-model",
                                "toolCalls": [
                                        {
                                            "id": "child-needs-permission",
                                            "name": "write_file",
                                            "arguments": {"path": rel, "content": "needs approval", "permissionTimeoutMs": 30_000},
                                        }
                                    ],
                                },
                            }
                    ],
                },
            ).json()

        query_thread = threading.Thread(target=run_query)
        query_thread.start()

        bubbled = wait_until(
            lambda: next(
                (
                    message["payload"]
                    for message in WS_SESSION_MANAGER.peek_messages(parent)
                    if message["type"] == "permission_request" and message["payload"].get("toolUseId") == "child-needs-permission"
                ),
                None,
            ),
            timeout=30,
        )
        if not bubbled:
            STATE.setdefault("permissionResponses", {})["child-needs-permission"] = {
                "toolUseId": "child-needs-permission",
                "decision": "deny",
                "allowed": False,
                "reason": "test cleanup",
                "updatedAt": time.time(),
            }
            query_thread.join(timeout=5)
        assert bubbled
        STATE.setdefault("permissionResponses", {})["child-needs-permission"] = {
            "toolUseId": "child-needs-permission",
            "decision": "allow",
            "allowed": True,
            "updatedAt": time.time(),
        }
        query_thread.join(timeout=30)
        assert not query_thread.is_alive()
        response = response_holder["response"]
    finally:
        TOOL_REGISTRY.policy.rules = old_rules

    child_session_id = f"subagent-{agent_id}"
    assert response["toolCalls"][0]["metadata"]["childSessionId"] == child_session_id
    parent_events = [
        message["payload"]
        for message in WS_SESSION_MANAGER.peek_messages(parent)
        if message["type"] == "permission_request" and message["payload"].get("toolUseId") == "child-needs-permission"
    ]
    assert parent_events
    assert parent_events[0]["sessionId"] == parent
    assert parent_events[0]["childSessionId"] == child_session_id
    assert parent_events[0]["bubbledFromSessionId"] == child_session_id
    assert response["toolCalls"][0]["metadata"]["status"] == "completed"
    assert (ROOT / rel).read_text(encoding="utf-8") == "needs approval"


def test_query_agent_tool_team_name_dispatches_to_swarm(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    team_name = f"dispatch-team-{uuid4().hex[:8]}"
    swarm = client.post(
        "/api/swarm",
        json={"teamName": team_name, "maxWorkers": 2, "sessionId": f"swarm-dispatch-{uuid4().hex}", "tasks": [], "workerUseLlm": False},
    ).json()
    parent = client.post("/api/sessions", json={"title": "team dispatch", "model": "test-model"}).json()["sessionId"]
    agent_id = f"agent-team-{uuid4().hex[:8]}"
    response = client.post(
        "/api/query",
        json={
            "sessionId": parent,
            "prompt": "dispatch to existing team",
            "toolCalls": [
                {
                    "id": "team-agent-1",
                    "name": "Agent",
                    "arguments": {
                        "prompt": "team worker task",
                        "agentId": agent_id,
                        "teamName": team_name,
                        "sessionId": parent,
                        "subagent_type": "reviewer",
                        "model": "test-model",
                    },
                }
            ],
        },
    ).json()
    result = response["toolCalls"][0]
    assert result["metadata"]["status"] == "completed"
    assert result["metadata"]["swarmId"] == swarm["swarmId"]
    assert result["metadata"]["workerId"].startswith("team-agent-")
    assert "team worker task" in result["content"]
    fetched = client.get(f"/api/swarm/{swarm['swarmId']}").json()
    assert result["metadata"]["workerId"] in fetched["workers"]
    assert fetched["results"][result["metadata"]["workerId"]]["status"] == "completed"
    events = client.get(f"/api/swarm/{swarm['swarmId']}/coordinator-events").json()["events"]
    assert any(event["eventType"] == "team_dispatch" for event in events)
    assert agent_id not in TOOL_REGISTRY._active_agents


def test_query_agent_team_permission_wait_is_excluded_from_agent_timeout(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    old_rules = list(TOOL_REGISTRY.policy.rules)
    TOOL_REGISTRY.policy.rules = []
    try:
        team_name = f"team-permission-timeout-{uuid4().hex[:8]}"
        swarm = client.post(
            "/api/swarm",
            json={
                "teamName": team_name,
                "maxWorkers": 1,
                "sessionId": f"swarm-team-permission-timeout-{uuid4().hex[:8]}",
                "tasks": [],
                "workerUseLlm": False,
                "permissionTimeoutMs": 1000,
            },
        ).json()
        parent = client.post("/api/sessions", json={"title": "team permission timeout", "model": "test-model"}).json()["sessionId"]
        agent_id = f"agent-team-permission-timeout-{uuid4().hex[:8]}"
        rel = f"backend-python/.test-workspace/{uuid4().hex}/team-permission-timeout.txt"
        response_holder: dict[str, dict] = {}

        def run_query() -> None:
            response_holder["response"] = client.post(
                "/api/query",
                json={
                    "sessionId": parent,
                    "prompt": "dispatch team worker needing permission",
                    "toolCalls": [
                        {
                            "id": "team-permission-timeout-agent",
                            "name": "Agent",
                            "arguments": {
                                "prompt": "team write after approval",
                                "agentId": agent_id,
                                "teamName": team_name,
                                "sessionId": parent,
                                "model": "test-model",
                                "timeoutMs": 200,
                                "toolCalls": [
                                    {
                                        "id": "team-needs-permission",
                                        "name": "write_file",
                                        "arguments": {"path": rel, "content": "team approved despite wait"},
                                    }
                                ],
                            },
                        }
                    ],
                },
            ).json()

        query_thread = threading.Thread(target=run_query)
        query_thread.start()
        pending = wait_until(lambda: client.get(f"/api/swarm/{swarm['swarmId']}").json().get("pendingPermissions"), timeout=5)
        assert pending
        time.sleep(0.3)
        assert client.post(f"/api/swarm/permission/{pending[0]['requestId']}", json={"approved": True}).json()["decision"] == "allow"
        query_thread.join(timeout=30)
        assert not query_thread.is_alive()
        response = response_holder["response"]
    finally:
        TOOL_REGISTRY.policy.rules = old_rules

    result = response["toolCalls"][0]
    assert result["metadata"]["status"] == "completed"
    assert result["metadata"]["swarmId"] == swarm["swarmId"]
    assert (ROOT / rel).read_text(encoding="utf-8") == "team approved despite wait"
    assert agent_id not in TOOL_REGISTRY._active_agents


def test_query_agent_tool_background_team_dispatch_is_tracked_and_merged(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    team_name = f"bg-dispatch-team-{uuid4().hex[:8]}"
    swarm = client.post(
        "/api/swarm",
        json={"teamName": team_name, "maxWorkers": 1, "sessionId": f"swarm-bg-dispatch-{uuid4().hex}", "tasks": [], "workerUseLlm": False},
    ).json()
    parent = client.post("/api/sessions", json={"title": "background team dispatch", "model": "test-model"}).json()["sessionId"]
    agent_id = f"agent-bg-team-{uuid4().hex[:8]}"
    response = client.post(
        "/api/query",
        json={
            "sessionId": parent,
            "prompt": "dispatch background to existing team",
            "toolCalls": [
                {
                    "id": "team-agent-bg-1",
                    "name": "Agent",
                    "arguments": {
                        "prompt": "background team worker task",
                        "agentId": agent_id,
                        "teamName": team_name,
                        "sessionId": parent,
                        "subagent_type": "reviewer",
                        "model": "test-model",
                        "run_in_background": True,
                    },
                }
            ],
        },
    ).json()

    result = response["toolCalls"][0]
    metadata = result["metadata"]
    assert metadata["status"] == "async_launched"
    assert metadata["swarmId"] == swarm["swarmId"]
    assert metadata["taskId"] in TOOL_REGISTRY._tasks
    assert "background team worker task" in response["answer"]
    assert any(event["type"] == "background_agents_wait_complete" and event["agentIds"] == [agent_id] for event in response["events"])
    status = client.get(f"/api/agents/background/{agent_id}").json()["agent"]
    assert status["status"] == "completed"
    assert status["teamName"] == team_name
    assert status["agentType"] == "reviewer"
    assert Path(status["outputFile"]).read_text(encoding="utf-8")
    assert agent_id not in TOOL_REGISTRY._active_agents


def test_query_agent_tool_worktree_isolation_merges_files(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)
    old_rules = list(TOOL_REGISTRY.policy.rules)
    TOOL_REGISTRY.policy.rules.insert(0, PermissionRule("write_file", PermissionDecision.ALLOW))
    try:
        parent = client.post("/api/sessions", json={"title": "worktree parent", "model": "test-model"}).json()["sessionId"]
        agent_id = f"agent-worktree-{uuid4().hex[:8]}"
        rel = f"backend-python/.test-workspace/{uuid4().hex}/worktree-merge.txt"
        response = client.post(
            "/api/query",
            json={
                "sessionId": parent,
                "prompt": "worktree delegation",
                "toolCalls": [
                    {
                        "id": "worktree-agent-1",
                        "name": "Agent",
                        "arguments": {
                            "prompt": "write from isolated worktree",
                            "agentId": agent_id,
                            "sessionId": parent,
                            "isolation": "WORKTREE",
                            "model": "test-model",
                            "toolCalls": [
                                {
                                    "id": "child-write-1",
                                    "name": "write_file",
                                    "arguments": {"path": rel, "content": "merged from worktree"},
                                }
                            ],
                        },
                    }
                ],
            },
        ).json()
    finally:
        TOOL_REGISTRY.policy.rules = old_rules

    metadata = response["toolCalls"][0]["metadata"]
    assert metadata["status"] == "completed"
    assert metadata["worktreeMode"] in {"git", "lightweight"}
    assert metadata["mergedFiles"] == [rel]
    assert (ROOT / rel).read_text(encoding="utf-8") == "merged from worktree"
    assert not Path(metadata["worktreePath"]).exists()
    assert STATE["agentWorktrees"][agent_id]["status"] == "cleaned"
    if metadata["worktreeMode"] == "git":
        assert metadata["worktreeBranch"]
        assert STATE["agentWorktrees"][agent_id]["changedPaths"] == [rel]
    child_session = STATE["sessions"][f"subagent-{agent_id}"]
    assert child_session["workingDirectory"] == metadata["worktreePath"]
    assert child_session["agentWorktreeMode"] == metadata["worktreeMode"]


def test_git_worktree_merge_strategy_commits_and_merges_in_temp_repo() -> None:
    root = workspace()
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, text=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, text=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, text=True, capture_output=True)
    (repo / "file.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True, text=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, text=True, capture_output=True)

    agent_id = f"merge-{uuid4().hex[:8]}"
    branch = f"agent-{agent_id}"
    worktree = root / "worktree"
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-b", branch, str(worktree), "HEAD"], check=True, text=True, capture_output=True)
    (worktree / "file.txt").write_text("from agent\n", encoding="utf-8")
    STATE.setdefault("agentWorktrees", {})[agent_id] = {"path": str(worktree), "branchName": branch, "mode": "git", "status": "active"}

    merged = merge_agent_worktree(agent_id, worktree, strategy="git", repo_root=repo)
    assert merged == ["file.txt"]
    assert (repo / "file.txt").read_text(encoding="utf-8") == "from agent\n"
    log = subprocess.run(["git", "-C", str(repo), "log", "--oneline", "-2"], check=True, text=True, capture_output=True).stdout
    assert f"Agent work: {branch}" in log
    assert STATE["agentWorktrees"][agent_id]["mergeMode"] == "git"
    subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(worktree)], check=True, text=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "branch", "-D", branch], check=True, text=True, capture_output=True)


def test_query_agent_fork_clones_and_merges_file_state_cache(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)
    old_rules = list(TOOL_REGISTRY.policy.rules)
    TOOL_REGISTRY.policy.rules.insert(0, PermissionRule("write_file", PermissionDecision.ALLOW))
    rel = f"backend-python/.test-workspace/{uuid4().hex}/file-state.txt"
    target = ROOT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("parent read", encoding="utf-8")
    try:
        parent = client.post("/api/sessions", json={"title": "file state parent", "model": "test-model"}).json()["sessionId"]
        client.post(
            "/api/query",
            json={
                "sessionId": parent,
                "prompt": "read file state",
                "toolCalls": [{"id": "read-state-1", "name": "read_file", "arguments": {"path": rel}}],
            },
        )
        assert STATE["sessions"][parent]["fileStateCache"][rel]["state"] == "read"
        agent_id = f"agent-state-{uuid4().hex[:8]}"
        response = client.post(
            "/api/query",
            json={
                "sessionId": parent,
                "prompt": "fork child file state",
                "toolCalls": [
                    {
                        "id": "agent-state-1",
                        "name": "Agent",
                        "arguments": {
                            "prompt": "modify cached file",
                            "agentId": agent_id,
                            "sessionId": parent,
                            "fork": True,
                            "model": "test-model",
                            "toolCalls": [
                                {
                                    "id": "child-state-write",
                                    "name": "write_file",
                                    "arguments": {"path": rel, "content": "child modified"},
                                }
                            ],
                        },
                    }
                ],
            },
        ).json()
    finally:
        TOOL_REGISTRY.policy.rules = old_rules

    metadata = response["toolCalls"][0]["metadata"]
    assert metadata["status"] == "completed"
    assert rel in metadata["mergedFileStates"]
    child_session_id = f"fork-{agent_id}"
    assert STATE["sessions"][child_session_id]["fileStateCache"][rel]["state"] == "modified"
    assert STATE["sessions"][parent]["fileStateCache"][rel]["state"] == "modified"
    assert target.read_text(encoding="utf-8") == "child modified"


def test_query_agent_fork_inherits_full_parent_message_history(monkeypatch) -> None:
    for env_name in [
        "LLM_PROVIDER_DASHSCOPE_API_KEY",
        "LLM_PROVIDER_DEEPSEEK_API_KEY",
        "LLM_PROVIDER_MOONSHOT_API_KEY",
        "LLM_API_KEY",
    ]:
        monkeypatch.delenv(env_name, raising=False)

    parent = client.post("/api/sessions", json={"title": "fork history parent", "model": "test-model"}).json()["sessionId"]
    STATE["sessions"][parent]["messages"] = [
        {
            "type": "user" if index % 2 == 0 else "assistant",
            "uuid": f"history-{index}",
            "timestamp": int(time.time() * 1000) + index,
            "content": [{"type": "text", "text": f"parent-history-{index}"}],
        }
        for index in range(25)
    ]
    agent_id = f"agent-fork-history-{uuid4().hex[:8]}"
    response = client.post(
        "/api/query",
        json={
            "sessionId": parent,
            "prompt": "launch fork with full history",
            "toolCalls": [
                {
                    "id": "agent-fork-history-1",
                    "name": "Agent",
                    "arguments": {
                        "prompt": "use full inherited history",
                        "agentId": agent_id,
                        "sessionId": parent,
                        "fork": True,
                        "model": "test-model",
                    },
                }
            ],
        },
    ).json()

    child_session_id = f"fork-{agent_id}"
    metadata = response["toolCalls"][0]["metadata"]
    child_messages = STATE["sessions"][child_session_id]["messages"]
    inherited_text = json.dumps(child_messages, ensure_ascii=False)
    assert metadata["childSessionId"] == child_session_id
    assert "parent-history-0" in inherited_text
    assert "parent-history-24" in inherited_text


def test_advanced_runtime_modules_task_sandbox_cost_bridge_lsp() -> None:
    task = client.post("/api/tasks", json={"title": "deep parity", "type": "analysis"}).json()
    task_id = task["id"]
    assert task["status"] == "pending"
    updated = client.patch(f"/api/tasks/{task_id}", json={"status": "running", "progress": 50}).json()
    assert updated["progress"] == 50
    output = client.post(f"/api/tasks/{task_id}/output", json={"content": "half done"}).json()
    assert output["entry"]["content"] == "half done"

    scheduled = client.post(
        "/api/query/tools/schedule",
        json={"toolCalls": [{"name": "FileEdit", "path": "a.py"}, {"name": "FileRead", "path": "a.py"}, {"name": "Bash"}]},
    ).json()
    assert [item["name"] for item in scheduled["toolCalls"]] == ["FileRead", "Bash", "FileEdit"]
    assert scheduled["hasConflict"] is True

    session_for_collapse = client.post("/api/sessions", json={"title": "collapse", "model": "test-model"}).json()["sessionId"]
    STATE["sessions"][session_for_collapse]["messages"] = [
        {"type": "user", "content": [{"type": "text", "text": "keep this instruction"}]},
        {"type": "assistant", "content": [{"type": "text", "text": "x" * 3000}]},
        {"type": "assistant", "content": [{"type": "text", "text": "recent"}]},
    ]
    collapsed = client.post(f"/api/query/session/{session_for_collapse}/collapse", json={"protectedTail": 1}).json()
    assert collapsed["collapsedCount"] == 1
    assert "collapsed" in collapsed["messages"][1]["content"][0]["text"]
    side = client.post("/api/query/side", json={"systemPrompt": "summarize", "content": "abc" * 100, "maxTokens": 10}).json()
    assert side["status"] == "completed"
    micro = client.post("/api/query/micro-compact", json={"messages": [{"type": "user", "toolUseResult": "x" * 100}]}).json()
    assert micro["compactedCount"] == 0 or "messages" in micro
    summary = client.post("/api/query/tool-result/summary", json={"toolName": "Bash", "content": "line1\nline2", "maxChars": 100}).json()
    assert summary["toolName"] == "Bash"
    abort = client.post(f"/api/query/session/{session_for_collapse}/abort", json={"reason": "USER_INTERRUPT"}).json()
    assert abort["success"] is True
    abort_status = client.get(f"/api/query/session/{session_for_collapse}/abort").json()
    assert abort_status["aborted"] is True

    cron = client.post("/api/cron/tasks", json={"cron": "*/10 * * * *", "prompt": "scheduled analysis"}).json()
    assert cron["success"] is True
    cron_id = cron["task"]["id"]
    cron_list = client.get("/api/cron/tasks").json()
    assert any(item["id"] == cron_id for item in cron_list["tasks"])
    assert client.post("/api/cron/tasks", json={"cron": "* * *", "prompt": "bad"}).status_code == 400
    assert client.delete(f"/api/cron/tasks/{cron_id}").json()["success"] is True

    sandbox = client.post("/api/sandbox/execute", json={"command": ["python", "-c", "print('ok')"], "timeoutMs": 5000}).json()
    assert sandbox["success"] is True
    assert "ok" in sandbox["stdout"]
    blocked = client.post("/api/sandbox/execute", json={"command": "rm -rf ."}).json()
    assert blocked["blocked"] is True

    cost = client.post("/api/cost/record", json={"sessionId": "cost-session", "inputTokens": 10, "outputTokens": 5, "costUsd": 0.01}).json()
    assert cost["success"] is True
    summary = client.get("/api/cost?sessionId=cost-session").json()
    assert summary["totalCostUsd"] >= 0.01

    anomaly = client.post("/api/anomalies", json={"type": "latency", "severity": "high", "message": "slow turn"}).json()
    assert anomaly["status"] == "open"
    closed = client.patch(f"/api/anomalies/{anomaly['id']}", json={"status": "resolved"}).json()
    assert closed["status"] == "resolved"

    device = client.post("/api/bridge/devices", json={"name": "phone"}).json()
    assert device["trusted"] is True
    bridge_status = client.get("/api/bridge/status").json()
    assert bridge_status["deviceCount"] >= 1

    bindings = client.put("/api/keybindings", json={"keybindings": {"ctrl+x": "cancel"}}).json()
    assert bindings["keybindings"]["ctrl+x"] == "cancel"
    resolved = client.post("/api/keybindings/resolve", json={"key": "ctrl+x"}).json()
    assert resolved["action"] == "cancel"

    root = workspace()
    source = root / "sample.py"
    source.write_text("class Demo:\n    pass\n\ndef target():\n    return Demo()\n", encoding="utf-8")
    rel = str(source.relative_to(ROOT))
    symbols = client.get(f"/api/lsp/symbols?path={rel}").json()["symbols"]
    assert any(item["name"] == "Demo" for item in symbols)
    refs = client.get(f"/api/lsp/references?symbol=Demo&path={rel}").json()["references"]
    assert refs
    servers = client.get("/api/lsp/servers").json()
    assert "pyright" in {server["name"] for server in servers["servers"]}
    opened = client.post("/api/lsp/open", json={"path": rel}).json()
    assert opened["success"] is True
    workspace_lsp_symbols = client.get("/api/lsp/workspace-symbols?query=target").json()["symbols"]
    assert any(item["name"] == "target" for item in workspace_lsp_symbols)
    hover = client.get(f"/api/lsp/hover?path={rel}&line=4&character=5").json()
    assert "hover" in hover
    definition = client.get(f"/api/lsp/definition?path={rel}&line=4&character=12").json()
    assert "definitions" in definition
    hierarchy = client.get(f"/api/lsp/call-hierarchy?symbol=target&path={rel}").json()
    assert hierarchy["symbol"] == "target"
    closed_lsp = client.post("/api/lsp/close", json={"path": rel}).json()
    assert closed_lsp["open"] is False

    journey = client.post("/api/verify/journey", json={"steps": [{"name": "open", "type": "http_api"}]}).json()
    assert journey["success"] is True
    assert journey["result"]["verdict"] == "verified"
    failed_journey = client.post("/api/verify/journey", json={"steps": [{"name": "open", "type": "http_api", "fail": True}]}).json()
    assert failed_journey["success"] is False
    assert failed_journey["results"][0]["status"] == "failed"
    unavailable = client.post(
        "/api/verify/journey",
        json={"featureFlags": {"RUNTIME_VERIFICATION": False}, "steps": [{"name": "open", "type": "http_api"}]},
    ).json()
    assert unavailable["verdict"] == "unavailable"

    correction = client.post(
        "/api/correction/parse",
        json={"output": "src/app.ts(23,5): error TS2304: Cannot find name 'xyz'\nFAILED tests/test_main.py::test_addition - AssertionError: assert 3 == 4"},
    ).json()
    assert correction["hasIssues"] is True
    assert correction["compileErrors"][0]["language"] == "typescript"
    assert correction["testFailures"][0]["framework"] == "pytest"
    fetched_correction = client.get(f"/api/correction/reports/{correction['id']}").json()
    assert fetched_correction["id"] == correction["id"]
    detected_correction = client.post(
        "/api/correction/detect",
        json={"output": "src/app.ts(23,5): error TS2304: Cannot find name 'xyz'", "previousAttempts": 1},
    ).json()
    assert detected_correction["instruction"]["type"] == "COMPILE_ERROR"
    assert detected_correction["instruction"]["attemptNumber"] == 2
    abort_decision = client.post(
        "/api/correction/should-abort",
        json={
            "previousOutput": "src/app.ts(23,5): error TS2304: Cannot find name 'xyz'",
            "newOutput": "src/app.ts(23,5): error TS2304: Cannot find name 'xyz'\nsrc/other.ts(1,1): error TS2304: Cannot find name 'abc'",
        },
    ).json()
    assert abort_decision["abort"] is True
    assert journey["results"][0]["status"] == "passed"


def test_hooks_execute_and_modify_prompt() -> None:
    hook_id = f"hook-{uuid4().hex}"
    hook = client.post(
        "/api/hooks",
        json={"id": hook_id, "event": "USER_PROMPT_SUBMIT", "matcher": "original", "action": "modify_input", "value": "modified prompt"},
    ).json()
    assert hook["id"] == hook_id

    executed = client.post("/api/hooks/execute", json={"event": "USER_PROMPT_SUBMIT", "context": {"input": "original prompt"}}).json()
    assert executed["proceed"] is True
    assert executed["context"]["input"] == "modified prompt"

    response = client.post("/ui/chat", data={"text": "original prompt"}, follow_redirects=False)
    assert response.status_code == 303
    follow = client.get(response.headers["location"])
    assert "modified prompt" in follow.text

    events = client.get("/api/hooks/events").json()
    assert any(event["hookId"] == hook_id for event in events["events"])
    assert client.delete(f"/api/hooks/{hook_id}").json()["success"] is True


def test_sockjs_stomp_chat_flow() -> None:
    with client.websocket_connect("/ws/000/test/websocket") as ws:
        assert ws.receive_text() == "o"

        connect = "CONNECT\naccept-version:1.2\nX-Session-Id:test-session\n\n\x00"
        ws.send_text(json.dumps([connect]))
        connected = ws.receive_text()
        assert "CONNECTED" in connected

        subscribe = "SUBSCRIBE\nid:sub-0\ndestination:/user/queue/messages\n\n\x00"
        ws.send_text(json.dumps([subscribe]))

        send = "SEND\ndestination:/app/chat\n\n{\"text\":\"hello\"}\x00"
        ws.send_text(json.dumps([send]))

        frames = [ws.receive_text() for _ in range(8)]
        assert any("stream_delta" in frame for frame in frames)


def test_sockjs_stomp_stateful_control_messages() -> None:
    with client.websocket_connect("/ws/000/control/websocket") as ws:
        assert ws.receive_text() == "o"

        connect = "CONNECT\naccept-version:1.2\nX-Session-Id:control-session\n\n\x00"
        ws.send_text(json.dumps([connect]))
        assert "CONNECTED" in ws.receive_text()

        subscribe = "SUBSCRIBE\nid:sub-0\ndestination:/user/queue/messages\n\n\x00"
        ws.send_text(json.dumps([subscribe]))

        cases = [
            ("SEND\ndestination:/app/model\n\n{\"model\":\"test-model\"}\x00", "model_changed"),
            ("SEND\ndestination:/app/permission-mode\n\n{\"mode\":\"acceptEdits\"}\x00", "permission_mode_changed"),
            ("SEND\ndestination:/app/permission\n\n{\"toolUseId\":\"tool-1\",\"decision\":\"allow\"}\x00", "permission_processed"),
            ("SEND\ndestination:/app/mcp\n\n{\"operation\":\"connect\",\"serverId\":\"demo\"}\x00", "mcp-connect"),
            ("SEND\ndestination:/app/mcp\n\n{\"operation\":\"list\"}\x00", "mcp_status"),
            ("SEND\ndestination:/app/rewind\n\n{\"messageId\":\"m1\",\"filePaths\":[\"a.py\"]}\x00", "rewind_complete"),
            ("SEND\ndestination:/app/elicitation\n\n{\"requestId\":\"q1\",\"answer\":\"ok\"}\x00", "elicitation_resolved"),
            ("SEND\ndestination:/app/activity-save\n\n{\"title\":\"work\"}\x00", "activity_saved"),
        ]
        for frame, expected in cases:
            ws.send_text(json.dumps([frame]))
            assert expected in ws.receive_text()
