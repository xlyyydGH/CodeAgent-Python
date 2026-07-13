from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


MAX_MCP_RESULT_SIZE = 1024 * 1024


class McpTransportType(StrEnum):
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"
    WS = "ws"
    STREAMABLE_HTTP = "streamable_http"
    WEBSOCKET = "websocket"
    IN_PROCESS = "in_process"
    DISABLED = "disabled"


class McpConfigScope(StrEnum):
    LOCAL = "local"
    USER = "user"
    PROJECT = "project"
    WORKSPACE = "workspace"
    DYNAMIC = "dynamic"
    PLUGIN = "plugin"
    REMOTE = "remote"


class McpConnectionStatus(StrEnum):
    PENDING = "pending"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass(slots=True)
class JsonRpcMessage:
    jsonrpc: str = "2.0"
    id: str | int | None = None
    method: str | None = None
    params: dict[str, Any] | list[Any] | None = None
    result: Any = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            payload["id"] = self.id
        if self.method is not None:
            payload["method"] = self.method
        if self.params is not None:
            payload["params"] = self.params
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        return payload

    @classmethod
    def request(cls, method: str, params: dict[str, Any] | None = None, request_id: str | int = 1) -> "JsonRpcMessage":
        return cls(id=request_id, method=method, params=params or {})

    @classmethod
    def parse(cls, raw: str | dict[str, Any]) -> "JsonRpcMessage":
        data = json.loads(raw) if isinstance(raw, str) else raw
        return cls(
            jsonrpc=str(data.get("jsonrpc") or "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass(slots=True)
class McpServerConfig:
    name: str
    type: McpTransportType
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    scope: McpConfigScope = McpConfigScope.DYNAMIC

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "McpServerConfig":
        raw_type = str(data.get("type") or data.get("transport") or "sse").lower().replace("-", "_")
        transport = McpTransportType(raw_type) if raw_type in set(item.value for item in McpTransportType) else McpTransportType.SSE
        return cls(
            name=str(data.get("name") or data.get("id") or "mcp-server"),
            type=transport,
            command=data.get("command"),
            args=[str(item) for item in data.get("args") or []],
            url=data.get("url"),
            env={str(key): str(value) for key, value in (data.get("env") or {}).items()},
            scope=McpConfigScope(str(data.get("scope") or "dynamic").lower()) if str(data.get("scope") or "dynamic").lower() in set(item.value for item in McpConfigScope) else McpConfigScope.DYNAMIC,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        data["scope"] = self.scope.value
        return data


@dataclass(slots=True)
class McpConnection:
    config: McpServerConfig
    status: McpConnectionStatus = McpConnectionStatus.PENDING
    tools: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)
    reconnectAttempts: int = 0
    lastError: str | None = None
    nextRetryAt: float | None = None
    updatedAt: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.config.to_dict(),
            "id": self.config.name,
            "status": self.status.value,
            "tools": self.tools,
            "resources": self.resources,
            "prompts": self.prompts,
            "reconnectAttempts": self.reconnectAttempts,
            "lastError": self.lastError,
            "nextRetryAt": self.nextRetryAt,
            "updatedAt": self.updatedAt,
        }

    def wrapped_tools(self) -> list[dict[str, Any]]:
        wrapped = []
        for tool in self.tools:
            name = str(tool.get("name") or tool.get("toolName") or "")
            schema = tool.get("inputSchema") or tool.get("schema") or {"type": "object"}
            if not name or not isinstance(schema, dict) or schema.get("type", "object") != "object":
                continue
            wrapped.append(
                {
                    "name": f"mcp__{self.config.name}__{name}",
                    "originalName": name,
                    "serverName": self.config.name,
                    "description": tool.get("description") or "",
                    "inputSchema": schema,
                    "group": "mcp",
                    "isMcp": True,
                }
            )
        return wrapped


class McpAuthFailureCache:
    def __init__(self, ttl_seconds: int = 900) -> None:
        self.ttl_seconds = ttl_seconds
        self._failures: dict[str, float] = {}

    def record(self, server: str) -> None:
        self._failures[server] = time.time()

    def is_cached(self, server: str) -> bool:
        timestamp = self._failures.get(server)
        if timestamp is None:
            return False
        if time.time() - timestamp > self.ttl_seconds:
            self._failures.pop(server, None)
            return False
        return True

    def clear(self, server: str | None = None) -> None:
        if server is None:
            self._failures.clear()
        else:
            self._failures.pop(server, None)

    def to_dict(self) -> dict[str, Any]:
        return {"size": len(self._failures), "servers": sorted(self._failures)}


class McpApprovalService:
    def __init__(self) -> None:
        self._trusted: set[str] = set()
        self._decisions: list[dict[str, Any]] = []

    def is_trusted(self, server: str) -> bool:
        return server in self._trusted

    def decide(self, server: str, decision: str, reason: str | None = None) -> dict[str, Any]:
        record = {"server": server, "decision": decision, "reason": reason, "createdAt": time.time()}
        self._decisions.append(record)
        if decision == "allow":
            self._trusted.add(server)
        if decision == "deny":
            self._trusted.discard(server)
        return record

    def decisions(self) -> list[dict[str, Any]]:
        return list(self._decisions)


class TokenEncryptionService:
    def __init__(self, secret: str = "zhikuncode-python") -> None:
        self.secret = secret.encode("utf-8") or b"secret"

    def encrypt(self, token: str) -> str:
        raw = token.encode("utf-8")
        mixed = bytes(value ^ self.secret[index % len(self.secret)] for index, value in enumerate(raw))
        return base64.urlsafe_b64encode(mixed).decode("ascii")

    def decrypt(self, encoded: str) -> str:
        raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        plain = bytes(value ^ self.secret[index % len(self.secret)] for index, value in enumerate(raw))
        return plain.decode("utf-8")


class McpStdioSession:
    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._request_id = 0
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._closed = False
        if not config.command:
            raise ValueError("stdio MCP server requires command")
        env = os.environ.copy()
        env.update(config.env)
        self.process = subprocess.Popen(
            [config.command, *config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        self._reader = threading.Thread(target=self._read_stdout, name=f"mcp-stdio-{config.name}", daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        if self.process.stdout is None:
            return
        for line in self.process.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                message = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict):
                self._responses.put(message)

    def request(self, method: str, params: dict[str, Any] | None = None, timeout_ms: int = 30_000) -> Any:
        self._request_id += 1
        request_id = f"{self.config.name}-{self._request_id}"
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        deadline = time.time() + max(timeout_ms, 1) / 1000
        while time.time() < deadline:
            try:
                message = self._responses.get(timeout=max(0.01, min(0.25, deadline - time.time())))
            except queue.Empty:
                continue
            if message.get("id") != request_id:
                continue
            if message.get("error"):
                error = message.get("error") or {}
                raise RuntimeError(str(error.get("message") or error))
            return message.get("result")
        raise TimeoutError(f"MCP stdio request timed out: {method}")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _send(self, payload: dict[str, Any]) -> None:
        if self._closed or self.process.stdin is None:
            raise RuntimeError("MCP stdio session is closed")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def close(self) -> None:
        self._closed = True
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


class McpHttpSession:
    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._request_id = 0
        if not config.url:
            raise ValueError("HTTP MCP server requires url")

    def request(self, method: str, params: dict[str, Any] | None = None, timeout_ms: int = 30_000) -> Any:
        self._request_id += 1
        request_id = f"{self.config.name}-{self._request_id}"
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        response = self._post(payload, timeout_ms)
        if response.get("error"):
            error = response.get("error") or {}
            raise RuntimeError(str(error.get("message") or error))
        return response.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None, timeout_ms: int = 30_000) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params or {}}, timeout_ms)

    def _post(self, payload: dict[str, Any], timeout_ms: int) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            str(self.config.url),
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=max(timeout_ms, 1) / 1000) as response:
            raw = response.read()
            if not raw:
                return {}
            parsed = json.loads(raw.decode("utf-8"))
            return parsed if isinstance(parsed, dict) else {}

    def close(self) -> None:
        return


class McpClientManager:
    def __init__(self) -> None:
        self.connections: dict[str, McpConnection] = {}
        self.authFailures = McpAuthFailureCache()
        self.approvals = McpApprovalService()
        self.tokens = TokenEncryptionService()
        self._result_cache: dict[str, tuple[float, str]] = {}
        self._stdio_sessions: dict[str, McpStdioSession] = {}
        self._http_sessions: dict[str, McpHttpSession] = {}

    @staticmethod
    def calculate_backoff(attempt: int) -> int:
        return min(30_000, 1000 * (2 ** max(0, attempt - 1)))

    def add_server(self, config: dict[str, Any]) -> McpConnection:
        server_config = McpServerConfig.from_dict(config)
        connection = McpConnection(
            config=server_config,
            status=McpConnectionStatus(str(config.get("status") or "pending").lower())
            if str(config.get("status") or "pending").lower() in set(item.value for item in McpConnectionStatus)
            else McpConnectionStatus.PENDING,
            tools=list(config.get("tools") or []),
            resources=list(config.get("resources") or []),
            prompts=list(config.get("prompts") or []),
            reconnectAttempts=int(config.get("reconnectAttempts") or 0),
            lastError=config.get("lastError"),
            nextRetryAt=float(config["nextRetryAt"]) if config.get("nextRetryAt") is not None else None,
        )
        self.connections[server_config.name] = connection
        return connection

    def remove_server(self, name: str) -> bool:
        self.close_server(name)
        return self.connections.pop(name, None) is not None

    def get(self, name: str) -> McpConnection | None:
        return self.connections.get(name)

    def list(self) -> list[dict[str, Any]]:
        return [connection.to_dict() for connection in self.connections.values()]

    def discover_wrapped_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for connection in self.connections.values():
            if connection.status == McpConnectionStatus.CONNECTED:
                tools.extend(connection.wrapped_tools())
        return tools

    def connect_server(self, name: str, timeout_ms: int = 30_000) -> McpConnection:
        connection = self.connections.get(name)
        if connection is None:
            raise KeyError(name)
        if connection.config.type == McpTransportType.DISABLED:
            connection.status = McpConnectionStatus.DISABLED
            return connection
        if connection.config.type in {McpTransportType.HTTP, McpTransportType.STREAMABLE_HTTP}:
            return self._connect_http_server(name, connection, timeout_ms)
        if connection.config.type != McpTransportType.STDIO:
            connection.status = McpConnectionStatus.CONNECTED
            connection.updatedAt = time.time()
            return connection
        connection.status = McpConnectionStatus.CONNECTING
        connection.updatedAt = time.time()
        try:
            self.close_server(name)
            session = McpStdioSession(connection.config)
            self._stdio_sessions[name] = session
            session.request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": "zhikuncode-python", "version": "0.1"},
                    "capabilities": {"tools": {}, "prompts": {}, "resources": {}},
                },
                timeout_ms=timeout_ms,
            )
            session.notify("notifications/initialized")
            tools_result = session.request("tools/list", {}, timeout_ms=timeout_ms)
            tools = tools_result.get("tools") if isinstance(tools_result, dict) else []
            connection.tools = list(tools) if isinstance(tools, list) else []
            connection.status = McpConnectionStatus.CONNECTED
            connection.lastError = None
            connection.nextRetryAt = None
            connection.updatedAt = time.time()
            return connection
        except Exception as exc:
            self.close_server(name)
            connection.status = McpConnectionStatus.FAILED
            connection.lastError = str(exc)
            connection.updatedAt = time.time()
            connection.reconnectAttempts += 1
            connection.nextRetryAt = connection.updatedAt + (self.calculate_backoff(connection.reconnectAttempts) / 1000)
            return connection

    def _connect_http_server(self, name: str, connection: McpConnection, timeout_ms: int) -> McpConnection:
        connection.status = McpConnectionStatus.CONNECTING
        connection.updatedAt = time.time()
        try:
            self.close_server(name)
            session = McpHttpSession(connection.config)
            self._http_sessions[name] = session
            session.request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": "zhikuncode-python", "version": "0.1"},
                    "capabilities": {"tools": {}, "prompts": {}, "resources": {}},
                },
                timeout_ms=timeout_ms,
            )
            session.notify("notifications/initialized", timeout_ms=timeout_ms)
            tools_result = session.request("tools/list", {}, timeout_ms=timeout_ms)
            tools = tools_result.get("tools") if isinstance(tools_result, dict) else []
            connection.tools = list(tools) if isinstance(tools, list) else []
            connection.status = McpConnectionStatus.CONNECTED
            connection.lastError = None
            connection.nextRetryAt = None
            connection.updatedAt = time.time()
            return connection
        except Exception as exc:
            self.close_server(name)
            connection.status = McpConnectionStatus.FAILED
            connection.lastError = str(exc)
            connection.updatedAt = time.time()
            connection.reconnectAttempts += 1
            connection.nextRetryAt = connection.updatedAt + (self.calculate_backoff(connection.reconnectAttempts) / 1000)
            return connection

    def close_server(self, name: str) -> bool:
        session = self._stdio_sessions.pop(name, None)
        http_session = self._http_sessions.pop(name, None)
        closed = False
        if session is not None:
            session.close()
            closed = True
        if http_session is not None:
            http_session.close()
            closed = True
        return closed

    def call_tool(self, server: str, tool_name: str, arguments: dict[str, Any] | None = None, timeout_ms: int = 30_000) -> dict[str, Any]:
        connection = self.connections.get(server)
        if connection is None:
            return {"status": "error", "error": f"MCP server not found: {server}", "serverName": server, "toolName": tool_name}
        args = arguments if isinstance(arguments, dict) else {}
        request = JsonRpcMessage.request("tools/call", {"name": tool_name, "arguments": args}).to_dict()
        cache_key = self._cache_key(server, tool_name, args)
        if connection.config.type == McpTransportType.STDIO and server not in self._stdio_sessions and connection.config.command:
            self.connect_server(server, timeout_ms=timeout_ms)
        if connection.config.type in {McpTransportType.HTTP, McpTransportType.STREAMABLE_HTTP} and server not in self._http_sessions and connection.config.url:
            configured_tool = self._find_tool(connection, tool_name)
            has_static_result = connection.status == McpConnectionStatus.CONNECTED and configured_tool is not None and configured_tool.get("result") is not None
            if not has_static_result:
                self.connect_server(server, timeout_ms=timeout_ms)
        if connection.status != McpConnectionStatus.CONNECTED:
            cached = self._cached_result(cache_key)
            if cached is not None and not self._is_realtime_tool(tool_name):
                return {
                    "status": "success",
                    "serverName": server,
                    "toolName": tool_name,
                    "connectionType": connection.config.type.value,
                    "request": request,
                    "content": f"[cached] {cached}",
                    "cached": True,
                    "metadata": {"mcpServer": server, "mcpTool": tool_name, "cached": "true"},
                }
            return {
                "status": "error",
                "serverName": server,
                "toolName": tool_name,
                "connectionType": connection.config.type.value,
                "request": request,
                "error": f"MCP server '{server}' is not connected (status: {connection.status.value})",
                "cached": False,
            }
        if connection.config.type in {McpTransportType.HTTP, McpTransportType.STREAMABLE_HTTP} and server in self._http_sessions:
            return self._call_connected_transport(server, tool_name, args, request, cache_key, timeout_ms, self._http_sessions[server])
        if connection.config.type == McpTransportType.STDIO and server in self._stdio_sessions:
            return self._call_connected_transport(server, tool_name, args, request, cache_key, timeout_ms, self._stdio_sessions[server])
        tool = self._find_tool(connection, tool_name)
        if tool is None:
            return {
                "status": "error",
                "serverName": server,
                "toolName": tool_name,
                "connectionType": connection.config.type.value,
                "request": request,
                "error": f"MCP tool not found: {tool_name}",
                "cached": False,
            }
        result_payload = tool.get("result")
        if result_payload is None:
            result_payload = {"content": [{"type": "text", "text": json.dumps(args, ensure_ascii=False)}], "isError": False}
        content = self._extract_result_content(result_payload)
        if len(content) > MAX_MCP_RESULT_SIZE:
            content = content[:MAX_MCP_RESULT_SIZE] + f"\n[Truncated: exceeded {MAX_MCP_RESULT_SIZE} chars]"
        is_error = bool(result_payload.get("isError")) if isinstance(result_payload, dict) else False
        if not is_error and not self._is_realtime_tool(tool_name) and content:
            self._result_cache[cache_key] = (time.time(), content)
        return {
            "status": "error" if is_error else "success",
            "serverName": server,
            "toolName": tool_name,
            "connectionType": connection.config.type.value,
            "request": request,
            "result": result_payload,
            "content": content,
            "cached": False,
            "timeoutMs": timeout_ms,
            "metadata": {"mcpServer": server, "mcpTool": tool_name},
        }

    def _call_connected_transport(
        self,
        server: str,
        tool_name: str,
        args: dict[str, Any],
        request: dict[str, Any],
        cache_key: str,
        timeout_ms: int,
        session: Any,
    ) -> dict[str, Any]:
        connection = self.connections[server]
        try:
            result_payload = session.request("tools/call", {"name": tool_name, "arguments": args}, timeout_ms=timeout_ms)
        except Exception as exc:
            failure = self.mark_connection_failure(server, str(exc))
            cached = self._cached_result(cache_key)
            if cached is not None and not self._is_realtime_tool(tool_name):
                return {
                    "status": "success",
                    "serverName": server,
                    "toolName": tool_name,
                    "connectionType": connection.config.type.value,
                    "request": request,
                    "content": f"[cached] {cached}",
                    "cached": True,
                    "metadata": {"mcpServer": server, "mcpTool": tool_name, "cached": "true", "failure": failure},
                }
            return {
                "status": "error",
                "serverName": server,
                "toolName": tool_name,
                "connectionType": connection.config.type.value,
                "request": request,
                "error": str(exc),
                "cached": False,
                "metadata": {"mcpServer": server, "mcpTool": tool_name, "failure": failure},
            }
        content = self._extract_result_content(result_payload)
        if len(content) > MAX_MCP_RESULT_SIZE:
            content = content[:MAX_MCP_RESULT_SIZE] + f"\n[Truncated: exceeded {MAX_MCP_RESULT_SIZE} chars]"
        is_error = bool(result_payload.get("isError")) if isinstance(result_payload, dict) else False
        if not is_error and not self._is_realtime_tool(tool_name) and content:
            self._result_cache[cache_key] = (time.time(), content)
        return {
            "status": "error" if is_error else "success",
            "serverName": server,
            "toolName": tool_name,
            "connectionType": connection.config.type.value,
            "request": request,
            "result": result_payload,
            "content": content,
            "cached": False,
            "timeoutMs": timeout_ms,
            "metadata": {"mcpServer": server, "mcpTool": tool_name},
        }

    def _find_tool(self, connection: McpConnection, tool_name: str) -> dict[str, Any] | None:
        for tool in connection.tools:
            if isinstance(tool, dict) and str(tool.get("name") or tool.get("toolName") or "") == tool_name:
                return tool
        return None

    def _cache_key(self, server: str, tool_name: str, arguments: dict[str, Any]) -> str:
        return f"{server}:{tool_name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)}"

    def _cached_result(self, key: str) -> str | None:
        record = self._result_cache.get(key)
        if record is None:
            return None
        created_at, content = record
        if time.time() - created_at > 300:
            self._result_cache.pop(key, None)
            return None
        return content

    def _is_realtime_tool(self, tool_name: str) -> bool:
        lowered = tool_name.lower()
        return any(part in lowered for part in ("search", "web", "fetch", "browse", "realtime", "live"))

    def _extract_result_content(self, result: Any) -> str:
        if not isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        content = result.get("content")
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                    chunks.append(str(item.get("text") or ""))
            if chunks:
                return "".join(chunks)
        if "text" in result:
            return str(result.get("text") or "")
        return json.dumps(result, ensure_ascii=False)

    def validate_tool_schemas(self, name: str | None = None) -> dict[str, Any]:
        targets = [self.connections[name]] if name and name in self.connections else list(self.connections.values())
        valid_tools: list[dict[str, Any]] = []
        invalid_tools: list[dict[str, Any]] = []
        for connection in targets:
            for tool in connection.tools:
                tool_name = str(tool.get("name") or tool.get("toolName") or "")
                schema = tool.get("inputSchema") or tool.get("schema") or {"type": "object"}
                errors: list[str] = []
                if not tool_name:
                    errors.append("missing_name")
                if not isinstance(schema, dict):
                    errors.append("schema_not_object")
                elif schema.get("type", "object") != "object":
                    errors.append("schema_type_must_be_object")
                elif "properties" in schema and not isinstance(schema.get("properties"), dict):
                    errors.append("properties_must_be_object")
                record = {"serverName": connection.config.name, "toolName": tool_name, "errors": errors}
                if errors:
                    invalid_tools.append(record)
                else:
                    valid_tools.append(record)
        return {
            "valid": not invalid_tools,
            "validCount": len(valid_tools),
            "invalidCount": len(invalid_tools),
            "validTools": valid_tools,
            "invalidTools": invalid_tools,
        }

    def mark_connection_failure(self, name: str, error: str, auth_failed: bool = False) -> dict[str, Any]:
        connection = self.connections.get(name)
        if connection is None:
            raise KeyError(name)
        now = time.time()
        connection.status = McpConnectionStatus.FAILED
        connection.lastError = error
        connection.reconnectAttempts += 1
        connection.updatedAt = now
        backoff_ms = self.calculate_backoff(connection.reconnectAttempts)
        connection.nextRetryAt = now + (backoff_ms / 1000)
        if auth_failed:
            self.authFailures.record(name)
        return {
            "server": name,
            "status": connection.status.value,
            "attempt": connection.reconnectAttempts,
            "backoffMs": backoff_ms,
            "nextRetryAt": connection.nextRetryAt,
            "updatedAt": connection.updatedAt,
            "authCached": self.authFailures.is_cached(name),
            "error": error,
        }

    def plan_reconnect(self, name: str) -> dict[str, Any]:
        connection = self.connections.get(name)
        if connection is None:
            raise KeyError(name)
        attempt = connection.reconnectAttempts + 1
        backoff_ms = self.calculate_backoff(attempt)
        if self.authFailures.is_cached(name):
            return {
                "server": name,
                "allowed": False,
                "reason": "auth_failure_cached",
                "attempt": attempt,
                "backoffMs": backoff_ms,
                "nextRetryAt": connection.nextRetryAt,
            }
        return {
            "server": name,
            "allowed": True,
            "reason": "ready",
            "attempt": attempt,
            "backoffMs": backoff_ms,
            "nextRetryAt": time.time() + (backoff_ms / 1000),
        }

    def load_state(self, servers: list[dict[str, Any]]) -> None:
        for name in list(self._stdio_sessions):
            self.close_server(name)
        for name in list(self._http_sessions):
            self.close_server(name)
        self.connections.clear()
        for server in servers:
            self.add_server(server)
