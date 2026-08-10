import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.mcp_runtime import (  # noqa: E402
    JsonRpcMessage,
    MAX_MCP_RESULT_SIZE,
    McpClientManager,
    McpConfigScope,
    McpConnectionStatus,
    McpServerConfig,
    McpTransportType,
    TokenEncryptionService,
)


class McpHttpEchoHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[str] = []

    def log_message(self, _format: str, *args) -> None:  # noqa: ANN002
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8")
        message = JsonRpcMessage.parse(body)
        self.__class__.requests.append(str(message.method))
        if message.method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "http-echo", "version": "0.1"},
                "capabilities": {"tools": {}},
            }
        elif message.method == "notifications/initialized":
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        elif message.method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "remote_echo",
                        "description": "Echo through HTTP MCP.",
                        "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}},
                    }
                ]
            }
        elif message.method == "tools/call":
            args = (message.params or {}).get("arguments", {}) if isinstance(message.params, dict) else {}
            result = {"content": [{"type": "text", "text": f"http:{args.get('message', '')}"}], "isError": False}
        else:
            payload = {"jsonrpc": "2.0", "id": message.id, "error": {"code": -32601, "message": "unknown method"}}
            self._send_json(payload)
            return
        self._send_json({"jsonrpc": "2.0", "id": message.id, "result": result})

    def _send_json(self, payload: dict) -> None:
        raw = JsonRpcMessage.parse(payload).to_dict() if "result" in payload or "error" in payload else payload
        body = __import__("json").dumps(raw, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_mcp_http_server() -> tuple[HTTPServer, str]:
    McpHttpEchoHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), McpHttpEchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/mcp"


def test_mcp_config_json_rpc_and_transport_contracts() -> None:
    assert len(McpTransportType) == 8
    assert McpTransportType.STDIO.value == "stdio"
    assert McpTransportType.SSE.value == "sse"
    assert McpTransportType.WS.value == "ws"
    assert len(McpConfigScope) == 7
    assert len(McpConnectionStatus) == 6

    stdio = McpServerConfig.from_dict({"name": "local", "type": "stdio", "command": "node", "args": ["server.js"]})
    assert stdio.name == "local"
    assert stdio.type == McpTransportType.STDIO
    assert stdio.command == "node"
    assert stdio.args == ["server.js"]

    message = JsonRpcMessage.request("tools/list", {"cursor": "1"}, request_id="req-1")
    parsed = JsonRpcMessage.parse(message.to_dict())
    assert parsed.method == "tools/list"
    assert parsed.params == {"cursor": "1"}
    assert parsed.id == "req-1"


def test_mcp_client_manager_tools_backoff_auth_and_tokens() -> None:
    manager = McpClientManager()
    assert manager.calculate_backoff(1) == 1000
    assert manager.calculate_backoff(6) == 30000

    conn = manager.add_server(
        {
            "name": "srv",
            "type": "sse",
            "url": "http://localhost:1/sse",
            "status": "connected",
            "tools": [{"name": "read_file", "description": "Read", "inputSchema": {"type": "object"}}],
            "resources": [{"uri": "file:///data.json", "name": "data"}],
        }
    )
    assert conn.status == McpConnectionStatus.CONNECTED
    wrapped = manager.discover_wrapped_tools()
    assert wrapped[0]["name"] == "mcp__srv__read_file"
    assert wrapped[0]["serverName"] == "srv"
    assert wrapped[0]["isMcp"] is True

    manager.authFailures.record("srv")
    assert manager.authFailures.is_cached("srv") is True
    manager.authFailures.clear("srv")
    assert manager.authFailures.is_cached("srv") is False

    decision = manager.approvals.decide("srv", "allow", "test")
    assert decision["decision"] == "allow"
    assert manager.approvals.is_trusted("srv") is True

    tokens = TokenEncryptionService("secret")
    encrypted = tokens.encrypt("plain-token")
    assert encrypted != "plain-token"
    assert tokens.decrypt(encrypted) == "plain-token"


def test_mcp_stdio_transport_connects_lists_tools_and_calls_remote_server() -> None:
    manager = McpClientManager()
    server_script = BACKEND_DIR / "tests" / "fixtures" / "mcp_stdio_echo_server.py"
    manager.add_server(
        {
            "name": "stdio_echo",
            "type": "stdio",
            "command": sys.executable,
            "args": [str(server_script)],
            "status": "pending",
        }
    )

    connection = manager.connect_server("stdio_echo")
    assert connection.status == McpConnectionStatus.CONNECTED
    assert [tool["name"] for tool in connection.tools] == ["remote_echo"]
    assert manager.discover_wrapped_tools()[0]["name"] == "mcp__stdio_echo__remote_echo"

    invoked = manager.call_tool("stdio_echo", "remote_echo", {"message": "hello"})
    assert invoked["status"] == "success"
    assert invoked["connectionType"] == "stdio"
    assert invoked["content"] == "stdio:hello"
    assert invoked["result"]["content"][0]["text"] == "stdio:hello"
    manager.close_server("stdio_echo")


def test_mcp_streamable_http_transport_connects_lists_tools_and_calls_remote_server() -> None:
    server, url = run_mcp_http_server()
    manager = McpClientManager()
    manager.add_server({"name": "http_echo", "type": "streamable_http", "url": url, "status": "pending"})
    try:
        connection = manager.connect_server("http_echo")
        assert connection.status == McpConnectionStatus.CONNECTED
        assert [tool["name"] for tool in connection.tools] == ["remote_echo"]
        assert "initialize" in McpHttpEchoHandler.requests
        assert "tools/list" in McpHttpEchoHandler.requests

        invoked = manager.call_tool("http_echo", "remote_echo", {"message": "hello"})
        assert invoked["status"] == "success"
        assert invoked["connectionType"] == "streamable_http"
        assert invoked["content"] == "http:hello"
        assert invoked["result"]["content"][0]["text"] == "http:hello"
        assert McpHttpEchoHandler.requests[-1] == "tools/call"
    finally:
        manager.close_server("http_echo")
        server.shutdown()


def test_mcp_manager_validates_tools_and_tracks_reconnect_backoff() -> None:
    manager = McpClientManager()
    manager.add_server(
        {
            "name": "deep",
            "type": "streamable-http",
            "url": "http://localhost:9/mcp",
            "status": "connected",
            "tools": [
                {"name": "valid_tool", "description": "ok", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}},
                {"name": "", "description": "missing name", "inputSchema": {"type": "object"}},
                {"name": "bad_schema", "description": "bad", "inputSchema": {"type": "array"}},
            ],
        }
    )

    validation = manager.validate_tool_schemas("deep")
    assert validation["valid"] is False
    assert validation["validCount"] == 1
    assert validation["invalidCount"] == 2
    assert {item["toolName"] for item in validation["invalidTools"]} == {"", "bad_schema"}
    assert [tool["originalName"] for tool in manager.discover_wrapped_tools()] == ["valid_tool"]

    first_failure = manager.mark_connection_failure("deep", "401 unauthorized", auth_failed=True)
    assert first_failure["status"] == "failed"
    assert first_failure["authCached"] is True
    assert first_failure["nextRetryAt"] > first_failure["updatedAt"]

    blocked = manager.plan_reconnect("deep")
    assert blocked["allowed"] is False
    assert blocked["reason"] == "auth_failure_cached"
    assert blocked["backoffMs"] > 0

    manager.authFailures.clear("deep")
    allowed = manager.plan_reconnect("deep")
    assert allowed["allowed"] is True
    assert allowed["attempt"] == 2


def test_mcp_manager_invokes_tools_with_json_rpc_semantics_and_cache_fallback() -> None:
    manager = McpClientManager()
    connection = manager.add_server(
        {
            "name": "deep",
            "type": "streamable-http",
            "url": "http://localhost:9/mcp",
            "status": "connected",
            "tools": [
                {
                    "name": "remote_echo",
                    "description": "Echo remotely",
                    "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}},
                    "result": {
                        "content": [
                            {"type": "text", "text": "remote "},
                            {"type": "text", "text": "reply"},
                        ],
                        "isError": False,
                    },
                }
            ],
        }
    )

    invoked = manager.call_tool("deep", "remote_echo", {"message": "hello"})
    assert invoked["status"] == "success"
    assert invoked["content"] == "remote reply"
    assert invoked["request"]["method"] == "tools/call"
    assert invoked["request"]["params"] == {"name": "remote_echo", "arguments": {"message": "hello"}}
    assert invoked["metadata"]["mcpServer"] == "deep"
    assert invoked["metadata"]["mcpTool"] == "remote_echo"
    assert invoked["cached"] is False

    connection.status = McpConnectionStatus.DISCONNECTED
    cached = manager.call_tool("deep", "remote_echo", {"message": "hello"})
    assert cached["status"] == "success"
    assert cached["cached"] is True
    assert cached["content"] == "[cached] remote reply"


def test_mcp_manager_truncates_oversized_tool_results() -> None:
    manager = McpClientManager()
    manager.add_server(
        {
            "name": "deep",
            "type": "sse",
            "status": "connected",
            "tools": [
                {
                    "name": "large_report",
                    "inputSchema": {"type": "object"},
                    "result": {"content": [{"type": "text", "text": "x" * (MAX_MCP_RESULT_SIZE + 32)}]},
                }
            ],
        }
    )

    invoked = manager.call_tool("deep", "large_report", {})
    assert invoked["status"] == "success"
    assert len(invoked["content"]) > MAX_MCP_RESULT_SIZE
    assert invoked["content"].endswith(f"[Truncated: exceeded {MAX_MCP_RESULT_SIZE} chars]")
