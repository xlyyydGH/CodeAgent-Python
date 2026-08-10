from __future__ import annotations

import json
import sys


TOOLS = [
    {
        "name": "remote_echo",
        "description": "Echo a message through stdio MCP.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
        },
    }
]


def write_response(request_id, result=None, error=None) -> None:
    payload = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        continue
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        write_response(
            request_id,
            {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "stdio-echo", "version": "0.1"},
                "capabilities": {"tools": {}},
            },
        )
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        write_response(request_id, {"tools": TOOLS})
    elif method == "tools/call":
        params = message.get("params") or {}
        args = params.get("arguments") or {}
        write_response(
            request_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": f"stdio:{args.get('message', '')}",
                    }
                ],
                "isError": False,
            },
        )
    else:
        write_response(request_id, error={"code": -32601, "message": f"unknown method: {method}"})
