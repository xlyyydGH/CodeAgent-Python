import json
import sys
from pathlib import Path
from uuid import uuid4

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.llm_tools import execute_tool_calls, parse_tool_arguments  # noqa: E402
from zhikun_py.mcp_runtime import McpClientManager  # noqa: E402
from zhikun_py.permissions import PermissionDecision, PermissionPolicy, PermissionRule  # noqa: E402
from zhikun_py.tools import ToolRegistry  # noqa: E402


def workspace() -> Path:
    root = BACKEND_DIR / ".test-workspace" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_parse_tool_arguments_accepts_json_objects() -> None:
    assert parse_tool_arguments('{"path":"README.md"}') == {"path": "README.md"}
    assert parse_tool_arguments("") == {}
    assert parse_tool_arguments("[1,2]") == {"value": [1, 2]}


def test_execute_tool_calls_returns_openai_tool_messages() -> None:
    root = workspace()
    (root / "README.md").write_text("hello tool loop", encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("edit_file", PermissionDecision.ALLOW)]))
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": json.dumps({"path": "README.md"})},
        }
    ]

    messages, executed = execute_tool_calls(registry, tool_calls)

    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call_1"
    assert messages[0]["name"] == "read_file"
    assert "hello tool loop" in messages[0]["content"]
    assert executed[0].name == "read_file"
    assert not executed[0].is_error


def test_tool_registry_registers_mcp_tools_for_llm_execution() -> None:
    root = workspace()
    manager = McpClientManager()
    manager.add_server(
        {
            "name": "deep",
            "type": "sse",
            "status": "connected",
            "tools": [
                {
                    "name": "remote_echo",
                    "description": "Remote echo",
                    "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}},
                    "result": {"content": [{"type": "text", "text": "remote reply"}], "isError": False},
                }
            ],
        }
    )
    registry = ToolRegistry(root, PermissionPolicy())

    registry.register_mcp_tools(manager)
    result = registry.call("mcp__deep__remote_echo", {"message": "hi"})

    assert result.isError is False
    assert result.content == "remote reply"
    assert result.metadata["mcpServer"] == "deep"
    assert result.metadata["mcpTool"] == "remote_echo"
    assert registry.get("mcp__deep__remote_echo") is not None
    assert any(item["function"]["name"] == "mcp__deep__remote_echo" for item in registry.llm_definitions())


def test_file_tools_track_hashes_and_report_conflicts() -> None:
    root = workspace()
    target = root / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("edit_file", PermissionDecision.ALLOW)]))

    read = registry.call("read_file", {"path": "module.py"})
    content_hash = read.metadata["contentHash"]
    assert content_hash

    target.write_text("value = 2\n", encoding="utf-8")
    conflict = registry.call("edit_file", {"path": "module.py", "old": "value = 1", "new": "value = 3", "expectedHash": content_hash})
    assert conflict.isError is True
    assert conflict.metadata["conflict"]["hasConflict"] is True
    assert conflict.metadata["recovery"]["action"] == "report_to_llm"

    fresh = registry.call("read_file", {"path": "module.py"}).metadata["contentHash"]
    edited = registry.call("edit_file", {"path": "module.py", "old": "value = 2", "new": "value = 3", "expectedHash": fresh})
    assert edited.isError is False
    assert edited.metadata["contentHash"] != fresh


def test_file_edit_returns_snapshot_before_write_and_unified_diff() -> None:
    root = workspace()
    target = root / "module.py"
    target.write_text("value = 1\n", encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("edit_file", PermissionDecision.ALLOW)]))

    edited = registry.call("edit_file", {"path": "module.py", "old": "value = 1", "new": "value = 2", "agentId": "agent-1"})

    assert edited.isError is False
    snapshot = edited.metadata["snapshotBeforeWrite"]
    assert snapshot["path"] == "module.py"
    assert snapshot["operation"] == "edit_file"
    assert snapshot["agentId"] == "agent-1"
    assert snapshot["content"] == "value = 1\n"
    assert snapshot["contentHash"]
    assert edited.metadata["diff"].startswith("--- module.py")
    assert "-value = 1" in edited.metadata["diff"]
    assert "+value = 2" in edited.metadata["diff"]
    assert edited.metadata["beforeHash"] == snapshot["contentHash"]
    assert edited.metadata["afterHash"] == edited.metadata["contentHash"]


def test_write_file_returns_snapshot_and_diff_when_overwriting_existing_file() -> None:
    root = workspace()
    target = root / "module.py"
    target.write_text("old = True\n", encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("write_file", PermissionDecision.ALLOW)]))

    written = registry.call("write_file", {"path": "module.py", "content": "old = False\n", "agentId": "writer"})

    assert written.isError is False
    assert written.metadata["snapshotBeforeWrite"]["content"] == "old = True\n"
    assert written.metadata["snapshotBeforeWrite"]["operation"] == "write_file"
    assert written.metadata["snapshotBeforeWrite"]["agentId"] == "writer"
    assert "-old = True" in written.metadata["diff"]
    assert "+old = False" in written.metadata["diff"]
    assert written.metadata["beforeHash"] == written.metadata["snapshotBeforeWrite"]["contentHash"]
    assert written.metadata["afterHash"] == written.metadata["contentHash"]


def test_file_edit_uses_whitespace_fuzzy_match() -> None:
    root = workspace()
    target = root / "module.py"
    target.write_text("def answer():\n    value = 1\n    return value\n", encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("edit_file", PermissionDecision.ALLOW)]))

    edited = registry.call(
        "edit_file",
        {
            "path": "module.py",
            "old": "def answer():\n value = 1\n return value",
            "new": "def answer():\n    value = 2\n    return value",
        },
    )

    assert edited.isError is False
    assert edited.metadata["matchStrategy"] == "fuzzy_whitespace"
    assert edited.metadata["atomic"] is True
    assert target.read_text(encoding="utf-8") == "def answer():\n    value = 2\n    return value\n"


def test_file_edit_uses_smart_quote_fuzzy_match() -> None:
    root = workspace()
    target = root / "module.py"
    target.write_text('msg = "hello"\n', encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("edit_file", PermissionDecision.ALLOW)]))

    edited = registry.call("edit_file", {"path": "module.py", "old": "msg = \u201Chello\u201D", "new": 'msg = "world"'})

    assert edited.isError is False
    assert edited.metadata["matchStrategy"] == "fuzzy_quotes"
    assert target.read_text(encoding="utf-8") == 'msg = "world"\n'


def test_file_edit_reports_trailing_whitespace_fuzzy_match() -> None:
    root = workspace()
    target = root / "module.py"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("edit_file", PermissionDecision.ALLOW)]))

    edited = registry.call("edit_file", {"path": "module.py", "old": "line2   ", "new": "LINE_TWO"})

    assert edited.isError is False
    assert edited.metadata["matchStrategy"] == "fuzzy_trailing_whitespace"
    assert target.read_text(encoding="utf-8") == "line1\nLINE_TWO\nline3\n"


def test_file_edit_reports_newline_fuzzy_match() -> None:
    root = workspace()
    target = root / "module.py"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("edit_file", PermissionDecision.ALLOW)]))

    edited = registry.call("edit_file", {"path": "module.py", "old": "alpha\r\nbeta", "new": "ALPHA\nBETA"})

    assert edited.isError is False
    assert edited.metadata["matchStrategy"] == "fuzzy_newline"
    assert target.read_text(encoding="utf-8") == "ALPHA\nBETA\n"


def test_file_edit_reports_tab_space_fuzzy_match() -> None:
    root = workspace()
    target = root / "module.py"
    target.write_text("if True:\n\treturn 1\n", encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("edit_file", PermissionDecision.ALLOW)]))

    edited = registry.call("edit_file", {"path": "module.py", "old": "if True:\n    return 1", "new": "if True:\n    return 2"})

    assert edited.isError is False
    assert edited.metadata["matchStrategy"] == "fuzzy_tab_space"
    assert target.read_text(encoding="utf-8") == "if True:\n    return 2\n"


def test_file_edit_keeps_original_when_atomic_replace_fails(monkeypatch) -> None:
    root = workspace()
    target = root / "module.py"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("edit_file", PermissionDecision.ALLOW)]))

    def fail_replace(self: Path, target_path: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    failed = registry.call("edit_file", {"path": "module.py", "old": "alpha", "new": "ALPHA"})

    assert failed.isError is True
    assert failed.metadata["atomic"] is True
    assert failed.metadata["recovery"]["action"] == "report_to_llm"
    assert "replace failed" in failed.content
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert list(root.glob(".module.py.*.tmp")) == []


def test_multi_edit_is_all_or_nothing_when_one_edit_misses() -> None:
    root = workspace()
    target = root / "module.py"
    original = "alpha = 1\nbeta = 2\ngamma = 3\n"
    target.write_text(original, encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("MultiEdit", PermissionDecision.ALLOW)]))

    result = registry.call(
        "MultiEdit",
        {
            "path": "module.py",
            "edits": [
                {"old": "alpha = 1", "new": "alpha = 10"},
                {"old": "missing = 9", "new": "missing = 10"},
            ],
        },
    )

    assert result.isError is True
    assert result.metadata["failedEditIndex"] == 1
    assert result.metadata["applied"] == 0
    assert result.metadata["atomic"] is True
    assert target.read_text(encoding="utf-8") == original


def test_multi_edit_applies_exact_and_fuzzy_matches_atomically() -> None:
    root = workspace()
    target = root / "module.py"
    target.write_text("def answer():\n    value = 1\n    return value\nstatus = 'old'\n", encoding="utf-8")
    registry = ToolRegistry(root, PermissionPolicy([PermissionRule("MultiEdit", PermissionDecision.ALLOW)]))

    result = registry.call(
        "MultiEdit",
        {
            "path": "module.py",
            "edits": [
                {"old": "def answer():\n value = 1\n return value", "new": "def answer():\n    value = 2\n    return value"},
                {"old": "status = 'old'", "new": "status = 'new'"},
            ],
        },
    )

    assert result.isError is False
    assert result.metadata["applied"] == 2
    assert result.metadata["atomic"] is True
    assert result.metadata["matchStrategies"] == ["fuzzy_whitespace", "exact"]
    assert result.metadata["snapshotBeforeWrite"]["content"].startswith("def answer():")
    assert "-status = 'old'" in result.metadata["diff"]
    assert "+status = 'new'" in result.metadata["diff"]
    assert target.read_text(encoding="utf-8") == "def answer():\n    value = 2\n    return value\nstatus = 'new'\n"
