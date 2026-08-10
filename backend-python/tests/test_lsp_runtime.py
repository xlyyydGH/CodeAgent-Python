import sys
from pathlib import Path
from uuid import uuid4

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.lsp_runtime import (  # noqa: E402
    LSPServerConfig,
    LSPServerManager,
    call_hierarchy,
    document_symbols,
    references,
    workspace_symbols,
)


def workspace() -> Path:
    root = BACKEND_DIR / ".test-workspace" / uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_lsp_server_config_manager_and_file_sync() -> None:
    assert LSPServerConfig.typescript().name == "tsserver"
    assert ".tsx" in LSPServerConfig.typescript().fileExtensions
    assert LSPServerConfig.python().name == "pyright"
    assert LSPServerConfig.go().command == "gopls"
    assert LSPServerConfig.java().startupTimeoutMs == 60_000

    manager = LSPServerManager(ROOT)
    assert manager.get_file_extension("src/app.ts") == ".ts"
    assert manager.get_server_for_file("src/main.py").config.name == "pyright"
    assert manager.get_server_for_file("src/main.rs").config.name == "rust-analyzer"
    assert manager.get_server_for_file("Makefile") is None
    manager.open_file("src/main.py")
    assert manager.is_file_open("src/main.py")
    manager.close_file("src/main.py")
    assert not manager.is_file_open("src/main.py")
    response = manager.send_request("src/main.py", "textDocument/definition", {"line": 1})
    assert response["method"] == "textDocument/definition"


def test_lsp_symbols_references_and_call_hierarchy() -> None:
    root = workspace()
    source = root / "service.py"
    source.write_text(
        "class Demo:\n    pass\n\ndef helper():\n    return Demo()\n\ndef target():\n    return helper()\n",
        encoding="utf-8",
    )
    consumer = root / "api.py"
    consumer.write_text("from service import target\n\ndef route():\n    return target()\n", encoding="utf-8")

    symbols = document_symbols(root, source)
    assert [item["name"] for item in symbols] == ["Demo", "helper", "target"]
    assert workspace_symbols(root, "target")[0]["filePath"] == "service.py"
    refs = references(root, "target")
    assert any(ref["filePath"] == "api.py" for ref in refs)
    hierarchy = call_hierarchy(root, "target", "api.py")
    assert hierarchy["symbol"] == "target"
    assert hierarchy["incomingCalls"]
