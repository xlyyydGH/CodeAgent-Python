from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LSPServerConfig:
    name: str
    command: str
    fileExtensions: list[str]
    startupTimeoutMs: int = 30_000

    @classmethod
    def typescript(cls) -> "LSPServerConfig":
        return cls("tsserver", "npx", [".ts", ".tsx", ".js", ".jsx"])

    @classmethod
    def python(cls) -> "LSPServerConfig":
        return cls("pyright", "pyright", [".py"])

    @classmethod
    def go(cls) -> "LSPServerConfig":
        return cls("gopls", "gopls", [".go"])

    @classmethod
    def rust(cls) -> "LSPServerConfig":
        return cls("rust-analyzer", "rust-analyzer", [".rs"])

    @classmethod
    def java(cls) -> "LSPServerConfig":
        return cls("jdtls", "jdtls", [".java"], startupTimeoutMs=60_000)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LSPServerInstance:
    def __init__(self, config: LSPServerConfig) -> None:
        self.config = config
        self.running = False
        self.requests: list[dict[str, Any]] = []

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def send_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.running:
            raise RuntimeError("LSP server is not running")
        record = {"server": self.config.name, "method": method, "params": params or {}}
        self.requests.append(record)
        return record

    def to_dict(self) -> dict[str, Any]:
        return {**self.config.to_dict(), "running": self.running, "requestCount": len(self.requests)}


class LSPServerManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.servers: dict[str, LSPServerInstance] = {}
        self.open_files: set[str] = set()
        for config in [LSPServerConfig.typescript(), LSPServerConfig.python(), LSPServerConfig.go(), LSPServerConfig.rust(), LSPServerConfig.java()]:
            self.register_and_start(config)

    @staticmethod
    def get_file_extension(path: str | None) -> str:
        if not path:
            return ""
        suffix = Path(path).suffix
        return suffix if suffix else ""

    def register_and_start(self, config: LSPServerConfig) -> LSPServerInstance:
        instance = LSPServerInstance(config)
        instance.start()
        self.servers[config.name] = instance
        return instance

    def get_server_for_file(self, file_path: str | Path) -> LSPServerInstance | None:
        extension = self.get_file_extension(str(file_path))
        for instance in self.servers.values():
            if extension in instance.config.fileExtensions:
                return instance
        return None

    def open_file(self, file_path: str) -> None:
        self.open_files.add(str(file_path))

    def close_file(self, file_path: str) -> None:
        self.open_files.discard(str(file_path))

    def is_file_open(self, file_path: str) -> bool:
        return str(file_path) in self.open_files

    def send_request(self, file_path: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        server = self.get_server_for_file(file_path)
        if not server:
            return None
        return server.send_request(method, params)

    def status(self) -> dict[str, Any]:
        return {
            "serverCount": len(self.servers),
            "supportedExtensions": sorted({ext for server in self.servers.values() for ext in server.config.fileExtensions}),
            "openFiles": sorted(self.open_files),
            "servers": [server.to_dict() for server in self.servers.values()],
        }


def safe_file(root: Path, path: str | Path) -> Path:
    candidate = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError("Path escapes workspace")
    return candidate


def document_symbols(root: Path, path: str | Path) -> list[dict[str, Any]]:
    file_path = safe_file(root, path)
    if not file_path.exists() or not file_path.is_file():
        return []
    if file_path.suffix == ".py":
        tree = ast.parse(file_path.read_text(encoding="utf-8", errors="ignore"))
        symbols = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append({"name": node.name, "kind": "class" if isinstance(node, ast.ClassDef) else "function", "line": node.lineno, "filePath": file_path.relative_to(root).as_posix()})
        return sorted(symbols, key=lambda item: item["line"])
    pattern = re.compile(r"^\s*(?:export\s+)?(?:class|function|interface|type|const|let|var)\s+([A-Za-z_$][\w$]*)", re.MULTILINE)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return [
        {"name": match.group(1), "kind": "symbol", "line": text[: match.start()].count("\n") + 1, "filePath": file_path.relative_to(root).as_posix()}
        for match in pattern.finditer(text)
    ]


def workspace_symbols(root: Path, query: str, limit: int = 100) -> list[dict[str, Any]]:
    if not query:
        return []
    matches: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if len(matches) >= limit:
            break
        if not path.is_file() or path.suffix.lower() not in {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs"}:
            continue
        if any(part in {".git", "node_modules", "__pycache__", ".pytest_cache", "venv", ".venv"} for part in path.relative_to(root).parts):
            continue
        for symbol in document_symbols(root, path):
            if query.lower() in symbol["name"].lower():
                matches.append(symbol)
                break
    return matches


def references(root: Path, symbol: str, path: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    search_root = safe_file(root, path).parent if path else root
    for file_path in search_root.rglob("*"):
        if len(refs) >= limit:
            break
        if not file_path.is_file() or file_path.suffix.lower() not in {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs"}:
            continue
        if any(part in {".git", "node_modules", "__pycache__", ".pytest_cache", "venv", ".venv"} for part in file_path.relative_to(root).parts):
            continue
        for line_no, line in enumerate(file_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            if re.search(rf"\b{re.escape(symbol)}\b", line):
                refs.append({"filePath": file_path.relative_to(root).as_posix(), "line": line_no, "preview": line.strip()})
                break
    return refs


def hover(root: Path, path: str, line: int, character: int) -> dict[str, Any]:
    file_path = safe_file(root, path)
    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines() if file_path.exists() else []
    text = lines[line - 1] if 0 < line <= len(lines) else ""
    word_match = re.search(r"[A-Za-z_][\w_]*", text[max(0, character - 30) : character + 60])
    word = word_match.group(0) if word_match else ""
    return {"contents": word or text.strip(), "line": line, "character": character, "filePath": str(path)}


def go_to_definition(root: Path, path: str, line: int, character: int) -> dict[str, Any]:
    h = hover(root, path, line, character)
    word = h.get("contents") or ""
    candidates = workspace_symbols(root, str(word), limit=20)
    return {"query": word, "definitions": candidates}


def call_hierarchy(root: Path, symbol: str, path: str | None = None) -> dict[str, Any]:
    refs = references(root, symbol, path, limit=50)
    incoming = [{"from": ref["filePath"], "line": ref["line"], "preview": ref["preview"]} for ref in refs]
    outgoing: list[dict[str, Any]] = []
    if path:
        file_path = safe_file(root, path)
        text = file_path.read_text(encoding="utf-8", errors="ignore") if file_path.exists() else ""
        for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", text):
            name = match.group(1)
            if name != symbol and name not in {"if", "for", "while", "return"}:
                outgoing.append({"to": name, "filePath": file_path.relative_to(root).as_posix()})
                if len(outgoing) >= 50:
                    break
    return {"symbol": symbol, "incomingCalls": incoming, "outgoingCalls": outgoing}
