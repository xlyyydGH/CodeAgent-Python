from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FileVersion:
    contentHash: str
    lastModified: float
    lastEditor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConflictCheckResult:
    hasConflict: bool
    currentHash: str | None = None
    expectedHash: str | None = None
    lastEditor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RecoveryDecision:
    action: str
    message: str
    escalateToUser: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FileVersionTracker:
    def __init__(self, max_entries: int = 10_000, eviction_ratio: float = 0.2) -> None:
        self.max_entries = max_entries
        self.eviction_ratio = eviction_ratio
        self.versions: dict[str, FileVersion] = {}

    def compute_hash(self, content_or_path: str | Path) -> str:
        if isinstance(content_or_path, Path):
            content = content_or_path.read_text(encoding="utf-8", errors="ignore")
        else:
            content = content_or_path
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def check_before_write(self, file_path: str | Path, expected_hash: str | None = None) -> ConflictCheckResult:
        path = Path(file_path)
        normalized = str(path.resolve() if path.exists() else path)
        stored = self.versions.get(normalized)
        if stored is None and expected_hash is None:
            return ConflictCheckResult(False)
        if not path.exists():
            return ConflictCheckResult(False, expectedHash=expected_hash)
        current_hash = self.compute_hash(path)
        effective_expected = expected_hash or (stored.contentHash if stored else None)
        if effective_expected and current_hash != effective_expected:
            return ConflictCheckResult(True, current_hash, effective_expected, stored.lastEditor if stored else None)
        return ConflictCheckResult(False, current_hash, effective_expected)

    def record_read(self, file_path: str | Path) -> str | None:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return None
        content_hash = self.compute_hash(path)
        self.versions[str(path.resolve())] = FileVersion(content_hash, time.time())
        self._evict_if_needed()
        return content_hash

    def record_write(self, file_path: str | Path, content_hash: str | None = None, agent_id: str | None = None) -> str:
        path = Path(file_path)
        new_hash = content_hash or self.compute_hash(path)
        self.versions[str(path.resolve())] = FileVersion(new_hash, time.time(), agent_id)
        self._evict_if_needed()
        return new_hash

    def version_for(self, file_path: str | Path) -> dict[str, Any] | None:
        path = Path(file_path)
        version = self.versions.get(str(path.resolve() if path.exists() else path))
        return version.to_dict() if version else None

    def _evict_if_needed(self) -> None:
        if len(self.versions) <= self.max_entries:
            return
        remove_count = max(1, int(self.max_entries * self.eviction_ratio))
        oldest = sorted(self.versions.items(), key=lambda item: item[1].lastModified)[:remove_count]
        for key, _ in oldest:
            self.versions.pop(key, None)


class FileEditRecoveryPolicy:
    def can_handle(self, tool_name: str) -> bool:
        return tool_name in {"FileEdit", "FileWrite", "edit_file", "write_file"}

    def recover(self, tool_name: str, error_message: str | None) -> RecoveryDecision:
        error = (error_message or "").lower()
        if any(token in error for token in ("conflict", "modified externally", "stale", "changed since")):
            return RecoveryDecision(
                "report_to_llm",
                "File conflict detected - re-read the file to get the current content before editing.",
            )
        if any(token in error for token in ("not found in file", "old_string", "content mismatch", "no match", "does not match", "string not found")):
            return RecoveryDecision(
                "report_to_llm",
                "Content mismatch - re-read the file and retry with exact text that exists in the file.",
            )
        if any(token in error for token in ("no such file", "file not found", "path does not exist", "does not exist", "not a file")):
            return RecoveryDecision(
                "report_to_llm",
                "File path does not exist. Verify the path with a file listing tool.",
            )
        if any(token in error for token in ("permission denied", "access denied", "operation not permitted", "read-only file system")):
            return RecoveryDecision(
                "escalate_to_user",
                "Permission denied when writing to file. Check file permissions and ownership.",
                escalateToUser=True,
            )
        return RecoveryDecision("report_to_llm", f"File operation failed: {error_message}. Verify file state and try again.")
