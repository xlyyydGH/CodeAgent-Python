from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class BlockLevel(StrEnum):
    ALLOWED = "allowed"
    AUDIT_LOG = "audit_log"
    HIGH_RISK_ASK = "high_risk_ask"
    ABSOLUTE_DENY = "absolute_deny"


@dataclass(slots=True)
class CommandCheck:
    level: BlockLevel
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.level == BlockLevel.ABSOLUTE_DENY

    @property
    def needs_confirmation(self) -> bool:
        return self.level == BlockLevel.HIGH_RISK_ASK


@dataclass(slots=True)
class CommandClassification:
    isReadOnly: bool
    isSearch: bool = False
    isRead: bool = False
    isList: bool = False
    category: str = "unknown"


READONLY_COMMANDS = {
    "basename",
    "cat",
    "cal",
    "column",
    "cut",
    "df",
    "diff",
    "dirname",
    "du",
    "echo",
    "find",
    "free",
    "grep",
    "head",
    "id",
    "ls",
    "md5sum",
    "nl",
    "nproc",
    "pwd",
    "readlink",
    "realpath",
    "rg",
    "sha1sum",
    "sort",
    "stat",
    "strings",
    "tail",
    "tac",
    "tree",
    "uname",
    "uniq",
    "wc",
    "whoami",
}
READ_COMMANDS = {"cat", "head", "tail", "nl", "sed", "less", "more"}
SEARCH_COMMANDS = {"grep", "rg", "find"}
LIST_COMMANDS = {"ls", "tree", "git", "docker", "kubectl", "npm", "pip", "yarn"}
WRITE_COMMANDS = {"rm", "mv", "cp", "chmod", "chown", "mkdir", "touch", "sed", "tee", "git", "npm", "pip", "docker", "kubectl"}


SECRET_PATTERNS = [
    re.compile(r"sk-proj-[A-Za-z0-9_-]{16,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_-]{16,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk_live_[A-Za-z0-9]{20,}"),
    re.compile(r"vercel_[A-Za-z0-9]{20,}"),
    re.compile(r"sbp_[A-Za-z0-9]{20,}"),
    re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*=\s*['\"]?[^'\"\s]{8,}"),
    re.compile(r"(?i)(postgres|mysql|mongodb)://[^:\s]+:[^@\s]+@"),
]


def command_parts(command: Any) -> list[str]:
    if isinstance(command, list):
        return [str(part) for part in command if str(part)]
    try:
        return shlex.split(str(command or ""), posix=False)
    except ValueError:
        return str(command or "").split()


def redact_token(token: str) -> str:
    prefix = token[:4]
    if "_" in token and token.index("_") < 8:
        prefix = token[: token.index("_") + 1]
    return f"{prefix}***REDACTED-{len(token)}***"


def filter_sensitive_data(text: str | None) -> str | None:
    if text is None:
        return None
    output = text
    for pattern in SECRET_PATTERNS:
        output = pattern.sub(lambda match: redact_token(match.group(0)), output)
    return output


def command_risk(command: Any) -> CommandCheck:
    text = " ".join(command) if isinstance(command, list) else str(command or "")
    lowered = text.lower()
    parts = command_parts(command)
    executable = Path(parts[0]).name.lower() if parts else ""
    if not text.strip():
        return CommandCheck(BlockLevel.ALLOWED)
    downloaded_script_execution_patterns = [
        r"\b(iex|invoke-expression)\b.*\b(downloadstring|downloadfile|new-object\s+net\.webclient|invoke-webrequest|iwr|invoke-restmethod|irm)\b",
        r"\b(downloadstring|downloadfile|new-object\s+net\.webclient|invoke-webrequest|iwr|invoke-restmethod|irm)\b.*\b(iex|invoke-expression)\b",
    ]
    if any(re.search(pattern, lowered) for pattern in downloaded_script_execution_patterns):
        return CommandCheck(BlockLevel.ABSOLUTE_DENY, "absolute deny command: downloaded script execution")
    encoded_shell_patterns = [
        r"\b(powershell(?:\.exe)?|pwsh)\b.*\s-(?:e|ec|encodedcommand)\b",
        r"\b(frombase64string)\b.*\b(iex|invoke-expression|powershell|pwsh)\b",
    ]
    if any(re.search(pattern, lowered) for pattern in encoded_shell_patterns):
        return CommandCheck(BlockLevel.ABSOLUTE_DENY, "absolute deny command: encoded shell execution")
    secret_env_patterns = [
        r"\benv:[a-z0-9_]*(api[_-]?key|token|secret|password)",
        r"\b(printenv|set)\b.*\b[a-z0-9_]*(api[_-]?key|token|secret|password)\b",
        r"\b(get-childitem|dir|ls)\s+env:\s*$",
        r"os\.environ(?:\.get)?\([^)]*(api[_-]?key|token|secret|password)",
        r"os\.environ\[[^\]]*(api[_-]?key|token|secret|password)",
        r"process\.env\.[a-z0-9_]*(api[_-]?key|token|secret|password)",
    ]
    sensitive_source_patterns = [
        r"(^|[/\\])\.ssh[/\\]id_",
        r"(^|[/\\])\.aws[/\\]credentials\b",
        r"(^|[/\\])\.kube[/\\]config\b",
        r"(^|[/\\])\.docker[/\\]config\.json\b",
        r"(^|[/\\])\.git-credentials\b",
        r"(^|[/\\])\.npmrc\b",
        r"(^|[/\\])\.env(\.|[\s'\"<>|&;]|$)",
        r"\b/etc/shadow\b",
    ]
    upload_command_patterns = [
        r"\b(curl|wget)\b.*(?:--data(?:-binary|-raw|-urlencode)?|-d\b|--form\b|-f\b|--upload-file|--post-data|--method\s+post|-x\s+post)",
        r"\b(invoke-webrequest|iwr|invoke-restmethod|irm)\b.*(?:-method\s+post|-infile\b|-body\b|-form\b)",
    ]
    has_secret_source = any(re.search(pattern, lowered) for pattern in secret_env_patterns)
    has_sensitive_source = has_secret_source or any(re.search(pattern, lowered) for pattern in sensitive_source_patterns)
    has_upload = any(re.search(pattern, lowered) for pattern in upload_command_patterns)
    if has_sensitive_source and has_upload:
        return CommandCheck(BlockLevel.ABSOLUTE_DENY, "absolute deny command: secret exfiltration")
    absolute_patterns = [
        r"(^|\s)(/usr/bin/|/bin/)?rm\s+-[a-z]*f[a-z]*\s+(/|~|\$home)(\s|$)",
        r"(^|\s)mkfs(\.\w+)?\s+",
        r"(^|\s)dd\s+.*of=/dev/",
        r">\s*/dev/[a-z]+",
        r":\(\)\s*\{\s*:\|:",
        r"(curl|wget)\s+[^|;&]+[|]\s*(bash|sh)",
        r"bash\s+-c\s+.*curl",
        r"chmod\s+(-r\s+)?777\s+/",
        r"(^|\s)(shred|wipefs)\s+/dev/",
        r"(^|\s)(reboot|shutdown|halt|poweroff|diskpart|format)(\s|$)",
        r"(^|\s)init\s+0(\s|$)",
    ]
    if any(re.search(pattern, lowered) for pattern in absolute_patterns):
        return CommandCheck(BlockLevel.ABSOLUTE_DENY, "absolute deny command")
    reverse_shell_patterns = [
        r"/dev/tcp/[^/\s]+/\d+",
        r"\bbash\s+-i\b.*>&",
        r"\bnc\s+.*\s-e\s+",
        r"\bncat\s+.*\s--exec\s+",
        r"\bpython\b.*socket.*dup2",
    ]
    if any(re.search(pattern, lowered) for pattern in reverse_shell_patterns):
        return CommandCheck(BlockLevel.ABSOLUTE_DENY, "absolute deny command: reverse shell pattern")
    windows_root_delete_patterns = [
        r"\b(remove-item|rd|rmdir)\b.*(?:-recurse|/s)\b.*(?:[a-z]:\\|[a-z]:/)\s*$",
        r"\b(del|erase)\b.*/s\b.*(?:[a-z]:\\|[a-z]:/)\s*$",
    ]
    if any(re.search(pattern, lowered) for pattern in windows_root_delete_patterns):
        return CommandCheck(BlockLevel.ABSOLUTE_DENY, "absolute deny command: windows root recursive delete")
    if any(re.search(pattern, lowered) for pattern in secret_env_patterns):
        return CommandCheck(BlockLevel.HIGH_RISK_ASK, "secret environment access requires confirmation")
    read_executables = READ_COMMANDS | {"type", "get-content", "gc"}
    if executable in read_executables:
        for part in parts[1:]:
            if sensitive_path_level(part) in {"forbidden", "protected"}:
                return CommandCheck(BlockLevel.HIGH_RISK_ASK, "sensitive file read requires confirmation")
    high_risk_patterns = [
        r"(^|\s)rm\s+-[a-z]*[rf][a-z]*\s+",
        r"git\s+push\b.*--force",
        r"git\s+reset\s+--hard",
        r"\b(drop|truncate)\s+table\b",
        r"(^|\s)kill\s+-9\b",
        r"(^|\s)killall\b",
        r"(^|\s)nc\s+.*\s-l",
        r"docker\s+system\s+prune",
        r"npm\s+publish",
        r"chmod\s+777\b",
        r"\b(remove-item|rd|rmdir)\b.*(^|\s)(-recurse|/s)(\s|$)",
    ]
    if any(re.search(pattern, lowered) for pattern in high_risk_patterns):
        return CommandCheck(BlockLevel.HIGH_RISK_ASK, "high risk command")
    audit_patterns = [r"^env$", r"^printenv$", r"git\s+push\b", r"npm\s+install\b", r"^curl\s+", r"^ssh\s+"]
    if any(re.search(pattern, lowered.strip()) for pattern in audit_patterns):
        return CommandCheck(BlockLevel.AUDIT_LOG, "audit command")
    if executable in {"reboot", "shutdown", "mkfs", "diskpart", "format"}:
        return CommandCheck(BlockLevel.ABSOLUTE_DENY, f"blocked command: {executable}")
    return CommandCheck(BlockLevel.ALLOWED)


def contains_unquoted_expansion(command: str | None) -> bool:
    if not command:
        return False
    single = False
    double = False
    escaped = False
    for index, char in enumerate(command):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not double:
            single = not single
            continue
        if char == '"' and not single:
            double = not double
            continue
        if single:
            continue
        if char == "$" or char in "*?":
            return True
        if char == "{" and "," in command[index : command.find("}", index) if "}" in command[index:] else len(command)]:
            return True
    return False


def classify_command(command: str | None) -> CommandClassification:
    if not command or not command.strip():
        return CommandClassification(False)
    risk = command_risk(command)
    if risk.blocked or risk.needs_confirmation:
        return CommandClassification(False, category=risk.level.value)
    segments = [part.strip() for part in re.split(r"\s*(?:&&|\|\||[|;])\s*", command) if part.strip()]
    if not segments:
        return CommandClassification(False)
    saw_search = saw_read = saw_list = False
    for segment in segments:
        parts = command_parts(segment)
        if not parts:
            continue
        cmd = Path(parts[0]).name.lower()
        if cmd == "git" and len(parts) > 1:
            read_only = parts[1] in {"status", "log", "diff", "show", "branch", "remote", "blame", "ls-files", "rev-parse", "config"}
        elif cmd == "docker" and len(parts) > 1:
            read_only = parts[1] in {"ps", "images", "logs", "inspect"}
        elif cmd == "kubectl" and len(parts) > 1:
            read_only = parts[1] in {"get", "describe", "logs"}
        elif cmd in {"npm", "yarn", "pip"} and len(parts) > 1:
            read_only = parts[1] in {"list", "info", "outdated", "audit", "show", "freeze"}
        else:
            read_only = cmd in READONLY_COMMANDS and not (cmd == "find" and any(part in {"-delete", "-exec", "-execdir"} for part in parts))
        if not read_only:
            return CommandClassification(False, category="write")
        saw_search = saw_search or cmd in SEARCH_COMMANDS
        saw_read = saw_read or cmd in READ_COMMANDS
        saw_list = saw_list or cmd in LIST_COMMANDS
    category = "search" if saw_search else "read" if saw_read else "list" if saw_list else "readonly"
    return CommandClassification(True, saw_search, saw_read, saw_list, category)


FORBIDDEN_PATH_PATTERNS = [
    r"(^|[/\\])\.ssh[/\\]id_",
    r"(^|[/\\])\.aws[/\\]credentials$",
    r"(^|[/\\])\.kube[/\\]config$",
    r"(^|[/\\])\.gnupg([/\\]|$)",
    r"(^|[/\\])\.npmrc$",
    r"(^|[/\\])\.pgpass$",
    r"(^|[/\\])\.git-credentials$",
    r"(^|[/\\])\.docker[/\\]config\.json$",
]
PROTECTED_PATH_PATTERNS = [r"(^|[/\\])\.(bashrc|zshrc)$", r"(^|[/\\])\.env(\..*)?$"]


def sensitive_path_level(path: str | None) -> str:
    if not path:
        return "allowed"
    normalized = path.replace("~", str(Path.home())).lower()
    if any(re.search(pattern, normalized) for pattern in FORBIDDEN_PATH_PATTERNS):
        return "forbidden"
    if any(re.search(pattern, normalized) for pattern in PROTECTED_PATH_PATTERNS):
        return "protected"
    if normalized in {"/etc/shadow", "/etc/passwd"}:
        return "audit"
    return "allowed"
