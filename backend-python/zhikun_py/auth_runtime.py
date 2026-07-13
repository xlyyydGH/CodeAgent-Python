from __future__ import annotations

import base64
import ipaddress
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REMOTE_COOKIE_NAME = "ai-coder-session"
REMOTE_COOKIE_MAX_AGE_DAYS = 30


@dataclass(slots=True)
class AuthDecision:
    authenticated: bool
    reason: str
    issueCookie: bool = False
    redirectUrl: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "authenticated": self.authenticated,
            "reason": self.reason,
            "issueCookie": self.issueCookie,
            "redirectUrl": self.redirectUrl,
        }


class RemoteAccessSecurity:
    def __init__(self, token_path: Path, allow_private_network: bool | None = None) -> None:
        self.token_path = token_path
        self.allow_private_network = bool(
            os.getenv("ALLOW_PRIVATE_NETWORK", "false").lower() == "true"
            if allow_private_network is None
            else allow_private_network
        )
        self.sessions: dict[str, datetime] = {}

    def access_token(self) -> str:
        env_token = os.getenv("AUTH_LAN_TOKEN") or os.getenv("ZHIKUN_ACCESS_TOKEN")
        if env_token:
            return env_token
        if self.token_path.exists():
            token = self.token_path.read_text(encoding="utf-8").strip()
            if token:
                return token
        token = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(token, encoding="utf-8")
        return token

    def validate(self, remote_addr: str, headers: dict[str, str] | None = None, cookies: dict[str, str] | None = None, query: dict[str, str] | None = None) -> AuthDecision:
        auth_mode = os.getenv("AUTH_MODE", "localhost")
        if auth_mode == "localhost":
            return AuthDecision(True, "localhost mode")
        if self.is_loopback(remote_addr):
            return AuthDecision(True, "loopback")
        if self.allow_private_network:
            return AuthDecision(True, "private network bypass")
        if not self.is_private_network(remote_addr):
            return AuthDecision(False, "non-private network")
        session_cookie = (cookies or {}).get(REMOTE_COOKIE_NAME)
        if session_cookie and self.validate_session(session_cookie):
            return AuthDecision(True, "session cookie")
        token = self.access_token()
        auth_header = (headers or {}).get("authorization") or (headers or {}).get("Authorization") or ""
        if auth_header.startswith("Bearer ") and auth_header[7:] == token:
            return AuthDecision(True, "bearer token", issueCookie=True)
        if (query or {}).get("token") == token:
            return AuthDecision(True, "query token", issueCookie=True, redirectUrl=self.remove_token_query(query or {}))
        return AuthDecision(False, "token required")

    def issue_session(self) -> str:
        session_id = secrets.token_urlsafe(24)
        self.sessions[session_id] = datetime.now(timezone.utc) + timedelta(days=REMOTE_COOKIE_MAX_AGE_DAYS)
        return session_id

    def validate_session(self, session_id: str) -> bool:
        expires = self.sessions.get(session_id)
        if not expires:
            return False
        if datetime.now(timezone.utc) > expires:
            self.sessions.pop(session_id, None)
            return False
        return True

    def token_preview(self) -> str:
        token = self.access_token()
        if len(token) <= 8:
            return token
        return f"{token[:4]}...{token[-4:]}"

    def is_loopback(self, value: str) -> bool:
        try:
            return ipaddress.ip_address(value).is_loopback
        except ValueError:
            return False

    def is_private_network(self, value: str) -> bool:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return False
        return address.is_private or address.is_loopback or address.is_link_local

    def remove_token_query(self, query: dict[str, str]) -> str:
        cleaned = {key: value for key, value in query.items() if key != "token"}
        if not cleaned:
            return ""
        return "?" + "&".join(f"{key}={value}" for key, value in cleaned.items())
