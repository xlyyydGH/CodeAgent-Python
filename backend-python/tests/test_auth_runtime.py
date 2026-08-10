import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.auth_runtime import REMOTE_COOKIE_NAME, RemoteAccessSecurity  # noqa: E402


def test_remote_access_security_private_network_token_and_cookie(monkeypatch) -> None:
    root = BACKEND_DIR / ".test-workspace" / "auth-runtime"
    root.mkdir(parents=True, exist_ok=True)
    token_path = root / "access-token"
    token_path.unlink(missing_ok=True)
    monkeypatch.setenv("AUTH_MODE", "lan_token")
    monkeypatch.setenv("AUTH_LAN_TOKEN", "secret-token")
    security = RemoteAccessSecurity(token_path)

    assert security.is_private_network("192.168.1.5") is True
    assert security.is_private_network("8.8.8.8") is False
    assert security.validate("8.8.8.8").authenticated is False

    bearer = security.validate("192.168.1.5", headers={"Authorization": "Bearer secret-token"})
    assert bearer.authenticated is True
    assert bearer.issueCookie is True

    session_id = security.issue_session()
    cookie = security.validate("192.168.1.5", cookies={REMOTE_COOKIE_NAME: session_id})
    assert cookie.authenticated is True
    assert cookie.reason == "session cookie"
