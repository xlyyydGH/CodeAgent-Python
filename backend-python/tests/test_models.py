import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.models import ContentBlock, Message, Session, Usage  # noqa: E402


def test_usage_shape_matches_frontend_contract() -> None:
    assert Usage().to_dict() == {
        "inputTokens": 0,
        "outputTokens": 0,
        "cacheReadInputTokens": 0,
        "cacheCreationInputTokens": 0,
    }


def test_message_text_and_dict() -> None:
    msg = Message(type="assistant", content=[ContentBlock(type="text", text="hello")])
    payload = msg.to_dict()
    assert msg.text() == "hello"
    assert payload["type"] == "assistant"
    assert payload["content"] == [{"type": "text", "text": "hello"}]
    assert payload["usage"]["inputTokens"] == 0


def test_session_summary() -> None:
    session = Session(id="session-1", model="model-a", messages=[Message(type="user", content=[ContentBlock(type="text", text="hi")])])
    assert session.summary()["messageCount"] == 1
    assert session.to_dict()["messages"][0]["type"] == "user"
