import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.memdir_runtime import MemdirService, MemoryCategory, MemorySearchEngine  # noqa: E402
from zhikun_py.permissions import PermissionDecision, PermissionPolicy, PermissionRule  # noqa: E402
from zhikun_py.tools import ToolRegistry  # noqa: E402


def test_memdir_bm25_chinese_search_category_and_prompt() -> None:
    state = {
        "memories": [
            {"id": "m1", "title": "Java编码规范", "content": "使用Java 21新特性，提交前运行测试", "category": "semantic"},
            {"id": "m2", "title": "Python项目", "content": "使用Python 3.12和pytest", "category": "procedural"},
            {"id": "m3", "title": "团队约定", "content": "评审时先看安全风险", "category": "team"},
        ]
    }
    service = MemdirService(BACKEND_DIR, state, MemorySearchEngine())

    results = service.search("Java编码", 2, rerank=True)
    assert results
    assert results[0].memory.id == "m1"
    assert results[0].matchedTokens
    procedural = service.search_by_category("procedural")
    assert procedural[0].id == "m2"
    assert MemoryCategory.from_tag("TEAM") == MemoryCategory.TEAM
    prompt = service.build_prompt(BACKEND_DIR)
    assert "Personal Memory" in prompt
    assert "Memory categories" in prompt


def test_memory_tool_read_write_search_delete() -> None:
    state = {"memories": []}
    service = MemdirService(BACKEND_DIR, state)
    registry = ToolRegistry(
        BACKEND_DIR,
        PermissionPolicy([PermissionRule("Memory", PermissionDecision.ALLOW)]),
        memdir_service=service,
    )

    written = registry.call("Memory", {"action": "write", "title": "偏好", "content": "用户偏好Python实现", "category": "semantic"})
    assert written.isError is False
    found = registry.call("Memory", {"action": "search", "query": "Python偏好", "limit": 3})
    assert found.isError is False
    assert found.metadata["results"]
    deleted = registry.call("Memory", {"action": "delete", "content": "Python实现"})
    assert deleted.metadata["deleted"] == 1
