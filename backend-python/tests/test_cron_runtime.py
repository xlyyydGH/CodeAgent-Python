import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.cron_runtime import CronTaskService, CronValidationError  # noqa: E402
from zhikun_py.permissions import PermissionDecision, PermissionPolicy, PermissionRule  # noqa: E402
from zhikun_py.tools import ToolRegistry  # noqa: E402


def test_cron_task_service_validates_persists_and_deletes() -> None:
    root = BACKEND_DIR / ".test-workspace" / "cron-runtime"
    root.mkdir(parents=True, exist_ok=True)
    store = root / "scheduled_tasks.json"
    store.unlink(missing_ok=True)
    service = CronTaskService(store)

    task = service.add_task("*/5 * * * *", "run analysis", durable=True, agent_id="agent-1")
    assert task.id
    assert task.nextRun
    assert service.task_count() == 1
    assert store.exists()

    loaded = CronTaskService(store)
    assert loaded.get_task(task.id) is not None
    assert loaded.remove(task.id).id == task.id

    try:
        service.add_task("* * *", "bad")
    except CronValidationError as exc:
        assert "5 fields" in str(exc)
    else:
        raise AssertionError("invalid cron should fail")


def test_cron_tools_share_runtime_service() -> None:
    root = BACKEND_DIR / ".test-workspace" / "cron-tools"
    root.mkdir(parents=True, exist_ok=True)
    service = CronTaskService(root / "scheduled_tasks.json")
    policy = PermissionPolicy(
        [
            PermissionRule("CronCreate", PermissionDecision.ALLOW),
            PermissionRule("CronDelete", PermissionDecision.ALLOW),
        ]
    )
    registry = ToolRegistry(BACKEND_DIR, policy, service)

    created = registry.call("CronCreate", {"cron": "0 * * * *", "prompt": "hourly job"})
    assert created.isError is False
    task_id = created.metadata["task"]["id"]

    listed = registry.call("CronList", {})
    assert listed.metadata["total"] == 1
    deleted = registry.call("CronDelete", {"id": task_id})
    assert deleted.isError is False
    assert service.task_count() == 0
