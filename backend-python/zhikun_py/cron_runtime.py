from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


MAX_CRON_JOBS = 50
DEFAULT_EXPIRY_DAYS = 30


@dataclass(slots=True)
class CronTask:
    id: str
    cron: str
    prompt: str
    recurring: bool
    durable: bool
    createdAt: str
    expiresAt: str
    agentId: str | None = None
    nextRun: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CronValidationError(ValueError):
    pass


class CronTaskService:
    FIELD_RANGES = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]

    def __init__(self, durable_store_path: Path) -> None:
        self.durable_store_path = durable_store_path
        self.tasks: dict[str, CronTask] = {}
        self.load_durable_tasks()

    def validate_cron(self, expression: str) -> None:
        fields = expression.split()
        if len(fields) != 5:
            raise CronValidationError("cron expression must have exactly 5 fields")
        for field, (minimum, maximum) in zip(fields, self.FIELD_RANGES):
            self._validate_field(field, minimum, maximum)

    def add_task(self, cron: str, prompt: str, recurring: bool = True, durable: bool = False, agent_id: str | None = None) -> CronTask:
        if len(self.tasks) >= MAX_CRON_JOBS:
            raise CronValidationError(f"Maximum number of scheduled tasks reached ({MAX_CRON_JOBS})")
        if not cron or not cron.strip():
            raise CronValidationError("cron expression is required")
        if not prompt or not prompt.strip():
            raise CronValidationError("prompt is required")
        self.validate_cron(cron)
        now = datetime.now(timezone.utc)
        task = CronTask(
            id=uuid.uuid4().hex[:8],
            cron=cron.strip(),
            prompt=prompt,
            recurring=recurring,
            durable=durable,
            createdAt=now.isoformat(),
            expiresAt=(now + timedelta(days=DEFAULT_EXPIRY_DAYS)).isoformat(),
            agentId=agent_id,
            nextRun=self.next_run(cron),
        )
        self.tasks[task.id] = task
        if durable:
            self.persist_durable_tasks()
        return task

    def list_all(self) -> list[CronTask]:
        self.cleanup_expired_tasks()
        return sorted(self.tasks.values(), key=lambda item: item.createdAt)

    def get_task(self, task_id: str) -> CronTask | None:
        return self.tasks.get(task_id)

    def remove(self, task_id: str) -> CronTask | None:
        removed = self.tasks.pop(task_id, None)
        if removed and removed.durable:
            self.persist_durable_tasks()
        return removed

    def task_count(self) -> int:
        return len(self.tasks)

    def cleanup_expired_tasks(self) -> int:
        now = datetime.now(timezone.utc)
        expired = [
            task_id
            for task_id, task in self.tasks.items()
            if self._parse_dt(task.expiresAt) and now > self._parse_dt(task.expiresAt)
        ]
        durable_removed = False
        for task_id in expired:
            durable_removed = durable_removed or bool(self.tasks[task_id].durable)
            self.tasks.pop(task_id, None)
        if durable_removed:
            self.persist_durable_tasks()
        return len(expired)

    def persist_durable_tasks(self) -> None:
        self.durable_store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.durable_store_path.with_suffix(self.durable_store_path.suffix + ".tmp")
        payload = [task.to_dict() for task in self.tasks.values() if task.durable]
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.durable_store_path)

    def load_durable_tasks(self) -> None:
        if not self.durable_store_path.exists():
            return
        try:
            payload = json.loads(self.durable_store_path.read_text(encoding="utf-8"))
        except Exception:
            return
        now = datetime.now(timezone.utc)
        for item in payload if isinstance(payload, list) else []:
            try:
                task = CronTask(**item)
            except TypeError:
                continue
            expires = self._parse_dt(task.expiresAt)
            if expires and now < expires:
                self.tasks[task.id] = task

    def next_run(self, expression: str, start: datetime | None = None) -> str:
        minute_field, hour_field, *_ = expression.split()
        current = (start or datetime.now(timezone.utc)).replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(366 * 24 * 60):
            if self._field_matches(current.minute, minute_field, 0, 59) and self._field_matches(current.hour, hour_field, 0, 23):
                return current.isoformat()
            current += timedelta(minutes=1)
        return "unknown"

    def _validate_field(self, field: str, minimum: int, maximum: int) -> None:
        for part in field.split(","):
            base = part.split("/", 1)[0]
            step = part.split("/", 1)[1] if "/" in part else None
            if step is not None and (not step.isdigit() or int(step) <= 0):
                raise CronValidationError(f"invalid cron step: {part}")
            if base == "*":
                continue
            values = base.split("-", 1)
            if not all(value.isdigit() for value in values):
                raise CronValidationError(f"invalid cron field: {part}")
            numbers = [int(value) for value in values]
            if any(number < minimum or number > maximum for number in numbers):
                raise CronValidationError(f"cron field out of range: {part}")
            if len(numbers) == 2 and numbers[0] > numbers[1]:
                raise CronValidationError(f"invalid cron range: {part}")

    def _field_matches(self, value: int, field: str, minimum: int, maximum: int) -> bool:
        for part in field.split(","):
            base, _, step_text = part.partition("/")
            step = int(step_text) if step_text else 1
            if base == "*":
                if (value - minimum) % step == 0:
                    return True
                continue
            if "-" in base:
                start, end = (int(item) for item in base.split("-", 1))
                if start <= value <= end and (value - start) % step == 0:
                    return True
                continue
            if int(base) == value and value <= maximum:
                return True
        return False

    def _parse_dt(self, value: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
