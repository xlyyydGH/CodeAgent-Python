from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class WorkflowStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class WorkflowPhaseName(StrEnum):
    RESEARCH = "Research"
    SYNTHESIS = "Synthesis"
    IMPLEMENTATION = "Implementation"
    VERIFICATION = "Verification"


PHASE_ORDER = [
    WorkflowPhaseName.RESEARCH,
    WorkflowPhaseName.SYNTHESIS,
    WorkflowPhaseName.IMPLEMENTATION,
    WorkflowPhaseName.VERIFICATION,
]


@dataclass(slots=True)
class WorkflowPhase:
    name: WorkflowPhaseName
    phaseIndex: int
    phasePrompt: str
    resultSummary: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "WorkflowPhase":
        payload = data or {}
        raw_name = str(payload.get("name") or WorkflowPhaseName.RESEARCH.value)
        try:
            name = WorkflowPhaseName(raw_name)
        except ValueError:
            name = next((item for item in WorkflowPhaseName if item.value.lower() == raw_name.lower()), WorkflowPhaseName.RESEARCH)
        default_index = PHASE_ORDER.index(name)
        try:
            phase_index = int(payload.get("phaseIndex", default_index))
        except (TypeError, ValueError):
            phase_index = default_index
        return cls(
            name=name,
            phaseIndex=phase_index,
            phasePrompt=str(payload.get("phasePrompt") or f"{name.value} phase"),
            resultSummary=payload.get("resultSummary"),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["name"] = self.name.value
        return data


@dataclass(slots=True)
class CoordinatorWorkflow:
    workflowId: str
    sessionId: str
    objective: str
    status: WorkflowStatus = WorkflowStatus.RUNNING
    currentPhase: WorkflowPhase | None = None
    history: list[WorkflowPhase] = field(default_factory=list)
    scratchpad: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    createdAt: float = field(default_factory=time.time)
    updatedAt: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoordinatorWorkflow":
        def _float(value: Any, fallback: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        raw_status = str(data.get("status") or WorkflowStatus.RUNNING.value)
        try:
            status = WorkflowStatus(raw_status)
        except ValueError:
            status = WorkflowStatus.RUNNING
        now = time.time()
        workflow = cls(
            workflowId=str(data.get("workflowId") or f"wf-{uuid.uuid4().hex[:8]}"),
            sessionId=str(data.get("sessionId") or "default"),
            objective=str(data.get("objective") or ""),
            status=status,
            currentPhase=WorkflowPhase.from_dict(data.get("currentPhase")) if data.get("currentPhase") else None,
            history=[WorkflowPhase.from_dict(item) for item in data.get("history") or [] if isinstance(item, dict)],
            scratchpad=list(data.get("scratchpad") or []),
            events=list(data.get("events") or []),
            createdAt=_float(data.get("createdAt"), now),
            updatedAt=_float(data.get("updatedAt"), now),
        )
        if workflow.currentPhase and not workflow.history:
            workflow.history.append(workflow.currentPhase)
        return workflow

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflowId": self.workflowId,
            "sessionId": self.sessionId,
            "objective": self.objective,
            "status": self.status.value,
            "currentPhase": self.currentPhase.to_dict() if self.currentPhase else None,
            "history": [phase.to_dict() for phase in self.history],
            "scratchpad": list(self.scratchpad),
            "events": list(self.events),
            "createdAt": self.createdAt,
            "updatedAt": self.updatedAt,
        }


@dataclass(slots=True)
class DelegationValidation:
    valid: bool
    warnings: list[str]
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CoordinatorWorkflowEngine:
    RESEARCH_KEYWORDS = {"explore", "investigate", "search", "find", "look for", "research", "调研", "搜索", "查找", "探索"}
    SYNTHESIS_KEYWORDS = {"synthesize", "summarize", "plan", "design", "craft", "综合", "总结", "规划", "设计"}
    IMPLEMENTATION_KEYWORDS = {"implement", "execute", "apply", "modify", "write", "create", "实现", "执行", "修改", "编写", "创建"}
    VERIFICATION_KEYWORDS = {"verify", "test", "validate", "check", "lint", "build", "验证", "测试", "校验", "检查"}
    VAGUE_PATTERNS = [
        re.compile(r"based on (your|the) (findings|research)", re.I),
        re.compile(r"fix the (bug|issue|problem)", re.I),
        re.compile(r"using what you (learned|found)", re.I),
        re.compile(r"implement the (solution|fix)", re.I),
    ]

    def __init__(self) -> None:
        self.active: dict[str, CoordinatorWorkflow] = {}

    def restore_workflow(self, data: dict[str, Any]) -> CoordinatorWorkflow:
        workflow = CoordinatorWorkflow.from_dict(data)
        self.active[workflow.sessionId] = workflow
        return workflow

    def start_workflow(self, session_id: str, objective: str) -> CoordinatorWorkflow:
        workflow = CoordinatorWorkflow(f"wf-{uuid.uuid4().hex[:8]}", session_id, objective)
        workflow.currentPhase = self._phase(WorkflowPhaseName.RESEARCH)
        workflow.history.append(workflow.currentPhase)
        workflow.events.append(self._event("phase_transition", {"from": None, "to": workflow.currentPhase.name.value}))
        self.active[session_id] = workflow
        return workflow

    def advance_workflow(self, session_id: str, result_summary: str = "") -> CoordinatorWorkflow | None:
        workflow = self.active.get(session_id)
        if not workflow or not workflow.currentPhase:
            return None
        current_index = workflow.currentPhase.phaseIndex
        workflow.currentPhase.resultSummary = result_summary
        if current_index + 1 >= len(PHASE_ORDER):
            workflow.status = WorkflowStatus.COMPLETED
            workflow.events.append(self._event("workflow_completed", {"phase": workflow.currentPhase.name.value, "summary": result_summary}))
        else:
            previous = workflow.currentPhase.name.value
            workflow.currentPhase = self._phase(PHASE_ORDER[current_index + 1])
            workflow.history.append(workflow.currentPhase)
            workflow.events.append(self._event("phase_transition", {"from": previous, "to": workflow.currentPhase.name.value, "summary": result_summary}))
        workflow.updatedAt = time.time()
        return workflow

    def cancel_workflow(self, session_id: str) -> CoordinatorWorkflow | None:
        workflow = self.active.get(session_id)
        if not workflow:
            return None
        workflow.status = WorkflowStatus.CANCELLED
        workflow.events.append(self._event("workflow_cancelled", {}))
        workflow.updatedAt = time.time()
        return workflow

    def add_scratchpad(self, session_id: str, author: str, content: str, phase: str | None = None) -> dict[str, Any]:
        workflow = self.active.get(session_id) or self.start_workflow(session_id, "scratchpad")
        item = {"id": f"note-{uuid.uuid4().hex[:8]}", "author": author, "content": content, "phase": phase or (workflow.currentPhase.name.value if workflow.currentPhase else None), "createdAt": time.time()}
        workflow.scratchpad.append(item)
        workflow.events.append(self._event("scratchpad_added", item))
        workflow.updatedAt = time.time()
        return item

    def detect_phase(self, output: str | None) -> WorkflowPhase:
        text = (output or "").lower()
        scores = {
            WorkflowPhaseName.RESEARCH: self._score(text, self.RESEARCH_KEYWORDS),
            WorkflowPhaseName.SYNTHESIS: self._score(text, self.SYNTHESIS_KEYWORDS),
            WorkflowPhaseName.IMPLEMENTATION: self._score(text, self.IMPLEMENTATION_KEYWORDS),
            WorkflowPhaseName.VERIFICATION: self._score(text, self.VERIFICATION_KEYWORDS),
        }
        if "fileedit" in text or "filewrite" in text:
            scores[WorkflowPhaseName.IMPLEMENTATION] += 3
        if "bash" in text and any(token in text for token in ("test", "build", "lint", "pytest")):
            scores[WorkflowPhaseName.VERIFICATION] += 3
        if "syntheticoutput" in text:
            scores[WorkflowPhaseName.SYNTHESIS] += 2
        selected = max(scores.items(), key=lambda item: item[1])
        return self._phase(selected[0] if selected[1] > 0 else WorkflowPhaseName.RESEARCH)

    def detect_and_validate_phase(self, session_id: str, output: str | None) -> dict[str, Any]:
        detected = self.detect_phase(output)
        workflow = self.active.get(session_id)
        warning = None
        if workflow and workflow.currentPhase and detected.phaseIndex > workflow.currentPhase.phaseIndex + 1:
            warning = f"Phase skip detected: {workflow.currentPhase.name.value} -> {detected.name.value}"
            workflow.events.append(self._event("phase_skip_warning", {"warning": warning, "detected": detected.name.value}))
        return {"detected": detected.to_dict(), "warning": warning}

    def validate_delegation(self, phase: str, prompt: str | None) -> DelegationValidation:
        text = prompt or ""
        warnings: list[str] = []
        min_len = 50 if self._contains_cjk(text) else 100
        if len(text) < min_len:
            warnings.append(f"Prompt too short ({len(text)} chars < {min_len} minimum). Likely delegating understanding.")
        has_specific = "/" in text or re.search(r":\d+", text) or re.search(r"\b[A-Z][A-Za-z]+\.[A-Za-z]+\(", text)
        if phase.lower().startswith("implementation") and not has_specific:
            warnings.append("Prompt lacks specific file paths, line numbers, or method names.")
        for pattern in self.VAGUE_PATTERNS:
            if pattern.search(text):
                warnings.append(f"Vague delegation detected: {pattern.pattern}")
        return DelegationValidation(not warnings, warnings, "WARN" if warnings else "OK")

    def _phase(self, name: WorkflowPhaseName) -> WorkflowPhase:
        index = PHASE_ORDER.index(name)
        prompts = {
            WorkflowPhaseName.RESEARCH: "Research the codebase and gather evidence before deciding.",
            WorkflowPhaseName.SYNTHESIS: "Synthesize findings into a concrete implementation plan.",
            WorkflowPhaseName.IMPLEMENTATION: "Implement the planned changes with specific file context.",
            WorkflowPhaseName.VERIFICATION: "Verify behavior with tests, checks, or runtime evidence.",
        }
        return WorkflowPhase(name, index, prompts[name])

    def _score(self, text: str, keywords: set[str]) -> int:
        return sum(1 for keyword in keywords if keyword in text)

    def _event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"id": f"event-{uuid.uuid4().hex[:8]}", "type": event_type, "payload": payload, "createdAt": time.time()}

    def _contains_cjk(self, text: str) -> bool:
        if not text:
            return False
        cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        return cjk > max(1, len(text)) * 0.3
