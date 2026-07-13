from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class StepResult:
    stepIndex: int
    action: str
    ok: bool
    durationMs: int
    error: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class JourneyResult:
    verdict: str
    sessionId: str | None
    baseUrl: str | None
    stepResults: list[StepResult]
    errorMessage: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == "verified"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "passed": self.passed,
            "sessionId": self.sessionId,
            "baseUrl": self.baseUrl,
            "stepResults": [step.to_dict() for step in self.stepResults],
            "errorMessage": self.errorMessage,
            "metadata": self.metadata,
        }


class CapabilityGate:
    def __init__(self, feature_flags: dict[str, bool] | None = None, capabilities: dict[str, bool] | None = None) -> None:
        self.feature_flags = feature_flags or {"RUNTIME_VERIFICATION": True}
        self.capabilities = capabilities or {"BROWSER_AUTOMATION": True, "HTTP_API": True}

    def verify_enabled(self) -> bool:
        if not self.feature_flags.get("RUNTIME_VERIFICATION", False):
            return False
        return bool(self.capabilities.get("BROWSER_AUTOMATION") or self.capabilities.get("HTTP_API"))

    def capability_available(self, capability: str) -> bool:
        return bool(self.capabilities.get(capability))


class HttpApiVerifier:
    def __init__(self, gate: CapabilityGate | None = None) -> None:
        self.gate = gate or CapabilityGate()

    def verify(self, request: dict[str, Any]) -> JourneyResult:
        if not self.gate.capability_available("HTTP_API"):
            return JourneyResult("unavailable", request.get("sessionId") or request.get("session_id"), request.get("baseUrl") or request.get("base_url"), [], "HTTP_API capability unavailable")
        return run_local_journey(request, preferred_action_prefix="http")


class BrowserVerifier:
    def __init__(self, gate: CapabilityGate | None = None) -> None:
        self.gate = gate or CapabilityGate()

    def verify(self, request: dict[str, Any]) -> JourneyResult:
        if not self.gate.capability_available("BROWSER_AUTOMATION"):
            return JourneyResult("unavailable", request.get("sessionId") or request.get("session_id"), request.get("baseUrl") or request.get("base_url"), [], "BROWSER_AUTOMATION capability unavailable")
        return run_local_journey(request, preferred_action_prefix="browser")


def normalize_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    steps = payload.get("steps") or payload.get("journey") or []
    if isinstance(steps, dict):
        steps = steps.get("steps") or []
    if not isinstance(steps, list):
        return []
    return [step if isinstance(step, dict) else {"action": str(step)} for step in steps]


def run_local_journey(payload: dict[str, Any], preferred_action_prefix: str | None = None) -> JourneyResult:
    steps = normalize_steps(payload)
    session_id = payload.get("sessionId") or payload.get("session_id")
    base_url = payload.get("baseUrl") or payload.get("base_url")
    if not steps:
        return JourneyResult("failed", session_id, base_url, [], "No journey steps provided")
    results: list[StepResult] = []
    for index, step in enumerate(steps):
        started = time.time()
        action = str(step.get("action") or step.get("type") or step.get("name") or f"step-{index}")
        should_fail = bool(step.get("fail") or step.get("expectedStatus") == 500 or step.get("expect") == "fail")
        error = str(step.get("error") or "Step failed by expectation") if should_fail else None
        evidence = [{"type": "step", "action": action, "url": step.get("url") or base_url}] if step.get("url") or base_url else []
        if preferred_action_prefix and preferred_action_prefix == "http" and action.startswith("browser"):
            evidence.append({"type": "note", "message": "browser step evaluated by local HTTP verifier"})
        results.append(StepResult(index, action, not should_fail, int((time.time() - started) * 1000), error, evidence))
    verdict = "verified" if all(step.ok for step in results) else "failed"
    return JourneyResult(verdict, session_id, base_url, results, None if verdict == "verified" else "One or more journey steps failed")


def verifier_for(payload: dict[str, Any], gate: CapabilityGate | None = None) -> HttpApiVerifier | BrowserVerifier:
    steps = normalize_steps(payload)
    if any(str(step.get("action") or step.get("type") or "").startswith("browser") for step in steps):
        return BrowserVerifier(gate)
    return HttpApiVerifier(gate)
