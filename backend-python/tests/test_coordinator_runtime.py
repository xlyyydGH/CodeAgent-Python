import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.coordinator_runtime import CoordinatorWorkflowEngine  # noqa: E402


def test_coordinator_workflow_phases_detection_and_delegation() -> None:
    engine = CoordinatorWorkflowEngine()
    workflow = engine.start_workflow("s1", "rewrite backend")
    assert workflow.currentPhase.name.value == "Research"

    advanced = engine.advance_workflow("s1", "found relevant files")
    assert advanced.currentPhase.name.value == "Synthesis"

    detected = engine.detect_phase("Run Bash pytest to verify build and tests")
    assert detected.name.value == "Verification"

    warning = engine.validate_delegation("Implementation", "fix the bug")
    assert warning.valid is False
    assert warning.severity == "WARN"

    note = engine.add_scratchpad("s1", "worker-1", "inspected app.py")
    assert note["phase"] == "Synthesis"
    assert engine.active["s1"].scratchpad
