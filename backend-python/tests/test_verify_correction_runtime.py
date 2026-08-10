import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.correction_runtime import CompileErrorParser, SelfCorrectionLoop, TestFailureParser, build_correction_instruction  # noqa: E402
from zhikun_py.verify_runtime import CapabilityGate, HttpApiVerifier, BrowserVerifier, run_local_journey  # noqa: E402


def test_compile_error_parser_handles_java_typescript_and_python() -> None:
    output = """
    [ERROR] /src/main/java/com/example/Main.java:15: error: cannot find symbol
    src/app.ts(23,5): error TS2304: Cannot find name 'xyz'
    Traceback (most recent call last):
      File "main.py", line 10
        x = 1 +
    SyntaxError: invalid syntax
    """
    errors = CompileErrorParser().parse(output)
    assert [error.language for error in errors] == ["java", "typescript", "python"]
    assert errors[0].fileName.endswith("Main.java")
    assert errors[1].code == "TS2304"
    assert errors[2].lineNumber == 10


def test_test_failure_parser_handles_junit_jest_and_pytest() -> None:
    junit = "testSomething(com.example.MyTest): expected:<true> but was:<false>"
    jest = """
    FAIL src/App.test.tsx
      ● App › renders
        Expected: "Hello"
        Received: "World"
    """
    pytest = "FAILED tests/test_main.py::test_addition - AssertionError: assert 3 == 4"
    parser = TestFailureParser()
    assert parser.parse(junit)[0].framework == "junit"
    assert parser.parse(jest)[0].framework == "jest"
    assert parser.parse(pytest)[0].testName == "test_addition"

    instruction = build_correction_instruction(junit + "\n" + pytest)
    assert instruction["hasIssues"] is True
    assert instruction["suggestions"]


def test_self_correction_loop_prefers_compile_errors_and_limits_attempts() -> None:
    output = """
    src/app.ts(23,5): error TS2304: Cannot find name 'xyz'
    FAILED tests/test_main.py::test_addition - AssertionError: assert 3 == 4
    """
    loop = SelfCorrectionLoop()
    instruction = loop.detect_and_prepare(output, previous_attempts=1)
    assert instruction is not None
    assert instruction.type == "COMPILE_ERROR"
    assert instruction.attemptNumber == 2
    assert instruction.errorContext["fileName"] == "src/app.ts"

    assert loop.detect_and_prepare(output, previous_attempts=3) is None


def test_self_correction_loop_handles_test_failures_truncation_and_abort_rules() -> None:
    loop = SelfCorrectionLoop(max_instruction_chars=120)
    long_output = "FAILED tests/test_main.py::test_addition - AssertionError: " + ("very long " * 50)
    instruction = loop.detect_and_prepare(long_output)
    assert instruction is not None
    assert instruction.type == "TEST_FAILURE"
    assert instruction.truncated is True
    assert len(instruction.instruction) <= 120

    previous = "src/app.ts(23,5): error TS2304: Cannot find name 'xyz'"
    more_errors = previous + "\nsrc/other.ts(10,1): error TS2322: Type mismatch"
    assert loop.should_abort(more_errors, previous) is True
    assert loop.should_abort(previous, more_errors) is False

    new_file = "src/new.ts(1,1): error TS2304: Cannot find name 'abc'"
    assert loop.should_abort(new_file, previous) is True


def test_verify_runtime_capability_gate_and_journey_results() -> None:
    disabled = CapabilityGate({"RUNTIME_VERIFICATION": False}, {"HTTP_API": True})
    assert disabled.verify_enabled() is False

    unavailable = HttpApiVerifier(CapabilityGate({"RUNTIME_VERIFICATION": True}, {"HTTP_API": False})).verify(
        {"sessionId": "s1", "baseUrl": "http://localhost", "steps": [{"action": "http_get", "url": "/health"}]}
    )
    assert unavailable.verdict == "unavailable"

    verified = HttpApiVerifier().verify({"sessionId": "s1", "baseUrl": "http://localhost", "steps": [{"action": "http_get", "url": "/health"}]})
    assert verified.verdict == "verified"
    assert verified.stepResults[0].ok is True

    browser = BrowserVerifier().verify({"sessionId": "s1", "steps": [{"action": "browser_click", "url": "/"}]})
    assert browser.verdict == "verified"

    failed = run_local_journey({"steps": [{"action": "http_get", "fail": True}]})
    assert failed.verdict == "failed"
    assert failed.stepResults[0].ok is False
