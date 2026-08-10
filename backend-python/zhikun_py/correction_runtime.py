from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ParsedError:
    language: str
    fileName: str
    lineNumber: int
    columnNumber: int = 0
    errorMessage: str = ""
    code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ParsedTestFailure:
    framework: str
    testName: str
    expected: str | None = None
    actual: str | None = None
    message: str = ""
    fileName: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CorrectionInstruction:
    type: str
    attemptNumber: int
    instruction: str
    errorContext: dict[str, Any]
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompileErrorParser:
    JAVA_PATTERNS = [
        re.compile(r"(?:\[ERROR]\s+)?(.+?\.java):(\d+):\s+error:\s+(.+)"),
    ]
    TS_PATTERN = re.compile(r"(.+?\.(?:ts|tsx|js|jsx))\((\d+),(\d+)\):\s+error\s+(TS\d+):\s+(.+)")
    PY_FILE_PATTERN = re.compile(r'File "([^"]+)", line (\d+)')
    PY_ERROR_PATTERN = re.compile(r"^\s*([A-Za-z_]*Error|SyntaxError|IndentationError):\s*(.+)$", re.MULTILINE)

    def parse(self, output: str | None, limit: int = 5) -> list[ParsedError]:
        if not output or not output.strip():
            return []
        errors: list[ParsedError] = []
        for line in output.splitlines():
            stripped = line.strip()
            match = self.TS_PATTERN.match(stripped)
            if match:
                errors.append(
                    ParsedError(
                        language="typescript",
                        fileName=match.group(1),
                        lineNumber=int(match.group(2)),
                        columnNumber=int(match.group(3)),
                        code=match.group(4),
                        errorMessage=match.group(5),
                    )
                )
                continue
            for pattern in self.JAVA_PATTERNS:
                match = pattern.match(stripped)
                if match:
                    errors.append(
                        ParsedError(
                            language="java",
                            fileName=match.group(1),
                            lineNumber=int(match.group(2)),
                            errorMessage=match.group(3),
                        )
                    )
                    break
            if len(errors) >= limit:
                return errors[:limit]
        py_file = self.PY_FILE_PATTERN.search(output)
        py_error = self.PY_ERROR_PATTERN.search(output)
        if py_file and py_error and len(errors) < limit:
            errors.append(
                ParsedError(
                    language="python",
                    fileName=py_file.group(1),
                    lineNumber=int(py_file.group(2)),
                    errorMessage=f"{py_error.group(1)}: {py_error.group(2)}",
                )
            )
        return errors[:limit]


class TestFailureParser:
    PYTEST_PATTERN = re.compile(r"FAILED\s+([^\s]+)::([^\s]+)\s+-\s+(.+)")
    JUNIT_PATTERN = re.compile(r"\s*([^(]+)\(([^)]+)\):\s+expected:<([^>]*)>\s+but was:<([^>]*)>")

    def parse(self, output: str | None, limit: int = 5) -> list[ParsedTestFailure]:
        if not output or not output.strip():
            return []
        if re.search(r"Failures:\s*0.*Errors:\s*0", output, re.DOTALL) and "FAILED" not in output and "FAIL " not in output:
            return []
        failures: list[ParsedTestFailure] = []
        for match in self.PYTEST_PATTERN.finditer(output):
            failures.append(
                ParsedTestFailure(
                    framework="pytest",
                    fileName=match.group(1),
                    testName=match.group(2),
                    message=match.group(3),
                )
            )
            if len(failures) >= limit:
                return failures
        for match in self.JUNIT_PATTERN.finditer(output):
            failures.append(
                ParsedTestFailure(
                    framework="junit",
                    testName=match.group(1).strip(),
                    fileName=match.group(2),
                    expected=match.group(3),
                    actual=match.group(4),
                    message=match.group(0).strip(),
                )
            )
            if len(failures) >= limit:
                return failures
        if "FAIL " in output or "Expected:" in output or "Received:" in output:
            blocks = re.split(r"\n\s*●\s+", output)
            for block in blocks:
                if "Expected:" not in block and "Received:" not in block:
                    continue
                expected = self._line_after("Expected:", block)
                actual = self._line_after("Received:", block)
                name = block.splitlines()[0].strip() if block.splitlines() else "jest failure"
                file_match = re.search(r"at .*?\(([^:]+):\d+:\d+\)", block)
                failures.append(
                    ParsedTestFailure(
                        framework="jest",
                        testName=name,
                        expected=expected,
                        actual=actual,
                        fileName=file_match.group(1) if file_match else None,
                        message=block.strip()[:1000],
                    )
                )
                if len(failures) >= limit:
                    return failures
        return failures[:limit]

    def _line_after(self, prefix: str, block: str) -> str | None:
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                return stripped[len(prefix) :].strip().strip('"')
        return None


def build_correction_instruction(output: str | None) -> dict[str, Any]:
    compile_errors = CompileErrorParser().parse(output)
    test_failures = TestFailureParser().parse(output)
    suggestions: list[str] = []
    for error in compile_errors:
        suggestions.append(f"Fix {error.language} error in {error.fileName}:{error.lineNumber}: {error.errorMessage}")
    for failure in test_failures:
        suggestions.append(f"Investigate {failure.framework} failure {failure.testName}: {failure.message}")
    return {
        "compileErrors": [item.to_dict() for item in compile_errors],
        "testFailures": [item.to_dict() for item in test_failures],
        "suggestions": suggestions[:10],
        "hasIssues": bool(compile_errors or test_failures),
    }


class SelfCorrectionLoop:
    def __init__(self, max_attempts: int = 3, max_instruction_chars: int = 4000) -> None:
        self.max_attempts = max_attempts
        self.max_instruction_chars = max_instruction_chars
        self.compile_parser = CompileErrorParser()
        self.test_parser = TestFailureParser()

    def detect_and_prepare(self, output: str | None, previous_attempts: int = 0) -> CorrectionInstruction | None:
        if not output or not output.strip() or previous_attempts >= self.max_attempts:
            return None
        compile_errors = self.compile_parser.parse(output)
        if compile_errors:
            first = compile_errors[0]
            instruction = (
                f"Fix the compile error in {first.fileName}:{first.lineNumber}.\n"
                f"Language: {first.language}\n"
                f"Error: {first.errorMessage}\n"
                "Make the smallest code change that resolves this error, then rerun the failing check."
            )
            instruction, truncated = self._truncate(instruction)
            return CorrectionInstruction("COMPILE_ERROR", previous_attempts + 1, instruction, first.to_dict(), truncated)
        failures = self.test_parser.parse(output)
        if failures:
            first = failures[0]
            instruction = (
                f"Fix the failing {first.framework} test: {first.testName}.\n"
                f"Expected: {first.expected or 'unknown'}\n"
                f"Actual: {first.actual or 'unknown'}\n"
                f"Message: {first.message}\n"
                "Preserve intended behavior and rerun the test."
            )
            instruction, truncated = self._truncate(instruction)
            return CorrectionInstruction("TEST_FAILURE", previous_attempts + 1, instruction, first.to_dict(), truncated)
        return None

    def should_abort(self, new_output: str | None, previous_output: str | None) -> bool:
        previous_errors = self.compile_parser.parse(previous_output) + [
            ParsedError("test", failure.fileName or failure.testName, 0, errorMessage=failure.message, code=failure.framework)
            for failure in self.test_parser.parse(previous_output)
        ]
        new_errors = self.compile_parser.parse(new_output) + [
            ParsedError("test", failure.fileName or failure.testName, 0, errorMessage=failure.message, code=failure.framework)
            for failure in self.test_parser.parse(new_output)
        ]
        if len(new_errors) > len(previous_errors):
            return True
        previous_files = {error.fileName for error in previous_errors}
        new_files = {error.fileName for error in new_errors}
        return bool(new_files - previous_files)

    def _truncate(self, instruction: str) -> tuple[str, bool]:
        if len(instruction) <= self.max_instruction_chars:
            return instruction, False
        suffix = "\n[truncated due to token limit]"
        return instruction[: self.max_instruction_chars - len(suffix)] + suffix, True
