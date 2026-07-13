import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.query_runtime import (  # noqa: E402
    AbortController,
    ContextCascadeService,
    ContextCollapseService,
    DefaultTerminationStrategy,
    MicroCompactService,
    PromptTooLongRecovery,
    QueryLoopState,
    QueryPhase,
    RecoveryEventType,
    SideQueryService,
    TerminationAction,
    TokenCounter,
    TokenBudget,
    ToolCallTracker,
    ToolPriorityScheduler,
    ToolResultSummarizer,
    estimate_tokens,
)
from zhikun_py.security import BlockLevel, command_risk  # noqa: E402


def test_token_budget_and_prompt_too_long_recovery() -> None:
    assert estimate_tokens("abcd", token_char_ratio=2.0) == 2
    budget = TokenBudget(contextWindow=1000, threshold=0.8, reservedOutputTokens=100, usedTokens=750)
    assert budget.max_input_tokens == 700
    assert budget.exceeded is True

    prompt = "x" * 10_000
    recovered, event = PromptTooLongRecovery(max_chars=2_000).recover(prompt, budget)
    assert len(recovered) < len(prompt)
    assert event is not None
    assert event.type == RecoveryEventType.PROMPT_TOO_LONG
    assert "compacted" in recovered


def test_token_counter_matches_source_content_heuristics() -> None:
    counter = TokenCounter()
    assert counter.detect_content_type('{"name":"zhikun","enabled":true}') == "json"
    assert counter.detect_content_type("你好，智能体会检索代码并调用工具。你好，智能体会检索代码并调用工具。") == "chinese"
    assert counter.detect_content_type("def run():\n    value = 1\n    value += 2\n    value += 3\n    return value\n\nclass Worker:\n    pass") == "code"
    assert counter.detect_content_type("short") == "text"

    assert counter.estimate_text_with_type('{"name":"zhikun","enabled":true}', "json") == 16
    assert counter.estimate_text_with_type("plain english text" * 4, "text") == 18
    assert counter.estimate_image_tokens(1024, 768) == 1049
    assert counter.estimate_image_tokens(0, 768) == 85


def test_token_counter_precise_tokenizer_and_model_chinese_adjustment() -> None:
    exact_counter = TokenCounter(exact_counter=lambda text, model: 42, precise_enabled=True)
    assert exact_counter.estimate_text("hello world") == 42

    fallback_counter = TokenCounter(exact_counter=lambda text, model: -1, precise_enabled=True)
    assert fallback_counter.estimate_text("x" * 35) == 10

    mixed_chinese = "你好智能体" * 8
    default_tokens = int(len(mixed_chinese) / 3.5)
    counter_tokens = fallback_counter.estimate_text_for_model(mixed_chinese, "qwen", 3.5)
    assert counter_tokens
    assert counter_tokens > default_tokens


def test_token_counter_messages_and_query_loop_precomputed_usage() -> None:
    counter = TokenCounter()
    messages = [
        {"type": "user", "content": [{"type": "text", "text": "hello"}]},
        {"type": "assistant", "content": [{"type": "tool_use", "toolName": "read_file", "input": {"path": "README.md"}}]},
        {"type": "user", "content": [{"type": "tool_result", "content": "file content"}], "toolUseResult": "extra"},
    ]
    assert counter.estimate_messages(messages) == 35

    loop = QueryLoopState.start("s1", "hello", "qwen", context_window=128_000, threshold=0.85, ratio=3.5, used_tokens=123)
    assert loop.tokenBudget.usedTokens == 123


def test_query_loop_state_and_tool_call_tracker() -> None:
    loop = QueryLoopState.start("s1", "hello", "qwen3.7-max", context_window=128_000, threshold=0.85, ratio=2.5)
    assert loop.status == "running"
    assert loop.tokenBudget.usedTokens >= 1
    loop.finish()
    payload = loop.to_dict()
    assert payload["status"] == "completed"
    assert payload["stopReason"] == "end_turn"
    assert payload["phase"] == "completed"
    assert payload["transitions"]

    loop2 = QueryLoopState.start("s1", "hello", "qwen3.7-max", context_window=128_000, threshold=0.85, ratio=2.5)
    loop2.transition(QueryPhase.STREAMING, "delta")
    event = loop2.event("stream_delta", {"delta": "hello", "messageId": "m1"})
    tool = loop2.record_tool_call("t1", "read_file", {"path": "README.md"})
    loop2.update_tool_call("t1", "completed", {"content": "ok", "isError": False}, "done")
    assert event.to_dict()["seq"] == 1
    assert tool["status"] == "completed"
    assert loop2.to_dict()["events"][0]["type"] == "stream_delta"

    tracker = ToolCallTracker()
    call = tracker.record("read_file", "success", duration_ms=12)
    assert call["name"] == "read_file"
    assert tracker.to_list()[0]["durationMs"] == 12


def test_default_termination_strategy_records_permission_abort_and_max_turns() -> None:
    strategy = DefaultTerminationStrategy()

    wait_loop = QueryLoopState.start("s1", "need permission", "qwen", context_window=128_000, threshold=0.85, ratio=3.5)
    wait_loop.transition(QueryPhase.WAITING_PERMISSION, "permission:write_file")
    wait_decision = strategy.decide(wait_loop)
    wait_loop.set_termination_decision(wait_decision)
    assert wait_decision.action == TerminationAction.WAIT
    assert wait_decision.reason == "permission_wait"
    assert wait_loop.to_dict()["terminationDecision"]["action"] == "wait"

    abort_loop = QueryLoopState.start("s2", "abort", "qwen", context_window=128_000, threshold=0.85, ratio=3.5)
    abort_loop.abort("USER_INTERRUPT")
    abort_decision = strategy.decide(abort_loop)
    assert abort_decision.action == TerminationAction.ABORT
    assert abort_decision.stopReason == "aborted"

    max_loop = QueryLoopState.start("s3", "too many turns", "qwen", context_window=128_000, threshold=0.85, ratio=3.5)
    max_loop.turns = max_loop.maxTurns
    max_decision = strategy.decide(max_loop)
    assert max_decision.action == TerminationAction.STOP
    assert max_decision.stopReason == "max_turns"

    completed_loop = QueryLoopState.start("s4", "done", "qwen", context_window=128_000, threshold=0.85, ratio=3.5)
    end_decision = strategy.decide(completed_loop, requested_stop_reason="end_turn")
    assert end_decision.action == TerminationAction.STOP
    assert end_decision.stopReason == "end_turn"


def test_default_termination_strategy_handles_max_tokens_recovery_lifecycle() -> None:
    strategy = DefaultTerminationStrategy()
    loop = QueryLoopState.start("s5", "continue long answer", "qwen", context_window=128_000, threshold=0.85, ratio=3.5)

    first = strategy.decide(loop, requested_stop_reason="max_tokens", metadata={"maxTokens": 8192})
    assert first.action == TerminationAction.CONTINUE
    assert first.reason == "max_tokens_recovery"
    assert first.retryable is True
    assert loop.maxTokensOverride == 65_536
    assert first.metadata["recoveryAction"] == "escalate_max_tokens"
    assert first.metadata["maxTokensOverride"] == 65_536

    second = strategy.decide(loop, requested_stop_reason="length", metadata={"maxTokens": loop.maxTokensOverride})
    assert second.action == TerminationAction.CONTINUE
    assert second.reason == "max_tokens_recovery"
    assert loop.maxOutputTokensRecoveryCount == 1
    assert second.metadata["recoveryAction"] == "inject_recovery_prompt"
    assert "Resume directly" in second.metadata["recoveryPrompt"]

    loop.maxOutputTokensRecoveryCount = 3
    exhausted = strategy.decide(loop, requested_stop_reason="max_tokens")
    assert exhausted.action == TerminationAction.STOP
    assert exhausted.reason == "max_tokens_recovery_limit"
    assert exhausted.stopReason == "max_tokens"
    assert exhausted.retryable is False


def test_default_termination_strategy_withholds_recoverable_prompt_errors() -> None:
    strategy = DefaultTerminationStrategy()
    loop = QueryLoopState.start("s6", "huge prompt", "qwen", context_window=128_000, threshold=0.85, ratio=3.5)
    loop.add_withheld_error("prompt_too_long", "413 prompt too long", retryable=True)

    decision = strategy.decide(loop, requested_stop_reason="withhold")

    assert decision.action == TerminationAction.CONTINUE
    assert decision.reason == "withhold"
    assert decision.retryable is True
    assert decision.metadata["withheld"] is True
    assert decision.metadata["withheldErrorCount"] == 1
    assert decision.metadata["incrementalCollapseNeeded"] is True


def test_default_termination_strategy_preserves_model_stop_reason_metadata() -> None:
    strategy = DefaultTerminationStrategy()
    loop = QueryLoopState.start("s7", "blocked answer", "qwen", context_window=128_000, threshold=0.85, ratio=3.5)

    decision = strategy.decide(loop, requested_stop_reason="content_filter", metadata={"provider": "dashscope"})

    assert decision.action == TerminationAction.STOP
    assert decision.reason == "model_stop_reason"
    assert decision.stopReason == "content_filter"
    assert decision.metadata["modelStopReason"] == "content_filter"
    assert decision.metadata["provider"] == "dashscope"


def test_tool_priority_scheduler_and_conflict_detection() -> None:
    scheduler = ToolPriorityScheduler()
    calls = ["FileEdit", "Bash", "FileRead", "LspDefinition"]
    assert scheduler.sort_by_priority(calls) == ["FileRead", "LspDefinition", "Bash", "FileEdit"]
    assert scheduler.sort_by_priority(["GrepSearch", "FileRead", "ListDir"]) == ["GrepSearch", "FileRead", "ListDir"]
    assert scheduler.has_conflict(["FileRead", "FileEdit"], ["app.py", "app.py"]) is True


def test_context_collapse_preserves_user_prompts_and_releases_withheld_state() -> None:
    service = ContextCollapseService(protected_tail=1, threshold=20, keep=5)
    messages = [
        {"type": "user", "content": [{"type": "text", "text": "do not use redux"}]},
        {"type": "assistant", "content": [{"type": "text", "text": "x" * 100}]},
        {"type": "assistant", "content": [{"type": "text", "text": "recent answer"}]},
    ]
    result = service.collapse_messages(messages)
    assert result["collapsedCount"] == 1
    assert result["messages"][0]["content"][0]["text"] == "do not use redux"
    assert "collapsed" in result["messages"][1]["content"][0]["text"]

    loop = QueryLoopState.start("s1", "x" * 1000, "model", context_window=100, threshold=0.8, ratio=1.0)
    recovered, event = PromptTooLongRecovery(max_chars=100).recover("x" * 1000, loop.tokenBudget)
    assert recovered
    loop.add_recovery(event)
    assert loop.promptTooLongWithheld is True
    assert loop.incrementalCollapseNeeded is True
    loop.release_withheld()
    assert loop.promptTooLongWithheld is False
    assert loop.incrementalCollapseNeeded is False


def test_query_side_micro_compact_summary_and_abort_services() -> None:
    side = SideQueryService().query("summarize", "important context" * 50, max_tokens=20)
    assert side["status"] == "completed"
    assert len(side["answer"]) <= 80

    compact = MicroCompactService(max_tool_result_chars=10).compact_tool_results([{"type": "user", "toolUseResult": "x" * 50}])
    assert compact["compactedCount"] == 1
    assert "micro-compact" in compact["messages"][0]["toolUseResult"]

    summary = ToolResultSummarizer().summarize("Bash", "line1\nline2\n" + ("x" * 1000), max_chars=20)
    assert summary["toolName"] == "Bash"
    assert summary["truncated"] is True

    aborts = AbortController()
    record = aborts.abort("s1", "USER_INTERRUPT")
    assert record["reason"] == "USER_INTERRUPT"
    assert aborts.is_aborted("s1") is True


def test_query_loop_attaches_tool_result_summary_for_large_outputs() -> None:
    loop = QueryLoopState.start("s-summary", "run checks", "qwen", context_window=128_000, threshold=0.85, ratio=3.5)
    loop.record_tool_call("tool-1", "Bash", {"command": "pytest"})
    large_output = "\n".join(f"line {index}" for index in range(200))

    updated = loop.update_tool_call("tool-1", "completed", {"content": large_output, "isError": False}, "completed")

    assert updated is not None
    summary = updated["summary"]
    assert summary["toolName"] == "Bash"
    assert summary["truncated"] is True
    assert summary["originalChars"] == len(large_output)
    assert "line 0" in summary["summary"]
    assert len(updated["result"]["content"]) == len(large_output)


def test_context_cascade_runs_five_named_layers_and_updates_budget() -> None:
    budget = TokenBudget(contextWindow=500, threshold=0.8, reservedOutputTokens=100, usedTokens=600)
    prompt = "写一个实现方案\n" + ("p" * 5_000)
    messages = [
        {"type": "assistant", "content": [{"type": "text", "text": "old context " + ("a" * 3_000)}]},
        {"type": "user", "toolUseResult": "tool output\n" + ("b" * 5_000)},
        {"type": "assistant", "content": [{"type": "text", "text": "recent answer"}]},
    ]

    result = ContextCascadeService(
        max_prompt_chars=1_200,
        max_message_chars=600,
        micro_compact=MicroCompactService(max_tool_result_chars=300),
        context_collapse=ContextCollapseService(protected_tail=1, threshold=200, keep=60),
    ).apply(prompt, messages, budget, token_char_ratio=2.0, protected_tail=1, force=True)

    assert [layer["name"] for layer in result["layers"]] == [
        "snip_selection",
        "micro_compact",
        "auto_compact",
        "collapse_drain",
        "reactive_compact",
    ]
    assert result["changed"] is True
    assert len(result["prompt"]) < len(prompt)
    assert result["messages"][1]["toolUseResult"] != messages[1]["toolUseResult"]
    assert result["usedTokens"] < budget.usedTokens
    assert any(event.type == RecoveryEventType.COMPACT_APPLIED for event in result["events"])
    assert any(event.type == RecoveryEventType.PROMPT_TOO_LONG for event in result["events"])


def test_context_cascade_strips_media_blocks_during_prompt_too_long_recovery() -> None:
    budget = TokenBudget(contextWindow=700, threshold=0.8, reservedOutputTokens=100, usedTokens=1_200)
    messages = [
        {
            "type": "user",
            "content": [
                {"type": "text", "text": "please inspect the image"},
                {"type": "image", "width": 2048, "height": 2048, "source": "data:image/png;base64," + ("x" * 8_000)},
            ],
        }
    ]

    result = ContextCascadeService(max_prompt_chars=600).apply(
        "prompt",
        messages,
        budget,
        token_char_ratio=2.0,
        force=True,
        recovery_cause="http_413",
    )

    stripped_block = result["messages"][0]["content"][1]
    assert stripped_block["type"] == "text"
    assert "media stripped" in stripped_block["text"]
    reactive = [layer for layer in result["layers"] if layer["name"] == "reactive_compact"][0]
    assert reactive["metadata"]["mediaStrippedCount"] == 1
    assert reactive["metadata"]["recoveryCause"] == "http_413"
    assert any(event.metadata.get("mediaStrippedCount") == 1 for event in result["events"])


def test_prompt_too_long_recovery_records_http_413_cause() -> None:
    budget = TokenBudget(contextWindow=300, threshold=0.8, reservedOutputTokens=100, usedTokens=500)

    recovered, event = PromptTooLongRecovery(max_chars=200).recover("x" * 2_000, budget, cause="http_413")

    assert len(recovered) < 2_000
    assert event is not None
    assert event.type == RecoveryEventType.PROMPT_TOO_LONG
    assert event.metadata["cause"] == "http_413"
    assert event.metadata["recoveryStage"] == "prompt_too_long"


def test_command_security_flags_secret_env_and_windows_root_delete() -> None:
    secret_env = command_risk("Get-ChildItem Env:OPENAI_API_KEY")
    assert secret_env.level == BlockLevel.HIGH_RISK_ASK
    assert "secret" in secret_env.reason.lower()

    root_delete = command_risk("Remove-Item -Recurse -Force C:\\")
    assert root_delete.level == BlockLevel.ABSOLUTE_DENY
