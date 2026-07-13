import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from zhikun_py.llm_runtime import (  # noqa: E402
    ApiCircuitBreaker,
    ApiKeyRotationManager,
    CircuitState,
    LlmErrorClassifier,
    LlmProviderRegistry,
    ModelDegradationChain,
    ModelAwareRetryPolicy,
    ModelCapabilityRegistry,
    RetryConfig,
)


def test_model_capability_registry_matches_original_contract() -> None:
    registry = ModelCapabilityRegistry()
    claude = registry.get_capability("claude-sonnet-4-6")
    assert claude.contextWindow == 200_000
    assert claude.outputMaxTokens == 64_000
    assert claude.supportsToolUse is True
    assert claude.supportsVision is True
    assert claude.supportsCache is True

    qwen = registry.get_capability("qwen3.7-max")
    assert qwen.contextWindow == 131_072
    assert qwen.tokenCharRatio == 2.5
    assert qwen.supportsVision is False
    assert registry.get_capability("unknown").modelId == "default"
    assert registry.compact_threshold("claude-sonnet-4-6") == 0.90
    assert registry.compact_threshold("qwen3.7-max") == 0.85
    assert registry.buffer_tokens("unknown") == 12_800


def test_provider_alias_and_fallback_chain() -> None:
    registry = LlmProviderRegistry(models=["qwen-plus", "qwen3.7-plus", "qwen3.7-max"])
    assert registry.resolve_model_alias("light") == "qwen3.7-plus"
    assert registry.resolve_model_alias("standard") == "qwen3.7-plus"
    assert registry.resolve_model_alias("premium") == "qwen3.7-max"
    assert registry.resolve_model_alias("custom-model") == "custom-model"
    assert registry.get_default_model() == "qwen3.7-max"
    assert registry.get_lightweight_model() == "qwen3.7-plus"
    assert registry.resolve_classifier_model()


def test_model_aware_retry_policy_and_circuit_breaker() -> None:
    policy = ModelAwareRetryPolicy(jitter_factor=0.0)
    default_config = policy.get_retry_config("unknown")
    assert default_config.maxRetries == 10
    assert default_config.baseDelayMs == 500
    assert policy.calculate_delay_ms("unknown", 0) == 500
    claude_config = policy.get_retry_config("claude-sonnet-4-6")
    assert claude_config.maxRetries == 5
    assert claude_config.baseDelayMs == 60_000
    assert policy.calculate_delay_ms("claude-sonnet-4-6", 0) == 30_000

    breaker = ApiCircuitBreaker()
    assert breaker.allow_request() is True
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False
    breaker.last_failure_time = time.time() - 61
    assert breaker.allow_request() is True
    assert breaker.state == CircuitState.HALF_OPEN
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 0


def test_model_aware_retry_policy_matches_source_model_rules_and_retry_after() -> None:
    policy = ModelAwareRetryPolicy(jitter_factor=0.0)

    qwen = policy.get_retry_config("qwen3.7-max")
    assert qwen.maxRetries == 8
    assert qwen.baseDelayMs == 10_000
    assert qwen.respectRetryAfter is False
    assert policy.calculate_delay_ms("qwen3.7-max", 1) == 20_000
    assert policy.calculate_delay_ms("qwen3.7-max", 3) == 30_000
    assert policy.should_respect_retry_after("qwen3.7-max", 60_000) is False
    assert policy.resolve_delay_ms("qwen3.7-max", 0, retry_after_ms=60_000) == 10_000

    deepseek = policy.get_retry_config("deepseek-v4-pro")
    assert deepseek.maxRetries == 6
    assert deepseek.baseDelayMs == 20_000
    assert deepseek.respectRetryAfter is True
    assert policy.should_respect_retry_after("deepseek-v4-pro", 45_000) is True
    assert policy.resolve_delay_ms("deepseek-v4-pro", 0, retry_after_ms=45_000) == 45_000

    custom = ModelAwareRetryPolicy(
        model_configs={
            "glm": RetryConfig(maxRetries=2, baseDelayMs=750, respectRetryAfter=False),
        },
        jitter_factor=0.0,
    )
    assert custom.get_retry_config("glm-5.1").maxRetries == 2
    assert custom.calculate_delay_ms("glm-5.1", 2) == 3_000


def test_error_classifier_and_degradation_chain_are_configurable() -> None:
    classifier = LlmErrorClassifier()
    assert classifier.classify(status_code=529)["type"] == "overloaded"
    assert classifier.classify(status_code=503)["retryable"] is True
    assert classifier.classify(status_code=429)["type"] == "rate_limit"
    assert classifier.classify(status_code=413)["fallbackAllowed"] is False
    assert classifier.classify(status_code=400, body="input is too long")["type"] == "prompt_too_long"
    assert classifier.classify(status_code=401)["retryable"] is False
    assert classifier.classify(error=TimeoutError("timed out"))["type"] == "timeout"

    chain = ModelDegradationChain()
    assert chain.get_next_fallback("qwen3.7-max", 0) == "qwen3.7-plus"
    assert chain.get_next_fallback("qwen3.7-max", 1) == "deepseek-v4-flash"
    assert chain.get_next_fallback("qwen3.7-max", 3) is None
    assert chain.chain_for("claude-sonnet-4-6") == ["qwen3.7-max", "deepseek-v4-flash"]

    custom_chain = ModelDegradationChain({"glm-5.1": ["qwen3.7-plus", "deepseek-v4-flash"]}, max_depth=1)
    assert custom_chain.get_next_fallback("glm-5.1", 0) == "qwen3.7-plus"
    assert custom_chain.get_next_fallback("glm-5.1", 1) is None


def test_api_key_rotation_manager_skips_failed_keys() -> None:
    manager = ApiKeyRotationManager(["k1", "k2"])
    assert manager.next_key() == "k1"
    assert manager.next_key() == "k2"
    manager.record_failure("k1")
    manager.record_failure("k1")
    manager.record_failure("k1")
    assert manager.next_key() == "k2"
    manager.record_success("k1")
    assert manager.next_key() in {"k1", "k2"}
