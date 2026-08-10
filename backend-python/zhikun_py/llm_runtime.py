from __future__ import annotations

import itertools
import random
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelCapability:
    modelId: str
    contextWindow: int = 128_000
    outputMaxTokens: int = 8192
    tokenCharRatio: float = 3.5
    supportsToolUse: bool = True
    supportsVision: bool = False
    supportsStreaming: bool = True
    supportsCache: bool = False
    supportsThinking: bool = False


DEFAULT_CAPABILITY = ModelCapability("default")
BUILTIN_CAPABILITIES = {
    "claude-sonnet-4-6": ModelCapability(
        "claude-sonnet-4-6",
        contextWindow=200_000,
        outputMaxTokens=64_000,
        tokenCharRatio=3.5,
        supportsToolUse=True,
        supportsVision=True,
        supportsStreaming=True,
        supportsCache=True,
        supportsThinking=True,
    ),
    "qwen3.7-max": ModelCapability(
        "qwen3.7-max",
        contextWindow=131_072,
        outputMaxTokens=8192,
        tokenCharRatio=2.5,
        supportsToolUse=True,
        supportsVision=False,
        supportsStreaming=True,
        supportsCache=False,
        supportsThinking=True,
    ),
    "qwen3.7-plus": ModelCapability("qwen3.7-plus", contextWindow=128_000, tokenCharRatio=2.5, supportsThinking=True),
    "qwen-plus": ModelCapability("qwen-plus", contextWindow=128_000, tokenCharRatio=2.5),
}


class ModelCapabilityRegistry:
    def __init__(self, overrides: dict[str, dict[str, Any]] | None = None) -> None:
        self.capabilities = dict(BUILTIN_CAPABILITIES)
        for model_id, data in (overrides or {}).items():
            self.capabilities[model_id] = ModelCapability(modelId=model_id, **data)

    def get_capability(self, model_id: str | None) -> ModelCapability:
        if not model_id or not model_id.strip():
            return DEFAULT_CAPABILITY
        return self.capabilities.get(model_id, DEFAULT_CAPABILITY)

    def compact_threshold(self, model_id: str | None) -> float:
        cap = self.get_capability(model_id)
        if cap.contextWindow >= 200_000:
            return 0.90
        if cap.contextWindow >= 128_000:
            return 0.85
        return 0.80

    def buffer_tokens(self, model_id: str | None) -> int:
        return int(self.get_capability(model_id).contextWindow * 0.10)

    def list_models(self) -> list[dict[str, Any]]:
        return [asdict(cap) for cap in self.capabilities.values()]


BUILTIN_ALIASES = {
    "light": "qwen3.7-plus",
    "standard": "qwen3.7-plus",
    "premium": "qwen3.7-max",
}


class LlmProviderRegistry:
    def __init__(self, models: list[str] | None = None, aliases: dict[str, str] | None = None, default_model: str = "qwen3.7-max") -> None:
        self.models = models or ["qwen-plus", "qwen3.7-plus", "qwen3.7-max"]
        self.aliases = {**BUILTIN_ALIASES, **(aliases or {})}
        self.default_model = default_model

    def resolve_model_alias(self, model: str | None) -> str:
        if not model:
            return self.default_model
        return self.aliases.get(model, model)

    def get_default_model(self) -> str:
        return self.default_model if self.default_model in self.models else self.models[0]

    def get_lightweight_model(self) -> str:
        resolved = self.resolve_model_alias("light")
        return resolved if resolved in self.models else self.get_default_model()

    def resolve_classifier_model(self) -> str:
        for candidate in ("light", "standard", self.default_model):
            resolved = self.resolve_model_alias(candidate)
            if resolved:
                return resolved
        return self.get_default_model()

    def list_available_models(self) -> list[str]:
        return list(self.models)


@dataclass(frozen=True, slots=True)
class RetryConfig:
    maxRetries: int
    baseDelayMs: int
    respectRetryAfter: bool = True
    maxDelayMs: int = 30_000
    jitterFactor: float = 0.25


class ModelAwareRetryPolicy:
    DEFAULT_CONFIG = RetryConfig(maxRetries=10, baseDelayMs=500, respectRetryAfter=True)
    DEFAULT_MODEL_CONFIGS = {
        "claude": RetryConfig(maxRetries=5, baseDelayMs=60_000, respectRetryAfter=True),
        "qwen": RetryConfig(maxRetries=8, baseDelayMs=10_000, respectRetryAfter=False),
        "deepseek": RetryConfig(maxRetries=6, baseDelayMs=20_000, respectRetryAfter=True),
    }

    def __init__(self, model_configs: dict[str, RetryConfig] | None = None, jitter_factor: float | None = None) -> None:
        self.model_configs = {**self.DEFAULT_MODEL_CONFIGS, **(model_configs or {})}
        self.jitter_factor = jitter_factor

    def get_retry_config(self, model_id: str | None) -> RetryConfig:
        if not model_id:
            return self.DEFAULT_CONFIG
        model_lower = model_id.lower()
        for key, config in self.model_configs.items():
            if key.lower() in model_lower:
                return config
        return self.DEFAULT_CONFIG

    def calculate_delay_ms(self, model_id: str | None, attempt: int) -> int:
        config = self.get_retry_config(model_id)
        base = min(config.baseDelayMs * (2 ** max(0, attempt)), config.maxDelayMs)
        jitter_factor = config.jitterFactor if self.jitter_factor is None else self.jitter_factor
        return int(base + base * max(0.0, jitter_factor) * random.random())

    def should_respect_retry_after(self, model_id: str | None, retry_after_ms: int | None) -> bool:
        return bool(retry_after_ms and retry_after_ms > 0 and self.get_retry_config(model_id).respectRetryAfter)

    def resolve_delay_ms(self, model_id: str | None, attempt: int, retry_after_ms: int | None = None) -> int:
        if self.should_respect_retry_after(model_id, retry_after_ms):
            return int(retry_after_ms or 0)
        return self.calculate_delay_ms(model_id, attempt)


class LlmErrorClassifier:
    RETRYABLE_TYPES = {"overloaded", "rate_limit", "timeout", "network", "server"}

    def classify(self, status_code: int | None = None, error: Exception | None = None, body: str | None = None) -> dict[str, Any]:
        text = " ".join(str(item or "") for item in (body, error)).lower()
        if status_code in {529, 503}:
            return self._result("overloaded", status_code, body or "Service overloaded")
        if status_code == 429:
            return self._result("rate_limit", status_code, body or "Rate limited")
        if status_code == 413 or (status_code == 400 and self._looks_prompt_too_long(text)):
            return self._result("prompt_too_long", status_code, body or "Prompt too long")
        if status_code in {401, 403}:
            return self._result("auth", status_code, body or "Authentication failed")
        if status_code is not None and status_code >= 500:
            return self._result("server", status_code, body or "Server error")
        if isinstance(error, TimeoutError) or "timeout" in text or "timed out" in text:
            return self._result("timeout", status_code, str(error) if error else body)
        if error is not None and any(marker in text for marker in ("connection", "network", "transport", "dns")):
            return self._result("network", status_code, str(error))
        if status_code is not None and status_code >= 400:
            return self._result("client", status_code, body or "Client error")
        return self._result("unknown", status_code, str(error) if error else body or "Unknown LLM failure")

    def _looks_prompt_too_long(self, text: str) -> bool:
        return any(marker in text for marker in ("max_tokens", "token limit", "request too large", "input is too long", "prompt too long", "too large"))

    def _result(self, category: str, status_code: int | None, message: str | None) -> dict[str, Any]:
        retryable = category in self.RETRYABLE_TYPES
        return {
            "type": category,
            "retryable": retryable,
            "fallbackAllowed": category not in {"auth", "prompt_too_long", "client"},
            "statusCode": status_code,
            "message": message or category,
        }

    def is_retryable(self, category: str) -> bool:
        return category in self.RETRYABLE_TYPES


class ModelDegradationChain:
    DEFAULT_CHAINS = {
        "claude-sonnet-4-6": ["qwen3.7-max", "deepseek-v4-flash"],
        "qwen3.7-max": ["qwen3.7-plus", "deepseek-v4-flash"],
        "qwen3.7-plus": ["deepseek-v4-flash", "qwen3.7-max"],
    }

    def __init__(self, chains: dict[str, list[str]] | None = None, max_depth: int = 3) -> None:
        self.chains = {**self.DEFAULT_CHAINS, **(chains or {})}
        self.max_depth = max(0, int(max_depth))

    def get_next_fallback(self, model_id: str | None, degradation_level: int) -> str | None:
        if not model_id or degradation_level < 0 or degradation_level >= self.max_depth:
            return None
        chain = self.chain_for(model_id)
        if degradation_level >= len(chain):
            return None
        return chain[degradation_level]

    def has_fallback(self, model_id: str | None, degradation_level: int) -> bool:
        return self.get_next_fallback(model_id, degradation_level) is not None

    def chain_for(self, model_id: str | None) -> list[str]:
        return list(self.chains.get(model_id or "", []))

    def fallback_chain(self, model_id: str | None) -> list[str]:
        return self.chain_for(model_id)[: self.max_depth]


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ApiCircuitBreaker:
    failure_threshold = 3
    open_timeout_seconds = 60

    def __init__(self) -> None:
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time = 0.0

    def allow_request(self) -> bool:
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.open_timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self) -> None:
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN or self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def reset(self) -> None:
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time = 0.0


class ApiKeyRotationManager:
    def __init__(self, keys: list[str] | None = None) -> None:
        self.keys = [key for key in (keys or []) if key]
        self._cycle = itertools.cycle(self.keys) if self.keys else None
        self.failures: dict[str, int] = {}

    def next_key(self) -> str | None:
        if not self._cycle:
            return None
        for _ in range(len(self.keys)):
            key = next(self._cycle)
            if self.failures.get(key, 0) < 3:
                return key
        return None

    def record_failure(self, key: str) -> None:
        self.failures[key] = self.failures.get(key, 0) + 1

    def record_success(self, key: str) -> None:
        self.failures[key] = 0
