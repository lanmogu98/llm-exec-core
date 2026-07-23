"""Legacy async LLM client behavior."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple
from urllib.parse import SplitResult, urlsplit

import httpx
import jsonschema

from .config import (
    ModelCapabilities,
    Pricing,
    PricingCalculationError,
    PricingContext,
    PricingContextRequiredError,
    PricingCost,
    PricingSchedule,
    _resolve_declared_pricing,
    calculate_pricing_cost,
    get_model_details,
    get_supported_models,
)
from .constants import (
    API_REQUEST_TIMEOUT_SECONDS,
    INITIAL_RETRY_DELAY_SECONDS,
    MAX_API_RETRIES,
    MAX_REQUESTS_PER_MINUTE,
    MIN_REQUEST_INTERVAL_SECONDS,
    RATE_LIMIT_WARNINGS_ENABLED,
    RESPONSE_CACHE_ENABLED,
    RESPONSE_CACHE_MAX_SIZE,
    RESPONSE_CACHE_TTL_SECONDS,
)
from .tokens import estimate_tokens
from .types import ExecutionMetadata, LLMResult, TokenUsage

logger = logging.getLogger(__name__)

_PROTECTED_REQUEST_FIELDS = {"model", "messages", "stream"}
_STRUCTURED_OUTPUT_MODES = {"require", "prefer", "off"}
_KNOWN_CAPABILITY_CONTROL_FIELDS = {
    "reasoning_effort",
    "thinking",
    "reasoning",
    "include_reasoning",
    "verbosity",
}
_MISSING = object()
_ASCII_EDGE_WHITESPACE = " \t\n\r\v\f"


@dataclass(frozen=True, slots=True)
class _PricingUsageParts:
    prompt_tokens: int
    completion_tokens: int
    ordinary_input_tokens: int
    cache_read_input_tokens: int
    cache_write_input_tokens: int


def _usage_token(
    values: Dict[str, Any], field_name: str, *, required: bool
) -> int:
    if field_name not in values:
        if required:
            raise PricingCalculationError(
                f"OpenAI Chat Completions usage requires {field_name}."
            )
        return 0
    value = values[field_name]
    if type(value) is not int or value < 0:
        raise PricingCalculationError(
            f"OpenAI Chat Completions usage {field_name} must be a plain "
            "nonnegative integer."
        )
    return value


def _parse_openai_chat_usage(
    usage: Any,
    cache_mode: Literal["none", "implicit", "explicit"],
) -> _PricingUsageParts:
    if type(usage) is not dict:
        raise PricingCalculationError(
            "OpenAI Chat Completions usage must be a JSON object."
        )

    prompt_tokens = _usage_token(usage, "prompt_tokens", required=True)
    completion_tokens = _usage_token(usage, "completion_tokens", required=True)
    details_value = usage.get("prompt_tokens_details", _MISSING)

    if cache_mode == "none":
        if details_value is _MISSING or details_value is None:
            details = {}
        elif type(details_value) is dict:
            details = details_value
        else:
            raise PricingCalculationError(
                "prompt_tokens_details must be a JSON object or null for "
                "none cache mode."
            )
        cache_read = _usage_token(details, "cached_tokens", required=False)
        cache_write = _usage_token(
            details, "cache_creation_input_tokens", required=False
        )
        if cache_read != 0 or cache_write != 0:
            raise PricingCalculationError(
                "none cache mode contradicts nonzero cache token buckets."
            )
    elif cache_mode == "implicit":
        if type(details_value) is not dict:
            raise PricingCalculationError(
                "implicit cache mode requires prompt_tokens_details."
            )
        details = details_value
        cache_read = _usage_token(details, "cached_tokens", required=True)
        cache_write = _usage_token(
            details, "cache_creation_input_tokens", required=False
        )
        if cache_write != 0:
            raise PricingCalculationError(
                "implicit cache mode contradicts cache creation tokens."
            )
    else:
        if type(details_value) is not dict:
            raise PricingCalculationError(
                "explicit cache mode requires prompt_tokens_details."
            )
        details = details_value
        cache_read = _usage_token(details, "cached_tokens", required=True)
        cache_write = _usage_token(
            details, "cache_creation_input_tokens", required=True
        )

    if cache_read + cache_write > prompt_tokens:
        raise PricingCalculationError(
            "Cache token buckets cannot exceed prompt_tokens."
        )
    return _PricingUsageParts(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ordinary_input_tokens=prompt_tokens - cache_read - cache_write,
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
    )


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not valid JSON.")


def _resolve_api_key(primary_name: str, aliases: list[str]) -> str:
    declared_names = [primary_name, *aliases]
    for environment_name in declared_names:
        value = os.environ.get(environment_name)
        if value is None or not any(
            not character.isspace() for character in value
        ):
            continue
        if any(
            ord(character) <= 0x1F or ord(character) == 0x7F
            for character in value
        ):
            raise ValueError(
                "API key environment variable "
                f"'{environment_name}' contains a control character."
            )
        return value

    if not aliases:
        raise ValueError(
            f"API key environment variable '{primary_name}' is not set."
        )

    rendered_names = ", ".join(
        f"'{environment_name}'" for environment_name in declared_names
    )
    raise ValueError(
        f"API key environment variables {rendered_names} "
        "are not set or are blank."
    )


def _invalid_endpoint_override(
    environment_name: str, reason: str
) -> ValueError:
    return ValueError(
        "Endpoint override environment variable "
        f"'{environment_name}' is invalid ({reason})."
    )


def _parse_endpoint_override(
    candidate: str, environment_name: str
) -> SplitResult:
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        parsed = None
    if parsed is None:
        raise _invalid_endpoint_override(environment_name, "URL syntax")
    return parsed


def _resolve_api_url(static_url: str, environment_name: str | None) -> str:
    if environment_name is None:
        return static_url

    value = os.environ.get(environment_name)
    if value is None or not any(
        not character.isspace() for character in value
    ):
        return static_url

    candidate = value.strip(_ASCII_EDGE_WHITESPACE)
    if any(
        ord(character) <= 0x20
        or ord(character) == 0x7F
        or character.isspace()
        or character == "\\"
        for character in candidate
    ):
        raise _invalid_endpoint_override(
            environment_name, "forbidden character"
        )
    if "?" in candidate or "#" in candidate:
        raise _invalid_endpoint_override(
            environment_name, "query or fragment delimiter"
        )

    parsed = _parse_endpoint_override(candidate, environment_name)
    if parsed.scheme.lower() != "https":
        raise _invalid_endpoint_override(environment_name, "HTTPS required")
    if not parsed.netloc:
        raise _invalid_endpoint_override(
            environment_name, "authority and hostname required"
        )
    if "@" in parsed.netloc:
        raise _invalid_endpoint_override(
            environment_name, "userinfo is not allowed"
        )
    if parsed.netloc.endswith(":"):
        raise _invalid_endpoint_override(environment_name, "invalid port")

    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        hostname = None
        port = None
        valid_authority = False
    else:
        valid_authority = True
    if not valid_authority:
        raise _invalid_endpoint_override(
            environment_name, "invalid authority or port"
        )
    if not hostname:
        raise _invalid_endpoint_override(
            environment_name, "authority and hostname required"
        )
    if port is not None and not 1 <= port <= 65535:
        raise _invalid_endpoint_override(environment_name, "invalid port")

    normalized = candidate.rstrip("/")
    normalized_path = _parse_endpoint_override(
        normalized, environment_name
    ).path
    if normalized_path.endswith("/chat/completions"):
        return normalized
    if normalized_path.endswith("/v1"):
        return f"{normalized}/chat/completions"
    raise _invalid_endpoint_override(
        environment_name, "unsupported endpoint path"
    )


class StructuredOutputValidationError(ValueError):
    """Raised when a structured response fails client-side validation."""


class ResponseCache:
    """LRU cache for LLM responses with TTL support."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, Tuple[str, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _make_key(self, request_payload: Mapping[str, Any]) -> str:
        """Create a cache key from a deterministic request payload."""
        content = json.dumps(
            request_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, request_payload: Mapping[str, Any]) -> Optional[str]:
        """Get cached response if exists and not expired."""
        key = self._make_key(request_payload)

        if key not in self._cache:
            self._misses += 1
            return None

        response, timestamp = self._cache[key]

        if (
            self._ttl_seconds > 0
            and time.time() - timestamp > self._ttl_seconds
        ):
            del self._cache[key]
            self._misses += 1
            return None

        self._cache.move_to_end(key)
        self._hits += 1
        return response

    def set(self, request_payload: Mapping[str, Any], response: str) -> None:
        """Store response in cache."""
        key = self._make_key(request_payload)

        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (response, time.time())

    def delete(self, request_payload: Mapping[str, Any]) -> None:
        """Delete a cached response if present."""
        key = self._make_key(request_payload)
        self._cache.pop(key, None)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self._cache),
            "max_size": self._max_size,
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


class LLMClient:
    """Client for interacting with the LLM API (Async)."""

    @staticmethod
    def get_supported_models(
        config_source: Path | Dict[str, Any] | None = None,
    ) -> list[str]:
        """Return models from a required explicit Path or dictionary catalog.

        Omitting ``config_source`` or passing ``None`` raises ``ValueError``.
        """
        return get_supported_models(config_source)

    def __init__(
        self,
        model_name: str,
        thinking_level: str | None = None,
        config_source: Path | Dict[str, Any] | None = None,
        *,
        pricing_context: PricingContext | Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initialize the LLM client for a specific model.
        The client automatically determines the service provider and settings.

        Args:
            model_name: The name of the model to use.
            thinking_level: Optional thinking/reasoning level override.
                For Gemini 3+, maps to reasoning_effort.
            config_source: Required complete config Path or raw dictionary.
                Omitting it or passing None raises ValueError.
            pricing_context: Exact context required by rich pricing schedules.
        """
        provider_name, provider_settings, model_details = get_model_details(
            model_name, config_source
        )
        if (
            thinking_level
            and model_details.capabilities is not None
            and "reasoning_effort"
            not in model_details.capabilities.reasoning_controls
        ):
            raise ValueError(
                f"{model_name} thinking_level requires reasoning_effort "
                "in model capabilities."
            )
        self._thinking_level = thinking_level

        self.api_key = _resolve_api_key(
            provider_settings.api_key_env_var,
            provider_settings.api_key_env_aliases,
        )

        self.context_window = (
            model_details.context_window
            if model_details.context_window is not None
            else provider_settings.context_window
        )
        self.max_tokens = (
            model_details.max_tokens
            if model_details.max_tokens is not None
            else provider_settings.max_tokens
        )
        self.model_name = model_name
        self.model = model_details.id
        self.capabilities = model_details.capabilities
        self.provider_name = provider_name
        self.pricing = model_details.pricing
        self._provider_pricing_currency = provider_settings.pricing_currency
        self.pricing_currency = provider_settings.pricing_currency
        self._usage_currency: str | None = (
            None
            if isinstance(self.pricing, PricingSchedule)
            else provider_settings.pricing_currency
        )
        self._pricing_context: PricingContext | None = None
        if isinstance(self.pricing, PricingSchedule):
            if pricing_context is None:
                raise PricingContextRequiredError(
                    f"Pricing context is required for model '{model_name}'."
                )
            resolved_pricing_context: PricingContext
            if isinstance(pricing_context, PricingContext):
                resolved_pricing_context = pricing_context.model_copy(
                    deep=True
                )
            elif isinstance(pricing_context, Mapping):
                resolved_pricing_context = PricingContext.model_validate(
                    deepcopy(dict(pricing_context))
                )
            else:
                raise PricingContextRequiredError(
                    f"Pricing context is required for model '{model_name}'."
                )
            if resolved_pricing_context.request_mode == "batch":
                raise PricingCalculationError(
                    "LLMClient supports realtime pricing contexts only."
                )
            self._pricing_context = resolved_pricing_context
        self.temperature = (
            model_details.temperature
            if model_details.temperature is not None
            else provider_settings.temperature
        )
        self.output_token_field = (
            model_details.output_token_field
            if model_details.output_token_field is not None
            else provider_settings.output_token_field
        )
        self.max_tokens_retry = provider_settings.max_tokens_retry

        self.api_url = _resolve_api_url(
            provider_settings.api_base_url,
            provider_settings.api_base_url_env_var,
        )
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        provider_request_overrides = self._normalize_request_options(
            provider_settings.request_overrides
        )
        model_request_overrides = self._normalize_request_options(
            model_details.request_overrides
        )
        provider_stream_options = provider_request_overrides.pop(
            "stream_options", _MISSING
        )
        model_stream_options = model_request_overrides.pop(
            "stream_options", _MISSING
        )
        self.request_overrides = provider_request_overrides
        self.request_overrides.update(model_request_overrides)
        stream_options = self._merge_stream_options(
            provider_stream_options, model_stream_options
        )
        if stream_options is not _MISSING:
            self.request_overrides["stream_options"] = stream_options
        if self._thinking_level:
            self.request_overrides["reasoning_effort"] = self._thinking_level

        self.token_usage: Dict[str, Any] = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "requests": [],
            "process_times": {"total_time": 0, "request_times": []},
            "cost": {"input_cost": 0, "output_cost": 0, "total_cost": 0},
        }

        if provider_settings.rate_limit:
            self._min_interval = (
                provider_settings.rate_limit.min_interval_seconds
            )
            self._max_rpm = (
                provider_settings.rate_limit.max_requests_per_minute
            )
        else:
            self._min_interval = MIN_REQUEST_INTERVAL_SECONDS
            self._max_rpm = MAX_REQUESTS_PER_MINUTE

        self._last_request_time = 0.0
        self._request_timestamps: deque[float] = deque(
            maxlen=self._max_rpm if self._max_rpm > 0 else 100
        )

        self._cache_enabled = RESPONSE_CACHE_ENABLED
        self._cache = ResponseCache(
            max_size=RESPONSE_CACHE_MAX_SIZE,
            ttl_seconds=RESPONSE_CACHE_TTL_SECONDS,
        )

        self._async_client: Optional[httpx.AsyncClient] = None

    @property
    def pricing_context(self) -> PricingContext | None:
        if self._pricing_context is None:
            return None
        return self._pricing_context.model_copy(deep=True)

    async def __aenter__(self):
        """Context manager entry."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=API_REQUEST_TIMEOUT_SECONDS
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=API_REQUEST_TIMEOUT_SECONDS
            )
        return self._async_client

    async def close(self):
        """Manually close the client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    async def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits (Async)."""
        current_time = time.time()

        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_interval:
            wait_time = self._min_interval - time_since_last
            if RATE_LIMIT_WARNINGS_ENABLED:
                logger.warning(
                    "Rate limiting: waiting %.2fs (min interval)",
                    wait_time,
                )
            await asyncio.sleep(wait_time)
            current_time = time.time()

        if self._max_rpm > 0:
            cutoff_time = current_time - 60
            while (
                self._request_timestamps
                and self._request_timestamps[0] < cutoff_time
            ):
                self._request_timestamps.popleft()

            if len(self._request_timestamps) >= self._max_rpm:
                wait_time = self._request_timestamps[0] + 60 - current_time
                if wait_time > 0:
                    if RATE_LIMIT_WARNINGS_ENABLED:
                        logger.warning(
                            "Rate limiting: waiting %.2fs (per-minute limit)",
                            wait_time,
                        )
                    await asyncio.sleep(wait_time)
                    current_time = time.time()

        self._last_request_time = current_time
        self._request_timestamps.append(current_time)

    def _normalize_request_options(
        self,
        options: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        if not options:
            return {}

        normalized: Dict[str, Any] = {}
        if "extra_body" in options:
            extra_body = options["extra_body"]
            if isinstance(extra_body, Mapping):
                normalized.update(deepcopy(dict(extra_body)))
            else:
                normalized["extra_body"] = deepcopy(extra_body)

        for key, value in options.items():
            if key == "extra_body":
                continue
            normalized[key] = deepcopy(value)

        return normalized

    def _validate_no_protected_fields(
        self,
        options: Mapping[str, Any],
        source: str,
    ) -> None:
        protected = _PROTECTED_REQUEST_FIELDS.intersection(options)
        if protected:
            field_list = ", ".join(sorted(protected))
            raise ValueError(f"{field_list} is a protected {source} field.")

    def _merge_stream_options(
        self,
        provider_stream_options: Any,
        request_stream_options: Any,
    ) -> Any:
        if request_stream_options is _MISSING:
            if provider_stream_options is _MISSING:
                return _MISSING
            return deepcopy(provider_stream_options)
        if provider_stream_options is _MISSING:
            return deepcopy(request_stream_options)
        if isinstance(provider_stream_options, Mapping) and isinstance(
            request_stream_options, Mapping
        ):
            merged = deepcopy(dict(provider_stream_options))
            for key, value in request_stream_options.items():
                if isinstance(merged.get(key), Mapping) and isinstance(
                    value, Mapping
                ):
                    merged[key] = self._merge_stream_options(
                        merged[key], value
                    )
                else:
                    merged[key] = deepcopy(value)
            return merged
        return deepcopy(request_stream_options)

    def _build_request_payload(
        self,
        prompt: str,
        stream: bool,
        request_options: Mapping[str, Any] | None,
    ) -> Tuple[Dict[str, Any], set[str]]:
        provider_options = self._normalize_request_options(
            self.request_overrides
        )
        per_call_options = self._normalize_request_options(request_options)

        self._validate_no_protected_fields(
            provider_options, "provider request override"
        )
        self._validate_no_protected_fields(per_call_options, "request option")

        provider_stream_options = provider_options.pop(
            "stream_options", _MISSING
        )
        request_stream_options = per_call_options.pop(
            "stream_options", _MISSING
        )
        stream_options = self._merge_stream_options(
            provider_stream_options, request_stream_options
        )

        has_explicit_token_limit = any(
            field in provider_options or field in per_call_options
            for field in ("max_tokens", "max_completion_tokens")
        )
        has_explicit_temperature = (
            "temperature" in provider_options
            or "temperature" in per_call_options
        )

        data: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": stream,
        }
        core_default_fields = {"temperature"}
        if not has_explicit_token_limit:
            data[self.output_token_field] = self.max_tokens
            core_default_fields.add(self.output_token_field)

        data.update(provider_options)
        data.update(per_call_options)
        if "max_tokens" in data and "max_completion_tokens" in data:
            raise ValueError(
                "Request payload cannot contain both max_tokens and "
                "max_completion_tokens; provide exactly one token-limit field."
            )
        if has_explicit_temperature:
            core_default_fields.discard("temperature")

        if stream_options is not _MISSING:
            data["stream_options"] = stream_options
        if stream:
            existing_stream_options = data.get("stream_options", _MISSING)
            if existing_stream_options is _MISSING:
                data["stream_options"] = {"include_usage": True}
            elif isinstance(existing_stream_options, Mapping):
                merged_stream_options = deepcopy(dict(existing_stream_options))
                merged_stream_options.setdefault("include_usage", True)
                data["stream_options"] = merged_stream_options

        return data, core_default_fields

    def _validate_structured_output_planner(
        self,
        structured_output: Mapping[str, Any] | None,
    ) -> str:
        if structured_output is None:
            return "off"
        if not isinstance(structured_output, Mapping):
            raise TypeError("structured_output must be a mapping.")

        mode = structured_output.get("mode", "require")
        if mode not in _STRUCTURED_OUTPUT_MODES:
            raise ValueError(
                "structured_output.mode must be one of: require, prefer, off."
            )
        return str(mode)

    def _planning_metadata(
        self,
        *,
        strategy: str = "raw",
        fallback_reason: str | None = None,
        validation_status: str = "not_applicable",
    ) -> Dict[str, Any]:
        capability_version = "untracked"
        if self.capabilities is not None:
            capability_version = self.capabilities.version
        return {
            "strategy": strategy,
            "fallback_reason": fallback_reason,
            "validation_status": validation_status,
            "capability_version": capability_version,
            "hook_applied": False,
        }

    def _is_openrouter_route(self) -> bool:
        return (
            "openrouter" in self.provider_name
            or "openrouter.ai" in self.api_url
        )

    def _is_gemini_route(self) -> bool:
        return self.provider_name.startswith("gemini") or (
            "generativelanguage.googleapis.com" in self.api_url
        )

    def _validate_gemini_thinking_controls(
        self,
        data: Mapping[str, Any],
    ) -> None:
        if not self._is_gemini_route() or "reasoning_effort" not in data:
            return

        google_options = [data.get("google")]
        extra_body = data.get("extra_body")
        if isinstance(extra_body, Mapping):
            google_options.append(extra_body.get("google"))

        for google in google_options:
            if not isinstance(google, Mapping):
                continue
            thinking_config = google.get("thinking_config")
            if not isinstance(thinking_config, Mapping):
                continue
            if "thinking_level" in thinking_config or (
                "thinking_budget" in thinking_config
            ):
                raise ValueError(
                    "reasoning_effort cannot be combined with "
                    "google.thinking_config thinking_level or "
                    "thinking_budget."
                )

    def _ensure_openrouter_require_parameters(
        self,
        data: Dict[str, Any],
    ) -> None:
        provider = data.get("provider", {})
        if provider is None:
            provider = {}
        if not isinstance(provider, Mapping):
            raise ValueError("OpenRouter provider routing must be a mapping.")
        provider_options = deepcopy(dict(provider))
        provider_options["require_parameters"] = True
        data["provider"] = provider_options

    def _has_correctness_dependent_openrouter_options(
        self,
        data: Mapping[str, Any],
    ) -> bool:
        return "response_format" in data or "tools" in data

    def _validate_openrouter_supported_parameter(
        self,
        parameter: str,
        capabilities: ModelCapabilities,
    ) -> None:
        supported = capabilities.openrouter_supported_parameters
        if parameter not in supported:
            raise ValueError(
                f"{self.model_name} does not support {parameter} through "
                "OpenRouter supported_parameters."
            )

    def _plan_openrouter_core_parameters(
        self,
        data: Dict[str, Any],
        core_default_fields: set[str],
    ) -> None:
        capabilities = self.capabilities
        if capabilities is None or not self._is_openrouter_route():
            return

        supported = capabilities.openrouter_supported_parameters
        for parameter in (
            "temperature",
            "max_tokens",
            "max_completion_tokens",
        ):
            if parameter not in data or parameter in supported:
                continue
            if parameter not in core_default_fields:
                self._validate_openrouter_supported_parameter(
                    parameter,
                    capabilities,
                )

            value = data.pop(parameter)
            if (
                parameter == "max_tokens"
                and "max_completion_tokens" in supported
                and "max_completion_tokens" not in data
            ):
                data["max_completion_tokens"] = value
                core_default_fields.add("max_completion_tokens")
            core_default_fields.discard(parameter)

    def _validate_capability_aware_request(
        self,
        data: Dict[str, Any],
        *,
        stream: bool,
    ) -> None:
        capabilities = self.capabilities
        if capabilities is None:
            return

        response_format = data.get("response_format")
        if isinstance(response_format, Mapping):
            response_format_type = response_format.get("type")
            if response_format_type == "json_schema":
                if not capabilities.strict_response_schema:
                    raise ValueError(
                        f"{self.model_name} does not support strict schema "
                        "response_format."
                    )
                if self._is_openrouter_route():
                    self._validate_openrouter_supported_parameter(
                        "response_format", capabilities
                    )
                    self._validate_openrouter_supported_parameter(
                        "structured_outputs", capabilities
                    )
            elif response_format_type == "json_object":
                if not capabilities.json_object_response:
                    raise ValueError(
                        f"{self.model_name} does not support json_object "
                        "response_format."
                    )
                if self._is_openrouter_route():
                    self._validate_openrouter_supported_parameter(
                        "response_format", capabilities
                    )

        if "tools" in data:
            if not capabilities.tools:
                raise ValueError(f"{self.model_name} does not support tools.")
            if stream and not capabilities.tool_streaming:
                raise ValueError(
                    f"{self.model_name} does not support tools with "
                    "stream=True."
                )
            if self._is_openrouter_route():
                self._validate_openrouter_supported_parameter(
                    "tools", capabilities
                )

        if "tool_choice" in data and not capabilities.tool_choice:
            raise ValueError(
                f"{self.model_name} does not support tool_choice."
            )
        if (
            "parallel_tool_calls" in data
            and not capabilities.parallel_tool_calls
        ):
            raise ValueError(
                f"{self.model_name} does not support parallel_tool_calls."
            )

        for field in _KNOWN_CAPABILITY_CONTROL_FIELDS:
            if field in data and field not in capabilities.reasoning_controls:
                raise ValueError(
                    f"{self.model_name} does not support {field}."
                )

        if self._is_openrouter_route() and (
            self._has_correctness_dependent_openrouter_options(data)
        ):
            self._ensure_openrouter_require_parameters(data)

    def _append_json_schema_instruction(
        self,
        data: Dict[str, Any],
        schema: Mapping[str, Any],
    ) -> None:
        schema_text = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        instruction = (
            "\n\nRespond with valid JSON matching this JSON Schema. "
            "Do not include markdown fences or explanatory prose.\n"
            f"JSON Schema: {schema_text}"
        )
        messages = data.get("messages", [])
        if messages:
            messages[0][
                "content"
            ] = f"{messages[0].get('content', '')}{instruction}"

    def _plan_structured_output(
        self,
        data: Dict[str, Any],
        structured_output: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        mode = self._validate_structured_output_planner(structured_output)
        if mode == "off":
            return self._planning_metadata()

        if not isinstance(structured_output, Mapping):
            raise TypeError("structured_output must be a mapping.")
        if "response_format" in data:
            raise ValueError(
                "structured_output cannot be combined with raw "
                "response_format."
            )

        schema = structured_output.get("schema")
        if not isinstance(schema, Mapping):
            raise ValueError("structured_output.schema must be a mapping.")
        name = str(structured_output.get("name", "structured_output"))

        capabilities = self.capabilities
        if capabilities is not None and capabilities.strict_response_schema:
            data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": deepcopy(dict(schema)),
                },
            }
            if self._is_openrouter_route():
                self._ensure_openrouter_require_parameters(data)
            return self._planning_metadata(
                strategy="strict_schema",
                validation_status="provider_enforced",
            )

        if mode == "require":
            raise ValueError(
                f"{self.model_name} does not support strict schema "
                "structured output."
            )

        if capabilities is not None and capabilities.json_object_response:
            data["response_format"] = {"type": "json_object"}
            self._append_json_schema_instruction(data, schema)
            return self._planning_metadata(
                strategy="json_object",
                fallback_reason="strict_schema_not_supported",
                validation_status="not_validated",
            )

        self._append_json_schema_instruction(data, schema)
        return self._planning_metadata(
            strategy="prompt_json",
            fallback_reason="response_format_not_supported",
            validation_status="not_validated",
        )

    def _build_cache_payload(
        self,
        data: Mapping[str, Any],
        planning_metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        cache_payload = deepcopy(dict(data))
        cache_payload["_llm_exec_core_plan"] = {
            "strategy": planning_metadata.get("strategy"),
            "fallback_reason": planning_metadata.get("fallback_reason"),
            "capability_version": planning_metadata.get("capability_version"),
        }
        return cache_payload

    def _process_response(
        self,
        response_text: str,
        planning_metadata: Mapping[str, Any],
        structured_output: Mapping[str, Any] | None,
        structured_output_hook: Optional[Callable[[str], Any]],
    ) -> Tuple[Any | None, Dict[str, Any]]:
        finalized = deepcopy(dict(planning_metadata))
        structured: Any | None = None
        mode = self._validate_structured_output_planner(structured_output)

        if mode != "off":
            if not isinstance(structured_output, Mapping):
                raise TypeError("structured_output must be a mapping.")
            schema = structured_output.get("schema")
            if not isinstance(schema, Mapping):
                raise ValueError("structured_output.schema must be a mapping.")
            try:
                structured = json.loads(
                    response_text,
                    parse_constant=_reject_non_finite_json_constant,
                )
                jsonschema.validate(instance=structured, schema=schema)
            except (
                ValueError,
                jsonschema.exceptions.SchemaError,
                jsonschema.exceptions.ValidationError,
            ) as error:
                raise StructuredOutputValidationError(
                    "Structured output validation failed: response is not "
                    "valid JSON matching the requested schema."
                ) from error
            finalized["validation_status"] = "client_validated"

        if structured_output_hook is not None:
            structured = structured_output_hook(response_text)
            finalized["hook_applied"] = True

        return structured, finalized

    def _build_request_plan(
        self,
        prompt: str,
        stream: bool,
        request_options: Mapping[str, Any] | None,
        structured_output: Mapping[str, Any] | None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        data, core_default_fields = self._build_request_payload(
            prompt,
            stream,
            request_options,
        )
        planning_metadata = self._plan_structured_output(
            data,
            structured_output,
        )
        self._plan_openrouter_core_parameters(data, core_default_fields)
        self._validate_gemini_thinking_controls(data)
        self._validate_capability_aware_request(data, stream=stream)
        return data, planning_metadata

    async def generate(
        self,
        prompt: str,
        request_name: str = "unnamed_request",
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
        structured_output_hook: Optional[Callable[[str], Any]] = None,
        trace_context: Optional[Dict[str, Any]] = None,
        run_id: str | None = None,
        request_id: str | None = None,
        *,
        request_options: Mapping[str, Any] | None = None,
        structured_output: Mapping[str, Any] | None = None,
    ) -> LLMResult:
        """Generate a structured LLM result."""
        request_pricing_context = (
            None
            if self._pricing_context is None
            else self._pricing_context.model_copy(deep=True)
        )
        if (
            request_pricing_context is not None
            and request_pricing_context.request_mode == "batch"
        ):
            raise PricingCalculationError(
                "LLMClient supports realtime pricing contexts only."
            )
        pricing_effective_at = datetime.now(timezone.utc)
        started_at = datetime.now()
        start_time = time.time()
        data, planning_metadata = self._build_request_plan(
            prompt,
            stream,
            request_options,
            structured_output,
        )

        if self._cache_enabled and not stream:
            cache_payload = self._build_cache_payload(data, planning_metadata)
            cached_response = self._cache.get(cache_payload)
            if cached_response is not None:
                logger.info("Cache hit for %s", request_name)
                try:
                    structured, finalized_planning = self._process_response(
                        cached_response,
                        planning_metadata,
                        structured_output,
                        structured_output_hook,
                    )
                except Exception:
                    self._cache.delete(cache_payload)
                    raise
                finished_at = datetime.now()
                result = LLMResult(
                    text=cached_response,
                    usage=self._build_token_usage(
                        input_tokens=0,
                        output_tokens=0,
                        input_cost=0.0,
                        output_cost=0.0,
                        total_cost=0.0,
                        requests=[],
                        process_times={"request_times": []},
                    ),
                    metadata=self._build_metadata(
                        request_name=request_name,
                        request_id=request_id,
                        run_id=run_id,
                        trace_context=trace_context,
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_seconds=finished_at.timestamp()
                        - started_at.timestamp(),
                        planning=finalized_planning,
                    ),
                    structured=structured,
                )
                return result

        await self._wait_for_rate_limit()

        retry_delay = INITIAL_RETRY_DELAY_SECONDS
        client = await self._get_client()

        for attempt in range(MAX_API_RETRIES):
            try:
                if stream:
                    response_text, legacy_usage = await self._stream_response(
                        client,
                        data,
                        start_time,
                        request_name,
                        pricing_effective_at,
                        request_pricing_context,
                        stream_callback,
                    )
                else:
                    legacy_result = await self._non_stream_response(
                        client,
                        data,
                        start_time,
                        request_name,
                        pricing_effective_at,
                        request_pricing_context,
                    )
                    response_text, legacy_usage = legacy_result

                structured, finalized_planning = self._process_response(
                    response_text,
                    planning_metadata,
                    structured_output,
                    structured_output_hook,
                )
                if self._cache_enabled and not stream:
                    self._cache.set(
                        self._build_cache_payload(data, planning_metadata),
                        response_text,
                    )

                finished_at = datetime.now()
                result = LLMResult(
                    text=response_text,
                    usage=self._usage_from_legacy_request(legacy_usage),
                    metadata=self._build_metadata(
                        request_name=request_name,
                        request_id=request_id,
                        run_id=run_id,
                        trace_context=trace_context,
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_seconds=legacy_usage["process_times"][
                            "total_time"
                        ],
                        planning=finalized_planning,
                    ),
                    structured=structured,
                )
                return result
            except asyncio.CancelledError:
                raise
            except httpx.RequestError as e:
                error_msg = str(e) or repr(e)
                if attempt == MAX_API_RETRIES - 1:
                    raise Exception(
                        "Failed to generate response after "
                        f"{MAX_API_RETRIES} attempts: {error_msg}"
                    )
                logger.warning(
                    "API request failed (%s), retrying in %s seconds...",
                    error_msg,
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            except httpx.HTTPStatusError as e:
                body = ""
                try:
                    body = e.response.text
                except Exception:
                    body = ""

                status_code = e.response.status_code
                retry_policy = self.max_tokens_retry
                token_limit_field = next(
                    (
                        field
                        for field in (
                            "max_tokens",
                            "max_completion_tokens",
                        )
                        if type(data.get(field)) is int
                    ),
                    None,
                )
                should_lower_token_limit = (
                    retry_policy is not None
                    and status_code == retry_policy.status_code
                    and retry_policy.body_contains.lower() in body.lower()
                    and token_limit_field is not None
                    and data[token_limit_field] > retry_policy.max_tokens_limit
                )
                if (
                    should_lower_token_limit
                    and retry_policy is not None
                    and token_limit_field is not None
                ):
                    old_limit = data[token_limit_field]
                    data[token_limit_field] = min(
                        retry_policy.max_tokens_limit,
                        max(1, old_limit // 2),
                    )
                    logger.warning(
                        "HTTP error %s; lowering %s %s -> %s "
                        "and retrying...",
                        status_code,
                        token_limit_field,
                        old_limit,
                        data[token_limit_field],
                    )
                elif status_code == 429:
                    logger.warning(
                        "Rate limit exceeded (429), retrying in %s seconds...",
                        retry_delay,
                    )
                else:
                    logger.warning("HTTP error %s, retrying...", status_code)

                if attempt == MAX_API_RETRIES - 1:
                    extra = f"\nResponse body: {body}" if body else ""
                    raise Exception(f"HTTP Error: {e}{extra}")

                await asyncio.sleep(retry_delay)
                retry_delay *= 2

        raise RuntimeError("Retry loop exited without returning or raising")

    async def generate_response(
        self,
        prompt: str,
        request_name: str = "unnamed_request",
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
        *,
        request_options: Mapping[str, Any] | None = None,
        structured_output: Mapping[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response using the LLM API (Async).

        Args:
            prompt: The prompt to send to the API
            request_name: Name of the request for token tracking
            stream: If True, stream the response
            stream_callback: Optional callback for streaming chunks.
            request_options: Per-call Chat Completions request payload fields.
            structured_output: Explicit semantic structured-output planner
                request with require/prefer/off modes.

        Returns:
            Tuple containing:
            - The response text
            - Dictionary with usage statistics for this specific request
        """
        result = await self.generate(
            prompt,
            request_name=request_name,
            stream=stream,
            stream_callback=stream_callback,
            request_options=request_options,
            structured_output=structured_output,
        )
        return result.to_legacy_tuple()

    async def _non_stream_response(
        self,
        client: httpx.AsyncClient,
        data: dict,
        start_time: float,
        request_name: str,
        pricing_effective_at: datetime,
        pricing_context: PricingContext | None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle non-streaming API response."""
        response = await client.post(
            self.api_url,
            headers=self.headers,
            json=data,
        )
        response.raise_for_status()

        result = response.json()
        response_text = result["choices"][0]["message"]["content"]

        pricing_cost = None
        if isinstance(self.pricing, PricingSchedule):
            assert pricing_context is not None
            usage_parts = _parse_openai_chat_usage(
                result.get("usage"), pricing_context.cache_mode
            )
            pricing_cost = self._calculate_rich_pricing(
                usage_parts, pricing_effective_at, pricing_context
            )
            input_tokens = usage_parts.prompt_tokens
            output_tokens = usage_parts.completion_tokens
        else:
            input_tokens = result.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = result.get("usage", {}).get("completion_tokens", 0)

        usage = self._track_usage(
            input_tokens,
            output_tokens,
            start_time,
            request_name,
            pricing_cost=pricing_cost,
        )
        return response_text, usage

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        data: dict,
        start_time: float,
        request_name: str,
        pricing_effective_at: datetime,
        pricing_context: PricingContext | None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle streaming API response with real-time output or callback."""
        full_content = []
        input_tokens = 0
        output_tokens = 0
        raw_usage = None

        async with client.stream(
            "POST", self.api_url, headers=self.headers, json=data
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                if line.startswith("data: "):
                    line = line[6:]

                if line.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)

                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")

                        if content:
                            full_content.append(content)
                            if stream_callback:
                                stream_callback(content)

                    if "usage" in chunk and chunk["usage"] is not None:
                        raw_usage = chunk["usage"]
                except json.JSONDecodeError:
                    continue

        response_text = "".join(full_content)

        pricing_cost = None
        if isinstance(self.pricing, PricingSchedule):
            assert pricing_context is not None
            usage_parts = _parse_openai_chat_usage(
                raw_usage, pricing_context.cache_mode
            )
            pricing_cost = self._calculate_rich_pricing(
                usage_parts, pricing_effective_at, pricing_context
            )
            input_tokens = usage_parts.prompt_tokens
            output_tokens = usage_parts.completion_tokens
        else:
            if raw_usage is not None:
                input_tokens = raw_usage.get("prompt_tokens", 0)
                output_tokens = raw_usage.get("completion_tokens", 0)
            if input_tokens == 0:
                prompt_content = ""
                for msg in data.get("messages", []):
                    prompt_content += msg.get("content", "")
                input_tokens = estimate_tokens(prompt_content)

            if output_tokens == 0:
                output_tokens = estimate_tokens(response_text)

        usage = self._track_usage(
            input_tokens,
            output_tokens,
            start_time,
            request_name,
            pricing_cost=pricing_cost,
        )
        return response_text, usage

    def _calculate_rich_pricing(
        self,
        usage_parts: _PricingUsageParts,
        effective_at: datetime,
        pricing_context: PricingContext,
    ) -> PricingCost:
        assert isinstance(self.pricing, PricingSchedule)
        resolved = _resolve_declared_pricing(
            self.model_name,
            self.model,
            self.pricing,
            self._provider_pricing_currency,
            pricing_context=pricing_context,
            input_tokens=usage_parts.prompt_tokens,
            effective_at=effective_at,
        )
        return calculate_pricing_cost(
            resolved,
            input_tokens=usage_parts.prompt_tokens,
            output_tokens=usage_parts.completion_tokens,
            cache_read_input_tokens=usage_parts.cache_read_input_tokens,
            cache_write_input_tokens=usage_parts.cache_write_input_tokens,
        )

    def _track_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        start_time: float,
        request_name: str,
        *,
        pricing_cost: PricingCost | None = None,
    ) -> Dict[str, Any]:
        """Track token usage and costs. Returns usage for this request."""
        if pricing_cost is None:
            assert isinstance(self.pricing, Pricing)
            input_cost = (input_tokens / 1_000_000) * self.pricing.input
            output_cost = (output_tokens / 1_000_000) * self.pricing.output
            total_cost = input_cost + output_cost
            request_currency = self._provider_pricing_currency
        else:
            input_cost = pricing_cost.input_cost
            output_cost = pricing_cost.output_cost
            total_cost = pricing_cost.total_cost
            request_currency = pricing_cost.currency

        if (
            self._usage_currency is not None
            and self._usage_currency != request_currency
        ):
            raise PricingCalculationError(
                "Client usage accumulation cannot combine currencies."
            )

        end_time = time.time()
        process_time = end_time - start_time
        request_record = {
            "name": request_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "process_time": process_time,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        request_time = {"name": request_name, "process_time": process_time}
        total_input_tokens = (
            self.token_usage["total_input_tokens"] + input_tokens
        )
        total_output_tokens = (
            self.token_usage["total_output_tokens"] + output_tokens
        )
        total_process_time = (
            self.token_usage["process_times"]["total_time"] + process_time
        )
        aggregate_input_cost = (
            self.token_usage["cost"]["input_cost"] + input_cost
        )
        aggregate_output_cost = (
            self.token_usage["cost"]["output_cost"] + output_cost
        )
        aggregate_total_cost = (
            self.token_usage["cost"]["total_cost"] + total_cost
        )
        requests = [*self.token_usage["requests"], request_record]
        request_times = [
            *self.token_usage["process_times"]["request_times"],
            request_time,
        ]

        self._usage_currency = request_currency
        self.pricing_currency = request_currency
        self.token_usage["total_input_tokens"] = total_input_tokens
        self.token_usage["total_output_tokens"] = total_output_tokens
        self.token_usage["process_times"]["total_time"] = total_process_time
        self.token_usage["process_times"]["request_times"] = request_times
        self.token_usage["cost"]["input_cost"] = aggregate_input_cost
        self.token_usage["cost"]["output_cost"] = aggregate_output_cost
        self.token_usage["cost"]["total_cost"] = aggregate_total_cost
        self.token_usage["requests"] = requests

        return {
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "cost": {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
            },
            "process_times": {"total_time": process_time},
        }

    def _usage_from_legacy_request(
        self, legacy_usage: Dict[str, Any]
    ) -> TokenUsage:
        requests = []
        request_times = []
        if self.token_usage["requests"]:
            requests = [dict(self.token_usage["requests"][-1])]
        if self.token_usage["process_times"]["request_times"]:
            request_times = [
                dict(self.token_usage["process_times"]["request_times"][-1])
            ]

        return self._build_token_usage(
            input_tokens=legacy_usage["total_input_tokens"],
            output_tokens=legacy_usage["total_output_tokens"],
            input_cost=legacy_usage["cost"]["input_cost"],
            output_cost=legacy_usage["cost"]["output_cost"],
            total_cost=legacy_usage["cost"]["total_cost"],
            requests=requests,
            process_times={"request_times": request_times},
        )

    def _build_token_usage(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        input_cost: float,
        output_cost: float,
        total_cost: float,
        requests: list[Dict[str, Any]],
        process_times: Dict[str, Any],
    ) -> TokenUsage:
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            currency=self.pricing_currency,
            requests=requests,
            process_times=process_times,
        )

    def _build_metadata(
        self,
        *,
        request_name: str,
        request_id: str | None,
        run_id: str | None,
        trace_context: Optional[Dict[str, Any]],
        started_at: datetime,
        finished_at: datetime,
        duration_seconds: float,
        planning: Mapping[str, Any] | None = None,
    ) -> ExecutionMetadata:
        return ExecutionMetadata(
            request_id=request_id,
            run_id=run_id,
            request_name=request_name,
            model_name=self.model_name,
            model_id=self.model,
            provider_name=self.provider_name,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            duration_seconds=duration_seconds,
            trace_context=dict(trace_context or {}),
            planning=dict(planning or {}),
        )

    def get_token_usage(self) -> Dict[str, Any]:
        """
        Get the current token usage statistics.

        Returns:
            Dictionary with token usage statistics
        """
        return self.token_usage

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hit/miss stats
        """
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
