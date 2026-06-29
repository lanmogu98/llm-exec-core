"""Legacy async LLM client behavior."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import httpx

from .config import get_model_details, get_supported_models
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

logger = logging.getLogger(__name__)


class ResponseCache:
    """LRU cache for LLM responses with TTL support."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, Tuple[str, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _make_key(self, prompt: str, model: str) -> str:
        """Create a cache key from prompt and model."""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response if exists and not expired."""
        key = self._make_key(prompt, model)

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

    def set(self, prompt: str, model: str, response: str) -> None:
        """Store response in cache."""
        key = self._make_key(prompt, model)

        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = (response, time.time())

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
    def get_supported_models():
        """Return list of supported model names from llm_config.yml."""
        return get_supported_models()

    def __init__(
        self,
        model_name: str,
        thinking_level: str = None,
        config_source: Path | dict | None = None,
    ):
        """
        Initialize the LLM client for a specific model.
        The client automatically determines the service provider and settings.

        Args:
            model_name: The name of the model to use.
            thinking_level: Optional thinking/reasoning level override.
                For Gemini 3+, maps to reasoning_effort.
            config_source: Optional config path or raw config dict.
        """
        self._thinking_level = thinking_level

        _, provider_settings, model_details = get_model_details(
            model_name, config_source
        )

        self.api_key = os.environ.get(provider_settings.api_key_env_var)
        if not self.api_key:
            raise ValueError(
                "API key environment variable "
                f"'{provider_settings.api_key_env_var}' is not set."
            )

        self.context_window = provider_settings.context_window
        self.max_tokens = provider_settings.max_tokens
        self.model_name = model_name
        self.model = model_details.id
        self.pricing = model_details.pricing
        self.pricing_currency = provider_settings.pricing_currency
        self.temperature = provider_settings.temperature

        self.api_url = provider_settings.api_base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        self.request_overrides = dict(
            provider_settings.request_overrides or {}
        )
        if self._thinking_level:
            self.request_overrides["reasoning_effort"] = self._thinking_level

        self.token_usage = {
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
        self._request_timestamps = deque(
            maxlen=self._max_rpm if self._max_rpm > 0 else 100
        )

        self._cache_enabled = RESPONSE_CACHE_ENABLED
        self._cache = ResponseCache(
            max_size=RESPONSE_CACHE_MAX_SIZE,
            ttl_seconds=RESPONSE_CACHE_TTL_SECONDS,
        )

        self._async_client: Optional[httpx.AsyncClient] = None

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

    async def generate_response(
        self,
        prompt: str,
        request_name: str = "unnamed_request",
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response using the LLM API (Async).

        Args:
            prompt: The prompt to send to the API
            request_name: Name of the request for token tracking
            stream: If True, stream the response
            stream_callback: Optional callback for streaming chunks.
                If None and stream=True, prints to stdout.

        Returns:
            Tuple containing:
            - The response text
            - Dictionary with usage statistics for this specific request
        """
        if self._cache_enabled:
            cached_response = self._cache.get(prompt, self.model)
            if cached_response is not None:
                logger.info("Cache hit for %s", request_name)
                empty_usage = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "cost": {
                        "input_cost": 0,
                        "output_cost": 0,
                        "total_cost": 0,
                    },
                    "process_times": {"total_time": 0},
                }
                return cached_response, empty_usage

        await self._wait_for_rate_limit()
        start_time = time.time()

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
            **self.request_overrides,
        }

        if stream:
            data["stream_options"] = {"include_usage": True}

        retry_delay = INITIAL_RETRY_DELAY_SECONDS
        client = await self._get_client()

        for attempt in range(MAX_API_RETRIES):
            try:
                if stream:
                    response_text, usage = await self._stream_response(
                        client, data, start_time, request_name, stream_callback
                    )
                else:
                    response_text, usage = await self._non_stream_response(
                        client, data, start_time, request_name
                    )

                if self._cache_enabled:
                    self._cache.set(prompt, self.model, response_text)

                return response_text, usage
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
                provider = data.get("provider", {})
                is_pinned_openrouter = (
                    "openrouter.ai" in self.api_url
                    and isinstance(provider, dict)
                    and provider.get("allow_fallbacks") is False
                )

                if status_code == 429:
                    logger.warning(
                        "Rate limit exceeded (429), retrying in %s seconds...",
                        retry_delay,
                    )
                elif (
                    status_code == 404
                    and is_pinned_openrouter
                    and "no allowed providers" in body.lower()
                    and isinstance(data.get("max_tokens"), int)
                    and data["max_tokens"] > 8192
                ):
                    old_max = data["max_tokens"]
                    data["max_tokens"] = max(1024, min(8192, old_max // 2))
                    logger.warning(
                        "HTTP error 404; lowering max_tokens %s -> %s "
                        "and retrying...",
                        old_max,
                        data["max_tokens"],
                    )
                else:
                    logger.warning("HTTP error %s, retrying...", status_code)

                if attempt == MAX_API_RETRIES - 1:
                    extra = f"\nResponse body: {body}" if body else ""
                    raise Exception(f"HTTP Error: {e}{extra}")

                await asyncio.sleep(retry_delay)
                retry_delay *= 2

        raise RuntimeError("Retry loop exited without returning or raising")

    async def _non_stream_response(
        self,
        client: httpx.AsyncClient,
        data: dict,
        start_time: float,
        request_name: str,
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

        input_tokens = result.get("usage", {}).get("prompt_tokens", 0)
        output_tokens = result.get("usage", {}).get("completion_tokens", 0)

        usage = self._track_usage(
            input_tokens, output_tokens, start_time, request_name
        )
        return response_text, usage

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        data: dict,
        start_time: float,
        request_name: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Handle streaming API response with real-time output or callback."""
        full_content = []
        input_tokens = 0
        output_tokens = 0

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
                            else:
                                print(content, end="", flush=True)

                    if "usage" in chunk and chunk["usage"] is not None:
                        input_tokens = chunk["usage"].get("prompt_tokens", 0)
                        output_tokens = chunk["usage"].get(
                            "completion_tokens", 0
                        )
                except json.JSONDecodeError:
                    continue

        if not stream_callback:
            print(flush=True)

        response_text = "".join(full_content)

        if input_tokens == 0:
            prompt_content = ""
            for msg in data.get("messages", []):
                prompt_content += msg.get("content", "")
            input_tokens = estimate_tokens(prompt_content)

        if output_tokens == 0:
            output_tokens = estimate_tokens(response_text)

        usage = self._track_usage(
            input_tokens, output_tokens, start_time, request_name
        )
        return response_text, usage

    def _track_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        start_time: float,
        request_name: str,
    ) -> Dict[str, Any]:
        """Track token usage and costs. Returns usage for this request."""
        input_cost = (input_tokens / 1_000_000) * self.pricing.input
        output_cost = (output_tokens / 1_000_000) * self.pricing.output
        total_cost = input_cost + output_cost

        end_time = time.time()
        process_time = end_time - start_time

        self.token_usage["total_input_tokens"] += input_tokens
        self.token_usage["total_output_tokens"] += output_tokens
        self.token_usage["process_times"]["total_time"] += process_time
        self.token_usage["cost"]["input_cost"] += input_cost
        self.token_usage["cost"]["output_cost"] += output_cost
        self.token_usage["cost"]["total_cost"] += total_cost

        self.token_usage["requests"].append(
            {
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
        )

        self.token_usage["process_times"]["request_times"].append(
            {"name": request_name, "process_time": process_time}
        )

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
