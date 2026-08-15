"""Provider usage normalization without embedding pricing assumptions in code."""

from __future__ import annotations

from typing import Any


def extract_usage_metadata(response: Any) -> dict[str, int]:
    """Normalize Gemini token usage fields across SDK response shapes."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        payload: dict[str, Any] = {}
    elif isinstance(usage, dict):
        payload = usage
    elif hasattr(usage, "model_dump"):
        payload = usage.model_dump()
    else:
        payload = {}

    def value(*keys: str) -> int:
        for key in keys:
            candidate = payload.get(key, getattr(usage, key, None))
            if candidate is not None:
                try:
                    return int(candidate)
                except (TypeError, ValueError):
                    return 0
        return 0

    return {
        "input_tokens": value("prompt_token_count", "input_token_count"),
        "output_tokens": value("candidates_token_count", "output_token_count"),
        "total_tokens": value("total_token_count"),
        "cached_tokens": value("cached_content_token_count"),
        "call_count": 1,
    }


def add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Add normalized usage counters."""
    return {
        key: int(left.get(key, 0)) + int(right.get(key, 0))
        for key in {"input_tokens", "output_tokens", "total_tokens", "cached_tokens", "call_count"}
    }
