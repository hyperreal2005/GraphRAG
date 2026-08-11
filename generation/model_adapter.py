"""Gemini model capability-aware request configuration."""

from __future__ import annotations

from typing import Any


NO_SAMPLING_PREFIXES = ("gemini-3.5-", "gemini-3.6-")


def supports_sampling_parameters(model_id: str) -> bool:
    """Return whether the selected model accepts legacy sampling controls."""
    normalized = model_id.strip().lower()
    return not normalized.startswith(NO_SAMPLING_PREFIXES)


def build_generate_config(
    types_module: Any | None,
    model_id: str,
    *,
    temperature: float | None = None,
    system_instruction: str | None = None,
    response_mime_type: str | None = None,
    response_schema: Any | None = None,
) -> Any | None:
    """Build GenerateContentConfig using only fields supported by the model."""
    if types_module is None:
        return None
    values: dict[str, Any] = {}
    if temperature is not None and supports_sampling_parameters(model_id):
        values["temperature"] = float(temperature)
    if system_instruction is not None:
        values["system_instruction"] = system_instruction
    if response_mime_type is not None:
        values["response_mime_type"] = response_mime_type
    if response_schema is not None:
        values["response_schema"] = response_schema
    return types_module.GenerateContentConfig(**values)

