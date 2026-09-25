"""Phase 10.1: query and ingestion tracing, exported to Phoenix over OTLP.

Architecture section 10.1 asked for Phoenix "from day one", and nothing was ever wired:
every debugging session in this project reconstructed what a query did from
`stage_timings_ms`, log lines and re-running it. Three of the errors recorded in
`docs/ERRORS.md` were measurements that reported a plausible number from a component that
never executed - a no-model fallback, an empty stub retriever, a wrong Qdrant collection -
and each would have been visible at a glance in a trace.

Design:

* **Opt-in and inert by default.** `tracing_enabled` is false. Every helper here is a
  no-op when it is, so callers instrument unconditionally and the test suite never needs
  a collector. Enabling it with no Phoenix listening is harmless: the exporter drops
  batches with a warning and the query still returns.
* **Stages are spans, provider calls are their children.** The orchestrator opens one span
  per stage it already times. `google-genai` is auto-instrumented through OpenInference,
  so each Gemini call - prompt, model, token counts - nests under the stage that made it.
* **Never raises.** Instrumentation that can fail a query is worse than none.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator

from config import Settings

logger = logging.getLogger(__name__)

_CONFIGURED = False
_ACTIVE = False
_TRACER: Any | None = None


def configure_tracing(settings: Settings) -> bool:
    """Install the tracer provider once. Returns whether tracing is active."""
    global _CONFIGURED, _ACTIVE, _TRACER
    if _CONFIGURED:
        return _ACTIVE
    _CONFIGURED = True

    if not bool(settings.tracing_enabled):
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Phoenix reads the project from this resource attribute.
        resource = Resource.create(
            {
                "service.name": "graphrag",
                "openinference.project.name": str(settings.tracing_project),
            }
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=str(settings.tracing_endpoint)))
        )
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("graphrag")
    except Exception as exc:
        logger.warning("tracing_unavailable error_type=%s", type(exc).__name__)
        return False

    try:
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

        GoogleGenAIInstrumentor().instrument(tracer_provider=provider)
    except Exception as exc:
        # Stage spans still work without provider-call spans; say so rather than fail.
        logger.warning("genai_instrumentation_unavailable error_type=%s", type(exc).__name__)

    _ACTIVE = True
    logger.info(
        "tracing_enabled endpoint=%s project=%s",
        settings.tracing_endpoint,
        settings.tracing_project,
    )
    return True


def is_active() -> bool:
    return _ACTIVE


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """A stage span, or nothing. Attribute values are coerced to what OTLP accepts."""
    if not _ACTIVE or _TRACER is None:
        yield None
        return
    try:
        with _TRACER.start_as_current_span(name) as current:
            for key, value in attributes.items():
                _set(current, key, value)
            yield current
    except Exception:
        # A failure inside the span body must propagate; a failure of the span itself
        # must not. Re-raise only if the body raised, which `start_as_current_span`
        # already records on the span before letting it through.
        raise


def annotate(current: Any, **attributes: Any) -> None:
    """Add attributes to a span obtained from `span()`; no-op for None."""
    if current is None:
        return
    for key, value in attributes.items():
        _set(current, key, value)


def _set(current: Any, key: str, value: Any) -> None:
    try:
        if value is None:
            return
        if isinstance(value, (str, bool, int, float)):
            current.set_attribute(key, value)
        elif isinstance(value, (list, tuple, set)):
            current.set_attribute(key, [str(item) for item in value][:50])
        else:
            current.set_attribute(key, str(value)[:500])
    except Exception:
        pass


def shutdown() -> None:
    """Flush pending spans. Called by long-lived processes on exit."""
    if not _ACTIVE:
        return
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        flush = getattr(provider, "force_flush", None)
        if callable(flush):
            flush(timeout_millis=5000)
    except Exception:
        pass
