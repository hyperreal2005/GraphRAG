"""Coreference resolution stage for Phase 1 ingestion."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from models.ingestion import ParsedDocument, SectionType


class CoreferenceResolver:
    """
    Resolve references in text sections before chunking.

    FastCoref is loaded lazily and only used for text sections.
    Table rows are intentionally skipped.
    """

    def __init__(self, profile: str = "balanced", strict: bool = True) -> None:
        self.profile = profile
        self.strict = strict
        self._model: Any | None = None
        self._model_initialized = False

    def resolve_document(self, document: ParsedDocument) -> ParsedDocument:
        """Apply coref section-by-section and return a copied document."""
        resolved_sections = []
        for section in document.sections:
            if section.section_type is not SectionType.TEXT:
                resolved_sections.append(section)
                continue
            if not section.text:
                resolved_sections.append(section)
                continue
            resolved_text = self.resolve_text(section.text)
            resolved_sections.append(replace(section, text=resolved_text))
        return replace(document, sections=resolved_sections)

    def resolve_text(self, text: str) -> str:
        """Resolve references in a single text section."""
        model = self._get_model()
        if model is None:
            if self.strict:
                raise RuntimeError(
                    "FastCoref is unavailable. Install fastcoref or run with strict_coref disabled."
                )
            return text

        try:
            prediction = self._predict(model, text)
            resolved = self._extract_resolved_text(prediction)
            return resolved or text
        except Exception:
            if self.strict:
                raise
            return text

    def _get_model(self) -> Any | None:
        if self._model_initialized:
            return self._model

        self._model_initialized = True

        try:
            from fastcoref import FCoref
        except Exception:
            self._model = None
            return self._model

        model_name = self._model_name_from_profile(self.profile)
        try:
            self._model = FCoref(model_name=model_name, device="cpu")
        except TypeError:
            # Some fastcoref builds do not accept model_name.
            self._model = FCoref(device="cpu")
        return self._model

    def _model_name_from_profile(self, profile: str) -> str:
        profile_normalized = profile.lower().strip()
        if profile_normalized == "quality":
            return "biu-nlp/lingmess-coref"
        if profile_normalized == "speed":
            return "biu-nlp/f-coref"
        return "biu-nlp/f-coref"

    def _predict(self, model: Any, text: str) -> Any:
        try:
            return model.predict(texts=[text])
        except TypeError:
            return model.predict([text])

    def _extract_resolved_text(self, prediction: Any) -> str | None:
        candidate = prediction
        if isinstance(prediction, (list, tuple)) and prediction:
            candidate = prediction[0]

        if isinstance(candidate, str):
            return candidate

        getter = getattr(candidate, "get_resolved_text", None)
        if callable(getter):
            try:
                value = getter()
                if isinstance(value, str):
                    return value
            except Exception:
                pass

        value = getattr(candidate, "resolved_text", None)
        if isinstance(value, str):
            return value

        return None
