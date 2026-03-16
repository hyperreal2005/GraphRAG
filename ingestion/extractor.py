"""Entity and relation extraction for Phase 2 ingestion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from config import Settings
from models.extraction import ExtractedEntity, ExtractionResult


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ExtractorStats:
    entities_count: int = 0
    relations_count: int = 0


class GeminiExtractor:
    """Gemini-backed extractor with GLiNER fallback for structured data."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._gliner_model = None
        self._gliner_loaded = False

        if not self.settings.gemini_api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required for extraction.")

        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("google-genai must be installed for extraction.") from exc

        self._genai = genai
        self._genai_types = types
        self._client = genai.Client(api_key=self.settings.gemini_api_key)

    def extract(self, chunk_text: str, doc_format: str) -> ExtractionResult:
        """Extract entities and relations from chunk text."""
        if doc_format in {"csv", "json"}:
            entities = self._gliner_entities(chunk_text)
            prompt = self._prompt_with_entities(chunk_text, entities)
        else:
            entities = None
            prompt = self._prompt_full(chunk_text)

        response_text = self._generate(prompt)
        parsed = self._parse_json(response_text)
        result = ExtractionResult.from_dict(parsed)

        if entities is not None and entities:
            result.entities = entities
        return result

    def _generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.settings.gemini_model,
            contents=prompt,
            config=self._genai_types.GenerateContentConfig(response_mime_type="application/json"),
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned empty response.")
        return text

    def _prompt_full(self, chunk_text: str) -> str:
        return (
            "SYSTEM:\n"
            "You are a knowledge graph construction engine.\n"
            "Your task is to extract structured knowledge from text.\n"
            "Rules:\n"
            "- Extract ALL meaningful entities (people, organisations, concepts,\n"
            "locations, events, technical terms, processes, etc.)\n"
            "- Extract ALL relationships between those entities\n"
            "- Do NOT use a predefined schema.\n"
            "Infer entity types and relation types from the text itself.\n"
            "- Assign a confidence score (0.0–1.0) to each triple\n"
            "- Return ONLY valid JSON. No explanation, no preamble.\n"
            "Output format:\n"
            "{\n"
            "\"entities\": [\n"
            "{\"id\": \"e1\", \"name\": \"...\", \"type\": \"...\", \"properties\": {...}}\n"
            "],\n"
            "\"relations\": [\n"
            "{\n"
            "\"head\": \"e1\",\n"
            "\"head_type\": \"Person\",\n"
            "\"relation\": \"FOUNDED\",\n"
            "\"tail\": \"e2\",\n"
            "\"tail_type\": \"Organisation\",\n"
            "\"properties\": {\"year\": 2003},\n"
            "\"confidence\": 0.95\n"
            "}\n"
            "]\n"
            "}\n"
            f"TEXT: {chunk_text}\n"
        )

    def _prompt_with_entities(self, chunk_text: str, entities: list[ExtractedEntity]) -> str:
        entities_payload = [
            {"id": entity.entity_id, "name": entity.name, "type": entity.entity_type, "properties": entity.properties}
            for entity in entities
        ]
        return (
            "SYSTEM:\n"
            "You are a knowledge graph construction engine.\n"
            "Your task is to infer relationships between a provided list of entities.\n"
            "Rules:\n"
            "- Use ONLY the provided entities. Do not invent new entities.\n"
            "- Extract relationships between them from the text.\n"
            "- Assign a confidence score (0.0–1.0) to each relation.\n"
            "- Return ONLY valid JSON. No explanation, no preamble.\n"
            "Output format:\n"
            "{\n"
            "\"entities\": [...],\n"
            "\"relations\": [...]\n"
            "}\n"
            f"ENTITIES: {json.dumps(entities_payload)}\n"
            f"TEXT: {chunk_text}\n"
        )

    def _parse_json(self, raw_text: str) -> dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = JSON_BLOCK_PATTERN.search(cleaned)
            if match:
                return json.loads(match.group(0))
        raise RuntimeError("Gemini response did not contain valid JSON.")

    def _gliner_entities(self, chunk_text: str) -> list[ExtractedEntity]:
        if not chunk_text.strip():
            return []

        model = self._get_gliner_model()
        if model is None:
            return []

        entities = model.predict_entities(chunk_text, list(self.settings.gliner_labels))
        results: list[ExtractedEntity] = []
        for idx, item in enumerate(entities, start=1):
            name = str(item.get("text", "")).strip()
            label = str(item.get("label", "")).strip() or "Entity"
            if not name:
                continue
            results.append(
                ExtractedEntity(
                    entity_id=f"e{idx}",
                    name=name,
                    entity_type=label,
                )
            )
        return results

    def _get_gliner_model(self) -> Any | None:
        if self._gliner_loaded:
            return self._gliner_model

        self._gliner_loaded = True
        try:
            from gliner import GLiNER
        except Exception:
            self._gliner_model = None
            return None

        self._gliner_model = GLiNER.from_pretrained(self.settings.gliner_model)
        return self._gliner_model
