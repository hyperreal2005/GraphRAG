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
        try:
            parsed = self._parse_json(response_text)
        except RuntimeError:
            repair_text = self._generate(self._repair_prompt(response_text))
            parsed = self._parse_json(repair_text)
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
            "Your task is to extract structured knowledge from text with high precision.\n"
            "Rules:\n"
            "- Extract ONLY salient, specific entities (people, organisations, systems,\n"
            "  methods, datasets, tools, publications, concepts, locations, events).\n"
            "- Avoid generic or deictic entities like \"this paper\", \"this study\",\n"
            "  \"this research\", \"the system\" unless clearly named.\n"
            "- If the paper itself must be referenced, use a single entity named\n"
            "  \"This Study\" with type \"Document\".\n"
            "- Extract ONLY explicit, meaningful relationships between entities.\n"
            "- Prefer precision over recall; omit uncertain relations.\n"
            "- Do NOT use a predefined schema.\n"
            "Infer entity types and relation types from the text itself.\n"
            "- Relation labels must be 1-3 words, uppercase SNAKE_CASE, base verb.\n"
            "  Drop auxiliaries like CAN, ABLE_TO, DESIGNED_TO, AIMS_TO.\n"
            "  Example: \"is designed to improve\" -> \"IMPROVES\".\n"
            "- Assign a confidence score (0.0-1.0) to each relation.\n"
            "- Limit output to the most important 5-15 entities and relations.\n"
            "- Properties must be JSON objects with primitive values only.\n"
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
            "- Extract ONLY explicit, meaningful relationships.\n"
            "- Relation labels must be 1-3 words, uppercase SNAKE_CASE, base verb.\n"
            "  Drop auxiliaries like CAN, ABLE_TO, DESIGNED_TO, AIMS_TO.\n"
            "- Assign a confidence score (0.0-1.0) to each relation.\n"
            "- Prefer precision over recall; omit uncertain relations.\n"
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
        cleaned = self._strip_code_fence(raw_text)
        try:
            return self._decode_json(cleaned)
        except json.JSONDecodeError:
            sanitized = self._sanitize_json_text(cleaned)
            try:
                return self._decode_json(sanitized)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Gemini response did not contain valid JSON: {exc}") from exc

    def _decode_json(self, text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        for start_char in ("{", "["):
            idx = text.find(start_char)
            if idx == -1:
                continue
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        match = JSON_BLOCK_PATTERN.search(text)
        if match:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        raise json.JSONDecodeError("No JSON object found", text, 0)

    def _strip_code_fence(self, raw_text: str) -> str:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        return cleaned.strip()

    def _sanitize_json_text(self, text: str) -> str:
        sanitized = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
        sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
        return sanitized.strip()

    def _repair_prompt(self, raw_text: str) -> str:
        return (
            "SYSTEM:\n"
            "You are a JSON repair tool.\n"
            "Fix the invalid JSON below and return ONLY valid JSON.\n"
            "Do not add any explanations or extra text.\n"
            "The JSON must match this schema:\n"
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
            f"INVALID_JSON:\n{raw_text}\n"
        )

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
