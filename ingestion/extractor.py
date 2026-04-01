"""Entity and relation extraction for Phase 2 ingestion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from config import Settings
from models.extraction import ExtractedEntity, ExtractionResult


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

CANONICAL_RELATION_LABELS = (
    "USES",
    "PROVIDES",
    "IMPROVES",
    "SUPPORTS",
    "REQUIRES",
    "INCLUDES",
    "INTEGRATES",
    "STORES",
    "PROCESSES",
    "ACCESSES",
    "ANALYZES",
    "TARGETS",
    "PART_OF",
    "TYPE_OF",
    "DEVELOPED_BY",
    "EVALUATED_WITH",
    "PUBLISHED_IN",
    "LOCATED_IN",
    "RELATED_TO",
)


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
        self._response_schema = self._build_response_schema()

    def extract(self, chunk_text: str, doc_format: str) -> ExtractionResult:
        """Extract entities and relations from chunk text."""
        if doc_format in {"csv", "json"}:
            entities = self._gliner_entities(chunk_text)
            prompt = self._prompt_with_entities(chunk_text, entities)
        else:
            entities = None
            prompt = self._prompt_full(chunk_text)

        try:
            parsed = self._generate(prompt)
        except RuntimeError:
            parsed = self._generate(self._repair_prompt(prompt))
        result = ExtractionResult.from_dict(parsed)

        if entities is not None and entities:
            result.entities = entities
        return result

    def _build_response_schema(self) -> Any:
        properties_schema = self._genai_types.Schema(type="OBJECT")
        entity_schema = self._genai_types.Schema(
            type="OBJECT",
            required=["id", "name", "type", "properties"],
            properties={
                "id": self._genai_types.Schema(type="STRING", min_length=1),
                "name": self._genai_types.Schema(type="STRING", min_length=1),
                "type": self._genai_types.Schema(type="STRING", min_length=1),
                "properties": properties_schema,
            },
        )
        relation_schema = self._genai_types.Schema(
            type="OBJECT",
            required=[
                "head",
                "head_type",
                "relation",
                "raw_relation",
                "tail",
                "tail_type",
                "properties",
                "confidence",
            ],
            properties={
                "head": self._genai_types.Schema(type="STRING", min_length=1),
                "head_type": self._genai_types.Schema(type="STRING", min_length=1),
                "relation": self._genai_types.Schema(
                    type="STRING",
                    min_length=1,
                    max_length=64,
                    pattern=r"^[A-Z][A-Z0-9_]*$",
                    enum=list(CANONICAL_RELATION_LABELS),
                ),
                "raw_relation": self._genai_types.Schema(type="STRING", min_length=1, max_length=128),
                "tail": self._genai_types.Schema(type="STRING", min_length=1),
                "tail_type": self._genai_types.Schema(type="STRING", min_length=1),
                "properties": properties_schema,
                "confidence": self._genai_types.Schema(type="NUMBER", minimum=0.0, maximum=1.0),
            },
        )
        return self._genai_types.Schema(
            type="OBJECT",
            required=["entities", "relations"],
            properties={
                "entities": self._genai_types.Schema(type="ARRAY", items=entity_schema),
                "relations": self._genai_types.Schema(type="ARRAY", items=relation_schema),
            },
        )

    def _generate(self, prompt: str) -> dict[str, Any]:
        response = self._client.models.generate_content(
            model=self.settings.gemini_model,
            contents=prompt,
            config=self._genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self._response_schema,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        if parsed is not None and hasattr(parsed, "model_dump"):
            dumped = parsed.model_dump()
            if isinstance(dumped, dict):
                return dumped
        text = getattr(response, "text", None)
        if text:
            return self._parse_json(text)
        raise RuntimeError("Gemini returned empty/invalid structured response.")

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
            "- For academic papers, ignore bibliography/reference entries and journal\n"
            "  header/footer metadata (ISSN, copyright lines, repeated page header text),\n"
            "  unless central to the chunk's main claim.\n"
            "- Infer entity types and relation types from the text itself.\n"
            "- Relation labels must be 1-3 words, uppercase SNAKE_CASE, base verb.\n"
            "  Drop auxiliaries like CAN, ABLE_TO, DESIGNED_TO, AIMS_TO.\n"
            "  Example: \"is designed to improve\" -> \"IMPROVES\".\n"
            "- relation must be chosen from this canonical set:\n"
            f"  {', '.join(CANONICAL_RELATION_LABELS)}.\n"
            "- If none clearly fits, use RELATED_TO.\n"
            "- raw_relation must preserve the exact surface phrasing from the chunk\n"
            "  before canonical mapping (e.g. \"is designed to improve\").\n"
            "- Assign a confidence score (0.0-1.0) to each relation.\n"
            "- Keep extraction concise. Output zero entities/relations when the chunk\n"
            "  lacks meaningful graph facts.\n"
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
            "\"relation\": \"DEVELOPED_BY\",\n"
            "\"raw_relation\": \"was developed by\",\n"
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
            "- Ignore bibliography/reference style statements and boilerplate metadata.\n"
            "- Relation labels must be 1-3 words, uppercase SNAKE_CASE, base verb.\n"
            "  Drop auxiliaries like CAN, ABLE_TO, DESIGNED_TO, AIMS_TO.\n"
            "- relation must be chosen from this canonical set:\n"
            f"  {', '.join(CANONICAL_RELATION_LABELS)}.\n"
            "- If none clearly fits, use RELATED_TO.\n"
            "- raw_relation must preserve the exact surface phrasing from the chunk\n"
            "  before canonical mapping.\n"
            "- Assign a confidence score (0.0-1.0) to each relation.\n"
            "- Prefer precision over recall; omit uncertain relations.\n"
            "- Return ONLY valid JSON. No explanation, no preamble.\n"
            "Output format:\n"
            "{\n"
            "\"entities\": [...],\n"
            "\"relations\": [{\"head\":\"e1\",\"head_type\":\"...\",\"relation\":\"USES\",\"raw_relation\":\"uses\",\"tail\":\"e2\",\"tail_type\":\"...\",\"properties\":{},\"confidence\":0.9}]\n"
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

    def _repair_prompt(self, previous_prompt: str) -> str:
        return (
            "SYSTEM:\n"
            "Re-run the extraction task below and return ONLY strict JSON with keys\n"
            "\"entities\" and \"relations\". Do not include any markdown or explanation.\n"
            "Each relation must include head, head_type, relation, raw_relation, tail,\n"
            "tail_type, properties, and confidence.\n"
            "If nothing can be extracted, return {\"entities\": [], \"relations\": []}.\n"
            f"TASK:\n{previous_prompt}\n"
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
