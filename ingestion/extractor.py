"""Entity and relation extraction for Phase 2 ingestion."""

from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from config import Settings
from generation.model_adapter import build_generate_config
from generation.usage import extract_usage_metadata
from models.extraction import ExtractedEntity, ExtractionResult


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

@dataclass
class ExtractorStats:
    entities_count: int = 0
    relations_count: int = 0
    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class GeminiExtractor:
    """Gemini-backed extractor with GLiNER fallback for structured data."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._gliner_model = None
        self._gliner_loaded = False
        self._stats = ExtractorStats()
        self._stats_lock = threading.Lock()

        if not self.settings.gemini_api_key:
            raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required for extraction.")

        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("google-genai must be installed for extraction.") from exc

        self._genai = genai
        self._genai_types = types
        self._client = genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=max(
                    1,
                    int(float(self.settings.ingestion_provider_timeout_seconds) * 1000),
                )
            ),
        )
        self._response_schema = self._build_response_schema()

    def extract(self, chunk_text: str, doc_format: str) -> ExtractionResult:
        """Extract entities and relations from chunk text."""
        if doc_format in {"csv", "json"}:
            entities = self._gliner_entities(chunk_text)
            prompt = self._prompt_with_entities(chunk_text, entities)
        else:
            entities = None
            prompt = self._prompt_full(chunk_text)

        attempts = max(1, int(self.settings.extraction_max_attempts))
        current_prompt = prompt
        last_error: Exception | None = None
        result: ExtractionResult | None = None
        for attempt in range(attempts):
            try:
                result = ExtractionResult.from_dict(self._generate(current_prompt))
                if entities is not None and entities:
                    result.entities = entities
                self._validate_result_integrity(result)
                break
            except Exception as exc:
                last_error = exc
                result = None
                if attempt + 1 >= attempts:
                    break
                current_prompt = self._repair_prompt(prompt, str(exc))
                base = max(0.0, float(self.settings.extraction_retry_base_seconds))
                delay = base * (2**attempt) + random.uniform(0.0, min(0.25, base))
                if delay > 0:
                    time.sleep(delay)
        if result is None:
            raise RuntimeError(f"Extraction failed after {attempts} attempts: {last_error}") from last_error

        with self._stats_lock:
            self._stats.entities_count += len(result.entities)
            self._stats.relations_count += len(result.relations)

        return result

    def snapshot_stats(self) -> ExtractorStats:
        """Return a thread-safe copy for per-run usage deltas."""
        with self._stats_lock:
            return ExtractorStats(**self._stats.__dict__)

    @staticmethod
    def _validate_result_integrity(result: ExtractionResult) -> None:
        entity_ids = [entity.entity_id for entity in result.entities]
        duplicate_ids = sorted(
            entity_id for entity_id in set(entity_ids) if entity_ids.count(entity_id) > 1
        )
        if duplicate_ids:
            raise ValueError(
                "Entity IDs must be unique within the entities array; "
                f"duplicate IDs: {', '.join(duplicate_ids[:5])}."
            )
        GeminiExtractor._validate_relation_endpoints(result)

    @staticmethod
    def _validate_relation_endpoints(result: ExtractionResult) -> None:
        entity_ids = {entity.entity_id for entity in result.entities}
        invalid = [
            f"{relation.head_id!r}->{relation.tail_id!r}"
            for relation in result.relations
            if relation.head_id not in entity_ids or relation.tail_id not in entity_ids
        ]
        if invalid:
            raise ValueError(
                "Relation endpoints must exactly match IDs from the entities array; "
                f"invalid endpoints: {', '.join(invalid[:5])}."
            )

    def _build_response_schema(self) -> Any:
        property_item_schema = self._genai_types.Schema(
            type="OBJECT",
            required=["name", "value_text", "value_type"],
            properties={
                "name": self._genai_types.Schema(
                    type="STRING",
                    min_length=1,
                    max_length=64,
                    pattern=r"^[a-z][a-z0-9_]*$",
                ),
                "value_text": self._genai_types.Schema(type="STRING", min_length=1),
                "value_type": self._genai_types.Schema(
                    type="STRING",
                    format="enum",
                    enum=["string", "integer", "number", "boolean"],
                ),
            },
        )
        properties_schema = self._genai_types.Schema(
            type="ARRAY",
            items=property_item_schema,
            max_items=12,
        )
        entity_schema = self._genai_types.Schema(
            type="OBJECT",
            required=["id", "name", "type", "properties"],
            properties={
                "id": self._genai_types.Schema(type="STRING", min_length=1),
                "name": self._genai_types.Schema(type="STRING", min_length=1),
                "type": self._genai_types.Schema(type="STRING", min_length=1),
                "properties": properties_schema,
                "confidence": self._genai_types.Schema(type="NUMBER", minimum=0.0, maximum=1.0),
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
                ),
                "raw_relation": self._genai_types.Schema(type="STRING", min_length=1, max_length=128),
                "tail": self._genai_types.Schema(type="STRING", min_length=1),
                "tail_type": self._genai_types.Schema(type="STRING", min_length=1),
                "properties": properties_schema,
                "confidence": self._genai_types.Schema(type="NUMBER", minimum=0.0, maximum=1.0),
                "valid_from": self._genai_types.Schema(type="STRING"),
                "valid_until": self._genai_types.Schema(type="STRING"),
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
            config=build_generate_config(
                self._genai_types,
                self.settings.gemini_model,
                response_mime_type="application/json",
                response_schema=self._response_schema,
            ),
        )
        self._record_response_usage(response)
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

    def _record_response_usage(self, response: Any) -> None:
        usage = extract_usage_metadata(response)
        with self._stats_lock:
            self._stats.call_count += usage["call_count"]
            self._stats.input_tokens += usage["input_tokens"]
            self._stats.output_tokens += usage["output_tokens"]
            self._stats.total_tokens += usage["total_tokens"]

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
            "- Infer relation labels from the evidence without using a predefined ontology.\n"
            "- Relation labels must be concise uppercase SNAKE_CASE base verbs.\n"
            "  Drop auxiliaries like CAN, ABLE_TO, DESIGNED_TO, AIMS_TO.\n"
            "- Reuse the SAME label for the same kind of relationship every time.\n"
            "  Two chunks stating one fact must produce one label, so prefer a short\n"
            "  general verb over specific phrasing: CREATED_BY for authored/founded/\n"
            "  built-by, USES for employs/leverages/applies. Keep the vocabulary small\n"
            "  and reuse it across the whole document.\n"
            "  Example: \"is designed to improve\" -> \"IMPROVES\".\n"
            "- raw_relation must preserve the exact surface phrasing from the chunk\n"
            "  before canonical mapping (e.g. \"is designed to improve\").\n"
            "- valid_from and valid_until are optional. Populate them only when the\n"
            "  text explicitly states a validity interval; otherwise omit them.\n"
            "- Assign a confidence score (0.0-1.0) to each relation.\n"
            "- Keep extraction concise. Output zero entities/relations when the chunk\n"
            "  lacks meaningful graph facts.\n"
            "- Entity properties are explicit literal attribute/value facts about that\n"
            "  entity which are not better represented as a relationship to another\n"
            "  entity (for example parameter_count, publication_year, or version).\n"
            "- Capture salient entity properties stated directly in the text; never\n"
            "  infer them. Property names must be specific lowercase snake_case keys.\n"
            "- Entity and relation properties must be JSON arrays of records with keys\n"
            "  name, value_text, and value_type. value_type must be string, integer,\n"
            "  number, or boolean; encode the primitive value exactly in value_text.\n"
            "  Use [] when there are no properties. Do not emit null values.\n"
            "- Return ONLY valid JSON. No explanation, no preamble.\n"
            "Output format:\n"
            "{\n"
            "\"entities\": [\n"
            "{\"id\": \"e1\", \"name\": \"ChatGPT\", \"type\": \"Language Model\", "
            "\"properties\": [{\"name\": \"parameter_count\", \"value_text\": "
            "\"175000000000\", \"value_type\": \"integer\"}], \"confidence\": 0.9}\n"
            "],\n"
            "\"relations\": [\n"
            "{\n"
            "\"head\": \"e1\",\n"
            "\"head_type\": \"Person\",\n"
            "\"relation\": \"DEVELOPED_BY\",\n"
            "\"raw_relation\": \"was developed by\",\n"
            "\"tail\": \"e2\",\n"
            "\"tail_type\": \"Organisation\",\n"
            "\"properties\": [{\"name\": \"year\", \"value_text\": \"2003\", "
            "\"value_type\": \"integer\"}],\n"
            "\"confidence\": 0.95,\n"
            "\"valid_from\": \"2003\",\n"
            "\"valid_until\": \"2019\"\n"
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
            "- Infer relation labels from the evidence without using a predefined ontology.\n"
            "- Relation labels must be concise uppercase SNAKE_CASE base verbs.\n"
            "  Drop auxiliaries like CAN, ABLE_TO, DESIGNED_TO, AIMS_TO.\n"
            "- Reuse the SAME label for the same kind of relationship every time.\n"
            "  Two chunks stating one fact must produce one label, so prefer a short\n"
            "  general verb over specific phrasing: CREATED_BY for authored/founded/\n"
            "  built-by, USES for employs/leverages/applies. Keep the vocabulary small\n"
            "  and reuse it across the whole document.\n"
            "- raw_relation must preserve the exact surface phrasing from the chunk\n"
            "  before canonical mapping.\n"
            "- valid_from and valid_until are optional and must be evidence-explicit.\n"
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

    def _repair_prompt(self, previous_prompt: str, validation_error: str = "") -> str:
        return (
            "SYSTEM:\n"
            "Re-run the extraction task below and return ONLY strict JSON with keys\n"
            "\"entities\" and \"relations\". Do not include any markdown or explanation.\n"
            "Each relation must include head, head_type, relation, raw_relation, tail,\n"
            "tail_type, properties, and confidence.\n"
            "Each entity properties array may contain only explicit literal facts as\n"
            "name/value_text/value_type records with lowercase snake_case names.\n"
            "CRITICAL: every entity id must be non-empty and unique within the entities array.\n"
            "CRITICAL: relation head and tail values must be exact entity id values\n"
            "from the entities array (for example e1), never entity names.\n"
            f"Previous validation error: {validation_error}\n"
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
