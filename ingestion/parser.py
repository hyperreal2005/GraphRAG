"""Document parsing for Phase 1 ingestion."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from models.ingestion import DocumentMetadata, ParsedDocument, Section, SectionType

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - optional dependency in dev environments
    DocumentConverter = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency in dev environments
    pd = None  # type: ignore[assignment]

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency in dev environments
    PdfReader = None  # type: ignore[assignment]


SUPPORTED_FORMATS = {"pdf", "txt", "md", "csv", "json"}
logger = logging.getLogger(__name__)


def detect_format(source_path: Path) -> str:
    """Infer document format from suffix."""
    ext = source_path.suffix.lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format '{ext}' for file {source_path}")
    return ext


def compute_sha256(source_path: Path) -> str:
    """Compute file-level SHA-256 checksum."""
    digest = hashlib.sha256()
    with source_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8192), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace to simplify downstream chunking/hash stability."""
    return re.sub(r"\s+", " ", text).strip()


def flatten_json(value: Any, prefix: str = "", out: dict[str, str] | None = None) -> dict[str, str]:
    """Flatten nested JSON into dot-path key/value pairs."""
    if out is None:
        out = {}

    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_json(child, next_prefix, out)
        return out

    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            out[prefix or "value"] = ", ".join(str(item) for item in value)
            return out
        for index, child in enumerate(value):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            flatten_json(child, next_prefix, out)
        return out

    out[prefix or "value"] = str(value)
    return out


class DocumentParser:
    """Parse source files into normalized section objects."""

    def __init__(self) -> None:
        self._docling_converter = DocumentConverter() if DocumentConverter is not None else None

    def parse_document(self, source_path: Path, doc_id: str, checksum: str) -> ParsedDocument:
        """Parse file into architecture-defined document schema."""
        fmt = detect_format(source_path)
        source = str(source_path.resolve())

        if fmt == "pdf":
            sections = self._parse_pdf(source_path)
        elif fmt == "txt":
            sections = self._parse_txt(source_path)
        elif fmt == "md":
            sections = self._parse_markdown(source_path)
        elif fmt == "csv":
            sections = self._parse_csv(source_path)
        elif fmt == "json":
            sections = self._parse_json(source_path)
        else:  # pragma: no cover - guarded by detect_format
            raise ValueError(f"Unsupported format: {fmt}")

        word_count = sum(self._section_word_count(section) for section in sections)
        metadata = DocumentMetadata(word_count=word_count)
        return ParsedDocument(
            doc_id=doc_id,
            source=source,
            format=fmt,
            checksum=checksum,
            sections=sections,
            metadata=metadata,
        )

    def _parse_txt(self, source_path: Path) -> list[Section]:
        text = normalize_whitespace(source_path.read_text(encoding="utf-8", errors="ignore"))
        return [
            Section(
                section_id="txt-1",
                section_type=SectionType.TEXT,
                heading=source_path.stem,
                text=text,
                page=1,
            )
        ]

    def _parse_markdown(self, source_path: Path) -> list[Section]:
        markdown = source_path.read_text(encoding="utf-8", errors="ignore")
        return self._parse_markdown_text(markdown)

    def _parse_markdown_text(self, markdown: str) -> list[Section]:
        sections: list[Section] = []
        current_heading = "Introduction"
        buffer: list[str] = []
        section_index = 1

        for raw_line in markdown.splitlines():
            line = raw_line.rstrip()
            if line.lstrip().startswith("#"):
                if buffer:
                    text = normalize_whitespace("\n".join(buffer))
                    if text:
                        sections.append(
                            Section(
                                section_id=f"md-{section_index}",
                                section_type=SectionType.TEXT,
                                heading=current_heading,
                                text=text,
                                page=1,
                            )
                        )
                        section_index += 1
                    buffer = []
                current_heading = line.lstrip("#").strip() or f"Section {section_index}"
                continue
            buffer.append(line)

        if buffer:
            text = normalize_whitespace("\n".join(buffer))
            if text:
                sections.append(
                    Section(
                        section_id=f"md-{section_index}",
                        section_type=SectionType.TEXT,
                        heading=current_heading,
                        text=text,
                        page=1,
                    )
                )

        if sections:
            return sections

        return [
            Section(
                section_id="md-1",
                section_type=SectionType.TEXT,
                heading="Introduction",
                text=normalize_whitespace(markdown),
                page=1,
            )
        ]

    def _parse_pdf(self, source_path: Path) -> list[Section]:
        sections_from_docling = self._parse_pdf_with_docling(source_path)
        if sections_from_docling:
            return sections_from_docling
        return self._parse_pdf_with_pypdf(source_path)

    def _parse_pdf_with_docling(self, source_path: Path) -> list[Section]:
        if self._docling_converter is None:
            return []

        try:
            result = self._docling_converter.convert(str(source_path))
        except Exception as exc:
            logger.warning("docling_pdf_parse_failed source=%s error_type=%s", source_path, type(exc).__name__)
            return []

        doc = getattr(result, "document", result)
        structured_sections = self._sections_from_docling(doc, source_path)
        if structured_sections:
            return structured_sections

        markdown: str | None = None
        for attr_name in ("export_to_markdown", "to_markdown"):
            candidate = getattr(doc, attr_name, None)
            if callable(candidate):
                try:
                    markdown = candidate()
                except Exception:
                    markdown = None
                if markdown:
                    break

        if isinstance(markdown, str) and markdown.strip():
            return self._parse_markdown_text(markdown)

        text_candidate = None
        for attr_name in ("text", "raw_text", "content"):
            value = getattr(doc, attr_name, None)
            if isinstance(value, str) and value.strip():
                text_candidate = value
                break

        if isinstance(text_candidate, str):
            return [
                Section(
                    section_id="pdf-1",
                    section_type=SectionType.TEXT,
                    heading=source_path.stem,
                    text=normalize_whitespace(text_candidate),
                    page=1,
                )
            ]

        return []

    def _sections_from_docling(self, doc: Any, source_path: Path) -> list[Section]:
        """Preserve Docling item order, page provenance, spans, and table rows."""
        iterate_items = getattr(doc, "iterate_items", None)
        if not callable(iterate_items):
            return []

        sections: list[Section] = []
        current_heading = source_path.stem
        section_index = 1
        try:
            items = iterate_items(with_groups=False)
            for item, _level in items:
                label = getattr(item, "label", None)
                label_value = str(getattr(label, "value", label) or "").casefold()
                item_text = normalize_whitespace(str(getattr(item, "text", "") or ""))
                page, char_start, char_end, bbox = self._docling_provenance(item)

                if label_value in {"title", "section_header"}:
                    if item_text:
                        current_heading = item_text
                    continue
                if label_value in {"page_header", "page_footer"}:
                    continue

                export_dataframe = getattr(item, "export_to_dataframe", None)
                if callable(export_dataframe):
                    try:
                        dataframe = export_dataframe(doc=doc)
                        headers = [str(column) for column in dataframe.columns]
                        for row in dataframe.itertuples(index=False, name=None):
                            values = [str(value) for value in row]
                            sections.append(
                                Section(
                                    section_id=f"pdf-{section_index}",
                                    section_type=SectionType.TABLE_ROW,
                                    heading=current_heading,
                                    page=page,
                                    table_headers=headers,
                                    table_row=values,
                                    char_start=char_start,
                                    char_end=char_end,
                                    bbox=bbox,
                                )
                            )
                            section_index += 1
                        continue
                    except Exception as exc:
                        logger.warning(
                            "docling_table_parse_failed source=%s error_type=%s",
                            source_path,
                            type(exc).__name__,
                        )

                if not item_text:
                    continue
                sections.append(
                    Section(
                        section_id=f"pdf-{section_index}",
                        section_type=SectionType.TEXT,
                        heading=current_heading,
                        text=item_text,
                        page=page,
                        char_start=char_start,
                        char_end=char_end,
                        bbox=bbox,
                    )
                )
                section_index += 1
        except Exception as exc:
            logger.warning("docling_item_iteration_failed source=%s error_type=%s", source_path, type(exc).__name__)
            return []
        return sections

    def _docling_provenance(
        self,
        item: Any,
    ) -> tuple[int | None, int | None, int | None, tuple[float, float, float, float] | None]:
        provenance = getattr(item, "prov", None)
        if not provenance:
            return None, None, None, None
        first = provenance[0]
        page_value = getattr(first, "page_no", None)
        page = int(page_value) if page_value is not None else None
        charspan = getattr(first, "charspan", None)
        char_start = int(charspan[0]) if isinstance(charspan, (list, tuple)) and len(charspan) == 2 else None
        char_end = int(charspan[1]) if isinstance(charspan, (list, tuple)) and len(charspan) == 2 else None
        bbox_value = getattr(first, "bbox", None)
        bbox = None
        if bbox_value is not None and all(hasattr(bbox_value, key) for key in ("l", "t", "r", "b")):
            bbox = (
                float(bbox_value.l),
                float(bbox_value.t),
                float(bbox_value.r),
                float(bbox_value.b),
            )
        return page, char_start, char_end, bbox

    def _parse_pdf_with_pypdf(self, source_path: Path) -> list[Section]:
        if PdfReader is None:
            raise RuntimeError(
                "PDF parsing requires either docling or pypdf. Install one of them before ingesting PDF files."
            )

        reader = PdfReader(str(source_path))
        sections: list[Section] = []
        for index, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            text = normalize_whitespace(extracted)
            sections.append(
                Section(
                    section_id=f"pdf-{index}",
                    section_type=SectionType.TEXT,
                    heading=f"Page {index}",
                    text=text,
                    page=index,
                )
            )

        if sections:
            return sections

        return [
            Section(
                section_id="pdf-1",
                section_type=SectionType.TEXT,
                heading="Page 1",
                text="",
                page=1,
            )
        ]

    def _parse_csv(self, source_path: Path) -> list[Section]:
        rows: list[dict[str, Any]] = []
        headers: list[str] = []

        if pd is not None:
            dataframe = pd.read_csv(source_path, dtype=str, keep_default_na=False)
            headers = [str(column) for column in dataframe.columns]
            rows = dataframe.to_dict(orient="records")
        else:
            with source_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.DictReader(handle)
                headers = [str(field) for field in reader.fieldnames or []]
                rows = [dict(record) for record in reader]

        sections: list[Section] = []
        for row_index, row in enumerate(rows, start=1):
            values = [str(row.get(header, "")) for header in headers]
            sections.append(
                Section(
                    section_id=f"csv-{row_index}",
                    section_type=SectionType.TABLE_ROW,
                    heading=source_path.stem,
                    page=1,
                    table_headers=headers,
                    table_row=values,
                )
            )
        return sections

    def _parse_json(self, source_path: Path) -> list[Section]:
        payload = json.loads(source_path.read_text(encoding="utf-8", errors="ignore"))
        rows = self._json_rows(payload)

        sections: list[Section] = []
        for row_index, row in enumerate(rows, start=1):
            headers = list(row.keys())
            values = [row[key] for key in headers]
            sections.append(
                Section(
                    section_id=f"json-{row_index}",
                    section_type=SectionType.TABLE_ROW,
                    heading=source_path.stem,
                    page=1,
                    table_headers=headers,
                    table_row=values,
                )
            )
        return sections

    def _json_rows(self, payload: Any) -> list[dict[str, str]]:
        if isinstance(payload, list):
            rows: list[dict[str, str]] = []
            for item in payload:
                if isinstance(item, dict):
                    flattened = flatten_json(item)
                    rows.append(flattened)
                else:
                    rows.append({"value": str(item)})
            return rows

        if isinstance(payload, dict):
            return [flatten_json(payload)]

        return [{"value": str(payload)}]

    def _section_word_count(self, section: Section) -> int:
        if section.section_type is SectionType.TEXT:
            return len((section.text or "").split())

        headers = section.table_headers or []
        values = section.table_row or []
        table_text = " ".join(f"{header}: {value}" for header, value in zip(headers, values))
        return len(table_text.split())
