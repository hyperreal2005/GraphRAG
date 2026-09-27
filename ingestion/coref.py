"""Coreference annotation for extraction.

FastCoref finds which mentions in a section refer to the same thing. Until 2026-09-27 this
module then read a rewritten text from the prediction, from attributes fastcoref's result
does not have, and silently kept the original, so coreference never changed a word in any
corpus. It also passed `model_name` to a constructor that takes `model_name_or_path`, so the
profile setting never chose a model either (REFACTOR3 WP20).

It now **annotates rather than rewrites.** A third-person pronoun or a definite description
("the company") is followed by the name its cluster resolves to: "it [Rusnano]". The
annotations are applied only to the text extraction reads (`Chunk.extraction_text`). The
section text - and so every stored, embedded and cited chunk - stays exactly as written.
Rewriting it instead would put words in a citation that the document never said, and would
break the verbatim gold phrases the evaluation resolves against.

Off unless `coref_enabled` is set; off means the model is not even loaded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Sequence

from models.ingestion import ParsedDocument, SectionType

# First and second person are never annotated: a paper's "we" is its authors, and nothing is
# gained by naming them in every sentence.
THIRD_PERSON = frozenset(
    {
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves",
    }
)
_DETERMINERS = frozenset({"the", "this", "that", "these", "those"})
_NOT_A_NAME = THIRD_PERSON | _DETERMINERS | {"a", "an"}
MAX_NAME_WORDS = 8
MAX_NAME_CHARS = 60
# Part of the pipeline fingerprint when coreference is on: a change to how annotations are
# chosen or written changes what extraction reads.
ANNOTATION_VERSION = "bracket-v1"


@dataclass
class CorefStats:
    sections: int = 0
    annotated_sections: int = 0
    annotations: int = 0
    failures: int = 0


def is_pronoun(mention: str) -> bool:
    return mention.strip().lower() in THIRD_PERSON


def is_definite_description(mention: str) -> bool:
    """"the company", "this firm": a determiner and lowercase words, so not itself a name."""
    words = mention.split()
    return (
        2 <= len(words) <= 4
        and words[0].lower() in _DETERMINERS
        and all(word[:1].islower() for word in words[1:])
    )


def _bare(word: str) -> str:
    bare = word.lower().strip(".,;:!?()\"")
    for suffix in ("'s", "’s"):
        if bare.endswith(suffix):
            return bare[: -len(suffix)]
    return bare


def is_name(mention: str) -> bool:
    """A mention carrying a capitalised word that is not a pronoun or determiner - so
    "Anatoly Chubais" and "the United States", but not a sentence-initial "This" or "He"."""
    words = mention.split()
    if not words or len(words) > MAX_NAME_WORDS or len(mention) > MAX_NAME_CHARS:
        return False
    if is_pronoun(mention) or is_definite_description(mention):
        return False
    return any(word[:1].isupper() and _bare(word) not in _NOT_A_NAME for word in words)


def clean_label(name: str) -> str:
    """The name as it reads in an annotation: parentheticals such as "( born 1963 )" and the
    empty "( )" tokenised sources leave behind are dropped, and quotes around a title too."""
    without_asides = re.sub(r"\s*\([^()]*\)", "", name)
    return " ".join(without_asides.split()).strip(" \"'“”")


def annotations_for(text: str, clusters: Sequence[Sequence[tuple[int, int]]]) -> list[tuple[int, str]]:
    """Insertions `(offset, " [Name]")` after each pronoun or definite description whose
    cluster contains a name. The name is the cluster's first name-like mention; a cluster with
    none (every "it" referring to "the company") is left alone rather than guessed at."""
    insertions: set[tuple[int, str]] = set()
    for cluster in clusters:
        spans = sorted((int(start), int(end)) for start, end in cluster)
        # Judged on the cleaned label: "Anatoly Borisovich Chubais (; born 16 June 1955 )" is
        # a three-word name, not a nine-word phrase, and was once skipped as the latter.
        named = next(((s, e) for s, e in spans if is_name(clean_label(text[s:e]))), None)
        if named is None:
            continue
        label = clean_label(text[named[0]:named[1]])
        for start, end in spans:
            if named[0] <= start and end <= named[1]:
                continue  # "itself" inside "the CD itself" is part of the name, not a reference to it
            mention = text[start:end]
            if mention.strip().lower() == label.lower():
                continue
            if is_pronoun(mention) or is_definite_description(mention):
                insertions.add((end, f" [{label}]"))
    return sorted(insertions)


def annotate(text: str, annotations: Sequence[tuple[int, str]]) -> str:
    out = text
    for offset, label in sorted(annotations, reverse=True):
        out = out[:offset] + label + out[offset:]
    return out


class CoreferenceResolver:
    """Annotate each prose section of a document with its coreference clusters."""

    def __init__(self, profile: str = "balanced", strict: bool = True, enabled: bool = False) -> None:
        self.profile = profile
        self.strict = strict
        self.enabled = enabled
        self._model: Any | None = None
        self._model_initialized = False
        self.last_stats = CorefStats()

    def resolve_document(self, document: ParsedDocument) -> ParsedDocument:
        """Return the document with `coref_annotations` set on its prose sections.

        Section text is never changed. `last_stats` records what was done, so a stage that
        silently annotates nothing shows up as zero rather than as success.
        """
        stats = CorefStats()
        self.last_stats = stats
        if not self.enabled:
            return document

        targets = [
            (index, (section.text or "").strip())
            for index, section in enumerate(document.sections)
            if section.section_type is SectionType.TEXT and (section.text or "").strip()
        ]
        if not targets:
            return document

        model = self._get_model()
        if model is None:
            if self.strict:
                raise RuntimeError(
                    "FastCoref is unavailable. Install fastcoref or set coref_enabled off."
                )
            stats.failures += len(targets)
            return document

        sections = list(document.sections)
        for (index, text), clusters in zip(targets, self._clusters(model, [t for _, t in targets], stats)):
            if clusters is None:
                continue
            stats.sections += 1
            annotations = annotations_for(text, clusters)
            if annotations:
                stats.annotated_sections += 1
                stats.annotations += len(annotations)
                sections[index] = replace(sections[index], coref_annotations=annotations)
        return replace(document, sections=sections)

    def _clusters(self, model: Any, texts: list[str], stats: CorefStats) -> list[Any]:
        """Clusters per text, predicted in one batch; on failure, one text at a time so a
        single bad section costs only itself."""
        try:
            return [p.get_clusters(as_strings=False) for p in model.predict(texts=texts)]
        except Exception:
            if self.strict:
                raise
        results: list[Any] = []
        for text in texts:
            try:
                results.append(model.predict(texts=[text])[0].get_clusters(as_strings=False))
            except Exception:
                stats.failures += 1
                results.append(None)
        return results

    def _get_model(self) -> Any | None:
        if self._model_initialized:
            return self._model
        self._model_initialized = True
        try:
            if self.profile.lower().strip() == "quality":
                from fastcoref import LingMessCoref

                self._model = LingMessCoref(
                    model_name_or_path="biu-nlp/lingmess-coref", device="cpu",
                    enable_progress_bar=False,
                )
            else:
                from fastcoref import FCoref

                self._model = FCoref(
                    model_name_or_path="biu-nlp/f-coref", device="cpu", enable_progress_bar=False
                )
        except Exception:
            if self.strict:
                raise
            self._model = None
        return self._model
