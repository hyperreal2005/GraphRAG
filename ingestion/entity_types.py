"""Map free-form LLM entity types onto a small, stable coarse class.

REVIEW1 finding `A2`/`B2`: the pipeline gated entity identity on string equality of a
type the model invents per mention. Measured, that type is not stable enough to carry
identity — 313 distinct type strings across 3,370 mentions, `"landing page"` alone
receiving 9 of them, and 115 of 172 cross-document candidates at >= 0.90 similarity
rejected *solely* on type conflict, including `("RAG", System Component)` against
`("RAG", System)` at 0.956.

Human annotation shows the opposite: in Re-DocRED only 9 of 9,684 gold entities (0.09%)
have mentions that disagree on type. Type *is* a stable property of an entity — it is
unstable only when assigned independently per mention.

So the remedy is not to trust type more, nor to discard it, but to
  1. derive it once per canonical entity rather than per mention, and
  2. reduce it to a coarse class before it is ever allowed to block a merge.

The closed set is Re-DocRED's, which is human-validated rather than invented
(docs/CORPUS.md section 7.1). `MISC` and `UNKNOWN` are compatible with everything: they
are where the entire technical-concept vocabulary lands (Concept, System, Method, Model,
Tool, Framework...), and that vocabulary is exactly what the old gate was shredding.
Blocking therefore only stops genuinely incompatible pairs — a person against a place.
Precision comes from name matching, not from the type.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

PERSON = "PER"
ORGANIZATION = "ORG"
LOCATION = "LOC"
TIME = "TIME"
NUMBER = "NUM"
MISC = "MISC"
UNKNOWN = "UNKNOWN"

COARSE_CLASSES = frozenset({PERSON, ORGANIZATION, LOCATION, TIME, NUMBER, MISC, UNKNOWN})

# Classes that never constrain a merge. See the module docstring.
PERMISSIVE_CLASSES = frozenset({MISC, UNKNOWN})

# Classes identified by their *value* rather than by a name. A name has variants and so
# tolerates fuzzy matching; a quantity or a date does not - two different values are two
# different things however similar their surface forms. Measured against Re-DocRED, fuzzy
# matching on these merged 908 with 919 and July 18 2007 with July 30 2008.
VALUE_COARSE_CLASSES = frozenset({TIME, NUMBER})

# Tokens whose whole job is to tell two otherwise identical names apart. Embedding
# similarity treats them as noise - they are short, and they sit in names that are
# identical everywhere else - which is exactly backwards.
#
# Confirmed in two unrelated domains: on arXiv, `Qwen3-4B` merged into `Qwen3-14B` and
# `Gemma 3 1B` into `Gemma 3 27B` (REFACTOR2 section 8.6); on Re-DocRED's human-validated
# clusters, `First Division` merged with `Second Division`, `Angleton High School` with
# `Angleton Intermediate School`, and `Paul Desmarais` with `Paul Desmarais Sr`.
_ORDINAL_TOKENS = frozenset({
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
})
_GENERATION_TOKENS = frozenset({"jr", "junior", "sr", "senior", "elder", "younger"})
_LEVEL_TOKENS = frozenset({
    "high", "intermediate", "primary", "secondary", "elementary", "middle",
    "upper", "lower", "north", "south", "east", "west", "old", "new",
})

DISCRIMINATIVE_TOKENS = _ORDINAL_TOKENS | _GENERATION_TOKENS | _LEVEL_TOKENS


_DIGIT_RUN = re.compile(r"\d+(?:\.\d+)?")


def _numbers_in(name: str) -> list[str]:
    """Every numeric run in a name, punctuation-independent.

    Comparing whole tokens made `GPT-4o-mini` and `GPT4o-mini` look different - they
    tokenize as {gpt, 4o, mini} against {gpt4o, mini} - and blocked a merge of one model
    with itself. The numbers are what carry the distinction, so compare those directly:
    both yield ["4"], while `Llama-2-70b` and `Llama-2-7b` yield ["2","70"] and ["2","7"].
    """
    return sorted(_DIGIT_RUN.findall(name.lower()))


def names_differ_discriminatively(left: str, right: str) -> bool:
    """True when two names differ by something that exists to tell them apart.

    Two independent checks. The numbers a name carries must match, punctuation aside, so
    `Qwen3-4B` against `Qwen3-14B` is refused while `Qwen2.5-14B` against `Qwen-2.5-14B`
    is not. Beyond that, a differing word must be one that distinguishes rather than
    describes, so `John Hopkins` against `John Hopkins University` is allowed through.
    """
    if _numbers_in(left) != _numbers_in(right):
        return True

    left_tokens = set(TOKEN_PATTERN.findall(left.lower()))
    right_tokens = set(TOKEN_PATTERN.findall(right.lower()))
    if not left_tokens or not right_tokens:
        return False
    differing = left_tokens.symmetric_difference(right_tokens)
    if not differing:
        return False
    return any(token in DISCRIMINATIVE_TOKENS for token in differing)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Collective nouns sit genuinely between PERSON and ORGANIZATION — a "user group" is
# both a set of people and an organisational unit, and the corpus labels the same
# entity "Group", "UserGroup" and "User Group" interchangeably. Resolving them to
# either specific class would make those spellings block each other, which is the
# defect this module exists to remove. They map to MISC so they never block.
_AMBIGUOUS_COLLECTIVES = frozenset({
    "group", "usergroup", "user group", "audience", "community", "cohort",
    "population", "crowd", "collective", "usergroups",
})

# Exact type strings take priority over keyword matching, so that a compound like
# "Sports Team" resolves to ORG rather than being pulled to PER by "sports".
_EXACT: dict[str, str] = {
    "person": PERSON, "people": PERSON, "human": PERSON, "individual": PERSON,
    "author": PERSON, "researcher": PERSON, "scientist": PERSON, "musician": PERSON,
    "artist": PERSON, "actor": PERSON, "player": PERSON, "athlete": PERSON,
    "politician": PERSON, "student": PERSON, "students": PERSON, "faculty": PERSON,
    "user": PERSON, "character": PERSON,
    "organisation": ORGANIZATION, "organization": ORGANIZATION, "company": ORGANIZATION,
    "corporation": ORGANIZATION, "institution": ORGANIZATION, "university": ORGANIZATION,
    "college": ORGANIZATION, "school": ORGANIZATION, "department": ORGANIZATION,
    "agency": ORGANIZATION, "publisher": ORGANIZATION, "team": ORGANIZATION,
    "sports team": ORGANIZATION, "football club": ORGANIZATION, "club": ORGANIZATION,
    "band": ORGANIZATION, "government": ORGANIZATION, "laboratory": ORGANIZATION,
    "lab": ORGANIZATION, "committee": ORGANIZATION,
    "location": LOCATION, "place": LOCATION, "country": LOCATION, "city": LOCATION,
    "region": LOCATION, "state": LOCATION, "province": LOCATION, "town": LOCATION,
    "village": LOCATION, "venue": LOCATION, "district": LOCATION, "county": LOCATION,
    "continent": LOCATION, "territory": LOCATION, "address": LOCATION,
    "date": TIME, "time": TIME, "year": TIME, "month": TIME, "day": TIME,
    "period": TIME, "era": TIME, "duration": TIME, "timestamp": TIME,
    "number": NUMBER, "quantity": NUMBER, "amount": NUMBER, "count": NUMBER,
    "percentage": NUMBER, "measurement": NUMBER, "statistic": NUMBER, "score": NUMBER,
    "entity": UNKNOWN, "unknown": UNKNOWN, "thing": UNKNOWN, "other": UNKNOWN, "": UNKNOWN,
}

# Checked against the type's tokens when no exact match applies. Order matters only in
# that the first class with a hit wins, so the more specific classes come first.
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (LOCATION, ("country", "city", "region", "location", "place", "state",
                "province", "town", "village", "venue", "district", "county",
                "continent", "territory", "geograph", "nation")),
    (ORGANIZATION, ("organisation", "organization", "company", "corporation",
                    "institution", "university", "college", "school", "department",
                    "agency", "publisher", "team", "club", "band", "government",
                    "laboratory", "committee", "association", "foundation",
                    "consortium", "faculty", "firm", "startup", "vendor",
                    "society", "union", "council", "board")),
    (PERSON, ("person", "people", "human", "author", "researcher", "scientist",
              "musician", "artist", "actor", "player", "athlete", "politician",
              "student", "user", "member", "employee", "founder", "engineer",
              "developer", "practitioner", "expert", "participant")),
    (TIME, ("date", "time", "year", "month", "day", "period", "era", "duration",
            "century", "decade", "season")),
    (NUMBER, ("number", "quantity", "amount", "count", "percentage", "measurement",
              "statistic", "score", "metric", "rate", "ratio")),
)


def normalize_type(raw_type: str) -> str:
    """Whitespace- and case-normalised type string."""
    return " ".join(str(raw_type or "").split()).casefold()


def coarse_class(raw_type: str) -> str:
    """Reduce a free-form type to one of ``COARSE_CLASSES``.

    Deterministic and offline: no provider call, no network, same answer every run.
    """
    normalized = normalize_type(raw_type)
    if not normalized:
        return UNKNOWN
    if normalized in _AMBIGUOUS_COLLECTIVES:
        return MISC
    if normalized in _EXACT:
        return _EXACT[normalized]

    tokens = TOKEN_PATTERN.findall(normalized)
    if not tokens:
        return UNKNOWN

    # The head noun of an English compound is its last token: "language model" is a
    # model, "sports team" is a team. Try the head first, then the modifiers.
    for candidate in (tokens[-1], *reversed(tokens[:-1])):
        if _singular(candidate) in _AMBIGUOUS_COLLECTIVES:
            return MISC
        klass = _classify_token(candidate)
        if klass is not None:
            return klass
    return MISC


def _singular(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _classify_token(token: str) -> str | None:
    """Classify one token, exactly or as the head of a closed compound.

    Matching is deliberately tight. An earlier version accepted a keyword that merely
    shared a prefix in either direction, which read "research question" as a PERSON
    (research/researcher) and "UserGroup" as a PERSON rather than an ORGANIZATION
    (user/usergroup) — then blocked it against "Group". Loose type matching is the very
    thing this module exists to stop.
    """
    singular = _singular(token)
    for klass, keywords in _KEYWORDS:
        for keyword in keywords:
            if singular == keyword or singular == _singular(keyword):
                return klass
            # Closed compound whose head noun is the keyword: usergroup -> group.
            if len(keyword) >= 4 and singular.endswith(keyword) and singular != keyword:
                return klass
    return None


def coarse_classes_compatible(left: str, right: str) -> bool:
    """Whether two coarse classes may describe the same entity.

    Permissive by design: only a confident disagreement between two specific classes
    blocks a merge. `MISC`/`UNKNOWN` never block, because that is where the technical
    vocabulary lands and blocking on it is precisely the measured defect.
    """
    if left == right:
        return True
    return left in PERMISSIVE_CLASSES or right in PERMISSIVE_CLASSES


def types_compatible(left_raw: str, right_raw: str) -> bool:
    """Convenience wrapper over two raw type strings."""
    return coarse_classes_compatible(coarse_class(left_raw), coarse_class(right_raw))


def dominant_coarse_class(raw_types: Iterable[str]) -> str:
    """The coarse class for a canonical entity, decided once from every observed type.

    A specific class outranks `MISC`/`UNKNOWN` even when those are more frequent: a
    single confident `PER` among a pile of generic labels is the informative signal.
    """
    counts: Counter[str] = Counter()
    for raw_type in raw_types:
        klass = coarse_class(raw_type)
        if klass != UNKNOWN:
            counts[klass] += 1
    if not counts:
        return UNKNOWN
    specific = {k: v for k, v in counts.items() if k not in PERMISSIVE_CLASSES}
    if specific:
        return max(sorted(specific), key=lambda key: specific[key])
    return max(sorted(counts), key=lambda key: counts[key])


def display_type(raw_types: Iterable[str]) -> str:
    """Most frequent raw type, kept as display metadata only.

    The emergent schema the architecture wants to observe lives in the full multiset;
    this is just the label shown on the node.
    """
    counts: Counter[str] = Counter(
        " ".join(str(item).split()) for item in raw_types if str(item or "").strip()
    )
    if not counts:
        return "Entity"
    top = max(counts.values())
    return sorted(name for name, count in counts.items() if count == top)[0]
