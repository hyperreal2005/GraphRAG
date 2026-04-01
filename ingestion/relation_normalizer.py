"""Relation label normalization and canonicalization."""

from __future__ import annotations


def sanitize_rel_type(relation: str) -> str:
    """Uppercase + replace non-alnum with underscore."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in relation.upper())
    cleaned = cleaned.strip("_") or "RELATED_TO"
    return cleaned


def canonicalize_relation(relation: str) -> str:
    """Map free-form relation labels into stable canonical labels."""
    label = sanitize_rel_type(relation)
    canonical_labels = {
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
    }
    if label in canonical_labels:
        return label

    tokens = [token for token in label.split("_") if token]
    token_set = set(tokens)

    exact_map = {
        "IS_A": "TYPE_OF",
        "IS_A_TYPE_OF": "TYPE_OF",
        "IS_TYPE_OF": "TYPE_OF",
        "IS_A_FORM_OF": "TYPE_OF",
        "IS_FORM_OF": "TYPE_OF",
        "IS_EXAMPLE_OF": "TYPE_OF",
        "IS_PART_OF": "PART_OF",
        "PART_OF": "PART_OF",
        "COMPONENT_OF": "PART_OF",
        "DEVELOPED_BY": "DEVELOPED_BY",
        "CREATED_BY": "DEVELOPED_BY",
        "AUTHORED_BY": "DEVELOPED_BY",
        "AUTHORED": "DEVELOPED_BY",
        "PUBLISHED_IN": "PUBLISHED_IN",
        "PUBLISHED_ON": "PUBLISHED_IN",
        "LOCATED_IN": "LOCATED_IN",
        "USED_FOR": "USES",
        "USED_IN": "USES",
        "UTILIZED_IN": "USES",
        "RELIES_ON": "REQUIRES",
        "RELY_ON": "REQUIRES",
        "DEPENDS_ON": "REQUIRES",
        "IS": "RELATED_TO",
        "FOR": "RELATED_TO",
        "BETWEEN": "RELATED_TO",
        "MAKES": "RELATED_TO",
        "PASSES": "RELATED_TO",
        "MATCHED": "RELATED_TO",
        "RELATED_TO": "RELATED_TO",
    }
    if label in exact_map:
        return exact_map[label]

    def has_prefix(*prefixes: str) -> bool:
        for token in tokens:
            for prefix in prefixes:
                if token.startswith(prefix):
                    return True
        return False

    if has_prefix("DEVELOP", "AUTHOR", "CREATE", "BUILD") and "BY" in token_set:
        return "DEVELOPED_BY"
    if has_prefix("TYPE", "FORM", "CLASS", "KIND", "EXAMPLE", "INSTANCE", "ITERATION"):
        return "TYPE_OF"
    if ({"PART", "OF"} <= token_set) or has_prefix("COMPONENT", "MEMBER"):
        return "PART_OF"
    if has_prefix("PUBLISH", "JOURNAL", "ISSUE"):
        return "PUBLISHED_IN"
    if has_prefix("LOCAT", "HOST") and ({"IN"} & token_set or {"AT"} & token_set):
        return "LOCATED_IN"

    if has_prefix("USE", "UTILIZ", "EMPLOY", "LEVERAG", "APPL", "POWER"):
        return "USES"
    if has_prefix("PROVID", "OFFER", "DELIVER", "RETURN"):
        return "PROVIDES"
    if has_prefix("REQUIR", "NEED", "MUST"):
        return "REQUIRES"
    if has_prefix("IMPROV", "ENHANC", "ADVANC", "ACCELERAT"):
        return "IMPROVES"
    if has_prefix("SUPPORT", "ENABLE", "FACILIT", "ALLOW", "HELP", "ASSIST", "EMPOWER"):
        return "SUPPORTS"
    if has_prefix("INCLUD", "COMPRIS", "CONTAIN", "CONSIST"):
        return "INCLUDES"
    if has_prefix("INTEGRAT", "COMBIN", "BLEND", "MERG"):
        return "INTEGRATES"
    if has_prefix("STORE", "SAVE", "INDEX", "EMBED"):
        return "STORES"
    if has_prefix("PROCESS", "CONVERT", "SPLIT", "EXTRACT", "TRANSFORM"):
        return "PROCESSES"
    if has_prefix("OPERAT", "RUN", "EXECUT", "HANDLE"):
        return "PROCESSES"
    if has_prefix("EVALUAT", "ASSESS", "TEST", "MEASURE", "VALIDAT"):
        return "EVALUATED_WITH"
    if has_prefix("ANALYZ", "INVESTIGAT", "EXAMIN", "STUDY", "SCRUTIN"):
        return "ANALYZES"
    if has_prefix("ACCESS", "QUERY", "SEARCH", "READ", "NAVIGAT", "ASK"):
        return "ACCESSES"
    if has_prefix("TARGET", "AIM", "PLAN", "PROPOSE"):
        return "TARGETS"
    if has_prefix("RELAT", "ASSOCIAT", "LINK", "CONNECT", "CORRELAT"):
        return "RELATED_TO"

    # Conservative fallback: avoid long-tail relation explosion.
    return "RELATED_TO"
