"""Deterministic, app-only cleanup of conversational filler in retrieval queries."""

import re

_LEADING_PLEASE = re.compile(r"(?i)^\s*please\b(?:\s*[,;:])?\s*")
_TRAILING_PLEASE = re.compile(r"(?i)(?:\s*[,;:]\s*|\s+)please[.!?]*\s*$")
_GREETING = re.compile(r"(?i)^\s*(?:hello|hi|hey)\b(?:\s+there\b)?[\s,;:!-]*")
_LEADING_FILLER = (
    re.compile(
        r"(?i)^\s*(?:can|could|would)\s+you\s+(?:please\s+)?"
        r"(?:(?:tell|show|explain)\s+me|help(?:\s+me)?(?:\s+figure\s+out)?)"
        r"(?:\s+with)?\b[\s,;:!-]*"
    ),
    re.compile(r"(?i)^\s*(?:tell|show)\s+me\b[\s,;:!-]*"),
    re.compile(r"(?i)^\s*help(?:\s+me)?(?:\s+figure\s+out)?(?:\s+with)?\b[\s,;:!-]*"),
    re.compile(r"(?i)^\s*i\s+(?:just\s+)?(?:want|need)\s+to\s+know\b[\s,;:!-]*"),
    re.compile(r"(?i)^\s*i(?:'d|\s+would)\s+like\s+to\s+know\b[\s,;:!-]*"),
    re.compile(r"(?i)^\s*is\s+it\s+possible\s+to\b[\s,;:!-]*"),
    re.compile(r"(?i)^\s*(?:do\s+you\s+know|i\s+was\s+wondering)\b[\s,;:!-]*"),
)
_DISTRO_DEVICE = re.compile(
    r"(?i)\bmy\s+"
    r"(?P<distro>(?:ubuntu|debian|fedora|arch(?:\s+linux)?|linux\s+mint|"
    r"open\s*suse|centos|rhel)(?:\s+\d+(?:\.\d+)*)?)\s+"
    r"(?:machine|computer|laptop|pc)\b"
)
_DEVICE_WITH_PREPOSITION = re.compile(
    r"(?i)\b(?:on|from|with|using)\s+my\s+(?:machine|computer|laptop|pc)\b"
)
_MY_DEVICE = re.compile(r"(?i)\bmy\s+(machine|computer|laptop|pc)\b")
_TRAILING_THANKS = re.compile(
    r"(?i)(?:\s*[,;.!-]\s*|\s+)(?:thanks|thank\s+you)(?:\s+very\s+much)?[.!?]*\s*$"
)
_TRAILING_QUESTION = re.compile(r"\s*\?+\s*$")


def normalize_query(query):
    """Remove only known conversational filler, preserving all retained token bytes."""
    original = str(query)
    normalized = original

    while True:
        before = normalized
        normalized = _GREETING.sub("", normalized)
        normalized = _LEADING_PLEASE.sub("", normalized)
        for pattern in _LEADING_FILLER:
            normalized = pattern.sub("", normalized)
        if normalized == before:
            break

    normalized = _DISTRO_DEVICE.sub(lambda match: match.group("distro"), normalized)
    normalized = _DEVICE_WITH_PREPOSITION.sub("", normalized)
    normalized = _MY_DEVICE.sub(lambda match: match.group(1), normalized)
    normalized = _TRAILING_THANKS.sub("", normalized)
    normalized = _TRAILING_PLEASE.sub("", normalized)
    normalized = _TRAILING_QUESTION.sub("", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized).strip(" \t,;:-")
    return normalized if normalized else original


def normalize_retrieval_query(question, contextual_query=None):
    """Normalize the current question while retaining any app-added retrieval context."""
    normalized_question = normalize_query(question)
    if contextual_query is None:
        return normalized_question
    contextual_query = str(contextual_query)
    if question and contextual_query.endswith(question):
        prefix = contextual_query[:-len(question)].rstrip()
        return " ".join(part for part in (prefix, normalized_question) if part)
    return normalize_query(contextual_query)
