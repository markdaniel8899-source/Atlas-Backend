from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .prompts import keyword_messages
from .types import (
    KeywordPlan,
    ResearchBundle,
    SearchIntent,
    TopicCandidate,
)

if TYPE_CHECKING:
    from .providers import BlogProviderClient, ProviderJsonResult


class KeywordValidationError(ValueError):
    pass


_INTENT_VALUES = {intent.value for intent in SearchIntent}


def validate_keyword_plan(data: Any) -> KeywordPlan:
    if not isinstance(data, dict):
        raise KeywordValidationError("Keyword response must be a JSON object.")

    primary = _clean_text(data.get("primary"), "primary")
    primary_words = primary.split()
    if len(primary_words) < 2 or len(primary_words) > 5:
        raise KeywordValidationError("Primary keyword must be 2-5 words.")
    if len(primary) > 100:
        raise KeywordValidationError("Primary keyword is too long.")

    secondary = _clean_list(data.get("secondary"), "secondary", 3, 6)
    long_tail = _clean_list(data.get("long_tail"), "long_tail", 4, 8)
    lsi = _clean_list(data.get("lsi"), "lsi", 5, 10)
    title_candidates = _clean_list(
        data.get("title_candidates"), "title_candidates", 2, 6
    )
    intent = _clean_text(
        data.get("search_intent") or "informational", "search_intent"
    ).lower()
    if intent not in _INTENT_VALUES:
        raise KeywordValidationError(
            "search_intent must be informational, commercial, "
            "transactional, or navigational."
        )

    return KeywordPlan(
        primary=primary,
        secondary=secondary,
        long_tail=long_tail,
        lsi=lsi,
        search_intent=SearchIntent(intent),
        title_candidates=title_candidates,
    )


def _clean_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KeywordValidationError(f"{name} must be a non-empty string.")
    return re.sub(r"\s+", " ", value).strip()


def _clean_list(
    value: Any,
    name: str,
    minimum: int,
    maximum: int,
) -> list[str]:
    if not isinstance(value, list):
        raise KeywordValidationError(f"{name} must be a list.")
    if not minimum <= len(value) <= maximum:
        raise KeywordValidationError(
            f"{name} must contain {minimum}-{maximum} items."
        )
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item, name)
        key = text.lower()
        if key in seen:
            raise KeywordValidationError(
                f"{name} must not contain duplicates."
            )
        seen.add(key)
        cleaned.append(text)
    return cleaned


async def generate_keywords(
    client: "BlogProviderClient",
    topic: TopicCandidate,
    research: ResearchBundle,
) -> "ProviderJsonResult[KeywordPlan]":
    from .providers import BlogStage

    messages = keyword_messages(topic=topic, research=research)
    return await client.complete_json(
        BlogStage.KEYWORDS,
        messages,
        validator=validate_keyword_plan,
    )
