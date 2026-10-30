from __future__ import annotations

import re
from typing import Any

from .prompts import meta_messages
from .providers import BlogProviderClient, BlogStage, ProviderChainError
from .types import (
    KeywordPlan,
    MetaPayload,
    MetaResult,
    TopicCandidate,
    WriterOutput,
)
from .writer import extract_h2s, first_words, plain_text


class MetaError(RuntimeError):
    pass


CLICKBAIT_RE = re.compile(
    r"\b(you won't believe|shocking|unbelievable|this one trick|"
    r"mind-blowing|insane|must-see|secret hack|guaranteed)\b",
    re.IGNORECASE,
)
CTA_RE = re.compile(
    r"\b(learn|start|explore|discover|read|see|build|master|try|use|get)\b",
    re.IGNORECASE,
)
QUOTE_RE = re.compile(r"[\"“”]")
ACRONYMS = {
    "AI",
    "API",
    "ATLAS",
    "CPU",
    "CSS",
    "GPU",
    "HTML",
    "LLM",
    "LLMS",
    "RAG",
    "SEO",
    "SQL",
}


def validate_meta_payload(
    data: Any,
    *,
    primary_keyword: str,
    source_text: str = "",
) -> MetaPayload:
    payload = (
        data if isinstance(data, MetaPayload) else MetaPayload.model_validate(data)
    )
    errors = meta_errors(
        payload,
        primary_keyword=primary_keyword,
        source_text=source_text,
    )
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def meta_errors(
    payload: MetaPayload,
    *,
    primary_keyword: str,
    source_text: str = "",
) -> list[str]:
    errors: list[str] = []
    primary = primary_keyword.strip()
    title = payload.meta_title.strip()
    description = payload.meta_description.strip()

    if not 50 <= len(title) <= 60:
        errors.append(f"Meta title must be 50-60 chars, got {len(title)}.")
    if not primary.lower() in title[:30].lower():
        errors.append("Primary keyword must appear in the first 30 title chars.")
    if not 150 <= len(description) <= 160:
        errors.append(
            f"Meta description must be 150-160 chars, got {len(description)}."
        )
    if not primary.lower() in description.lower():
        errors.append("Primary keyword is missing from the meta description.")
    if not CTA_RE.search(description):
        errors.append("Meta description needs a clear CTA.")
    if CLICKBAIT_RE.search(f"{title} {description}"):
        errors.append("Meta contains clickbait.")
    if QUOTE_RE.search(f"{title} {description}") or "—" in f"{title} {description}":
        errors.append("Meta contains quotation marks or an em dash.")
    if _has_shouting(title) or _has_shouting(description):
        errors.append("Meta contains an all-caps word.")
    unsupported = _new_numbers(
        f"{title} {description}",
        f"{primary} {source_text}",
    )
    if unsupported:
        errors.append(f"Meta introduces unsupported number(s): {unsupported}.")
    return errors


def fallback_meta(
    *,
    title: str,
    excerpt: str,
    primary_keyword: str,
) -> MetaPayload:
    primary = _sanitize(primary_keyword)
    if not primary:
        raise MetaError("Primary keyword is required for metadata.")
    if len(primary) > 30:
        raise MetaError("Primary keyword is too long for the title prefix.")

    title_candidates = [_clean(title)]
    if not title_candidates[0].lower().startswith(primary.lower()):
        title_candidates.insert(0, f"{primary}: {_clean(title)}")
    meta_title = ""
    for candidate in title_candidates:
        try:
            meta_title = _fit_title(candidate, primary)
            break
        except MetaError:
            continue
    if not meta_title:
        raise MetaError("Could not build a valid fallback meta title.")

    body = _sanitize(excerpt)
    description = f"Learn {primary} with ATLAS: {body}".strip()
    pads = (
        " Explore practical steps and examples.",
        " Start learning with ATLAS today.",
        " Build a repeatable study system with clear next steps.",
    )
    for pad in pads:
        if len(description) >= 150:
            break
        description += pad
    if len(description) > 160:
        description = _trim_words(description, 159)
    description = _sentence_end(description)
    if len(description) < 150:
        description += " Learn more with ATLAS."
        description = _trim_words(description, 159)
        description = _sentence_end(description)

    payload = MetaPayload(
        meta_title=meta_title,
        meta_description=description,
    )
    errors = meta_errors(
        payload,
        primary_keyword=primary,
        source_text=f"{title} {excerpt} {primary}",
    )
    if errors:
        raise MetaError("Fallback metadata is invalid: " + "; ".join(errors))
    return payload


async def generate_meta(
    client: BlogProviderClient,
    topic: TopicCandidate,
    keywords: KeywordPlan,
    output: WriterOutput,
) -> MetaResult:
    headings = [heading.text for heading in extract_h2s(output.content)]
    source_text = " ".join(
        (
            output.title,
            output.excerpt,
            plain_text(output.content),
            keywords.primary,
        )
    )
    messages = meta_messages(
        title=output.title,
        excerpt=output.excerpt,
        primary_keyword=keywords.primary,
        secondary_keywords=keywords.secondary,
        first_words=first_words(output.content, 500),
        headings=headings,
    )

    def validator(data: Any) -> MetaPayload:
        return validate_meta_payload(
            data,
            primary_keyword=keywords.primary,
            source_text=source_text,
        )

    try:
        result = await client.complete_json(
            BlogStage.META,
            messages,
            validator=validator,
        )
        return MetaResult(
            meta_title=result.data.meta_title,
            meta_description=result.data.meta_description,
            fallback=False,
        )
    except ProviderChainError:
        fallback = fallback_meta(
            title=output.title,
            excerpt=output.excerpt,
            primary_keyword=keywords.primary,
        )
        return MetaResult(
            meta_title=fallback.meta_title,
            meta_description=fallback.meta_description,
            fallback=True,
        )


def _fit_title(candidate: str, primary: str) -> str:
    title = _sanitize(candidate)
    if not primary.lower() in title[:30].lower():
        title = _sanitize(f"{primary}: {title}")
    if len(title) > 60:
        title = _trim_words(title, 60).rstrip(" ,.;:-")
    suffixes = (
        " | ATLAS",
        " Guide | ATLAS",
        " for Learners | ATLAS",
        " - A Practical Guide",
    )
    for suffix in suffixes:
        if 50 <= len(title) <= 60:
            break
        trial = title + suffix
        if len(trial) <= 60:
            title = trial
    if not 50 <= len(title) <= 60:
        raise MetaError("Could not fit the title into 50-60 characters.")
    if not primary.lower() in title[:30].lower():
        raise MetaError("Primary keyword is not in the first 30 title characters.")
    return title


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _sanitize(value: str) -> str:
    value = QUOTE_RE.sub("", value or "")
    value = value.replace("—", "-")

    def soften(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if token in ACRONYMS else token.title()

    value = re.sub(r"\b[A-Z]{4,}\b", soften, value)
    return _clean(value)


def _trim_words(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    clipped = value[: limit + 1]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,.;:-")


def _sentence_end(value: str) -> str:
    value = value.rstrip(" ,.;:-")
    if value and value[-1] not in ".!?":
        value += "."
    return value


def _has_shouting(value: str) -> bool:
    return any(
        token.isupper() and len(token) >= 4 and token not in ACRONYMS
        for token in re.findall(r"[A-Z]{2,}", value)
    )


def _new_numbers(text: str, source: str) -> list[str]:
    source_numbers = {
        _normalize_number(token) for token in _number_tokens(source)
    }
    return sorted(
        {
            token
            for token in _number_tokens(text)
            if _normalize_number(token) not in source_numbers
        }
    )


def _number_tokens(value: str) -> list[str]:
    return re.findall(r"\d[\d,]*(?:\.\d+)?", value or "")


def _normalize_number(value: str) -> str:
    return value.replace(",", "").rstrip(".")
