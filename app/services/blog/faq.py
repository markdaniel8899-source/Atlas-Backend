from __future__ import annotations

import html
import re
from typing import Any, Sequence

from .prompts import faq_messages
from .providers import BlogProviderClient, BlogStage, ProviderChainError
from .types import (
    FAQAnswerSchema,
    FAQBundle,
    FAQItem,
    FAQPageSchema,
    FAQQuestionSchema,
    KeywordPlan,
    ResearchBundle,
)
from .writer import extract_h2s, normalize_heading, plain_text


class FAQError(RuntimeError):
    pass


ADVICE_RE = re.compile(
    r"\b(you should|i recommend|you must|guaranteed|risk[- ]free|"
    r"get rich|buy now|invest in|personalized advice)\b",
    re.IGNORECASE,
)


def validate_faq_payload(
    data: Any,
    *,
    content: str,
    research: ResearchBundle,
) -> list[FAQItem]:
    raw = data.get("faqs") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise ValueError("FAQ payload must contain a faqs array.")
    items = [FAQItem.model_validate(item) for item in raw]
    cleaned = [
        FAQItem(
            question=_clean(item.question),
            answer=_clean(item.answer),
        )
        for item in items
    ]
    errors = faq_errors(cleaned, content=content, research=research)
    if errors:
        raise ValueError("; ".join(errors))
    return cleaned


def faq_errors(
    faqs: Sequence[FAQItem],
    *,
    content: str,
    research: ResearchBundle,
) -> list[str]:
    errors: list[str] = []
    if not 4 <= len(faqs) <= 6:
        errors.append(f"FAQ count must be 4-6, got {len(faqs)}.")
    seen: set[str] = set()
    source_numbers = {
        _normalize_number(token)
        for token in _number_tokens(
            " ".join(
                [
                    plain_text(content),
                    *(
                        f"{fact.value:g} {fact.unit} {fact.context}"
                        for fact in research.numeric_facts
                    ),
                ]
            )
        )
    }
    for index, item in enumerate(faqs, start=1):
        question = item.question.strip()
        answer = item.answer.strip()
        if not question:
            errors.append(f"FAQ {index} has an empty question.")
        if not question.endswith("?"):
            errors.append(f"FAQ {index} question must end with '?'.")
        if not 8 <= len(question) <= 160:
            errors.append(f"FAQ {index} question length is invalid.")
        if not answer:
            errors.append(f"FAQ {index} has an empty answer.")
        answer_words = len(re.findall(r"\S+", answer))
        if not 40 <= answer_words <= 80:
            errors.append(
                f"FAQ {index} answer must be 40-80 words, got {answer_words}."
            )
        key = _question_key(question)
        if key in seen:
            errors.append(f"FAQ {index} duplicates another question.")
        seen.add(key)
        unsupported = [
            token
            for token in _number_tokens(answer)
            if _normalize_number(token) not in source_numbers
        ]
        if unsupported:
            errors.append(
                f"FAQ {index} introduces unsupported number(s): "
                f"{', '.join(unsupported)}."
            )
        if ADVICE_RE.search(answer):
            errors.append(f"FAQ {index} contains personalized advice.")
    return errors


def build_faq_bundle(faqs: Sequence[FAQItem]) -> FAQBundle:
    return FAQBundle(
        faqs=list(faqs),
        faq_schema=FAQPageSchema(
            main_entity=[
                FAQQuestionSchema(
                    name=item.question,
                    accepted_answer=FAQAnswerSchema(text=item.answer),
                )
                for item in faqs
            ]
        ),
    )


async def generate_faqs(
    client: BlogProviderClient,
    *,
    content: str,
    keywords: KeywordPlan,
    research: ResearchBundle,
) -> FAQBundle:
    messages = faq_messages(
        content=content,
        keywords=keywords,
        research=research,
    )

    def validator(data: Any) -> list[FAQItem]:
        return validate_faq_payload(
            data,
            content=content,
            research=research,
        )

    try:
        result = await client.complete_json(
            BlogStage.FAQ,
            messages,
            validator=validator,
        )
    except ProviderChainError as exc:
        raise FAQError("All FAQ providers failed.") from exc
    return build_faq_bundle(result.data)


def insert_faq_section(content: str, faqs: Sequence[FAQItem]) -> str:
    if not faqs:
        raise FAQError("Cannot insert an empty FAQ section.")
    body = "\n".join(
        f"<h3>{html.escape(item.question)}</h3>\n"
        f"<p>{html.escape(item.answer)}</p>"
        for item in faqs
    )
    section = f"<h2>FAQ</h2>\n{body}"

    headings = extract_h2s(content)
    existing = next(
        (
            heading
            for heading in headings
            if normalize_heading(heading.text)
            in {"faq", "frequently asked questions"}
        ),
        None,
    )
    if existing is not None:
        content = content[: existing.start] + content[existing.body_end :]

    sources = next(
        (
            heading
            for heading in extract_h2s(content)
            if normalize_heading(heading.text) == "sources"
        ),
        None,
    )
    if sources is None:
        return f"{content.rstrip()}\n\n{section}\n"
    before = content[: sources.start].rstrip()
    after = content[sources.start :].lstrip()
    return f"{before}\n\n{section}\n\n{after}"


def _clean(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _question_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _number_tokens(value: str) -> list[str]:
    return re.findall(r"\d[\d,]*(?:\.\d+)?", value or "")


def _normalize_number(value: str) -> str:
    return value.replace(",", "").rstrip(".")
