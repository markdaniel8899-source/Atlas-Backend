"""Keyword planning for the dynamic AutoBlog pipeline.

One Groq call produces the full keyword map: one primary keyword, 3-5
supporting keywords, 5-7 long-tail queries and 5-7 LSI terms. A deterministic
fallback derived from the headline keeps the pipeline alive if the model
returns junk.
"""

from __future__ import annotations

import re

from app.autoblog.llm.clients import (
    BlogPipelineError,
    extract_json,
    groq_research_with_nemotron_fallback,
)
from app.autoblog.models import KeywordPlan, TopicTrend
from app.autoblog.research import FactSheet
from app.autoblog.sources import STOPWORDS


_KEYWORD_SYSTEM = """You are an SEO strategist for ATLAS, an AI learning OS for students.

Build one keyword map for the article described by the user.

Return ONLY strict JSON (no fences, no commentary):
{
  "primary_keyword": "...",              // 1-4 words, lowercase, exact phrase a student would search
  "secondary_keywords": ["...", "..."],  // 3-5 supporting phrases, lowercase
  "long_tail_keywords": ["...", "..."],  // 5-7 question-style or 4+ word search phrases
  "lsi_keywords": ["...", "..."],        // 5-7 semantically related terms (entities, concepts, tools)
  "search_intent": "..."                 // informational | commercial | transactional | navigational
}

RULES:
- Primary keyword must appear naturally in a headline and must be specific to this story
- No keyword stuffing and no duplicate phrases across buckets
- Long-tail keywords must read like real searches, not keyword salad
- LSI terms must come from the supplied facts and entities, never invented
- Lowercase everything except product names and acronyms
"""


def _clean_keyword(value: str, *, max_words: int = 8) -> str:
    clean = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;!?-")
    if len(clean.split()) > max_words:
        clean = " ".join(clean.split()[:max_words])
    return clean


def _as_list(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _dedupe(values: list[str], *, exclude: set[str] | None = None) -> list[str]:
    blocked = exclude or set()
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        clean = _clean_keyword(value)
        key = clean.lower()
        if clean and key not in seen and key not in blocked:
            seen.add(key)
            out.append(clean)
    return out


def _fallback_primary(trend: TopicTrend) -> str:
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", trend.headline.lower())
        if word not in STOPWORDS and len(word) > 2
    ]
    if not words:
        return "ai learning"
    return " ".join(words[:3])


def _fallback_plan(trend: TopicTrend) -> KeywordPlan:
    primary = _fallback_primary(trend)
    return KeywordPlan(
        primary_keyword=primary,
        secondary_keywords=[
            f"{primary} 2026",
            f"{primary} for students",
            f"{primary} explained",
        ],
        long_tail_keywords=[
            f"what is {primary}",
            f"how does {primary} affect students",
            f"is {primary} worth using",
            f"{primary} benefits and risks",
            f"how to use {primary} for studying",
        ],
        lsi_keywords=[
            "AI learning tools",
            "study workflow",
            "productivity",
            "critical thinking",
            "academic integrity",
        ],
        search_intent=trend.search_intent or "informational",
    )


def _format_facts(fact_sheet: FactSheet) -> str:
    claims = "\n".join(f"- {claim}" for claim in fact_sheet.claims[:12]) or "(none)"
    numbers = "\n".join(
        f"- {item.get('value')} {item.get('unit')}: {item.get('context', '')[:160]}"
        for item in fact_sheet.numbers[:15]
    ) or "(none)"
    entities = "\n".join(
        f"- {item.get('name')} ({item.get('type', 'entity')}): {item.get('context', '')[:120]}"
        for item in fact_sheet.entities[:12]
    ) or "(none)"
    return f"Claims:\n{claims}\n\nNumbers:\n{numbers}\n\nEntities:\n{entities}"


async def generate_keyword_plan(trend: TopicTrend, fact_sheet: FactSheet) -> KeywordPlan:
    """Step 2: produce the keyword map used by the writer and validator."""
    fallback = _fallback_plan(trend)
    prompt = f"""Headline: {trend.headline}
Angle: {trend.angle}
Category: {trend.category}
Why now: {trend.why_now}
Suggested intent: {trend.search_intent}

FACT SHEET:
{_format_facts(fact_sheet)}

Build the keyword map now."""

    try:
        result = await groq_research_with_nemotron_fallback(
            _KEYWORD_SYSTEM,
            prompt,
            temperature=0.3,
            max_tokens=2048,
        )
        parsed = extract_json(result.text)
        data = parsed.data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            raise BlogPipelineError("Keyword planner returned invalid JSON.")

        primary = _clean_keyword(str(data.get("primary_keyword") or ""), max_words=5)
        if not primary:
            primary = fallback.primary_keyword

        secondary = _dedupe(
            _as_list(data.get("secondary_keywords")),
            exclude={primary.lower()},
        )[:5]
        long_tail = _dedupe(
            _as_list(data.get("long_tail_keywords")),
            exclude={primary.lower(), *(item.lower() for item in secondary)},
        )[:7]
        lsi = _dedupe(
            _as_list(data.get("lsi_keywords")),
            exclude={
                primary.lower(),
                *(item.lower() for item in secondary),
                *(item.lower() for item in long_tail),
            },
        )[:7]

        if not secondary:
            secondary = fallback.secondary_keywords[:3]
        if not long_tail:
            long_tail = fallback.long_tail_keywords[:5]
        if not lsi:
            lsi = fallback.lsi_keywords[:5]

        intent = str(data.get("search_intent") or trend.search_intent).strip().lower()
        if intent not in {"informational", "commercial", "transactional", "navigational"}:
            intent = "informational"

        plan = KeywordPlan(
            primary_keyword=primary,
            secondary_keywords=secondary,
            long_tail_keywords=long_tail,
            lsi_keywords=lsi,
            search_intent=intent,
        )
    except Exception as exc:
        print(
            f"[AutoBlog] Keyword planner failed "
            f"({type(exc).__name__}: {exc}) - using deterministic fallback.",
            flush=True,
        )
        plan = fallback

    print(
        f"[AutoBlog] Keywords: primary={plan.primary_keyword!r}, "
        f"secondary={len(plan.secondary_keywords)}, "
        f"long_tail={len(plan.long_tail_keywords)}, lsi={len(plan.lsi_keywords)}",
        flush=True,
    )
    return plan


__all__ = ["generate_keyword_plan"]
