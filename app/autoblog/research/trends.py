"""Groq-powered trend discovery for the dynamic AutoBlog pipeline.

The weekly-queries era is gone: every run asks Groq for 5-7 specific,
newsworthy headlines, filters out generic or duplicate ideas, scores the
remaining candidates with deterministic signals, then hands the winner to the
existing Tavily + fact-extraction research flow.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.autoblog.llm.clients import (
    BlogPipelineError,
    extract_json,
    groq_research_with_nemotron_fallback,
)
from app.autoblog.models import TopicTrend
from app.autoblog.research import FactSheet, run_research
from app.autoblog.sources import (
    AI_MODEL_RE,
    FALLBACK_TOPICS,
    STOPWORDS,
    fetch_recent_posts,
    research_chunk,
)


# ─── Constants ────────────────────────────────────────────────────────────────

MIN_TOPIC_WORDS = 4
DUPLICATE_OVERLAP = 0.65
MAX_RECENT_TITLES = 25


_TREND_SCOUT_SYSTEM = """You are the trend scout for ATLAS, an AI learning OS for students.

Find 5-7 SPECIFIC, newsworthy topics worth publishing today. Focus on AI models, AI products, AI research, AI at school or work, and the tools students actually use.

Return ONLY strict JSON (no fences, no commentary):
{
  "topics": [
    {
      "headline": "...",        // specific news-style headline, 40-120 chars, names the model/company/product
      "angle": "...",           // the lens: what it means for students, workers or learners
      "category": "...",        // one of: model_releases, industry_news, research_papers, education_tech, productivity, hardware_chips, ai_safety, open_source
      "why_now": "...",         // one sentence on why this is timely
      "search_intent": "..."    // informational | commercial | transactional | navigational
    }
  ]
}

RULES:
- Every headline must be specific: named models, companies, papers, products, measurable events
- Never return generic pillar terms such as "AI", "technology", "the future of AI"
- Do not repeat or lightly reword any recent post title
- Prefer stories from the last 7 days with a clear learning or work angle
- Mix categories; do not return 5 versions of the same story
- Headline must be readable by a student and free of hype
"""


# ─── Text helpers ─────────────────────────────────────────────────────────────


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clean_text(text: str, limit: int = 240) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    clean = clean.replace("\u2014", ",").replace("\u2013", ",")
    clean = re.sub(r"\s*,\s*,+", ", ", clean)
    return clean[:limit].strip(" ,")


def _tokens(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(word) > 2 and word not in STOPWORDS
    ]


def _overlap(left: str, right: str) -> float:
    left_words = set(_tokens(left))
    right_words = set(_tokens(right))
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / min(len(left_words), len(right_words))


def _is_generic(headline: str) -> bool:
    words = _tokens(headline)
    if len(words) < MIN_TOPIC_WORDS - 1:
        return True
    return len(set(words)) < 3


def _is_duplicate(headline: str, recent_titles: list[str]) -> bool:
    return any(_overlap(headline, title) >= DUPLICATE_OVERLAP for title in recent_titles)


def _normalize_topic(raw: Any) -> TopicTrend | None:
    if isinstance(raw, str):
        raw = {"headline": raw}
    if not isinstance(raw, dict):
        return None

    headline = _clean_text(str(raw.get("headline") or raw.get("title") or ""), 140)
    if not headline:
        return None

    angle = _clean_text(str(raw.get("angle") or raw.get("why_it_matters") or ""), 240)
    if not angle:
        angle = "What this means for students and self-directed learners"

    category = _clean_text(str(raw.get("category") or "industry_news"), 40).lower()
    valid_categories = {
        "model_releases",
        "industry_news",
        "research_papers",
        "education_tech",
        "productivity",
        "hardware_chips",
        "ai_safety",
        "open_source",
    }
    if category not in valid_categories:
        category = "industry_news"

    why_now = _clean_text(str(raw.get("why_now") or raw.get("reason") or ""), 200)
    if not why_now:
        why_now = "Fresh development with active search interest"

    intent = _clean_text(str(raw.get("search_intent") or "informational"), 30).lower()
    if intent not in {"informational", "commercial", "transactional", "navigational"}:
        intent = "informational"

    return TopicTrend(
        headline=headline,
        angle=angle,
        category=category,
        why_now=why_now,
        search_intent=intent,
    )


# ─── Discovery ────────────────────────────────────────────────────────────────


def _recent_titles(posts: list[dict[str, Any]]) -> list[str]:
    titles: list[str] = []
    for post in posts[:MAX_RECENT_TITLES]:
        title = _clean_text(str(post.get("title") or ""), 200)
        if title:
            titles.append(title)
    return titles


async def discover_trending_topics(
    recent_posts: list[dict[str, Any]] | None = None,
) -> list[TopicTrend]:
    """Ask Groq for today's candidates and keep only specific, fresh ideas."""
    if recent_posts is None:
        recent_posts = await fetch_recent_posts(MAX_RECENT_TITLES)

    recent_titles = _recent_titles(recent_posts)
    today = _utc_now().strftime("%B %d, %Y")
    recent_text = "\n".join(f"- {title}" for title in recent_titles) or "(none yet)"

    prompt = f"""Today is {today}.

Recent post titles (never duplicate or lightly reword these):
{recent_text}

Find today's 5-7 best topics now."""

    topics: list[TopicTrend] = []
    try:
        result = await groq_research_with_nemotron_fallback(
            _TREND_SCOUT_SYSTEM,
            prompt,
            temperature=0.55,
            max_tokens=2048,
        )
        parsed = extract_json(result.text)
        data = parsed.data
        if isinstance(data, dict):
            raw_topics = data.get("topics") or data.get("ideas") or []
        elif isinstance(data, list):
            raw_topics = data
        else:
            raw_topics = []

        if isinstance(raw_topics, dict):
            raw_topics = [raw_topics]
        if not isinstance(raw_topics, list):
            raw_topics = []

        for raw_topic in raw_topics:
            topic = _normalize_topic(raw_topic)
            if topic:
                topics.append(topic)
    except Exception as exc:
        print(
            f"[AutoBlog] Trend scout failed "
            f"({type(exc).__name__}: {exc}) - using fallback topics.",
            flush=True,
        )

    filtered: list[TopicTrend] = []
    seen: set[str] = set()
    for topic in topics:
        key = " ".join(_tokens(topic.headline))
        if not key or key in seen:
            continue
        if _is_generic(topic.headline):
            continue
        if _is_duplicate(topic.headline, recent_titles):
            print(f"[AutoBlog] Skipping duplicate trend: {topic.headline!r}", flush=True)
            continue
        seen.add(key)
        filtered.append(topic)

    if not filtered:
        month_year = _utc_now().strftime("%B %Y")
        for fallback in FALLBACK_TOPICS:
            headline = f"{fallback}: what changed in {month_year}"
            if _is_duplicate(headline, recent_titles):
                continue
            filtered.append(
                TopicTrend(
                    headline=headline,
                    angle="A practical look at what this means for students and lifelong learners",
                    category="education_tech",
                    why_now="Evergreen learning question with steady search demand",
                    search_intent="informational",
                )
            )
        if not filtered:
            filtered.append(
                TopicTrend(
                    headline=f"AI learning tools worth watching in {month_year}",
                    angle="How students can separate useful AI study tools from hype",
                    category="productivity",
                    why_now="Monthly roundup of tools students are actively testing",
                    search_intent="informational",
                )
            )

    print(
        f"[AutoBlog] Trend scout kept {len(filtered)} candidate topic(s).",
        flush=True,
    )
    return filtered[:7]


def _topic_score(topic: TopicTrend, recent_titles: list[str]) -> float:
    now = _utc_now()
    score = 0.0
    headline_lower = topic.headline.lower()

    if str(now.year) in headline_lower:
        score += 3.0
    if now.strftime("%B").lower() in headline_lower:
        score += 2.0
    for word in ("this week", "today", "launch", "launched", "release", "released", "update"):
        if word in headline_lower:
            score += 1.0

    score += min(len(AI_MODEL_RE.findall(topic.headline)) * 2.0, 4.0)

    if topic.category in {"model_releases", "research_papers", "education_tech", "industry_news"}:
        score += 2.0
    if topic.search_intent in {"informational", "commercial"}:
        score += 1.0

    for title in recent_titles:
        overlap = _overlap(topic.headline, title)
        if overlap >= 0.35:
            score -= 3.0
        elif overlap >= 0.2:
            score -= 1.0

    words = set(_tokens(topic.headline))
    score += min(len(words) * 0.15, 2.0)
    if any(char.isdigit() for char in topic.headline):
        score += 0.5
    return score


def select_best_topic(
    topics: list[TopicTrend],
    recent_posts: list[dict[str, Any]] | None = None,
) -> TopicTrend:
    """Pick the strongest candidate with deterministic, explainable signals."""
    if not topics:
        raise BlogPipelineError("Trend discovery produced no usable topics.")

    recent_titles = _recent_titles(recent_posts or [])
    ranked = sorted(
        topics,
        key=lambda topic: _topic_score(topic, recent_titles),
        reverse=True,
    )
    best = ranked[0]
    print(
        f"[AutoBlog] Selected trend: {best.headline!r} "
        f"(category={best.category}, why_now={best.why_now!r})",
        flush=True,
    )
    return best


# ─── Fact sheet fallback ──────────────────────────────────────────────────────


_NUMBER_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(%|percent|billion|million|trillion|thousand|ms|GB|TB|MB|tokens?|users?|USD|\$)",
    re.IGNORECASE,
)


def _fallback_fact_sheet(raw_results: list[dict[str, str]]) -> FactSheet:
    """Build a usable fact sheet from Tavily snippets when extraction is empty."""
    claims: list[str] = []
    numbers: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    source_urls: list[str] = []

    for result in raw_results[:15]:
        url = str(result.get("url") or "").strip()
        title = str(result.get("title") or "").strip()
        content = re.sub(r"\s+", " ", str(result.get("content") or "")).strip()
        if url:
            source_urls.append(url)

        for sentence in re.split(r"(?<=[.!?])\s+", content):
            sentence = sentence.strip()
            if 40 <= len(sentence) <= 240 and len(claims) < 12:
                claims.append(sentence)

        if title:
            for model in AI_MODEL_RE.findall(title)[:2]:
                entities.append(
                    {
                        "name": " ".join(model.split()),
                        "type": "model",
                        "context": title[:160],
                        "source_url": url,
                    }
                )

        for match in _NUMBER_RE.finditer(content):
            if len(numbers) >= 20:
                break
            value, unit = match.group(1), match.group(2)
            start = max(match.start() - 100, 0)
            end = min(match.end() + 100, len(content))
            numbers.append(
                {
                    "value": value,
                    "unit": unit,
                    "context": content[start:end].strip(),
                    "source_url": url,
                }
            )

    deduped_entities: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    for entity in entities:
        key = entity["name"].lower()
        if key not in seen_entities:
            seen_entities.add(key)
            deduped_entities.append(entity)

    return FactSheet(
        claims=claims[:12],
        numbers=numbers[:20],
        entities=deduped_entities[:12],
        source_urls=list(dict.fromkeys(source_urls))[:15],
        raw_snippets=[
            f"{result.get('title', '')} ({result.get('url', '')}): {result.get('content', '')[:500]}"
            for result in raw_results[:30]
        ],
    )


async def research_topic(trend: TopicTrend) -> tuple[FactSheet, list[dict[str, str]]]:
    """Run the existing Tavily + fact-extraction flow for the chosen topic."""
    try:
        fact_sheet, raw_results = await run_research(
            trend.headline,
            trend.angle,
            "analysis",
        )
    except Exception as exc:
        print(
            f"[AutoBlog] Full research failed "
            f"({type(exc).__name__}: {exc}) - falling back to raw Tavily snippets.",
            flush=True,
        )
        raw_results = await research_chunk(trend.headline)
        fact_sheet = _fallback_fact_sheet(raw_results)

    if not fact_sheet.claims and not fact_sheet.numbers:
        fact_sheet = _fallback_fact_sheet(raw_results)

    if not fact_sheet.claims and not fact_sheet.numbers and not fact_sheet.source_urls:
        raise BlogPipelineError(
            f"Research returned no usable facts for {trend.headline!r}."
        )

    return fact_sheet, raw_results


async def run_trend_research() -> tuple[TopicTrend, FactSheet, list[dict[str, str]]]:
    """Step 1: discover a fresh topic, then ground it with Tavily research."""
    recent_posts = await fetch_recent_posts(MAX_RECENT_TITLES)
    candidates = await discover_trending_topics(recent_posts)
    trend = select_best_topic(candidates, recent_posts)
    fact_sheet, raw_results = await research_topic(trend)
    return trend, fact_sheet, raw_results


__all__ = [
    "discover_trending_topics",
    "select_best_topic",
    "research_topic",
    "run_trend_research",
]
