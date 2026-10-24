"""Research modules for AutoBlog v2 - Topic selection, Keyword extraction, Data analysis."""

from __future__ import annotations

import asyncio
import difflib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.llm.clients import nim_chat, extract_json, BlogPipelineError
from app.autoblog.sources import (
    ordered_research_queries,
    research_chunk,
    fetch_recent_posts,
    research_snippets,
    BLOCKED_TOPIC_JUNK,
    FALLBACK_TOPICS,
    STOPWORDS,
    _result_line,
    _dedupe_results,
)
from app.autoblog.planner import (
    ARTICLE_PROFILES,
    ARTICLE_TYPES,
    upgrade_article_type,
    build_profile,
    MIN_CHART_NUMBERS,
    MIN_TABLE_ENTITIES,
    MIN_TABLE_ATTRIBUTES,
)


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Research:
    topic: str
    keywords: list[str]
    snippets: list[str]
    source_urls: list[str] | None = None
    article_type: str = "explainer"
    profile: Any = None
    data_availability: Any = None


@dataclass(slots=True)
class KeywordPlan:
    keywords: list[str]
    article_type: str


@dataclass(slots=True)
class DataAvailability:
    numbers: list[str]
    examples: list[str]
    named_items: list[str]
    attributes: list[str]
    has_chart_data: bool
    has_table_data: bool
    summary: str


# ─── Constants ────────────────────────────────────────────────────────────────


TOPIC_RETRY_LIMIT = 2
SIMILARITY_THRESHOLD = 0.70


# ─── Helper Functions ─────────────────────────────────────────────────────────


def _topic_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _topic_tokens(value: str) -> set[str]:
    return {
        token
        for token in _topic_text(value).split()
        if token not in STOPWORDS and len(token) > 1
    }


def _topic_sequence_similarity(first: str, second: str) -> float:
    first_text, second_text = _topic_text(first), _topic_text(second)
    if not first_text or not second_text:
        return 0.0
    return difflib.SequenceMatcher(None, first_text, second_text).ratio()


def _topic_similarity(first: str, second: str) -> float:
    sequence = _topic_sequence_similarity(first, second)
    first_tokens, second_tokens = _topic_tokens(first), _topic_tokens(second)
    if not first_tokens or not second_tokens:
        return sequence
    overlap = len(first_tokens & second_tokens)
    jaccard = overlap / len(first_tokens | second_tokens)
    containment = overlap / min(len(first_tokens), len(second_tokens))
    return max(sequence, jaccard, containment)


def _find_duplicate(
    topic: str, recent_posts: list[dict[str, Any]]
) -> tuple[float, str]:
    best_score = 0.0
    best_title = ""
    for post in recent_posts:
        title = str(post.get("title") or "").strip()
        if title:
            score = _topic_similarity(topic, title)
            if score > best_score:
                best_score, best_title = score, title
        keywords = post.get("keywords")
        if not isinstance(keywords, list):
            continue
        for keyword in keywords:
            text = str(keyword or "").strip()
            if not text or text.lower() in BLOCKED_TOPIC_JUNK:
                continue
            score = _topic_sequence_similarity(topic, text)
            if score > best_score:
                best_score, best_title = score, text
    return best_score, best_title


def _blocked_topics(recent_posts: list[dict[str, Any]]) -> list[str]:
    blocked: list[str] = []
    seen: set[str] = set()
    for post in recent_posts:
        values: list[Any] = [post.get("title")]
        keywords = post.get("keywords")
        if isinstance(keywords, list):
            values.extend(keywords)
        for value in values:
            text = str(value or "").strip()
            key = text.lower()
            if text and key not in seen and key not in BLOCKED_TOPIC_JUNK:
                seen.add(key)
                blocked.append(text[:160])
    return blocked[:30]


def _heuristic_keywords(text: str, limit: int = 5) -> list[str]:
    words = re.findall(r"[a-z][a-z'-]{2,}", text.lower())
    counts = Counter(w for w in words if w not in STOPWORDS)
    out: list[str] = []
    for word, _ in counts.most_common():
        if word not in out:
            out.append(word)
        if len(out) >= limit:
            break
    return out


def _pad_keywords(keywords: list[str], topic: str) -> list[str]:
    settings = get_autoblog_settings()
    tokens = [t for t in re.findall(r"[a-z0-9]{2,}", topic.lower()) if t not in STOPWORDS]
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    out: list[str] = []
    for value in [*keywords, *bigrams, *tokens, *_heuristic_keywords(topic)]:
        clean = value.strip().lower()
        if clean and clean not in out:
            out.append(clean)
        if len(out) >= settings.max_results:
            break
    return out or [topic.lower()]


# ─── Topic Selection (Nemotron Ultra) ─────────────────────────────────────────


async def topic_chunk(
    results: list[dict[str, str]],
    blocked_topics: list[str] | None = None,
    category: str | None = None,
) -> str:
    """Nemotron Ultra picks the day's trending topic."""
    if not results:
        from app.autoblog.sources import _day_index
        topic = FALLBACK_TOPICS[_day_index() % len(FALLBACK_TOPICS)]
        print(f"[AutoBlog] research fallback -> topic={topic!r}")
        return topic

    blob = "\n".join(_result_line(result) for result in results)
    system = """You are an SEO trend analyst for ATLAS, a learning OS for students.
From the research snippets, pick ONE specific, currently trending, searchable blog topic (not a generic pillar term).

Return ONLY strict JSON (no fences, no commentary):
{"topic": "..."}

- topic: a concrete angle a reader would click, max 120 chars.
- Prefer a specific model, company, research paper, or measurable news event over generic advice.
- Never repeat an angle from the recently published posts supplied in the prompt."""
    if blocked_topics:
        avoid = "\n".join(f"- {topic[:160]}" for topic in blocked_topics[:20])
        system += (
            "\n\nRECENTLY PUBLISHED (strictly avoid these angles):\n"
            f"{avoid}\n"
            "Choose a clearly different story, model, company, or event."
        )
    category_line = f"Preferred category: {category}\n" if category else ""
    try:
        text = await nim_chat(
            system,
            f"{category_line}Today's research:\n{blob}",
            temperature=0.3,
            max_tokens=2048,
        )
        result = extract_json(text, prefer="topic")
        data = result.data
        topic = str(data.get("topic") or "").strip() if isinstance(data, dict) else ""
        if topic:
            return topic[:120]
        print("[AutoBlog] topic chunk returned no topic - using headline fallback.")
    except Exception as exc:
        print(f"[AutoBlog] topic extraction failed (Nemotron): {type(exc).__name__}: {exc}")
    headline = results[0]["title"].split("|")[0].split(" - ")[0].strip()[:120]
    return headline or FALLBACK_TOPICS[0]


# ─── Keyword Extraction (Nemotron Ultra) ──────────────────────────────────────


async def keyword_chunk(topic: str, results: list[dict[str, str]]) -> KeywordPlan:
    """Nemotron Ultra extracts the top 3-5 SEO keywords + article type."""
    blob = "\n".join(_result_line(result) for result in results) or "(no research snippets)"
    system = """You are an SEO content strategist for ATLAS, a learning OS for students.
Extract the top 3 to 5 SEO keywords for the given blog topic.

Return ONLY strict JSON (no fences, no commentary):
{"keywords": ["...", "...", "..."], "article_type": "..."}

- keywords: 3 to 5, each 1-3 words, lowercase, no dupes, primary keyword first.
- article_type: one of: news, explainer, how_to, opinion, comparison, benchmark, financial, growth, research.
  Choose the type that best fits the topic. Prefer specific types (comparison, benchmark, financial, growth, research) when the topic clearly fits.
  If unsure, use "explainer"."""
    try:
        text = await nim_chat(
            system,
            f"Topic: {topic}\nResearch snippets:\n{blob}",
            temperature=0.3,
            max_tokens=2048,
        )
        result = extract_json(text, prefer=("keywords", "article_type"))
        data = result.data
        if isinstance(data, dict):
            raw_keywords = data.get("keywords")
            keywords = _pad_keywords(
                [str(k) for k in raw_keywords] if isinstance(raw_keywords, list) else [], topic
            )
            article_type = str(data.get("article_type") or "explainer").strip().lower()
            if article_type not in ARTICLE_TYPES:
                article_type = "explainer"
            article_type = upgrade_article_type(topic, article_type)
            print(f"[AutoBlog] keyword extraction OK -> {', '.join(keywords)} (type: {article_type})")
            return KeywordPlan(keywords=keywords, article_type=article_type)
    except Exception as exc:
        print(f"[AutoBlog] keyword extraction failed (Nemotron): {type(exc).__name__}: {exc}")
    return KeywordPlan(keywords=_pad_keywords([], topic), article_type="explainer")


# ─── Data Availability Analysis ───────────────────────────────────────────────


def _extract_numbers(text: str) -> list[str]:
    patterns = [
        r"\$[\d,]+\.?\d*\s*(?:billion|million|bn|m|k)?",
        r"[\d,]+\.?\d*\s*(?:billion|million|bn|m|k)\s*(?:dollars?|usd|eur|users?|downloads?)",
        r"\d+(?:\.\d+)?\s*%",
        r"\d+(?:\.\d+)?\s*(?:out of|/)\s*\d+(?:\.\d+)?",
        r"\b\d+(?:\.\d+)?\s*(?:points?|score|rating)\b",
        r"\b(?:19|20)\d{2}\b",
        r"\b\d+(?:\.\d+)?\s*(?:x|times|fold)\b",
        r"\bv?\d+(?:\.\d+)+(?:\s*\+\s*\d+(?:\.\d+)?)*\b",
        r"\b\d+(?:,\d{3})+(?:\.\d+)?\b",
    ]
    combined = "|".join(patterns)
    matches = re.findall(combined, text, re.IGNORECASE)
    normalized = []
    for m in matches:
        clean = m.strip().replace(",", "").replace("$", "")
        normalized.append(clean)
    return normalized


def _extract_named_items(text: str) -> list[str]:
    pattern = r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s+(?:v?\d+(?:\.\d+)*))?\b"
    matches = re.findall(pattern, text)
    common = {"The", "This", "That", "These", "Those", "Then", "There", "Their", "We", "You", "It", "AI", "API", "URL", "HTTP", "CEO", "CTO"}
    items = []
    for m in matches:
        if m not in common and len(m) > 2:
            items.append(m)
    from app.autoblog.sources import AI_MODEL_RE
    model_matches = AI_MODEL_RE.findall(text)
    for m in model_matches:
        m_clean = " ".join(m.split())
        if m_clean not in items:
            items.append(m_clean)
    seen = set()
    unique = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique[:10]


def _extract_attributes(text: str) -> list[str]:
    attr_keywords = [
        "price", "cost", "speed", "latency", "accuracy", "performance",
        "context", "tokens", "parameters", "benchmark", "throughput",
        "memory", "storage", "bandwidth", "resolution", "fps", "tps",
        "revenue", "growth", "market share", "users", "customers",
        "valuation", "funding", "investment", "profit", "margin",
        "release date", "launch", "availability", "pricing",
        "features", "specs", "specifications", "capabilities",
    ]
    text_lower = text.lower()
    found = [attr for attr in attr_keywords if attr in text_lower]
    return found


def analyze_data_availability(research: Research) -> DataAvailability:
    full_text = " ".join(research.snippets)
    numbers = _extract_numbers(full_text)
    named_items = _extract_named_items(full_text)
    attributes = _extract_attributes(full_text)

    unique_numbers = list(dict.fromkeys(numbers))
    has_chart_data = len(unique_numbers) >= MIN_CHART_NUMBERS
    has_table_data = len(named_items) >= MIN_TABLE_ENTITIES and len(attributes) >= MIN_TABLE_ATTRIBUTES

    summary_parts = []
    if unique_numbers:
        summary_parts.append(f"{len(unique_numbers)} numeric facts")
    if named_items:
        summary_parts.append(f"{len(named_items)} named items ({', '.join(named_items[:3])})")
    if attributes:
        summary_parts.append(f"{len(attributes)} attributes ({', '.join(attributes[:3])})")

    return DataAvailability(
        numbers=unique_numbers,
        examples=unique_numbers[:10],
        named_items=named_items,
        attributes=attributes,
        has_chart_data=has_chart_data,
        has_table_data=has_table_data,
        summary="; ".join(summary_parts) if summary_parts else "No structured data found",
    )


# ─── High-level Research Functions ────────────────────────────────────────────


async def research_and_select_topic() -> tuple[Research, list[dict[str, str]]]:
    """Research, generate a topic, reject repeats, then force a new category."""
    settings = get_autoblog_settings()
    queries = ordered_research_queries()
    recent_posts = await asyncio.to_thread(fetch_recent_posts, settings.recent_posts_limit)
    blocked = _blocked_topics(recent_posts)
    attempted: list[str] = []
    results: list[dict[str, str]] = []
    topic = ""

    for attempt in range(TOPIC_RETRY_LIMIT + 1):
        category, query = queries[attempt % len(queries)]
        print(
            f"[AutoBlog] research attempt {attempt + 1}/{TOPIC_RETRY_LIMIT + 1}: "
            f"{category} -> {query!r}",
            flush=True,
        )
        results = await research_chunk(query)
        topic = await topic_chunk(results, [*blocked, *attempted], category=category)
        score, match = _find_duplicate(topic, recent_posts)
        print(
            f"[AutoBlog] candidate topic={topic!r} similarity={score:.2f}"
            + (f" match={match!r}" if match else ""),
            flush=True,
        )
        if score <= SIMILARITY_THRESHOLD:
            return (
                Research(
                    topic=topic,
                    keywords=[],
                    snippets=research_snippets(results),
                ),
                results,
            )
        attempted.append(topic)
        print(
            f"[AutoBlog] duplicate topic rejected ({score:.0%} similar) - retrying.",
            flush=True,
        )

    used_categories = {category for category, _ in queries[: TOPIC_RETRY_LIMIT + 1]}
    forced_category, forced_query = next(
        (
            (category, query)
            for category, query in queries
            if category not in used_categories
        ),
        queries[-1],
    )
    print(
        f"[AutoBlog] forcing a different category: {forced_category} -> "
        f"{forced_query!r}",
        flush=True,
    )
    results = await research_chunk(forced_query)
    topic = await topic_chunk(results, [*blocked, *attempted], category=forced_category)
    score, match = _find_duplicate(topic, recent_posts)
    if score > SIMILARITY_THRESHOLD:
        print(
            f"[AutoBlog] forced topic still {score:.0%} similar to {match!r}; "
            "accepting it to avoid an empty run.",
            flush=True,
        )
    return (
        Research(
            topic=topic,
            keywords=[],
            snippets=research_snippets(results),
        ),
        results,
    )


async def extract_keywords(topic: str, results: list[dict[str, str]]) -> KeywordPlan:
    return await keyword_chunk(topic, results)


async def analyze_data_and_profile(
    topic: str, article_type: str, snippets: list[str]
) -> tuple[DataAvailability, dict[str, Any]]:
    research = Research(topic=topic, keywords=[], snippets=snippets, article_type=article_type)
    data = analyze_data_availability(research)
    profile = build_profile(article_type, topic, data)
    return data, profile


# ─── Exports ──────────────────────────────────────────────────────────────────

__all__ = [
    "research_and_select_topic",
    "extract_keywords",
    "analyze_data_and_profile",
    "topic_chunk",
    "keyword_chunk",
    "analyze_data_availability",
    "Research",
    "KeywordPlan",
    "DataAvailability",
    "TOPIC_RETRY_LIMIT",
    "SIMILARITY_THRESHOLD",
    "FALLBACK_TOPICS",
]