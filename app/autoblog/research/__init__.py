"""Research for AutoBlog v2 - Planner-driven queries and fact sheet building."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.llm.clients import (
    groq_research_with_nemotron_fallback,
    extract_json,
    BlogPipelineError,
)
from app.autoblog.sources import (
    ordered_research_queries,
    research_chunk,
    research_snippets,
    fetch_recent_posts,
)


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class FactSheet:
    """Structured facts extracted from sources for the writer."""
    claims: list[str] = field(default_factory=list)           # Key claims/statements
    numbers: list[dict[str, Any]] = field(default_factory=list)  # {value, unit, context, source_url}
    entities: list[dict[str, Any]] = field(default_factory=list)  # {name, type, context, source_url}
    source_urls: list[str] = field(default_factory=list)
    raw_snippets: list[str] = field(default_factory=list)


# ─── Query Planning (Groq) ────────────────────────────────────────────────────


_QUERY_PLANNER_SYSTEM = """You are a research strategist. Given a topic angle and recent posts, write the exact search queries to fetch fresh, authoritative sources.

Return ONLY strict JSON (no fences, no commentary):
{
  "news_query": "...",           // general news search for the topic
  "official_query": "...",       // official sources (company blogs, arxiv, docs)
  "benchmarks_query": "...",     // ONLY if comparison/benchmark/financial/growth/research type: numbers, benchmarks, specs, pricing
  "reasoning": "..."             // brief: why these queries
}

RULES:
- Queries must be specific (model names, companies, paper titles, measurable events)
- Use current month/year context
- Official query should target known authoritative domains
- Benchmarks query ONLY for data-heavy article types
- Keep queries under 100 chars for Tavily"""


async def plan_research_queries(
    topic: str,
    angle: str,
    article_type: str,
    recent_posts: list[dict[str, Any]],
) -> dict[str, str]:
    """Groq plans the exact search queries."""
    recent_titles = [p.get("title", "") for p in recent_posts[:10] if p.get("title")]
    recent_text = "\n".join(f"- {t}" for t in recent_titles) or "(none)"

    prompt = f"""Topic: {topic}
Angle: {angle}
Article Type: {article_type}

Recent Post Titles (avoid duplication):
{recent_text}

Write the search queries now."""

    result = await groq_research_with_nemotron_fallback(
        _QUERY_PLANNER_SYSTEM,
        prompt,
        temperature=0.2,
        max_tokens=1024,
    )

    parsed = extract_json(result.text)
    data = parsed.data
    if not isinstance(data, dict):
        raise BlogPipelineError("Query planner returned invalid JSON.")

    queries = {
        "news_query": str(data.get("news_query", "")).strip(),
        "official_query": str(data.get("official_query", "")).strip(),
        "benchmarks_query": str(data.get("benchmarks_query", "")).strip() if article_type in ("comparison", "benchmark", "financial", "growth", "research") else "",
    }

    # Fallback to rotating queries if planner fails
    if not queries["news_query"] or not queries["official_query"]:
        from app.autoblog.sources import _ordered_research_queries, _month_year
        month_year = _month_year()
        fallback = ordered_research_queries()
        cat, q = fallback[0]
        queries["news_query"] = queries["news_query"] or q.format(month_year=month_year)
        queries["official_query"] = queries["official_query"] or q.format(month_year=month_year)

    return queries


# ─── Fact Extraction (Groq) ───────────────────────────────────────────────────


_FACT_EXTRACTOR_SYSTEM = """You are a fact extractor. From the research snippets, extract a structured fact sheet.

Return ONLY strict JSON (no fences, no commentary):
{
  "claims": ["...", "..."],                    // 5-15 key claims/statements, each one sentence
  "numbers": [                                 // numbers WITH units and context
    {"value": "...", "unit": "...", "context": "...", "source_url": "..."}
  ],
  "entities": [                                // named entities: models, companies, products, papers
    {"name": "...", "type": "model|company|product|paper|person", "context": "...", "source_url": "..."}
  ],
  "source_urls": ["...", "..."]
}

RULES:
- Every claim must be traceable to a source_url in the snippets
- Numbers MUST include unit (%, $, tokens, ms, GB, etc.) and context (what it measures)
- Entities: classify type, include context
- Source URLs must come from the provided snippets
- Deduplicate: same claim/number/entity from multiple sources -> one entry with first source
- Max 15 claims, 30 numbers, 20 entities"""


async def extract_fact_sheet(
    snippets: list[str],
    source_urls: list[str],
) -> FactSheet:
    """Groq extracts structured facts from raw snippets."""
    snippets_text = "\n\n".join(snippets[:30]) or "(none)"
    urls_text = "\n".join(f"- {u}" for u in source_urls[:20]) or "(none)"

    prompt = f"""Research Snippets:
{snippets_text}

Source URLs:
{urls_text}

Extract the fact sheet now."""

    result = await groq_research_with_nemotron_fallback(
        _FACT_EXTRACTOR_SYSTEM,
        prompt,
        temperature=0.1,
        max_tokens=4096,
    )

    parsed = extract_json(result.text)
    data = parsed.data
    if not isinstance(data, dict):
        raise BlogPipelineError("Fact extractor returned invalid JSON.")

    claims = [str(c).strip() for c in data.get("claims", []) if str(c).strip()][:15]
    numbers = []
    for n in data.get("numbers", []):
        if isinstance(n, dict) and n.get("value") and n.get("unit"):
            numbers.append({
                "value": str(n.get("value", "")).strip(),
                "unit": str(n.get("unit", "")).strip(),
                "context": str(n.get("context", "")).strip(),
                "source_url": str(n.get("source_url", "")).strip(),
            })
    entities = []
    for e in data.get("entities", []):
        if isinstance(e, dict) and e.get("name"):
            entities.append({
                "name": str(e.get("name", "")).strip(),
                "type": str(e.get("type", "")).strip(),
                "context": str(e.get("context", "")).strip(),
                "source_url": str(e.get("source_url", "")).strip(),
            })
    source_urls = [str(u).strip() for u in data.get("source_urls", []) if str(u).strip()]

    # Deduplicate numbers by (value, unit, context)
    seen_nums = set()
    unique_numbers = []
    for n in numbers:
        key = (n["value"], n["unit"], n["context"])
        if key not in seen_nums:
            seen_nums.add(key)
            unique_numbers.append(n)

    # Deduplicate entities by name
    seen_ent = set()
    unique_entities = []
    for e in entities:
        key = e["name"].lower()
        if key not in seen_ent:
            seen_ent.add(key)
            unique_entities.append(e)

    return FactSheet(
        claims=claims,
        numbers=unique_numbers[:30],
        entities=unique_entities[:20],
        source_urls=source_urls[:15],
        raw_snippets=snippets,
    )


# ─── High-level Research Flow ─────────────────────────────────────────────────


async def run_research(
    topic: str,
    angle: str,
    article_type: str,
) -> tuple[FactSheet, list[dict[str, str]]]:
    """
    1. Fetch recent posts for dedupe context
    2. Planner writes search queries
    3. Execute queries via Tavily
    4. Extract fact sheet from results
    Returns (fact_sheet, raw_results_for_logging)
    """
    settings = get_autoblog_settings()

    # Step 1: Recent posts for context
    recent_posts = await asyncio.to_thread(fetch_recent_posts, settings.recent_posts_limit)

    # Step 2: Plan queries
    queries = await plan_research_queries(topic, angle, article_type, recent_posts)
    print(f"[AutoBlog] Research queries: news={queries['news_query']!r} official={queries['official_query']!r} benchmarks={queries['benchmarks_query']!r}", flush=True)

    # Step 3: Execute searches
    all_results: list[dict[str, str]] = []

    # News query
    news_results = await research_chunk(queries["news_query"])
    all_results.extend(news_results)

    # Official sources query
    official_results = await research_chunk(queries["official_query"])
    all_results.extend(official_results)

    # Benchmarks query (if applicable)
    if queries["benchmarks_query"]:
        bench_results = await research_chunk(queries["benchmarks_query"])
        all_results.extend(bench_results)

    # Dedupe by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    print(f"[AutoBlog] Research complete: {len(unique_results)} unique sources", flush=True)

    # Step 4: Extract fact sheet
    snippets = research_snippets(unique_results)
    source_urls = [r.get("url", "") for r in unique_results if r.get("url")]
    fact_sheet = await extract_fact_sheet(snippets, source_urls)

    print(
        f"[AutoBlog] Fact sheet: {len(fact_sheet.claims)} claims, "
        f"{len(fact_sheet.numbers)} numbers, {len(fact_sheet.entities)} entities, "
        f"{len(fact_sheet.source_urls)} source URLs",
        flush=True,
    )

    return fact_sheet, unique_results


# ─── Exports ──────────────────────────────────────────────────────────────────

__all__ = [
    "run_research",
    "plan_research_queries",
    "extract_fact_sheet",
    "FactSheet",
]