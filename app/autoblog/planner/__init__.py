"""Editor-planner for AutoBlog v2 - Returns full article plan."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.llm.clients import (
    nemotron_with_gemini_fallback,
    extract_json,
    BlogPipelineError,
)
from app.autoblog.sources import fetch_recent_posts


logger = logging.getLogger(__name__)


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class SectionPlan:
    heading: str
    word_budget: int
    purpose: str
    needs_visual: bool = False
    visual_type: str | None = None  # "table", "bar", "line", "photo"


@dataclass(slots=True)
class VisualPlan:
    kind: str  # "table", "bar", "line"
    title: str
    caption: str
    columns: list[str] | None = None  # for tables
    rows: list[list[str]] | None = None  # for tables
    labels: list[str] | None = None  # for charts
    values: list[float | int] | None = None  # for charts
    unit: str | None = None
    source_url: str = ""
    intro_sentence: str = ""
    takeaway: str = ""


@dataclass(slots=True)
class ImagePlan:
    section_heading: str
    keyword: str
    caption_hint: str


@dataclass(slots=True)
class ArticlePlan:
    topic: str
    angle: str
    domain: str
    article_type: str  # "news", "explainer", "how_to", "opinion", "comparison", "benchmark", "financial", "growth", "research"
    tone: str  # "analytical", "conversational", "authoritative", "investigative"
    target_words: int  # 800-1150
    primary_keyword: str
    secondary_keywords: list[str]
    outline: list[SectionPlan]
    visual_plan: list[VisualPlan] = field(default_factory=list)
    image_plan: list[ImagePlan] = field(default_factory=list)


# ─── Constants ────────────────────────────────────────────────────────────────

ARTICLE_TYPES = (
    "news",
    "explainer",
    "how_to",
    "opinion",
    "comparison",
    "benchmark",
    "financial",
    "growth",
    "research",
)

DOMAINS = (
    "model_releases",
    "industry_news",
    "research_papers",
    "education_tech",
    "fintech_regulation",
    "open_source",
    "hardware_chips",
    "ai_safety",
    "productivity_tools",
    "startups_funding",
)


_PLANNER_SYSTEM = """You are the Editor-in-Chief of ATLAS, a learning OS for students. Your job is to plan a single high-quality blog post from a research fact sheet.

Return ONLY strict JSON (no fences, no commentary):
{
  "topic": "...",                    // specific, clickable headline angle (max 120 chars)
  "angle": "...",                    // the unique lens: e.g. "why students should care about X", "what X means for learning"
  "domain": "...",                   // one of: model_releases, industry_news, research_papers, education_tech, fintech_regulation, open_source, hardware_chips, ai_safety, productivity_tools, startups_funding
  "article_type": "...",             // one of: news, explainer, how_to, opinion, comparison, benchmark, financial, growth, research
  "tone": "...",                     // one of: analytical, conversational, authoritative, investigative
  "target_words": 950,               // integer 800-1150
  "primary_keyword": "...",          // exact keyword, 1-3 words, lowercase
  "secondary_keywords": ["...", "..."], // 2-4 additional keywords, lowercase
  "outline": [                       // 4-6 sections
    {"heading": "...", "word_budget": 180, "purpose": "...", "needs_visual": false, "visual_type": null}
  ],
  "visual_plan": [                   // ONLY if fact sheet has real structured data (numbers with units, comparable entities)
    {"kind": "table|bar|line", "title": "...", "caption": "...", "columns": [...], "labels": [...], "unit": "...", "source_url": "...", "intro_sentence": "...", "takeaway": "..."}
  ],
  "image_plan": [                    // for non-data sections that need a photo
    {"section_heading": "...", "keyword": "...", "caption_hint": "..."}
  ]
}

RULES:
- topic must be specific (model name, company, paper, measurable event), NOT a generic pillar term
- angle must differentiate from the last 60 posts (avoid their domains/topics)
- article_type: prefer specific types (comparison, benchmark, financial, growth, research) when fact sheet has data; otherwise explainer
- target_words: 800-1150, scale with outline complexity
- outline: 4-6 sections, each with word_budget summing to ~target_words
- needs_visual=true only for sections where visual_plan has a matching entry
- visual_plan: ONLY when fact sheet contains real numbers with units (>=3 data points for charts, >=2 entities with >=3 attributes for tables). Every visual MUST have source_url from fact sheet, intro_sentence, and takeaway.
- image_plan: for sections without data visuals, max 3 total images including cover
- Primary keyword must appear in topic and be 1-3 words
- Secondary keywords: 2-4, lowercase, no duplicates
- Tone: analytical for data-heavy, conversational for explainers, authoritative for news, investigative for deep-dives
"""


# ─── Domain Variety from Recent Posts ─────────────────────────────────────────


def _get_recent_domains(limit: int = 60) -> list[str]:
    """Extract domains from last 60 posts to ensure variety."""
    try:
        posts = fetch_recent_posts(limit)
        domains = []
        for p in posts:
            keywords = p.get("keywords", [])
            if isinstance(keywords, list):
                domains.extend(keywords)
        # Normalize and deduplicate
        seen = set()
        unique = []
        for d in domains:
            key = str(d).strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(key)
        return unique[:60]
    except Exception:
        return []


def _build_planner_prompt(fact_sheet: dict[str, Any] | Any, recent_domains: list[str]) -> str:
    # Support both dict and FactSheet dataclass
    if hasattr(fact_sheet, "claims"):
        claims = fact_sheet.claims
        numbers = fact_sheet.numbers
        entities = fact_sheet.entities
        source_urls = fact_sheet.source_urls
    else:
        claims = fact_sheet.get("claims", [])
        numbers = fact_sheet.get("numbers", [])
        entities = fact_sheet.get("entities", [])
        source_urls = fact_sheet.get("source_urls", [])

    claims_text = "\n".join(f"- {c}" for c in claims[:20]) or "(none)"
    numbers_text = "\n".join(f"- {n}" for n in numbers[:30]) or "(none)"
    entities_text = "\n".join(f"- {e}" for e in entities[:20]) or "(none)"
    sources_text = "\n".join(f"- {s}" for s in source_urls[:10]) or "(none)"
    recent_text = ", ".join(recent_domains[:30]) or "(none)"

    return f"""FACT SHEET (writer may ONLY use these facts, must link 2-4 source URLs inline):

Claims:
{claims_text}

Numbers (with units):
{numbers_text}

Named Entities (models, companies, products, papers):
{entities_text}

Source URLs:
{sources_text}

RECENT POST DOMAINS/KEYWORDS (avoid these for variety):
{recent_text}

Plan the article now."""


async def create_article_plan(fact_sheet: Any) -> ArticlePlan:
    """Run editor-planner (Nemotron -> Gemini fallback)."""
    recent_domains = await asyncio.to_thread(_get_recent_domains, 60)
    prompt = _build_planner_prompt(fact_sheet, recent_domains)

    result = await nemotron_with_gemini_fallback(
        _PLANNER_SYSTEM,
        prompt,
        temperature=0.3,
        max_tokens=4096,
    )

    parsed = extract_json(result.text)
    data = parsed.data
    if not isinstance(data, dict):
        raise BlogPipelineError("Planner returned invalid JSON.")

    # Build outline
    outline = []
    for sec in data.get("outline", []):
        outline.append(SectionPlan(
            heading=str(sec.get("heading", "")).strip(),
            word_budget=int(sec.get("word_budget", 200)),
            purpose=str(sec.get("purpose", "")).strip(),
            needs_visual=bool(sec.get("needs_visual", False)),
            visual_type=sec.get("visual_type") if sec.get("needs_visual") else None,
        ))

    # Build visual plan
    visual_plan = []
    for vis in data.get("visual_plan", []):
        visual_plan.append(VisualPlan(
            kind=str(vis.get("kind", "")).strip(),
            title=str(vis.get("title", "")).strip(),
            caption=str(vis.get("caption", "")).strip(),
            columns=vis.get("columns") if isinstance(vis.get("columns"), list) else None,
            rows=vis.get("rows") if isinstance(vis.get("rows"), list) else None,
            labels=vis.get("labels") if isinstance(vis.get("labels"), list) else None,
            values=vis.get("values") if isinstance(vis.get("values"), list) else None,
            unit=str(vis.get("unit", "")).strip() or None,
            source_url=str(vis.get("source_url", "")).strip(),
            intro_sentence=str(vis.get("intro_sentence", "")).strip(),
            takeaway=str(vis.get("takeaway", "")).strip(),
        ))

    # Build image plan
    image_plan = []
    for img in data.get("image_plan", []):
        image_plan.append(ImagePlan(
            section_heading=str(img.get("section_heading", "")).strip(),
            keyword=str(img.get("keyword", "")).strip(),
            caption_hint=str(img.get("caption_hint", "")).strip(),
        ))

    # Validate required fields
    topic = str(data.get("topic", "")).strip()
    if not topic:
        raise BlogPipelineError("Planner returned empty topic.")
    if len(topic) > 120:
        topic = topic[:120]

    article_type = str(data.get("article_type", "explainer")).strip().lower()
    if article_type not in ARTICLE_TYPES:
        article_type = "explainer"

    tone = str(data.get("tone", "analytical")).strip().lower()
    valid_tones = {"analytical", "conversational", "authoritative", "investigative"}
    if tone not in valid_tones:
        tone = "analytical"

    target_words = int(data.get("target_words", 950))
    target_words = max(800, min(1150, target_words))

    primary_keyword = str(data.get("primary_keyword", "")).strip().lower()
    if not primary_keyword:
        # fallback to first secondary or topic
        secondary = data.get("secondary_keywords", [])
        primary_keyword = secondary[0] if secondary else topic.lower().split()[0]

    secondary_keywords = []
    for kw in data.get("secondary_keywords", []):
        kw_clean = str(kw).strip().lower()
        if kw_clean and kw_clean != primary_keyword and kw_clean not in secondary_keywords:
            secondary_keywords.append(kw_clean)
    secondary_keywords = secondary_keywords[:4]

    domain = str(data.get("domain", "")).strip().lower()
    if domain not in DOMAINS:
        domain = "industry_news"

    plan = ArticlePlan(
        topic=topic,
        angle=str(data.get("angle", "")).strip(),
        domain=domain,
        article_type=article_type,
        tone=tone,
        target_words=target_words,
        primary_keyword=primary_keyword,
        secondary_keywords=secondary_keywords,
        outline=outline,
        visual_plan=visual_plan,
        image_plan=image_plan,
    )

    logger.info(
        f"[AutoBlog] Plan: topic={plan.topic!r} type={plan.article_type} "
        f"words={plan.target_words} sections={len(plan.outline)} "
        f"visuals={len(plan.visual_plan)} images={len(plan.image_plan)}"
    )
    return plan


# ─── Exports ──────────────────────────────────────────────────────────────────

__all__ = [
    "create_article_plan",
    "ArticlePlan",
    "SectionPlan",
    "VisualPlan",
    "ImagePlan",
    "ARTICLE_TYPES",
    "DOMAINS",
]