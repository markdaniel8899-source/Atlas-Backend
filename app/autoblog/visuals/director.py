"""Visual Director - LLM-based visual planning for ATLAS articles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from app.autoblog.llm.clients import (
    BlogPipelineError,
    extract_json,
    gemini_chat,
    nemotron_with_gemini_fallback,
)
from app.autoblog.models import KeywordPlan
from app.autoblog.research import FactSheet
from app.autoblog.visuals.labels import clean_label
from pydantic import BaseModel, Field


class TableSpec(BaseModel):
    type: Literal["table"] = "table"
    after_section_heading: str
    caption: str
    columns: list[str] = Field(min_items=2, max_items=5)
    rows: list[list[str]] = Field(min_items=2, max_items=8)
    source_url: Optional[str] = None


class ChartSpec(BaseModel):
    type: Literal["chart"] = "chart"
    after_section_heading: str
    chart_type: Literal["bar", "hbar", "line", "doughnut"] = "bar"
    title: str
    unit: str
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    points: list[dict] = Field(min_items=3, max_items=10)
    source_url: Optional[str] = None
    why: str


class ImageSpec(BaseModel):
    type: Literal["image"] = "image"
    after_section_heading: str
    query: str
    alt: str
    caption: Optional[str] = None


class StatCardsSpec(BaseModel):
    type: Literal["stat_cards"] = "stat_cards"
    after_section_heading: str
    cards: list[dict] = Field(min_items=2, max_items=4)


class VisualPlan(BaseModel):
    visuals: list[Any] = Field(default_factory=list)


_VISUAL_DIRECTOR_SYSTEM = """You are the Visual Director for ATLAS, an AI learning OS for students.
Your job is to plan visuals for a finished article based on the article text and research data.

You read the full article text and the research fact_sheet. You return a strict JSON visual plan.

OUTPUT SCHEMA:
{
  "visuals": [
    {
      "type": "table",
      "after_section_heading": "Exact H2 heading text from article",
      "caption": "Brief caption",
      "columns": ["Metric", "Value", "Source"],
      "rows": [["Metric 1", "Value 1", "Source 1"], ...],
      "source_url": "https://..."
    },
    {
      "type": "chart",
      "after_section_heading": "Exact H2 heading text from article",
      "chart_type": "bar|hbar|line|doughnut",
      "title": "Short title under 10 words",
      "unit": "$B|%|ms|users|...",
      "x_label": "Category",
      "y_label": "Value",
      "points": [{"label": "Short Label", "value": 12.5}, ...],
      "source_url": "https://...",
      "why": "One sentence: what insight this chart shows"
    },
    {
      "type": "image",
      "after_section_heading": "Exact H2 heading text from article",
      "query": "concrete visual subject (3-6 words)",
      "alt": "Descriptive alt text",
      "caption": "Optional caption"
    },
    {
      "type": "stat_cards",
      "after_section_heading": "Exact H2 heading text from article",
      "cards": [{"value": "$40B", "label": "Raise target"}, ...]
    }
  ]
}

RULES:
1. ONLY add a visual if it explains something better than text. Zero visuals is valid. Max 3 total.
2. NEVER two visuals of the same kind back-to-back.
3. Every number MUST appear in the article text or fact_sheet. Do not invent or estimate.
4. Table: use for specs, pricing, side-by-side comparison.
5. Chart: only if >= 3 data points with the SAME unit and a clear message.
   - If values have different units or only 1-2 points, use stat_cards or table instead.
   - Chart type: time series -> line; part of whole -> doughnut (max 6 slices);
     ranking/comparison with long labels -> hbar; few short labels -> bar.
6. Provide short readable label (<= 18 chars) per point; no truncation by code.
7. Image: query must describe a concrete visual (e.g. "student using laptop classroom",
   not "AI"). Different query per image. Place images only where they add context.
7. Image queries must be different from each other.
8. Stat cards: use for 2-4 key metrics that don't need a full table/chart.
9. after_section_heading MUST match an H2 heading in the article EXACTLY (fuzzy match OK).
10. Provide a source_quote for each data point (1-2 sentence quote from article/fact_sheet).
11. Provide a 'why' sentence for each chart explaining the insight.
12. NO images/charts/tables in intro, conclusion, or FAQ sections.
13. Place visuals after H2 headings only, never after H3 or in intro/conclusion/FAQ.
14. Max 3 visuals total per article.
15. Minimum 250 words between visuals.

Return ONLY the JSON object. No markdown fences, no commentary.
"""

_VISUAL_DIRECTOR_REPAIR = """You repair malformed JSON from another language model.
Return ONLY a valid VisualPlan JSON object. Do not add commentary.
"""

_DIRECTOR_USER_PROMPT = """Analyze the article and research data to create a visual plan.

ARTICLE TEXT (HTML):
{article_text}

RESEARCH DATA:
Key Data Points:
{key_data_points}

Fact Sheet Numbers:
{numbers}

Fact Sheet Claims:
{claims}

Keywords:
Primary: {primary_keyword}
Secondary: {secondary_keywords}

Return a VisualPlan JSON with up to 3 visuals. Follow all rules in the system prompt.
"""


async def plan_visuals(
    draft: Any,
    fact_sheet: Any,
    keywords: Any,
    used_image_ids: set[str] | None = None,
) -> dict:
    """
    Node 3a: Visual Director - LLM plans visuals based on article + research.

    Returns VisualPlan as dict.
    """
    # Extract article text
    article_html = draft.content_html if hasattr(draft, 'content_html') else draft.get('content_html', '')

    # Prepare data for prompt
    key_data_points = fact_sheet.raw_snippets if hasattr(fact_sheet, 'raw_snippets') else []
    numbers = fact_sheet.numbers if hasattr(fact_sheet, 'numbers') else []
    claims = fact_sheet.claims if hasattr(fact_sheet, 'claims') else []

    primary = getattr(keywords, 'primary_keyword', '') if hasattr(keywords, 'primary_keyword') else keywords.get('primary_keyword', '')
    secondary = getattr(keywords, 'secondary_keywords', []) if hasattr(keywords, 'secondary_keywords') else keywords.get('secondary_keywords', [])

    prompt = _DIRECTOR_USER_PROMPT.format(
        article_text=article_html,
        key_data_points="\n".join(f"- {p}" for p in key_data_points[:10]) or "(none)",
        numbers="\n".join(
            f"- {n.get('value', '')} {n.get('unit', '')}: {n.get('context', '')[:120]} (source: {n.get('source_url', '')})"
            for n in fact_sheet.numbers[:10]
        ) if hasattr(fact_sheet, 'numbers') else "(none)",
        claims="\n".join(f"- {c}" for c in fact_sheet.claims[:8]) if hasattr(fact_sheet, 'claims') else "(none)",
        primary_keyword=primary,
        secondary_keywords=", ".join(secondary[:5]) if secondary else "(none)",
    )

    # Use Gemini 3.8-flash as primary, Groq as fallback
    try:
        result = await nemotron_with_gemini_fallback(
            _VISUAL_DIRECTOR_SYSTEM,
            prompt,
            temperature=0.3,
            max_tokens=4096,
        )
    except Exception as exc:
        # Fallback to Groq
        from app.autoblog.llm.clients import groq_research_with_nemotron_fallback
        result = await groq_research_with_nemotron_fallback(
            _VISUAL_DIRECTOR_SYSTEM,
            prompt,
            temperature=0.3,
            max_tokens=4096,
        )

    # Parse JSON
    try:
        parsed = extract_json(result.text)
        data = parsed.data
        if isinstance(data, list):
            data = {"visuals": data}
        elif not isinstance(data, dict):
            data = {"visuals": []}
        # Validate against VisualPlan
        plan = data.get("visuals", [])
        # Filter valid visuals
        valid = []
        for v in plan:
            if isinstance(v, dict) and v.get("type") in ("table", "chart", "image", "stat_cards"):
                valid.append(v)
        return {"visuals": valid[:3]}
    except Exception:
        return {"visuals": []}


__all__ = ["plan_visuals"]