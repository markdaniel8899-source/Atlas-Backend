"""Shared data models for the dynamic SEO/AEO/GEO AutoBlog pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ─── Research & keywords ──────────────────────────────────────────────────────


@dataclass(slots=True)
class TopicTrend:
    """A specific, timely topic selected for today's article."""

    headline: str
    angle: str
    category: str
    why_now: str
    search_intent: str = "informational"


@dataclass(slots=True)
class KeywordPlan:
    """Primary, secondary, long-tail and LSI keywords for one article."""

    primary_keyword: str
    secondary_keywords: list[str] = field(default_factory=list)
    long_tail_keywords: list[str] = field(default_factory=list)
    lsi_keywords: list[str] = field(default_factory=list)
    search_intent: str = "informational"

    def all_keywords(self) -> list[str]:
        """Flat, de-duplicated keyword list for storage and prompts."""
        seen: set[str] = set()
        flat: list[str] = []
        for keyword in (
            self.primary_keyword,
            *self.secondary_keywords,
            *self.long_tail_keywords,
            *self.lsi_keywords,
        ):
            clean = str(keyword).strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                flat.append(clean)
        return flat

    def as_json(self) -> dict[str, Any]:
        return {
            "primary": self.primary_keyword,
            "secondary": list(self.secondary_keywords),
            "long_tail": list(self.long_tail_keywords),
            "lsi": list(self.lsi_keywords),
            "search_intent": self.search_intent,
        }


# ─── Links, meta & visuals ────────────────────────────────────────────────────


@dataclass(slots=True)
class InternalLink:
    title: str
    url: str


@dataclass(slots=True)
class MetaBundle:
    meta_title: str
    meta_description: str
    og_title: str
    og_description: str
    twitter_title: str
    twitter_description: str
    excerpt: str


@dataclass(slots=True)
class VisualPayload:
    """A rendered data visual (table or chart) ready to embed in the article."""

    kind: str  # "table" | "chart"
    html: str
    alt: str = ""
    source_url: str = ""
    intro_sentence: str = ""


@dataclass(slots=True)
class VisualStrategy:
    """Everything needed to make the article visual, decided before writing."""

    mode: str  # "data" | "news"
    cover_url: str
    cover_provider_id: str
    content_images: list[dict[str, str]] = field(default_factory=list)
    data_visuals: list[VisualPayload] = field(default_factory=list)


# ─── Article & validation ─────────────────────────────────────────────────────


@dataclass(slots=True)
class DynamicArticleDraft:
    title: str
    intro_html: str
    sections: list[dict[str, str]]
    conclusion_html: str
    faq: list[dict[str, str]]
    tags: list[str]
    content_html: str
    word_count: int
    meta_title_raw: str = ""
    meta_description_raw: str = ""
    excerpt_raw: str = ""
    chart_spec: dict[str, Any] | None = None


@dataclass(slots=True)
class SeoValidation:
    total: int
    passed: bool
    breakdown: dict[str, int]
    issues: list[str]
