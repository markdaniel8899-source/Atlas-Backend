"""Planner module for AutoBlog v2 - Article profiles and visual planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ─── Article Type Taxonomy ────────────────────────────────────────────────────

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

# ─── Profile Definitions ──────────────────────────────────────────────────────

# cover_photo, max_content_photos, required_visuals, optional_visuals, max_visuals
ARTICLE_PROFILES: dict[str, dict[str, Any]] = {
    "comparison": {
        "cover_photo": True,
        "max_content_photos": 0,
        "required_visuals": ("table",),
        "optional_visuals": ("bar",),
        "max_visuals": 2,
    },
    "benchmark": {
        "cover_photo": True,
        "max_content_photos": 0,
        "required_visuals": ("table", "bar"),
        "max_visuals": 2,
    },
    "financial": {
        "cover_photo": True,
        "max_content_photos": 0,
        "required_visuals": ("bar", "table"),
        "max_visuals": 2,
    },
    "growth": {
        "cover_photo": True,
        "max_content_photos": 0,
        "required_visuals": ("bar", "line"),
        "max_visuals": 2,
    },
    "research": {
        "cover_photo": True,
        "max_content_photos": 0,
        "required_visuals": ("table", "bar", "line"),
        "max_visuals": 1,
    },
    "news": {
        "cover_photo": True,
        "max_content_photos": 1,
        "required_visuals": (),
        "max_visuals": 1,
    },
    "explainer": {
        "cover_photo": True,
        "max_content_photos": 1,
        "required_visuals": (),
        "max_visuals": 1,
    },
    "how_to": {
        "cover_photo": True,
        "max_content_photos": 1,
        "required_visuals": (),
        "max_visuals": 1,
    },
    "opinion": {
        "cover_photo": True,
        "max_content_photos": 1,
        "required_visuals": (),
        "max_visuals": 1,
    },
}

# ─── Heuristic Upgrade Signals ────────────────────────────────────────────────

_UPGRADE_SIGNALS: dict[str, list[str]] = {
    "comparison": ["vs", "versus", "compare", "comparison"],
    "benchmark": ["benchmark", "benchmarks", "performance test", "speed test"],
    "financial": ["revenue", "funding", "valuation", "investment", "series a", "series b", "ipo", "raised $", "million", "billion"],
    "growth": ["growth", "growing", "growth rate", "yoy", "quarter over quarter", "month over month"],
    "research": ["study", "paper", "research", "arxiv", "findings", "methodology"],
}

# ─── Minimum Data Requirements ────────────────────────────────────────────────

MIN_CHART_NUMBERS = 3
MIN_TABLE_ENTITIES = 2
MIN_TABLE_ATTRIBUTES = 3


def upgrade_article_type(topic: str, article_type: str) -> str:
    """Upgrade article type based on topic text signals."""
    topic_lower = topic.lower()
    generic_types = {"news", "explainer", "how_to", "opinion"}
    if article_type not in generic_types:
        return article_type

    for target_type, signals in _UPGRADE_SIGNALS.items():
        if any(signal in topic_lower for signal in signals):
            return target_type
    return article_type


def build_profile(
    article_type: str, topic: str, data: Any
) -> dict[str, Any]:
    """Build the article profile based on type and data availability."""
    base = ARTICLE_PROFILES.get(article_type, ARTICLE_PROFILES["explainer"])
    profile = dict(base)

    required = list(profile.get("required_visuals", []))
    downgraded = False
    downgrade_reason = ""

    if "table" in required and not data.has_table_data:
        downgraded = True
        downgrade_reason = (
            f"Table required but only {len(data.named_items)} items and "
            f"{len(data.attributes)} attributes found "
            f"(need >={MIN_TABLE_ENTITIES} items and >={MIN_TABLE_ATTRIBUTES} attributes)"
        )
    if "bar" in required and not data.has_chart_data:
        downgraded = True
        downgrade_reason = (
            f"Bar chart required but only {len(data.numbers)} numeric facts found "
            f"(need >={MIN_CHART_NUMBERS})"
        )
    if "line" in required and not data.has_chart_data:
        downgraded = True
        downgrade_reason = (
            f"Line chart required but only {len(data.numbers)} numeric facts found "
            f"(need >={MIN_CHART_NUMBERS})"
        )

    if downgraded:
        profile = dict(ARTICLE_PROFILES["explainer"])
        profile["downgraded"] = True
        profile["downgrade_reason"] = downgrade_reason
        print(f"[AutoBlog] Profile downgraded to explainer: {downgrade_reason}")
    else:
        profile["downgraded"] = False
        profile["downgrade_reason"] = ""

    profile["article_type"] = article_type
    profile["data"] = data
    return profile


def get_article_profile(article_type: str) -> dict[str, Any]:
    """Get the base article profile for a type."""
    return ARTICLE_PROFILES.get(article_type, ARTICLE_PROFILES["explainer"])