"""Image handling for AutoBlog v2 - Caps, provider IDs, fact-sheet grounding."""

from __future__ import annotations

import asyncio
import random
import re
import html
from dataclasses import dataclass
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.sources import (
    fetch_relevant_image,
    persist_cover_image,
    _figure_html,
    UNSPLASH_FALLBACK_IMAGES,
)
from app.autoblog.visuals import replace_visual_markers


# ─── Constants ────────────────────────────────────────────────────────────────

MAX_TOTAL_IMAGES = 4  # including cover
MAX_CONTENT_IMAGES_UNDER_1000 = 2
MAX_CONTENT_IMAGES_OVER_1000 = 3

_H2_RE = re.compile(r"(<h2[^>]*>.*?</h2>)", re.IGNORECASE | re.DOTALL)
_FAQ_H2_RE = re.compile(
    r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b)", re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(slots=True)
class ImageResult:
    url: str
    caption: str
    provider_id: str  # Pixabay image ID or "unsplash:N" for deduping


# ─── Regex & Helpers ─────────────────────────────────────────────────────────


def _get_limits(word_count: int) -> tuple[int, int]:
    """Return (max_content_images, max_total_images) based on word count."""
    max_content = MAX_CONTENT_IMAGES_OVER_1000 if word_count >= 1000 else MAX_CONTENT_IMAGES_UNDER_1000
    return max_content, MAX_TOTAL_IMAGES


def _extract_provider_id(url: str) -> str:
    """Extract provider-specific ID for deduplication."""
    # Pixabay URLs contain image ID: https://pixabay.com/get/.../image-1234567_...
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    if "pixabay.com" in parsed.netloc:
        # Try to extract from path
        match = re.search(r"/(\d+)_", parsed.path)
        if match:
            return f"pixabay:{match.group(1)}"
        # Fallback to full URL hash
        return f"pixabay:{hash(url) % 1000000}"
    return f"unsplash:{hash(url) % 1000000}"


async def fetch_cover_image(
    title: str,
    keywords: list[str],
    used_provider_ids: set[str] | None = None,
) -> tuple[ImageResult, str]:
    """Build the cover, returning (ImageResult, source URL for content dedupe)."""
    used = used_provider_ids or set()
    candidate = await fetch_relevant_image(title, keywords, category="technology", exclude_urls=set())
    if not candidate:
        provider_id = "unsplash:cover_fallback"
        return ImageResult(
            url=UNSPLASH_FALLBACK_IMAGES[0],
            caption=title,
            provider_id=provider_id,
        ), UNSPLASH_FALLBACK_IMAGES[0]

    # Check if provider ID already used
    pid = _extract_provider_id(candidate)
    if pid in used:
        # Try to get another
        candidate = await fetch_relevant_image(title, keywords, category="technology", exclude_urls={candidate})
        if not candidate:
            return ImageResult(
                url=UNSPLASH_FALLBACK_IMAGES[0],
                caption=title,
                provider_id="unsplash:cover_fallback",
            ), UNSPLASH_FALLBACK_IMAGES[0]
        pid = _extract_provider_id(candidate)

    cover = await persist_cover_image(candidate, slugify(title))
    return ImageResult(
        url=cover,
        caption=title,
        provider_id=pid,
    ), candidate


async def fetch_content_image(
    section_title: str,
    keyword: str,
    used_provider_ids: set[str] | None = None,
    used_urls: set[str] | None = None,
) -> ImageResult | None:
    """Fetch in-article image; None when every available photo is already used."""
    used = used_provider_ids or set()
    excluded = used_urls or set()

    url = await fetch_relevant_image(
        section_title,
        [keyword] if keyword else None,
        category="technology",
        exclude_urls=excluded,
    )
    if not url:
        return None

    pid = _extract_provider_id(url)
    if pid in used:
        return None

    return ImageResult(
        url=url,
        caption=section_title,
        provider_id=pid,
    )


async def inject_images_into_content(
    content_html: str,
    plan: Any,  # ArticlePlan
    fact_sheet: Any,  # FactSheet
    keyword: str | None = None,
    used_provider_ids: set[str] | None = None,
) -> tuple[str, list[ImageResult], int]:
    """
    Inject content images per image_plan.
    Returns (html_with_images, image_results, content_images_count).
    """
    word_count = len(re.sub(r"<[^>]+>", " ", content_html).split())
    max_content, _ = _get_limits(word_count)

    image_plan = getattr(plan, "image_plan", [])
    if not image_plan or max_content == 0:
        return content_html, [], 0

    used = used_provider_ids or set()
    used_urls = set()
    images: list[ImageResult] = []

    for img_plan in image_plan[:max_content]:
        section_heading = getattr(img_plan, "section_heading", "")
        kw = getattr(img_plan, "keyword", keyword or "technology")
        caption_hint = getattr(img_plan, "caption_hint", section_heading)

        img = await fetch_content_image(section_heading, kw, used, used_urls)
        if not img:
            print(f"[AutoBlog] no unique content image for {section_heading!r} - skipping", flush=True)
            continue

        used.add(img.provider_id)
        used_urls.add(img.url)
        images.append(img)

        # Insert after the section heading
        marker = f"<h2[^>]*>{re.escape(section_heading)}</h2>"
        match = re.search(marker, content_html, re.IGNORECASE)
        if match:
            insert_pos = match.end()
            figure = _figure_html(img.url, img.caption)
            content_html = content_html[:insert_pos] + f"\n{figure}\n" + content_html[insert_pos:]
        else:
            # Fallback: after first paragraph
            if "</p>" in content_html:
                head, tail = content_html.split("</p>", 1)
                content_html = f"{head}</p>{_figure_html(img.url, img.caption)}{tail}"

    return content_html, images, len(images)


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")[:80].strip("-")
    return slug or f"blog-{_utc_str()}"


def _utc_str() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d")


# ─── Exports ──────────────────────────────────────────────────────────────────

__all__ = [
    "fetch_cover_image",
    "fetch_content_image",
    "inject_images_into_content",
    "ImageResult",
    "MAX_TOTAL_IMAGES",
    "MAX_CONTENT_IMAGES_UNDER_1000",
    "MAX_CONTENT_IMAGES_OVER_1000",
]