"""Image handling for AutoBlog v2 - Cover and content images."""

from __future__ import annotations

import asyncio
import random
import re
import html
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.sources import (
    fetch_relevant_image,
    persist_cover_image,
    _figure_html,
    UNSPLASH_FALLBACK_IMAGES,
)
from app.autoblog.visuals import replace_visual_markers


# ─── Regex ─────────────────────────────────────────────────────────────────────

_H2_RE = re.compile(r"(<h2[^>]*>.*?</h2>)", re.IGNORECASE | re.DOTALL)
_FAQ_H2_RE = re.compile(
    r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b)", re.IGNORECASE
)
_TAG_RE = re.compile(r"<[^>]+>")


def _get_limits() -> tuple[int, int, int]:
    settings = get_autoblog_settings()
    return settings.min_content_images, settings.max_content_images, settings.image_pause_seconds


async def fetch_cover_image(title: str, keywords: list[str]) -> tuple[str, str]:
    """Build the cover, returning (durable URL, source URL to dedupe against)."""
    candidate = await fetch_relevant_image(
        title, keywords, category="technology"
    )
    if not candidate:
        return UNSPLASH_FALLBACK_IMAGES[0], UNSPLASH_FALLBACK_IMAGES[0]
    cover = await persist_cover_image(candidate, slugify(title))
    return cover, candidate


async def fetch_cover_image_url(title: str, keywords: list[str]) -> str:
    """Fetch a cover and return only a durable R2 or Unsplash URL."""
    cover, _ = await fetch_cover_image(title, keywords)
    return cover


async def fetch_content_image(
    section_title: str,
    keyword: str,
    exclude_urls: set[str] | None = None,
) -> str | None:
    """Fetch in-article image; None when every available photo is already used."""
    url = await fetch_relevant_image(
        section_title,
        [keyword] if keyword else None,
        category="technology",
        exclude_urls=exclude_urls,
    )
    return url


async def inject_images_into_content(
    content_html: str,
    keywords: list[str],
    *,
    max_images: int | None = None,
    exclude_urls: set[str] | None = None,
    profile: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Inject 1-3 unique in-article images with pauses between fetches."""
    min_img, max_img, img_pause = _get_limits()
    target = (
        max_images
        if max_images is not None
        else random.randint(min_img, max_img)
    )
    target = max(0, min(target, max_img))

    if profile:
        max_profile = profile.get("max_content_photos", max_img)
        target = min(target, max_profile)

    used = set(exclude_urls or ())

    parts = _H2_RE.split(content_html)
    if len(parts) < 3:
        if not keywords or target == 0:
            return content_html, []
        caption = _TAG_RE.sub(" ", parts[0])[:120].strip() or "Article illustration"
        keyword = keywords[0] if keywords else "technology"
        url = await fetch_content_image(caption, keyword, exclude_urls=used)
        if not url or url in used:
            print("[AutoBlog] no unique content image for the intro - skipping.", flush=True)
            return content_html, []
        used.add(url)
        figure = _figure_html(url, caption)
        if "</p>" in content_html:
            head, tail = content_html.split("</p>", 1)
            html_out = f"{head}</p>{figure}{tail}"
        else:
            html_out = f"{content_html}{figure}"
        return html_out, [{"url": url, "caption": caption}]

    images: list[dict[str, str]] = []
    out: list[str] = [parts[0]]
    for section, i in enumerate(range(1, len(parts), 2)):
        h2 = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        should_fetch = (
            len(images) < target and keywords and not _FAQ_H2_RE.search(h2)
        )
        if should_fetch:
            caption = _TAG_RE.sub(" ", h2).strip()[:120] or f"Section {section + 1}"
            url = await fetch_content_image(
                caption,
                keywords[section % len(keywords)],
                exclude_urls=used,
            )
            if images:
                await asyncio.sleep(img_pause)
            if not url or url in used:
                print(
                    f"[AutoBlog] duplicate content image skipped: {caption!r}",
                    flush=True,
                )
                out.append(h2)
                out.append(body)
                continue
            used.add(url)
            print(
                f"[AutoBlog] content image {len(images) + 1}/{target} "
                f"ready: {caption!r}"
            )
            body = f"{body}{_figure_html(url, caption)}"
            images.append({"url": url, "caption": caption})
        out.append(h2)
        out.append(body)
    return "".join(out), images


def slugify(title: str) -> str:
    """URL slug matching the blog_posts check constraint ^[a-z0-9]+(-[a-z0-9]+)*$."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")[:80].strip("-")
    return slug or f"blog-{_utc_str()}"


def _utc_str() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d")