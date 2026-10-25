"""Image fetching with deduplication via Supabase."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.sources import fetch_relevant_image


@dataclass
class ImageResult:
    url: str
    caption: str
    provider_id: str
    provider: str = "pixabay"


async def _get_used_image_ids() -> set[str]:
    """Get set of already used image IDs from Supabase."""
    # In production, query Supabase:
    # SELECT provider_image_id FROM used_images WHERE provider = 'pixabay'
    # For now, return empty set (in-memory fallback)
    return set()


async def _mark_image_used(provider: str, provider_id: str, url: str, post_slug: str, role: str) -> None:
    """Mark image as used in Supabase."""
    # In production:
    # INSERT INTO used_images (provider, provider_id, url, post_slug, role)
    # ON CONFLICT (provider, provider_image_id) DO NOTHING
    pass


def _extract_provider_id(url: str) -> str:
    """Extract provider-specific image ID from URL."""
    # Pixabay: https://pixabay.com/photos/ai-technology-robot-1234567/
    # Extract the numeric ID at the end
    match = re.search(r'/(\d+)/?$', url)
    if match:
        return match.group(1)
    # Fallback: use hash of URL
    return hashlib.md5(url.encode()).hexdigest()[:16]


async def fetch_unique_image(
    query: str,
    role: str = "content",
    fallback_query: str | None = None,
    post_slug: str = "",
    used_image_ids: set[str] | None = None,
) -> Optional[ImageResult]:
    """
    Fetch a unique image not used before.

    Args:
        query: Search query for the image
        role: "cover" or "content"
        fallback_query: Alternative query if first fails
        post_slug: Post slug for tracking
        used_image_ids: Pre-fetched set of used image IDs (for performance)

    Returns:
        ImageResult or None if no unique image found
    """
    if used_image_ids is None:
        used_image_ids = await _get_used_image_ids()

    # Try primary query
    for q in [query] + ([fallback_query] if fallback_query else []):
        try:
            # Request more results to find unused ones
            candidate = await fetch_relevant_image(q, "technology", count=5)
            if not candidate:
                continue

            # Check if already used
            provider_id = _extract_provider_id(candidate.url)
            if provider_id in used_image_ids:
                continue

            # Found unused image
            result = ImageResult(
                url=candidate.url,
                caption=candidate.caption,
                provider_id=provider_id,
                provider="pixabay",
            )

            # Mark as used
            await _mark_image_used("pixabay", provider_id, candidate.url, "", "cover" if role == "cover" else "content")

            return result
        except Exception:
            continue

    # Try fallback provider (Unsplash) if Pixabay fails
    # This would require Unsplash API integration
    # For now, return None

    return None


async def fetch_unique_cover_image(
    title: str,
    keywords: list[str],
    post_slug: str,
    used_image_ids: set[str] | None = None,
) -> Optional[str]:
    """Fetch a unique cover image for the article."""
    # Try primary query
    query = " ".join(keywords[:3]) if keywords else title
    result = await fetch_unique_image(
        query=query,
        role="cover",
        fallback_query=title,
        post_slug=post_slug,
        used_image_ids=used_image_ids,
    )
    return result.url if result else None


async def fetch_unique_content_images(
    queries: list[str],
    post_slug: str,
    used_image_ids: set[str] | None = None,
    count: int = 2,
) -> list[str]:
    """Fetch multiple unique content images."""
    urls = []
    used = set(used_image_ids) if used_image_ids else await _get_used_image_ids()

    for i, query in enumerate(queries[:count * 2]):  # Try more queries than needed
        if len(urls) >= count:
            break

        result = await fetch_unique_image(
            query=query,
            role="content",
            post_slug=post_slug,
            used_image_ids=used,
        )
        if result:
            urls.append(result.url)
            used.add(_extract_provider_id(result.url))

    return urls


__all__ = [
    "fetch_unique_image",
    "fetch_unique_cover_image",
    "fetch_unique_content_images",
    "ImageResult",
]