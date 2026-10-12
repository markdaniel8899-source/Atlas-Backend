"""Public blog reads - cached list/detail endpoints for web + mobile.

Built for the 1GB server and a future React Native client:

  * the LIST never ships `content` / `content_images` (multi-KB HTML) -
    mobile gets a light payload (id, slug, title, excerpt, cover, tags),
    the full body is detail-only,
  * every GET runs through a strict TTL+LRU cache (maxsize=50, ttl=300s),
    so hot traffic never touches Supabase,
  * supabase-py is sync, so all DB work runs via asyncio.to_thread and
    never blocks the event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from supabase import Client, create_client

from app.cache import ttl_lru_cache
from app.config import get_settings

router = APIRouter(prefix="/api/blog", tags=["blog"])

# Light columns for the list - deliberately no `content` / `content_images`.
_LIST_COLUMNS = "id, slug, title, excerpt, cover_image_url, tags, keywords, published_at"
_LIST_DEFAULT_LIMIT = 12

_client: Client | None = None


def _get_client() -> Client:
    """Reuse one supabase client (service-role preferred, anon fallback)."""
    settings = get_settings()
    if not settings.supabase_url:
        raise HTTPException(
            status_code=503,
            detail="Blog storage is not configured: set SUPABASE_URL.",
        )
    key = settings.supabase_service_role_key or settings.supabase_anon_key
    if not key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Blog storage is not configured: set SUPABASE_SERVICE_ROLE_KEY "
                "or SUPABASE_ANON_KEY."
            )
        )
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, key)
    return _client


def _fetch_list(limit: int) -> list[dict[str, Any]]:
    """Sync Supabase read - wrapped by the cache and asyncio.to_thread."""
    result = (
        _get_client()
        .table("blog_posts")
        .select(_LIST_COLUMNS)
        .order("published_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(result.data or [])


def _fetch_one(slug: str) -> dict[str, Any] | None:
    """Sync Supabase read of one post (full body) - cached + to_thread."""
    result = (
        _get_client()
        .table("blog_posts")
        .select("*")
        .eq("slug", slug)
        .limit(1)
        .execute()
    )
    rows = list(result.data or [])
    return rows[0] if rows else None


# LiteSpeed-style cache: strictly 50 entries, 5-minute freshness.
_cached_list = ttl_lru_cache(maxsize=50, ttl=300)(_fetch_list)
_cached_one = ttl_lru_cache(maxsize=50, ttl=300)(_fetch_one)


@router.get("")
async def list_published_blogs(
    limit: int = Query(default=_LIST_DEFAULT_LIMIT, ge=1, le=50),
) -> dict[str, Any]:
    """Latest published posts (lightweight - no body HTML). Cached 5 min."""
    items = await asyncio.to_thread(_cached_list, limit)
    return {"items": items, "count": len(items)}


@router.get("/{slug}")
async def get_published_blog(slug: str) -> dict[str, Any]:
    """One post by slug, full content included. Cached 5 min."""
    post = await asyncio.to_thread(_cached_one, slug)
    if post is None:
        raise HTTPException(status_code=404, detail="Blog post not found.")
    return post
