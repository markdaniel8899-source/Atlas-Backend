"""Publisher for AutoBlog v2 - Supabase insert with RLS bypass."""

from __future__ import annotations

import asyncio
import uuid
from functools import lru_cache
from typing import Any

from supabase import Client, create_client

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.llm.clients import BlogPipelineError
from app.autoblog.images import slugify
from app.cache import clear_all as clear_read_caches


@lru_cache(maxsize=1)
def _supabase_client() -> Client:
    """Service-role client (bypasses RLS - the only role that may write blog_posts)."""
    settings = get_autoblog_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise BlogPipelineError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required to save posts.",
            status_code=503,
        )
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def _insert_post(payload: dict[str, Any]) -> dict[str, Any]:
    """Sync supabase-py insert (runs in a worker thread). Retries slug clashes."""
    client = _supabase_client()
    base_slug = payload.pop("slug_base")
    last_error: Exception | None = None

    for attempt in range(4):
        slug = (
            base_slug
            if attempt == 0
            else f"{base_slug}-{_utc_str()}"
            if attempt == 1
            else f"{base_slug}-{uuid.uuid4().hex[:6]}"
        )
        try:
            result = client.table("blog_posts").insert({**payload, "slug": slug}).execute()
            if result.data:
                return dict(result.data[0])
            raise BlogPipelineError("Supabase insert returned no row.")
        except BlogPipelineError:
            raise
        except Exception as exc:
            last_error = exc
            if "duplicate" in str(exc).lower() and attempt < 3:
                continue
            break

    raise BlogPipelineError(f"Supabase insert failed: {last_error}") from last_error


async def publish_post(
    title: str,
    content_html: str,
    excerpt: str,
    cover_image_url: str,
    content_images: list[dict[str, str]],
    tags: list[str],
    keywords: list[str],
) -> dict[str, Any]:
    """Save the finished post to Supabase."""
    payload = {
        "slug_base": slugify(title),
        "title": title,
        "content": content_html,
        "excerpt": excerpt,
        "cover_image_url": cover_image_url,
        "content_images": content_images,
        "tags": tags,
        "keywords": keywords,
    }
    post = await asyncio.to_thread(_insert_post, payload)
    clear_read_caches()  # new post -> drop the public read caches
    print(
        f"[AutoBlog] Step 6/6: Saved post id={post.get('id')} slug={post.get('slug')}",
        flush=True,
    )
    return post


def _utc_str() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d")