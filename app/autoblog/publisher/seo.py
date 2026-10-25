"""SEO-aware publisher: Supabase insert + Vercel ISR revalidation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.images import slugify
from app.autoblog.llm.clients import BlogPipelineError
from app.autoblog.models import DynamicArticleDraft, KeywordPlan, MetaBundle
from app.autoblog.publisher import _insert_post
from app.cache import clear_all as clear_read_caches


_NEW_COLUMNS = frozenset(
    {"meta_title", "meta_description", "keywords_json", "word_count", "seo_score"}
)


def _missing_new_columns(exc: Exception) -> bool:
    message = str(exc).lower()
    if "duplicate" in message:
        return False
    if any(column in message for column in _NEW_COLUMNS):
        return True
    return "pgrst204" in message or "schema cache" in message


async def trigger_revalidation(slug: str) -> None:
    """Ask Vercel to rebuild the blog routes. Failures never block publishing."""
    settings = get_autoblog_settings()
    if not settings.vercel_deploy_hook_url and not settings.vercel_revalidate_url:
        return

    payload = {"slug": slug, "paths": ["/blog", f"/blog/{slug}"]}

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            if settings.vercel_deploy_hook_url:
                response = await client.post(settings.vercel_deploy_hook_url, json={})
                response.raise_for_status()
                print("[AutoBlog] Vercel deploy hook triggered.", flush=True)

            if settings.vercel_revalidate_url:
                params = (
                    {"secret": settings.vercel_revalidate_secret}
                    if settings.vercel_revalidate_secret
                    else None
                )
                headers = (
                    {
                        "x-revalidate-secret": settings.vercel_revalidate_secret,
                        "Authorization": f"Bearer {settings.vercel_revalidate_secret}",
                    }
                    if settings.vercel_revalidate_secret
                    else {}
                )
                response = await client.post(
                    settings.vercel_revalidate_url,
                    params=params,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                print(
                    f"[AutoBlog] Vercel revalidation sent for /blog/{slug}.",
                    flush=True,
                )
    except Exception as exc:
        print(
            f"[AutoBlog] Vercel revalidation failed "
            f"({type(exc).__name__}: {exc}) - post is still live.",
            flush=True,
        )


async def publish_seo_post(
    *,
    article: DynamicArticleDraft,
    meta: MetaBundle,
    keywords: KeywordPlan,
    cover_image_url: str,
    content_images: list[dict[str, str]],
    seo_score: int,
) -> dict[str, Any]:
    """Step 7: write the complete row, clear read caches and revalidate."""
    full_payload: dict[str, Any] = {
        "slug_base": slugify(article.title),
        "title": article.title,
        "content": article.content_html,
        "excerpt": meta.excerpt,
        "cover_image_url": cover_image_url,
        "content_images": content_images,
        "tags": article.tags,
        "keywords": keywords.all_keywords(),
        "keywords_json": keywords.as_json(),
        "meta_title": meta.meta_title,
        "meta_description": meta.meta_description,
        "word_count": article.word_count,
        "seo_score": seo_score,
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    try:
        post = await asyncio.to_thread(_insert_post, dict(full_payload))
    except BlogPipelineError as exc:
        if not _missing_new_columns(exc):
            raise
        print(
            "[AutoBlog] SEO columns are not in Supabase yet - retrying with the "
            "legacy schema. Run migration 0020_blog_posts_seo_fields.sql.",
            flush=True,
        )
        legacy_payload = {
            key: value
            for key, value in full_payload.items()
            if key not in _NEW_COLUMNS
        }
        post = await asyncio.to_thread(_insert_post, legacy_payload)

    clear_read_caches()
    await trigger_revalidation(str(post.get("slug") or ""))

    print(
        f"[AutoBlog] Published post id={post.get('id')} "
        f"slug={post.get('slug')} score={seo_score}",
        flush=True,
    )
    return post


__all__ = ["publish_seo_post", "trigger_revalidation"]
