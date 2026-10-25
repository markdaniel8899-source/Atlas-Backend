"""Durable image persistence for the dynamic pipeline.

Pixabay hands back temporary ``/get/...`` URLs. Every content image must be
downloaded and re-uploaded to R2 before it is embedded in a stored article.
"""

from __future__ import annotations

import asyncio

import httpx

from app.autoblog.sources import MAX_COVER_DOWNLOAD_BYTES, UNSPLASH_FALLBACK_IMAGES
from app.services.r2_storage import r2_storage


async def persist_image(url: str, slug: str, prefix: str = "content") -> str | None:
    """Download a remote image, upload a compressed WebP to R2, return its URL.

    Unsplash fallbacks stay usable when R2 is not configured. Temporary Pixabay
    URLs are dropped instead of being written into the database.
    """
    if not url:
        return None
    if not url.startswith("http"):
        return url

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = (
                response.headers.get("content-type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )
            if not content_type.startswith("image/"):
                raise ValueError(
                    f"unexpected content type: {content_type or 'unknown'}"
                )
            image_bytes = response.content
            if not image_bytes:
                raise ValueError("download returned no bytes")
            if len(image_bytes) > MAX_COVER_DOWNLOAD_BYTES:
                raise ValueError("image exceeds the 15 MB limit")

        stored = await asyncio.to_thread(
            r2_storage.compress_and_upload,
            image_bytes,
            slug,
            prefix,
        )
        if not stored:
            raise RuntimeError("R2 upload returned an empty URL")
        return stored
    except Exception as exc:
        print(
            f"[AutoBlog] image persistence failed for {prefix}/{slug} "
            f"({type(exc).__name__}: {exc})",
            flush=True,
        )
        if url in UNSPLASH_FALLBACK_IMAGES:
            return url
        return None


__all__ = ["persist_image"]
