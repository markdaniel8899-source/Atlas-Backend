"""One-off migration: replace Pollinations URLs in blog_posts with Pixabay photos.

Every row whose cover_image_url / content_images / content HTML still point at
image.pollinations.ai gets a relevant Pixabay photo (reusing the pipeline's
fetch_relevant_image). Only URLs are swapped — no image data touches the DB.

Usage (from backend/):
    .\\venv\\Scripts\\python.exe -m scripts.migrate_pollinations_to_pixabay --dry-run
    .\\venv\\Scripts\\python.exe -m scripts.migrate_pollinations_to_pixabay

Requires PIXABAY_API_KEY + SUPABASE_SERVICE_ROLE_KEY in backend/.env.
Rows with no Pixabay hit for their query are left untouched and reported.
Idempotent — safe to re-run; only rows still containing Pollinations URLs move.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supabase import Client, create_client  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.auto_blog import PIXABAY_API_KEY, fetch_relevant_image  # noqa: E402

POLLINATIONS = "image.pollinations.ai"
PIXABAY_PAUSE_SECONDS = 1.1  # stay well under the free-tier rate limit


def _supabase() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise SystemExit("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing in backend/.env")
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def _needs_migration(post: dict) -> bool:
    blob = " ".join(
        filter(
            None,
            [
                str(post.get("cover_image_url") or ""),
                str(post.get("content") or ""),
                str(post.get("content_images") or ""),
            ],
        )
    )
    return POLLINATIONS in blob


async def _pixabay(query: str) -> str | None:
    url = await fetch_relevant_image(query)
    await asyncio.sleep(PIXABAY_PAUSE_SECONDS)
    return url


async def _migrate(dry_run: bool) -> int:
    client = _supabase()
    rows = (
        client.table("blog_posts")
        .select("id, title, slug, keywords, cover_image_url, content, content_images")
        .execute()
        .data
        or []
    )
    targets = [r for r in rows if _needs_migration(r)]
    print(f"Found {len(targets)}/{len(rows)} posts with Pollinations URLs.")
    if not targets:
        return 0

    updated = 0
    would_update = 0
    for post in targets:
        slug = post["slug"]
        keywords = post.get("keywords") or []
        images = post.get("content_images") or []
        cover_old = str(post.get("cover_image_url") or "")
        content = str(post.get("content") or "")
        patch: dict = {}

        # ── cover ──
        cover_new = None
        if POLLINATIONS in cover_old:
            query = " ".join(keywords[:3]) if keywords else str(post["title"])
            cover_new = await _pixabay(query)
            if cover_new:
                patch["cover_image_url"] = cover_new
                print(f"  [{slug}] cover <- {cover_new}")
            else:
                print(f"  [{slug}] cover: no Pixabay hit for {query!r} — kept old URL")

        # ── in-article content_images ──
        url_map: list[tuple[str, str]] = []
        new_images = []
        for img in images:
            old_url = str(img.get("url") or "")
            if POLLINATIONS in old_url:
                caption = str(img.get("caption") or "")[:80]
                new_url = await _pixabay(caption)
                if new_url:
                    new_images.append({**img, "url": new_url})
                    url_map.append((old_url, new_url))
                    print(f"  [{slug}] image {caption!r} <- {new_url}")
                    continue
                print(f"  [{slug}] image {caption!r}: no Pixabay hit — kept old URL")
            new_images.append(img)
        if url_map:
            patch["content_images"] = new_images

        # ── swap URLs inside the article HTML ──
        new_content = content
        if cover_new:
            new_content = new_content.replace(cover_old, cover_new)
        for old_url, new_url in url_map:
            new_content = new_content.replace(old_url, new_url)
        if new_content != content:
            patch["content"] = new_content

        if dry_run:
            if patch:
                would_update += 1
                print(f"  [{slug}] DRY RUN — would update keys: {sorted(patch)}")
            continue

        if patch:
            client.table("blog_posts").update(patch).eq("id", post["id"]).execute()
            updated += 1
            remaining = new_content.count(POLLINATIONS) if "content" in patch else content.count(POLLINATIONS)
            if remaining:
                print(f"  [{slug}] WARNING: {remaining} Pollinations ref(s) still in HTML")

    if dry_run:
        print(f"Done — would update {would_update}/{len(targets)} posts (dry run).")
    else:
        print(f"Done — updated {updated}/{len(targets)} posts.")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print actions without writing")
    args = parser.parse_args()
    if not PIXABAY_API_KEY:
        raise SystemExit("PIXABAY_API_KEY is empty in backend/.env — add it first.")
    asyncio.run(_migrate(args.dry_run))


if __name__ == "__main__":
    main()
