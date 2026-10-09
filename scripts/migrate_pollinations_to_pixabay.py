"""One-off migration: fix blog_posts images (Pollinations → Pixabay / refresh).

Modes:
    (default)      Replace any remaining image.pollinations.ai URLs with
                   Pixabay photos. Idempotent — only touches rows that still
                   contain Pollinations URLs.
    --refresh      Re-fetch cover + in-article images for EVERY post using
                   the title-first Pixabay search (fixes irrelevant photos
                   such as cake-for-"AI Is Eating Software") and strips
                   <figcaption> text out of the stored article HTML.

Only URLs are swapped — no image data touches the DB. Rows where a query
finds nothing keep their existing URL and are reported.

Usage (from backend/):
    .\\venv\\Scripts\\python.exe -m scripts.migrate_pollinations_to_pixabay --dry-run
    .\\venv\\Scripts\\python.exe -m scripts.migrate_pollinations_to_pixabay
    .\\venv\\Scripts\\python.exe -m scripts.migrate_pollinations_to_pixabay --refresh

Requires PIXABAY_API_KEY + SUPABASE_SERVICE_ROLE_KEY in backend/.env.
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supabase import Client, create_client  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.auto_blog import PIXABAY_API_KEY, fetch_relevant_image  # noqa: E402

POLLINATIONS = "image.pollinations.ai"
FIGCAPTION_RE = re.compile(r"<figcaption\b.*?</figcaption>", re.IGNORECASE | re.DOTALL)
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


async def _pixabay(title: str, keywords: list[str]) -> str | None:
    url = await fetch_relevant_image(title, keywords, category="technology")
    await asyncio.sleep(PIXABAY_PAUSE_SECONDS)
    return url


async def _migrate(dry_run: bool, refresh: bool) -> int:
    client = _supabase()
    rows = (
        client.table("blog_posts")
        .select("id, title, slug, keywords, cover_image_url, content, content_images")
        .execute()
        .data
        or []
    )
    if refresh:
        targets = rows
        print(f"REFRESH mode: re-fetching images for all {len(targets)} posts.")
    else:
        targets = [r for r in rows if _needs_migration(r)]
        print(f"Found {len(targets)}/{len(rows)} posts with Pollinations URLs.")
    if not targets:
        return 0

    updated = 0
    would_update = 0
    for post in targets:
        slug = post["slug"]
        keywords = post.get("keywords") or []
        title = str(post["title"])
        images = post.get("content_images") or []
        cover_old = str(post.get("cover_image_url") or "")
        content = str(post.get("content") or "")
        patch: dict = {}

        # ── cover ──
        cover_new = None
        if refresh or POLLINATIONS in cover_old:
            cover_new = await _pixabay(title, keywords)
            if cover_new and cover_new != cover_old:
                patch["cover_image_url"] = cover_new
                print(f"  [{slug}] cover <- {cover_new}")
            elif not cover_new:
                print(f"  [{slug}] cover: no image found — kept old URL")

        # ── in-article content_images ──
        url_map: list[tuple[str, str]] = []
        new_images = []
        for img in images:
            old_url = str(img.get("url") or "")
            caption = str(img.get("caption") or "")
            if refresh or POLLINATIONS in old_url:
                new_url = await _pixabay(caption[:100] or title, keywords)
                if new_url and new_url != old_url:
                    new_images.append({**img, "url": new_url})
                    url_map.append((old_url, new_url))
                    print(f"  [{slug}] image {caption[:50]!r} <- {new_url}")
                    continue
                if not new_url:
                    print(f"  [{slug}] image {caption[:50]!r}: no image found — kept")
            new_images.append(img)
        if url_map:
            patch["content_images"] = new_images

        # ── swap URLs inside the article HTML ──
        new_content = content
        if cover_new and cover_old:
            new_content = new_content.replace(cover_old, cover_new)
        for old_url, new_url in url_map:
            new_content = new_content.replace(old_url, new_url)

        # ── strip rendered captions from stored HTML (image-only figures) ──
        if refresh and "<figcaption" in new_content.lower():
            new_content = FIGCAPTION_RE.sub("", new_content)
            print(f"  [{slug}] stripped figcaptions from stored HTML")

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
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-fetch ALL posts' images with the title-first search + strip figcaptions",
    )
    args = parser.parse_args()
    if not PIXABAY_API_KEY:
        raise SystemExit("PIXABAY_API_KEY is empty in backend/.env — add it first.")
    asyncio.run(_migrate(args.dry_run, args.refresh))


if __name__ == "__main__":
    main()
