"""Internal-link discovery and injection for SEO.

The writer is asked to include two internal links. This module supplies the
real targets (recent published posts plus evergreen app pages) and guarantees
the links exist before validation runs.
"""

from __future__ import annotations

import asyncio
import html as html_lib
import re
from typing import Any

from app.autoblog.models import InternalLink


DEFAULT_LINKS: tuple[InternalLink, ...] = (
    InternalLink(title="Browse all ATLAS AI articles", url="/blog"),
    InternalLink(title="Open the ATLAS learning workspace", url="/app"),
    InternalLink(title="How ATLAS builds your learning roadmap", url="/about"),
)

_HREF_RE = re.compile(r"""<a\b[^>]*\bhref=["']([^"']+)["']""", re.IGNORECASE)
_FAQ_RE = re.compile(
    r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b).*?</h2>",
    re.IGNORECASE | re.DOTALL,
)


def _fetch_recent_link_rows(limit: int) -> list[dict[str, Any]]:
    from app.autoblog.publisher import _supabase_client

    client = _supabase_client()
    response = (
        client.table("blog_posts")
        .select("title, slug, published_at")
        .order("published_at", desc=True)
        .limit(max(limit * 3, limit))
        .execute()
    )
    rows = getattr(response, "data", None)
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict) and row.get("published_at")
    ]


async def gather_internal_links(limit: int = 4) -> list[InternalLink]:
    """Recent posts first, then evergreen pages, de-duplicated by URL."""
    links: list[InternalLink] = []

    try:
        rows = await asyncio.to_thread(_fetch_recent_link_rows, limit)
        for row in rows:
            title = str(row.get("title") or "").strip()
            slug = str(row.get("slug") or "").strip()
            if title and slug:
                links.append(InternalLink(title=title, url=f"/blog/{slug}"))
    except Exception as exc:
        print(
            f"[AutoBlog] Internal-link lookup failed "
            f"({type(exc).__name__}: {exc}) - using evergreen pages only.",
            flush=True,
        )

    for fallback in DEFAULT_LINKS:
        links.append(fallback)

    deduped: list[InternalLink] = []
    seen: set[str] = set()
    for link in links:
        if link.url not in seen:
            seen.add(link.url)
            deduped.append(link)
    return deduped[: max(limit, 2)]


def count_internal_links(content_html: str) -> int:
    return len(
        [
            href
            for href in _HREF_RE.findall(content_html)
            if href.startswith("/") and not href.startswith("//")
        ]
    )


def ensure_internal_links(
    content_html: str,
    links: list[InternalLink],
    *,
    minimum: int = 2,
) -> str:
    """Add a compact 'Continue learning' block if the article is short on links."""
    existing = {
        href
        for href in _HREF_RE.findall(content_html)
        if href.startswith("/") and not href.startswith("//")
    }
    if len(existing) >= minimum:
        return content_html

    needed = minimum - len(existing)
    targets: list[InternalLink] = []
    seen = set(existing)
    for link in links:
        if link.url not in seen:
            seen.add(link.url)
            targets.append(link)
        if len(targets) >= needed:
            break

    if not targets:
        return content_html

    items = "".join(
        f'<li><a href="{html_lib.escape(link.url, quote=True)}">'
        f"{html_lib.escape(link.title)}</a></li>"
        for link in targets
    )
    block = (
        "<h2>Continue learning with ATLAS</h2>"
        f"<ul>{items}</ul>"
    )

    faq_match = _FAQ_RE.search(content_html)
    if faq_match:
        position = faq_match.start()
        return content_html[:position] + block + "\n" + content_html[position:]
    return content_html + "\n" + block


__all__ = [
    "DEFAULT_LINKS",
    "count_internal_links",
    "ensure_internal_links",
    "gather_internal_links",
]
