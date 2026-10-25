"""Content-image fetching for the dynamic pipeline.

Unlike the older injector, every image here is persisted to R2 first, so a
published article never points at a temporary Pixabay URL.
"""

from __future__ import annotations

import html as html_lib
import re

from app.autoblog.images import ImageResult, _extract_provider_id
from app.autoblog.images.storage import persist_image
from app.autoblog.sources import fetch_relevant_image


_H2_END_RE = re.compile(r"</h2>", re.IGNORECASE)


async def fetch_content_image(
    section_title: str,
    keyword: str,
    *,
    slug: str,
    index: int,
    used_urls: set[str] | None = None,
) -> ImageResult | None:
    """Fetch one unique Pixabay photo and persist it to R2."""
    used = used_urls if used_urls is not None else set()

    source_url = await fetch_relevant_image(
        section_title,
        [keyword] if keyword else None,
        category="technology",
        exclude_urls=used,
    )
    if not source_url or source_url in used:
        return None

    stored_url = await persist_image(source_url, f"{slug}-{index}", prefix="content")
    if not stored_url:
        return None

    used.add(source_url)
    used.add(stored_url)
    return ImageResult(
        url=stored_url,
        caption=section_title,
        provider_id=_extract_provider_id(source_url),
    )


def figure_html(url: str, caption: str, alt: str) -> str:
    return (
        '<figure class="blog-visual">'
        f'<img src="{html_lib.escape(url, quote=True)}" '
        f'alt="{html_lib.escape(alt, quote=True)}" loading="lazy" '
        'width="800" height="450" />'
        f"<figcaption>{html_lib.escape(caption)}</figcaption>"
        "</figure>"
    )


def inject_content_images(
    content_html: str,
    images: list[ImageResult],
    *,
    keyword: str,
    positions: tuple[int, ...] = (0, 2, 4),
) -> str:
    """Insert figures after evenly spread H2 headings."""
    if not images:
        return content_html

    html_text = content_html
    for image, position in zip(images, positions):
        figure = figure_html(
            image.url,
            image.caption,
            alt=f"{keyword} illustration: {image.caption}",
        )
        matches = list(_H2_END_RE.finditer(html_text))
        if matches:
            index = min(position, len(matches) - 1)
            insert_at = matches[index].end()
            html_text = (
                html_text[:insert_at] + f"\n{figure}\n" + html_text[insert_at:]
            )
        elif "</p>" in html_text:
            head, tail = html_text.split("</p>", 1)
            html_text = f"{head}</p>\n{figure}\n{tail}"
        else:
            html_text = f"{html_text}\n{figure}"

    return html_text


__all__ = ["fetch_content_image", "figure_html", "inject_content_images"]
