"""Visual strategy for the dynamic pipeline.

Data-rich stories get a fact-grounded table plus a QuickChart image. News and
explainer stories get a Pixabay cover with one or two persisted in-article
photos. When photo lookup fails, a compact 'At a glance' table guarantees the
article still ships with a visual.
"""

from __future__ import annotations

import html as html_lib
import json
import re
import urllib.parse
from typing import Any

from app.autoblog.images import (
    ImageResult,
    _extract_provider_id,
    fetch_cover_image,
    slugify,
)
from app.autoblog.images.dynamic import fetch_content_image, inject_content_images
from app.autoblog.internal_links import ensure_internal_links
from app.autoblog.llm.clients import (
    extract_json,
    groq_research_with_nemotron_fallback,
)
from app.autoblog.models import (
    DynamicArticleDraft,
    InternalLink,
    KeywordPlan,
    TopicTrend,
    VisualPayload,
    VisualStrategy,
)
from app.autoblog.research import FactSheet
from app.autoblog.sources import UNSPLASH_FALLBACK_IMAGES
from app.autoblog.visuals import render_table


_TABLE_SYSTEM = """You turn a research fact sheet into ONE clean HTML table.

Return ONLY strict JSON:
{
  "title": "...",
  "caption": "...",
  "columns": ["...", "..."],
  "rows": [["...", "..."], ["...", "..."]],
  "source_url": "...",
  "intro_sentence": "...",
  "takeaway": "..."
}

RULES:
- 2-5 columns and 2-8 rows
- Every cell must be short enough to render on mobile (under 70 characters when possible)
- Use ONLY numbers and entities from the fact sheet
- source_url MUST be one of the supplied source URLs
- intro_sentence sets up the table in one sentence
- takeaway states what the reader should notice in one sentence
- No HTML inside cells
"""


# ─── Small helpers ────────────────────────────────────────────────────────────


def _clean(value: Any, limit: int = 120) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _source_host(url: str) -> str:
    host = urllib.parse.urlsplit(str(url or "")).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _number(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _has_numeric_data(fact_sheet: FactSheet) -> bool:
    values = {
        round(number, 6)
        for number in (_number(item.get("value")) for item in fact_sheet.numbers)
        if number is not None
    }
    return len(values) >= 3


def _has_html_table(content_html: str) -> bool:
    return bool(re.search(r"<table\b", content_html, re.IGNORECASE))


def _has_mermaid_chart(content_html: str) -> bool:
    return bool(
        re.search(
            r"""<[a-z][^>]*\bclass\s*=\s*["'][^"']*\bmermaid\b""",
            content_html,
            re.IGNORECASE,
        )
        or re.search(r"```mermaid\b", content_html, re.IGNORECASE)
    )


def _cover_fallback(headline: str) -> tuple[str, str]:
    index = sum(ord(char) for char in headline) % len(UNSPLASH_FALLBACK_IMAGES)
    return UNSPLASH_FALLBACK_IMAGES[index], f"unsplash:topic_{index}"


# ─── Tables ───────────────────────────────────────────────────────────────────


def _table_payload(vis: dict[str, Any], *, altitude: str = "") -> VisualPayload | None:
    html_text = render_table(vis)
    if not html_text:
        return None
    return VisualPayload(
        kind="table",
        html=html_text,
        alt=_clean(vis.get("title") or "Data table", 160),
        source_url=_clean(vis.get("source_url"), 500),
        intro_sentence=_clean(vis.get("intro_sentence") or altitude, 240),
    )


def _deterministic_table(fact_sheet: FactSheet) -> VisualPayload | None:
    rows: list[list[str]] = []
    for item in fact_sheet.numbers[:5]:
        value = _clean(item.get("value"), 30)
        unit = _clean(item.get("unit"), 20)
        context = _clean(item.get("context"), 90)
        if not value or not context:
            continue
        rows.append([context, f"{value} {unit}".strip(), _source_host(item.get("source_url", ""))])

    if len(rows) < 2:
        return None

    return _table_payload(
        {
            "title": "Key numbers at a glance",
            "caption": "Reported figures from the sources used for this article.",
            "columns": ["Metric", "Reported value", "Source"],
            "rows": rows,
            "source_url": fact_sheet.source_urls[0] if fact_sheet.source_urls else "",
            "intro_sentence": "The numbers below come directly from the research used for this story.",
            "takeaway": "Compare the scale of the change before acting on any single figure.",
        }
    )


def _glance_table(fact_sheet: FactSheet, trend: TopicTrend) -> VisualPayload | None:
    claims = [_clean(claim, 160) for claim in fact_sheet.claims[:3] if _clean(claim)]
    if len(claims) < 2:
        return None

    rows = [
        ["What happened", claims[0]],
        ["Why it matters", claims[1]],
    ]
    if len(claims) >= 3:
        rows.append(["What to watch", claims[2]])
    rows.append(["Next step", "Test one small workflow change and measure the result."])

    return _table_payload(
        {
            "title": "At a glance",
            "caption": "The key points from the source material, without the hype.",
            "columns": ["Question", "What the sources say"],
            "rows": rows,
            "source_url": fact_sheet.source_urls[0] if fact_sheet.source_urls else "",
            "intro_sentence": f"{trend.why_now}.",
            "takeaway": "Focus on the practical change, not the announcement.",
        }
    )


async def _ai_table(trend: TopicTrend, fact_sheet: FactSheet) -> VisualPayload | None:
    numbers = "\n".join(
        f"- {item.get('value')} {item.get('unit')}: {item.get('context', '')[:140]} "
        f"(source: {item.get('source_url', '')})"
        for item in fact_sheet.numbers[:15]
    ) or "(none)"
    entities = "\n".join(
        f"- {item.get('name')}: {item.get('context', '')[:100]}"
        for item in fact_sheet.entities[:10]
    ) or "(none)"
    sources = "\n".join(f"- {url}" for url in fact_sheet.source_urls[:10]) or "(none)"

    prompt = f"""Topic: {trend.headline}
Angle: {trend.angle}

NUMBERS:
{numbers}

ENTITIES:
{entities}

SOURCE URLS:
{sources}

Build the table now."""

    try:
        result = await groq_research_with_nemotron_fallback(
            _TABLE_SYSTEM,
            prompt,
            temperature=0.1,
            max_tokens=2048,
        )
        parsed = extract_json(result.text)
        data = parsed.data
        if not isinstance(data, dict):
            return None

        columns = [_clean(column, 40) for column in data.get("columns", []) if _clean(column, 40)]
        raw_rows = data.get("rows", [])
        if not isinstance(raw_rows, list) or not 2 <= len(columns) <= 5:
            return None

        rows: list[list[str]] = []
        for raw_row in raw_rows[:8]:
            if not isinstance(raw_row, list):
                continue
            row = [_clean(cell, 70) for cell in raw_row]
            if len(row) == len(columns):
                rows.append(row)
        if len(rows) < 2:
            return None

        source_url = _clean(data.get("source_url"), 500)
        if source_url not in fact_sheet.source_urls:
            source_url = fact_sheet.source_urls[0] if fact_sheet.source_urls else ""

        return _table_payload(
            {
                "title": _clean(data.get("title") or "Key numbers", 120),
                "caption": _clean(data.get("caption") or "Figures from the cited sources.", 200),
                "columns": columns,
                "rows": rows,
                "source_url": source_url,
                "intro_sentence": _clean(data.get("intro_sentence"), 240),
                "takeaway": _clean(data.get("takeaway"), 240),
            }
        )
    except Exception as exc:
        print(
            f"[AutoBlog] AI table failed "
            f"({type(exc).__name__}: {exc}) - using deterministic table.",
            flush=True,
        )
        return None


async def generate_data_table(trend: TopicTrend, fact_sheet: FactSheet) -> VisualPayload | None:
    return (
        await _ai_table(trend, fact_sheet)
        or _deterministic_table(fact_sheet)
        or _glance_table(fact_sheet, trend)
    )


# ─── Charts ───────────────────────────────────────────────────────────────────


def _chart_from_facts(fact_sheet: FactSheet) -> dict[str, Any] | None:
    groups: dict[str, list[tuple[str, float, str]]] = {}
    seen_labels: dict[str, set[str]] = {}

    for item in fact_sheet.numbers:
        value = _number(item.get("value"))
        label = _clean(item.get("context") or item.get("value"), 42)
        unit = _clean(item.get("unit"), 24) or "value"
        if value is None or not label:
            continue
        labels_seen = seen_labels.setdefault(unit, set())
        if label.lower() in labels_seen:
            continue
        labels_seen.add(label.lower())
        groups.setdefault(unit, []).append((label, value, item.get("source_url", "")))

    if not groups:
        return None

    unit, points = max(groups.items(), key=lambda pair: len(pair[1]))
    if len(points) < 3:
        return None

    return {
        "title": f"{unit.capitalize()} figures in context",
        "labels": [point[0] for point in points[:8]],
        "values": [point[1] for point in points[:8]],
        "unit": unit,
        "source_url": points[0][2] or (fact_sheet.source_urls[0] if fact_sheet.source_urls else ""),
        "alt": "",
    }


def _quickchart_url(spec: dict[str, Any]) -> str:
    config = {
        "type": "bar",
        "data": {
            "labels": spec["labels"],
            "datasets": [
                {
                    "label": spec.get("title") or "Reported value",
                    "data": spec["values"],
                    "backgroundColor": "#7c5cff",
                    "borderRadius": 6,
                }
            ],
        },
        "options": {
            "plugins": {"legend": {"display": False}},
            "scales": {
                "x": {"beginAtZero": True, "grid": {"color": "#e2e8f0"}},
                "y": {"grid": {"display": False}},
            },
        },
    }
    encoded = urllib.parse.quote(json.dumps(config, separators=(",", ":")), safe="")
    return f"https://quickchart.io/chart?w=800&h=450&bkg=white&c={encoded}"


def _chart_payload(spec: dict[str, Any] | None) -> VisualPayload | None:
    if not spec:
        return None
    labels = spec.get("labels")
    values = spec.get("values")
    if not isinstance(labels, list) or not isinstance(values, list):
        return None
    if len(labels) != len(values) or len(labels) < 3:
        return None
    if any(_number(value) is None for value in values):
        return None

    title = _clean(spec.get("title") or "Key numbers", 120)
    alt = _clean(spec.get("alt"), 200) or f"{title}: bar chart comparing {', '.join(str(label) for label in labels)}"
    url = _quickchart_url(spec)
    source_url = _clean(spec.get("source_url"), 500)

    figure = (
        '<figure class="blog-visual blog-chart">'
        f'<img src="{html_lib.escape(url, quote=True)}" '
        f'alt="{html_lib.escape(alt, quote=True)}" loading="lazy" '
        'width="800" height="450" />'
        f"<figcaption>{html_lib.escape(title)}</figcaption>"
        "</figure>"
    )
    if source_url.startswith("http"):
        figure += (
            f'<p class="blog-visual-source">Source: '
            f'<a href="{html_lib.escape(source_url, quote=True)}" '
            f'rel="noopener" target="_blank">{html_lib.escape(_source_host(source_url))}</a></p>'
        )

    return VisualPayload(kind="chart", html=figure, alt=alt, source_url=source_url)


# ─── News images ──────────────────────────────────────────────────────────────


async def _news_content_images(
    draft: DynamicArticleDraft,
    keywords: KeywordPlan,
    *,
    used_urls: set[str],
) -> list[ImageResult]:
    count = 2 if draft.word_count >= 1000 else 1
    headings = [
        section["heading"] for section in draft.sections if section.get("heading")
    ]
    slug = slugify(draft.title)
    images: list[ImageResult] = []

    for index in range(count):
        section_title = headings[(index * 2) % len(headings)] if headings else draft.title
        image = await fetch_content_image(
            section_title,
            keywords.primary_keyword,
            slug=slug,
            index=index + 1,
            used_urls=used_urls,
        )
        if image:
            images.append(image)
    return images


# ─── Strategy + assembly ──────────────────────────────────────────────────────


async def build_visual_strategy(
    trend: TopicTrend,
    keywords: KeywordPlan,
    fact_sheet: FactSheet,
    draft: DynamicArticleDraft,
    *,
    used_image_urls: set[str] | None = None,
) -> VisualStrategy:
    """Step 5: decide the visual mode and prepare every asset once."""
    used_urls = used_image_urls if used_image_urls is not None else set()
    has_data = _has_numeric_data(fact_sheet)

    if has_data:
        cover_url, provider_id = _cover_fallback(draft.title)
        payloads: list[VisualPayload] = []

        table = await generate_data_table(trend, fact_sheet)
        if table:
            payloads.append(table)

        chart_spec = draft.chart_spec or _chart_from_facts(fact_sheet)
        if isinstance(chart_spec, dict):
            chart_source = _clean(chart_spec.get("source_url"), 500)
            if chart_source not in fact_sheet.source_urls:
                chart_spec = {
                    **chart_spec,
                    "source_url": (
                        fact_sheet.source_urls[0] if fact_sheet.source_urls else ""
                    ),
                }
        chart = _chart_payload(chart_spec)
        if chart:
            payloads.append(chart)

        if not payloads:
            fallback = _deterministic_table(fact_sheet) or _glance_table(fact_sheet, trend)
            if fallback:
                payloads.append(fallback)

        print(
            f"[AutoBlog] Visual mode=data, {len(payloads)} data visual(s), "
            "cover=Unsplash fallback (no Pixabay)",
            flush=True,
        )
        return VisualStrategy(
            mode="data",
            cover_url=cover_url,
            cover_provider_id=provider_id,
            content_images=[],
            data_visuals=payloads,
        )

    used_provider_ids = {
        _extract_provider_id(url) for url in used_urls if url
    }
    cover, cover_source = await fetch_cover_image(
        draft.title,
        [keywords.primary_keyword, *keywords.secondary_keywords[:3]],
        used_provider_ids,
    )
    used_provider_ids.add(cover.provider_id)
    used_urls.add(cover_source)
    used_urls.add(cover.url)

    images = await _news_content_images(draft, keywords, used_urls=used_urls)
    payloads: list[VisualPayload] = []
    if not images:
        fallback = _glance_table(fact_sheet, trend) or _deterministic_table(fact_sheet)
        if fallback:
            payloads.append(fallback)

    print(
        f"[AutoBlog] Visual mode=news, cover={'R2' if cover.url.startswith('http') else 'fallback'}, "
        f"{len(images)} content image(s)",
        flush=True,
    )
    return VisualStrategy(
        mode="news",
        cover_url=cover.url,
        cover_provider_id=cover.provider_id,
        content_images=[
            {"url": image.url, "caption": image.caption} for image in images
        ],
        data_visuals=payloads,
    )


def assemble_content(
    draft: DynamicArticleDraft,
    strategy: VisualStrategy,
    links: list[InternalLink],
    *,
    keyword: str = "article",
) -> tuple[str, list[dict[str, str]]]:
    """Inject data visuals, content photos and internal links into the draft."""
    html_text = draft.content_html

    has_table = _has_html_table(html_text)
    has_mermaid = _has_mermaid_chart(html_text)
    visual_html = "\n".join(
        payload.html
        for payload in strategy.data_visuals
        if payload.html
        and not (payload.kind == "table" and has_table)
        and not (payload.kind == "chart" and has_mermaid)
    )
    if visual_html:
        if "<!--DATA_VISUAL-->" in html_text:
            html_text = html_text.replace("<!--DATA_VISUAL-->", visual_html, 1)
        else:
            faq_match = re.search(
                r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b).*?</h2>",
                html_text,
                re.IGNORECASE | re.DOTALL,
            )
            if faq_match:
                html_text = (
                    html_text[: faq_match.start()]
                    + visual_html
                    + "\n"
                    + html_text[faq_match.start():]
                )
            else:
                html_text = f"{html_text}\n{visual_html}"

    html_text = html_text.replace("<!--DATA_VISUAL-->", "")

    image_results = [
        ImageResult(url=image["url"], caption=image["caption"], provider_id="")
        for image in strategy.content_images
    ]
    if image_results:
        html_text = inject_content_images(
            html_text,
            image_results,
            keyword=keyword,
        )

    html_text = ensure_internal_links(html_text, links, minimum=2)
    return html_text, [dict(image) for image in strategy.content_images]


__all__ = [
    "assemble_content",
    "build_visual_strategy",
    "generate_data_table",
]
