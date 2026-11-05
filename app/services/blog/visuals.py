from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import httpx

from app.services.blog.charts import ChartError, render_chart
from app.services.blog.media import (
    generate_branded_content,
    generate_branded_cover,
)
from app.services.blog.prompts import visual_placement_messages
from app.services.blog.providers import BlogProviderClient, BlogStage
from app.services.blog.research import canonicalize_url
from app.services.blog.types import (
    Category,
    ContentImage,
    ContentVisualPlan,
    ProducedVisual,
    ResearchBundle,
    StoredImage,
    VisualDataset,
    VisualPlan,
    VisualProduction,
    VisualType,
)
from app.services.r2_storage import r2_storage
from app.services.blog.writer import (
    extract_headings,
    normalize_heading,
    word_count,
)

PIXABAY_BRANDS = {
    "adobe",
    "android",
    "apple",
    "canva",
    "chatgpt",
    "google",
    "iphone",
    "microsoft",
    "notion",
    "openai",
    "samsung",
    "windows",
}
PHOTO_QUERY_WORDS = (2, 5)
PARAGRAPH_CLOSE_RE = re.compile(r"</p\s*>", re.IGNORECASE)
IMAGE_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
FORBIDDEN_SECTIONS = {"faq", "sources", "frequently asked questions"}
MIN_VISUAL_COVERAGE_WORDS = 800
MAX_TEXT_GAP_WORDS = 450
SAME_TYPE_MIN_GAP_WORDS = 150
PHOTO_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "your",
    "you",
    "how",
    "what",
    "why",
    "when",
    "where",
    "from",
    "into",
    "that",
    "this",
    "are",
    "can",
    "does",
    "using",
    "use",
    "best",
    "top",
    "guide",
    "introduction",
    "conclusion",
    "final",
    "thoughts",
    "summary",
    "overview",
}


class VisualError(RuntimeError):
    pass


class VisualPlanError(VisualError):
    pass


class MediaRegistry(Protocol):
    def is_used(
        self,
        provider: str,
        *,
        external_id: str | None = None,
        url: str | None = None,
    ) -> bool: ...

    def add(
        self,
        provider: str,
        *,
        external_id: str | None = None,
        url: str | None = None,
    ) -> None: ...


class InMemoryMediaRegistry:
    def __init__(
        self,
        used_ids: Sequence[str] = (),
        used_urls: Sequence[str] = (),
    ) -> None:
        self._ids = set(used_ids)
        self._urls = {self._hash(url) for url in used_urls if url}

    def is_used(
        self,
        provider: str,
        *,
        external_id: str | None = None,
        url: str | None = None,
    ) -> bool:
        if external_id and f"{provider}:{external_id}" in self._ids:
            return True
        return bool(url and self._hash(url) in self._urls)

    def add(
        self,
        provider: str,
        *,
        external_id: str | None = None,
        url: str | None = None,
    ) -> None:
        if external_id:
            self._ids.add(f"{provider}:{external_id}")
        if url:
            self._urls.add(self._hash(url))

    @staticmethod
    def _hash(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PixabayAsset:
    external_id: str
    url: str
    width: int
    height: int
    page_url: str = ""


class PixabayClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://pixabay.com/api/",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self._client = client
        self.base_url = base_url
        self.timeout = timeout

    async def find_photo(
        self,
        query: str,
        *,
        registry: MediaRegistry | None = None,
    ) -> PixabayAsset | None:
        if not self.api_key:
            return None
        for variant in self._query_variants(query):
            payload = await self._get_json(
                {
                    "key": self.api_key,
                    "q": variant,
                    "image_type": "photo",
                    "orientation": "horizontal",
                    "safesearch": "true",
                    "per_page": 30,
                    "order": "popular",
                    "min_width": 1200,
                }
            )
            for hit in payload.get("hits", []) if payload else []:
                if not isinstance(hit, dict):
                    continue
                external_id = str(hit.get("id") or "")
                width = int(hit.get("imageWidth") or 0)
                height = int(hit.get("imageHeight") or 0)
                url = str(
                    hit.get("largeImageURL")
                    or hit.get("fullHDURL")
                    or hit.get("webformatURL")
                    or ""
                )
                if not external_id or not url or width < 1200 or height < 1:
                    continue
                if registry and registry.is_used(
                    "pixabay", external_id=external_id, url=url
                ):
                    continue
                return PixabayAsset(
                    external_id=external_id,
                    url=url,
                    width=width,
                    height=height,
                    page_url=str(hit.get("pageURL") or ""),
                )
        return None

    async def download(self, url: str) -> bytes:
        response = await self._get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if content_type and not content_type.startswith("image/"):
            raise VisualError("Pixabay returned a non-image response.")
        return response.content

    def _query_variants(self, query: str) -> list[str]:
        words = query.strip().split()
        variants = []
        for count in (len(words), 3, 2):
            if 2 <= count <= len(words):
                variant = " ".join(words[:count])
                if variant and variant not in variants:
                    variants.append(variant)
        return variants

    async def _get_json(self, params: dict[str, Any]) -> dict[str, Any]:
        response = await self._get(self.base_url, params=params)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    async def _get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        if self._client is not None:
            return await self._client.get(
                url, params=params, timeout=self.timeout
            )
        async with httpx.AsyncClient(
            timeout=self.timeout, follow_redirects=True
        ) as client:
            return await client.get(url, params=params)


class VisualService:
    def __init__(
        self,
        *,
        storage: Any | None = None,
        pixabay: PixabayClient | None = None,
        registry: MediaRegistry | None = None,
    ) -> None:
        self.storage = storage or r2_storage
        self.pixabay = pixabay or PixabayClient()
        self.registry = registry or InMemoryMediaRegistry()

    async def produce(
        self,
        plan: VisualPlan,
        content: str,
        research: ResearchBundle,
        *,
        slug: str,
        title: str,
        category: Category,
    ) -> VisualProduction:
        validate_visual_plan(
            plan,
            content,
            research,
            required_count=len(plan.visuals),
        )
        cover = await self._cover(plan, slug, title, category)
        insertions: list[tuple[int, str]] = []
        produced: list[ProducedVisual] = []
        content_images: list[ContentImage] = []
        headings = {normalize_heading(item.text): item for item in extract_headings(content)}

        for item in plan.visuals:
            heading = headings.get(normalize_heading(item.after_heading))
            if heading is None:
                raise VisualPlanError(
                    f"Heading not found: {item.after_heading}"
                )
            block = ""
            if item.type is VisualType.CHART:
                if item.data is None:
                    raise VisualPlanError("Chart visual has no data.")
                try:
                    block = render_chart(
                        item.data, alt=item.alt, caption=item.caption
                    )
                except ChartError as exc:
                    raise VisualPlanError(str(exc)) from exc
                produced.append(
                    ProducedVisual(
                        type=VisualType.CHART,
                        after_heading=item.after_heading,
                        after_paragraph=item.after_paragraph,
                        alt=item.alt,
                        caption=item.caption,
                        provider="generated",
                        source_url=item.data.source_url,
                    )
                )
            elif item.type is VisualType.TABLE:
                if item.data is None:
                    raise VisualPlanError("Table visual has no data.")
                block = render_table(item.data, caption=item.caption)
                produced.append(
                    ProducedVisual(
                        type=VisualType.TABLE,
                        after_heading=item.after_heading,
                        after_paragraph=item.after_paragraph,
                        alt=item.alt,
                        caption=item.caption,
                        provider="generated",
                        source_url=item.data.source_url,
                    )
                )
            else:
                image = await self._content_photo(
                    item, slug, category, title
                )
                block = _photo_figure(image, item.alt, item.caption)
                content_images.append(
                    ContentImage(
                        url=image.url,
                        alt=item.alt,
                        caption=item.caption,
                        width=image.width,
                        height=image.height,
                    )
                )
                produced.append(
                    ProducedVisual(
                        type=VisualType.PHOTO,
                        after_heading=item.after_heading,
                        after_paragraph=item.after_paragraph,
                        alt=item.alt,
                        caption=item.caption,
                        url=image.url,
                        provider=image.provider,
                        external_id=image.external_id,
                        width=image.width,
                        height=image.height,
                    )
                )
            offset = _insertion_offset(content, heading, item.after_paragraph)
            insertions.append((offset, block))

        updated = content
        for offset, block in sorted(insertions, key=lambda item: item[0], reverse=True):
            updated = f"{updated[:offset]}{block}{updated[offset:]}"

        production = VisualProduction(
            content=updated,
            cover_image_url=cover.url,
            cover_alt=plan.cover.alt,
            cover_caption=plan.cover.caption,
            cover_width=cover.width,
            cover_height=cover.height,
            content_images=content_images,
            visuals=produced,
        )
        errors = qa_visuals(production, plan)
        if errors:
            raise VisualError("; ".join(errors))
        return production

    async def _cover(
        self,
        plan: VisualPlan,
        slug: str,
        title: str,
        category: Category,
    ) -> StoredImage:
        try:
            asset = await self.pixabay.find_photo(
                plan.cover.query, registry=self.registry
            )
            if asset is not None:
                data = await self.pixabay.download(asset.url)
                url = self.storage.compress_and_upload(
                    data, slug, prefix="covers"
                )
                stored = StoredImage(
                    url=url,
                    width=1200,
                    height=630,
                    provider="pixabay",
                    external_id=asset.external_id,
                    kind="cover",
                )
                self.registry.add(
                    "pixabay",
                    external_id=asset.external_id,
                    url=asset.url,
                )
                return stored
        except Exception:
            pass
        data = generate_branded_cover(title, category.value)
        url = self.storage.compress_and_upload(data, slug, prefix="covers")
        stored = StoredImage(
            url=url,
            width=1200,
            height=630,
            provider="branded",
            external_id=None,
            kind="cover",
        )
        self.registry.add("branded", url=stored.url)
        return stored

    async def _content_photo(
        self,
        item: ContentVisualPlan,
        slug: str,
        category: Category,
        title: str,
    ) -> StoredImage:
        query = item.query or item.alt
        try:
            asset = await self.pixabay.find_photo(
                query, registry=self.registry
            )
            if asset is not None:
                data = await self.pixabay.download(asset.url)
                url = self.storage.compress_and_upload(
                    data, slug, prefix="content"
                )
                stored = StoredImage(
                    url=url,
                    width=1200,
                    height=630,
                    provider="pixabay",
                    external_id=asset.external_id,
                    kind="content",
                )
                self.registry.add(
                    "pixabay",
                    external_id=asset.external_id,
                    url=asset.url,
                )
                return stored
        except Exception:
            pass
        data = generate_branded_content(
            item.data.title if item.data and item.data.title else title,
            category.value,
        )
        url = self.storage.compress_and_upload(data, slug, prefix="content")
        stored = StoredImage(
            url=url,
            width=1200,
            height=630,
            provider="branded",
            external_id=None,
            kind="content",
        )
        self.registry.add("branded", url=stored.url)
        return stored


async def plan_visuals(
    client: BlogProviderClient,
    content: str,
    research: ResearchBundle,
    *,
    required_count: int,
) -> VisualPlan:
    headings = [item.text for item in extract_headings(content)]
    messages = visual_placement_messages(
        content,
        research,
        required_count=required_count,
        headings=headings,
    )
    result = await client.complete_json(
        BlogStage.VISUAL_PLACEMENT,
        messages,
        validator=lambda data: VisualPlan.model_validate(data),
    )
    validate_visual_plan(
        result.data, content, research, required_count=required_count
    )
    plan = _expand_visual_coverage(result.data, content, research)
    if len(plan.visuals) != required_count:
        validate_visual_plan(
            plan, content, research, required_count=len(plan.visuals)
        )
    return plan


def _expand_visual_coverage(
    plan: VisualPlan,
    content: str,
    research: ResearchBundle,
) -> VisualPlan:
    """Add photos where long stretches of the article have no visual."""
    total_words = word_count(content)
    if total_words < MIN_VISUAL_COVERAGE_WORDS:
        return plan
    headings = [
        heading
        for heading in extract_headings(content)
        if heading.level == 2
        and normalize_heading(heading.text) not in FORBIDDEN_SECTIONS
    ]
    if not headings:
        return plan
    points = _visual_word_points(plan, content, headings)
    points.extend(_table_word_points(content))
    points.sort()
    gaps = _visual_gaps(points, total_words)
    if not gaps:
        return plan
    candidate = plan.model_copy(deep=True)
    max_total_visuals = 4
    for gap_start, gap_end in reversed(gaps):
        if len(candidate.visuals) >= max_total_visuals:
            break
        section = _section_for_gap(headings, content, gap_start, gap_end)
        if section is None:
            continue
        key = normalize_heading(section.text)
        photo_sections = {
            normalize_heading(item.after_heading)
            for item in candidate.visuals
            if item.type is VisualType.PHOTO
        }
        if key in photo_sections:
            continue
        body = content[section.body_start : section.body_end]
        if _paragraph_count(body) < 1:
            continue
        addition = ContentVisualPlan(
            after_heading=section.text,
            after_paragraph=1,
            type=VisualType.PHOTO,
            purpose="Support a long text-only stretch of the article.",
            alt=f"{section.text} for students",
            caption=f"Reference image for the {section.text.lower()} section.",
            query=_fallback_photo_query(section.text, body),
        )
        trial = candidate.model_copy(deep=True)
        trial.visuals.append(addition)
        try:
            validate_visual_plan(
                trial,
                content,
                research,
                required_count=len(trial.visuals),
            )
        except VisualPlanError:
            continue
        candidate = trial
    return candidate


def _visual_word_points(
    plan: VisualPlan,
    content: str,
    headings: Sequence[Any],
) -> list[int]:
    by_text = {normalize_heading(heading.text): heading for heading in headings}
    points: list[int] = []
    for item in plan.visuals:
        heading = by_text.get(normalize_heading(item.after_heading))
        if heading is None:
            continue
        try:
            offset = _insertion_offset(content, heading, item.after_paragraph)
        except VisualPlanError:
            continue
        points.append(word_count(content[:offset]))
    return sorted(points)


def _table_word_points(content: str) -> list[int]:
    points: list[int] = []
    for match in re.finditer(r"<table\b", content, re.IGNORECASE):
        points.append(word_count(content[: match.start()]))
    return points


def _visual_gaps(points: Sequence[int], total_words: int) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for point in points:
        if point - cursor > MAX_TEXT_GAP_WORDS:
            gaps.append((cursor, point))
        cursor = max(cursor, point)
    if total_words - cursor > MAX_TEXT_GAP_WORDS:
        gaps.append((cursor, total_words))
    return gaps


def _section_for_gap(
    headings: Sequence[Any],
    content: str,
    gap_start: int,
    gap_end: int,
) -> Any | None:
    starts: list[tuple[int, Any]] = []
    overlaps: list[tuple[int, Any]] = []
    for heading in headings:
        start_words = word_count(content[: heading.body_start])
        end_words = word_count(content[: heading.body_end])
        if end_words <= gap_start:
            continue
        if start_words >= gap_end:
            continue
        overlaps.append((start_words, heading))
        if start_words >= gap_start:
            starts.append((start_words, heading))
    pool = starts or overlaps
    if not pool:
        return None
    return max(pool, key=lambda item: item[0])[1]


def _fallback_photo_query(heading_text: str, section_body: str) -> str:
    haystack = f"{heading_text} {section_body[:1200]}".lower()
    mapping = (
        (("code", "coding", "developer", "programming"), "coding screen"),
        (("laptop", "hardware", "device", "computer"), "laptop desk"),
        (("study", "studying", "student", "revision"), "student studying"),
        (("research", "paper", "academic", "thesis"), "books research"),
        (("finance", "money", "budget", "loan", "salary"), "finance planning"),
        (("productivity", "planner", "schedule", "time"), "calendar planning"),
        (("data", "analytic", "metric", "chart", "statistic"), "data dashboard"),
        (("design", "ui", "ux", "interface"), "design workspace"),
        (("team", "collaborat", "group"), "team collaboration"),
        (("ai", "automation", "workflow"), "automation workflow"),
    )
    for needles, query in mapping:
        if any(needle in haystack for needle in needles):
            return query
    return "student desk study"


def validate_visual_plan(
    plan: VisualPlan,
    content: str,
    research: ResearchBundle,
    *,
    required_count: int,
) -> None:
    if len(plan.visuals) != required_count:
        raise VisualPlanError(
            f"Expected {required_count} content visuals, got {len(plan.visuals)}."
        )
    plan.cover.query = _validate_photo_query(plan.cover.query)
    if not plan.cover.alt.strip() or not plan.cover.caption.strip():
        raise VisualPlanError("Cover needs alt text and a caption.")

    headings = extract_headings(content)
    by_text: dict[str, Any] = {}
    for heading in headings:
        key = normalize_heading(heading.text)
        if key in by_text:
            raise VisualPlanError(f"Headings must be unique: {heading.text}")
        by_text[key] = heading

    research_urls = {
        canonicalize_url(source.url) for source in research.sources
    }
    charts = [
        item for item in plan.visuals if item.type is VisualType.CHART
    ]
    for extra in charts[1:]:
        extra.type = VisualType.TABLE
    photo_counts: dict[str, int] = {}
    locations: list[tuple[int, VisualType]] = []

    for item in plan.visuals:
        heading_key = normalize_heading(item.after_heading)
        heading = by_text.get(heading_key)
        if heading is None:
            raise VisualPlanError(
                f"Visual heading not found: {item.after_heading}"
            )
        if normalize_heading(heading.text) in FORBIDDEN_SECTIONS:
            raise VisualPlanError("Visuals cannot be placed in FAQ or Sources.")
        if heading.level != 2:
            raise VisualPlanError(
                "Visuals must be placed under an H2 section."
            )
        if not item.alt.strip() or not item.caption.strip():
            raise VisualPlanError("Every visual needs alt text and a caption.")
        paragraphs = _paragraph_count(
            content[heading.body_start : heading.body_end]
        )
        if not 1 <= item.after_paragraph <= paragraphs:
            raise VisualPlanError(
                f"Invalid after_paragraph for {item.after_heading}."
            )
        if item.type is VisualType.PHOTO:
            item.query = _validate_photo_query(item.query or "")
            key = normalize_heading(item.after_heading)
            photo_counts[key] = photo_counts.get(key, 0) + 1
        elif item.type is VisualType.CHART:
            _validate_chart_data(item.data, research_urls)
        else:
            _validate_table_data(item.data, research_urls)
        offset = _insertion_offset(content, heading, item.after_paragraph)
        locations.append((offset, item.type))

    for key, count in photo_counts.items():
        if count > 1:
            raise VisualPlanError(
                f"Only one photo is allowed per H2: {key}"
            )
    ordered = sorted(locations)
    previous_kind: VisualType | None = None
    previous_words = 0
    for offset, kind in ordered:
        current_words = word_count(content[:offset])
        if (
            previous_kind is not None
            and previous_kind is kind
            and current_words - previous_words < SAME_TYPE_MIN_GAP_WORDS
        ):
            raise VisualPlanError(
                "Adjacent visuals cannot have the same type."
            )
        previous_kind = kind
        previous_words = current_words


def qa_visuals(
    production: VisualProduction,
    plan: VisualPlan,
) -> list[str]:
    errors: list[str] = []
    if len(production.visuals) != len(plan.visuals):
        errors.append("Produced visual count does not match the plan.")
    if not production.cover_image_url:
        errors.append("Cover image URL is missing.")
    if (production.cover_width, production.cover_height) != (1200, 630):
        errors.append("Cover image must be 1200x630.")
    if not production.cover_alt.strip() or not production.cover_caption.strip():
        errors.append("Cover alt text and caption are required.")
    urls = [production.cover_image_url]
    for image in production.content_images:
        if not image.url or not image.alt.strip() or not image.caption.strip():
            errors.append("Content images need URL, alt text, and caption.")
        urls.append(image.url)
    if len(urls) != len(set(urls)):
        errors.append("Duplicate media URLs detected.")
    if production.visuals:
        types = {visual.type for visual in production.visuals}
        if len(production.visuals) >= 2 and len(types) < 2:
            errors.append("Content visual types should vary.")
    for image in IMAGE_RE.findall(production.content):
        if not re.search(r'\balt\s*=', image, re.IGNORECASE):
            errors.append("An image is missing alt text.")
        if not re.search(r'\bwidth\s*=\s*"', image, re.IGNORECASE):
            errors.append("An image is missing a width attribute.")
        if not re.search(r'\bheight\s*=\s*"', image, re.IGNORECASE):
            errors.append("An image is missing a height attribute.")
    for visual in production.visuals:
        if visual.type is VisualType.CHART:
            if '<svg' not in production.content:
                errors.append("Chart SVG is missing.")
            if 'viewBox="0 0 800 450"' not in production.content:
                errors.append("Chart viewBox is invalid.")
            if "<title>" not in production.content or "<desc>" not in production.content:
                errors.append("Chart title or desc is missing.")
        if visual.type is VisualType.TABLE:
            if "<table" not in production.content:
                errors.append("Table is missing.")
            if "<caption>" not in production.content:
                errors.append("Table caption is missing.")
            if "scope=" not in production.content:
                errors.append("Table headers need scope.")
    return errors


def render_table(data: VisualDataset, *, caption: str) -> str:
    if not data.source_url:
        raise VisualPlanError("Table data needs a source URL.")
    source = data.source_label or data.source_url
    rows = data.rows
    headers = data.headers
    if not rows:
        if len(data.labels) != len(data.values) or len(data.labels) < 2:
            raise VisualPlanError("Table needs rows or matching labels/values.")
        headers = ["Item", data.title or "Value"]
        rows = [
            [label, f"{value:g}{data.unit}"]
            for label, value in zip(data.labels, data.values)
        ]
    if not headers or any(len(row) != len(headers) for row in rows):
        raise VisualPlanError("Table rows must match the header count.")
    header_html = "".join(
        f'<th scope="col">{html.escape(str(header))}</th>'
        for header in headers
    )
    body_html = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
        + "</tr>"
        for row in rows
    )
    caption_text = html.escape(caption)
    return (
        '<div class="table-wrap"><table>'
        f"<caption>{caption_text} Source: {html.escape(source)}</caption>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table></div>"
    )


def _photo_figure(image: StoredImage, alt: str, caption: str) -> str:
    return (
        "<figure>"
        f'<img src="{html.escape(image.url, quote=True)}" '
        f'alt="{html.escape(alt, quote=True)}" '
        f'width="{image.width}" height="{image.height}" loading="lazy">'
        f"<figcaption>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def _validate_photo_query(query: str) -> str:
    """Return a safe photo query (2-5 words, no brands). Never raises."""
    words: list[str] = []
    for token in (query or "").split():
        word = token.strip(".,!?\"'")
        if word and word.lower() not in PIXABAY_BRANDS:
            words.append(word)
    if len(words) > PHOTO_QUERY_WORDS[1]:
        words = words[:3]
    if len(words) < PHOTO_QUERY_WORDS[0]:
        return "study desk"
    return " ".join(words)


def _validate_chart_data(
    data: VisualDataset | None,
    research_urls: set[str | None],
) -> None:
    if data is None:
        raise VisualPlanError("Chart data is required.")
    if not data.title.strip() or not data.source_url.strip():
        raise VisualPlanError("Chart needs a title and source URL.")
    if not 2 <= len(data.labels) <= 8 or len(data.labels) != len(data.values):
        raise VisualPlanError("Chart needs 2-8 matching labels and values.")
    chart_type = (data.chart_type or "bar").strip().lower()
    if chart_type not in ("bar", "line"):
        chart_type = "bar"
    if _is_time_series(data.labels):
        chart_type = "line"
    data.chart_type = chart_type
    _validate_source(data.source_url, research_urls)


def _is_time_series(labels: Sequence[str]) -> bool:
    cleaned = [str(label).strip().lower() for label in labels]
    if not cleaned:
        return False
    year_like = sum(
        1 for label in cleaned if re.fullmatch(r"(19|20)\d{2}", label)
    )
    quarter_like = sum(
        1
        for label in cleaned
        if re.fullmatch(r"q[1-4](\s+)?(19|20)?\d{2}", label)
        or re.fullmatch(r"q[1-4]", label)
    )
    month_like = sum(
        1
        for label in cleaned
        if label[:3]
        in {
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        }
    )
    return (year_like + quarter_like + month_like) >= max(2, len(cleaned) - 1)


def _validate_table_data(
    data: VisualDataset | None,
    research_urls: set[str | None],
) -> None:
    if data is None:
        raise VisualPlanError("Table data is required.")
    if not data.source_url.strip():
        raise VisualPlanError("Table needs a source URL.")
    if data.rows:
        if not data.headers or any(
            len(row) != len(data.headers) for row in data.rows
        ):
            raise VisualPlanError("Table rows must match the header count.")
    elif (
        len(data.labels) < 2
        or len(data.labels) != len(data.values)
    ):
        raise VisualPlanError("Table needs rows or matching labels and values.")
    _validate_source(data.source_url, research_urls)


def _validate_source(source_url: str, research_urls: set[str | None]) -> None:
    if canonicalize_url(source_url) not in research_urls:
        raise VisualPlanError("Visual data must cite a research source.")


def _paragraph_count(body: str) -> int:
    return len(PARAGRAPH_CLOSE_RE.findall(body))


def _insertion_offset(content: str, heading: Any, after_paragraph: int) -> int:
    body = content[heading.body_start : heading.body_end]
    matches = list(PARAGRAPH_CLOSE_RE.finditer(body))
    if after_paragraph > len(matches):
        raise VisualPlanError(
            f"Paragraph {after_paragraph} does not exist in {heading.text}."
        )
    return heading.body_start + matches[after_paragraph - 1].end()

