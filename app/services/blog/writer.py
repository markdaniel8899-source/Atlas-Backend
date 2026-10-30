from __future__ import annotations

import html as html_lib
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from .prompts import writer_messages, writer_repair_messages
from .providers import BlogProviderClient, BlogStage, ProviderTextResult
from .types import Category, GateReport, KeywordPlan, ResearchBundle, TopicCandidate, WriterOutput


class WriterError(RuntimeError):
    pass


class WriterParseError(WriterError):
    pass


@dataclass(frozen=True, slots=True)
class WriterResult:
    output: WriterOutput
    provider: str
    model: str
    attempts: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class Heading:
    level: int
    text: str
    start: int
    end: int
    body_start: int
    body_end: int


WRITER_TAGS = {
    "h2",
    "h3",
    "p",
    "ul",
    "ol",
    "li",
    "strong",
    "em",
    "a",
    "blockquote",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "caption",
    "div",
}

VISUAL_TAGS = WRITER_TAGS | {
    "figure",
    "figcaption",
    "img",
    "svg",
    "g",
    "title",
    "desc",
    "rect",
    "line",
    "polyline",
    "text",
    "tspan",
    "path",
    "circle",
}

VOID_TAGS = {"img"}
GLOBAL_FORBIDDEN_ATTRS = {"style", "id"}

ATLAS_INTERNAL_LINKS: tuple[tuple[str, str], ...] = (
    ("ATLAS study notes", "/app/notes"),
    ("ATLAS courses", "/app/courses"),
    ("ATLAS roadmap", "/app/roadmap"),
    ("ATLAS quiz practice", "/app/quiz"),
    ("ATLAS AI study assistant", "/app/assistant"),
    ("ATLAS focus timer", "/app/timer"),
    ("ATLAS blog", "/blog"),
)

_HEADING_RE = re.compile(r"<h([23])\b[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_ANCHOR_RE = re.compile(r"<a\b[^>]*\bhref\s*=\s*([\"'])(.*?)\1", re.IGNORECASE | re.DOTALL)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def slugify(value: str, *, max_length: int = 80) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:max_length].rstrip("-") or "article"


def _clean_text(value: str) -> str:
    text = html_lib.unescape(value or "")
    text = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_writer_output(raw: str) -> WriterOutput:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:html|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    # Auto-fix missing scope attribute on <th> tags
    text = re.sub(
        r"<th(?![^>]*\bscope=)(?![^>]*\s/\s*>)(>|[^>]*>)",
        r'<th scope="col"',
        text,
        flags=re.IGNORECASE,
    )
    title = _extract_tag(text, "TITLE")
    slug = _extract_tag(text, "SLUG")
    excerpt = _extract_tag(text, "EXCERPT")
    content = _extract_content(text)
    if not title or not content:
        raise WriterParseError("Writer output is missing TITLE or CONTENT.")

    title = _clean_text(title)
    if not title:
        raise WriterParseError("Writer output has an empty title.")
    clean_excerpt = _clean_text(excerpt)
    if len(clean_excerpt) > 200:
        clean_excerpt = _trim_words(clean_excerpt, 200)
    if len(clean_excerpt) < 100:
        fallback = content_excerpt(content)
        if len(fallback) >= 100:
            clean_excerpt = fallback

    content = content.strip()
    if not content:
        raise WriterParseError("Writer output has an empty body.")
    html_errors = validate_html(content, allow_visuals=False)
    if html_errors:
        raise WriterParseError("; ".join(html_errors))
    return WriterOutput(
        title=title,
        slug=slugify(slug or title),
        excerpt=clean_excerpt,
        content=content,
    )


def _extract_tag(text: str, name: str) -> str:
    match = re.search(
        rf"<{name}>(.*?)</{name}>", text, flags=re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else ""


def _extract_content(text: str) -> str:
    match = re.search(
        r"<CONTENT>(.*)</CONTENT>", text, flags=re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else ""


def _trim_words(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[: limit + 1]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,.;:-") + "."


def content_excerpt(content: str, target: int = 155) -> str:
    text = plain_text(content)
    sentences = [item.strip() for item in _SENTENCE_RE.findall(text) if item.strip()]
    result = ""
    for sentence in sentences:
        candidate = f"{result} {sentence}".strip()
        if len(candidate) >= target:
            break
        result = candidate
    if not result:
        result = text
    result = re.sub(r"\s+", " ", result).strip()
    if len(result) < 100:
        result = f"{result} Learn the practical steps and apply them with ATLAS."
    return _trim_words(result, 200)


def plain_text(content: str) -> str:
    parser = _TextParser()
    parser.feed(content or "")
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def word_count(content: str) -> int:
    return len(_WORD_RE.findall(plain_text(content)))


def first_words(content: str, limit: int) -> str:
    words = plain_text(content).split()
    return " ".join(words[:limit])


def extract_headings(content: str) -> list[Heading]:
    matches = list(_HEADING_RE.finditer(content or ""))
    headings: list[Heading] = []
    for index, match in enumerate(matches):
        level = int(match.group(1))
        body_start = match.end()
        body_end = len(content)
        for later in matches[index + 1 :]:
            if later.group(1) == "2" or level == 3:
                body_end = later.start()
                break
        headings.append(
            Heading(
                level=level,
                text=_clean_text(match.group(2)),
                start=match.start(),
                end=match.end(),
                body_start=body_start,
                body_end=body_end,
            )
        )
    return headings


def extract_h2s(content: str) -> list[Heading]:
    return [heading for heading in extract_headings(content) if heading.level == 2]


def normalize_heading(text: str) -> str:
    value = _clean_text(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def extract_links(content: str) -> list[str]:
    return [html_lib.unescape(match.group(2)).strip() for match in _ANCHOR_RE.finditer(content or "")]


def is_internal_link(href: str) -> bool:
    value = (href or "").strip()
    if value.startswith("/") and not value.startswith("//"):
        return True
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and parsed.netloc.endswith("atlas.app")


def validate_html(content: str, *, allow_visuals: bool = False) -> list[str]:
    parser = _HtmlValidator(allow_visuals=allow_visuals)
    parser.feed(content or "")
    parser.close()
    return parser.errors


def sanitize_html(content: str, *, allow_visuals: bool = True) -> str:
    parser = _HtmlSanitizer(allow_visuals=allow_visuals)
    parser.feed(content or "")
    parser.close()
    return "".join(parser.parts).strip()


def internal_links_for(category: Category, topic: str = "") -> list[tuple[str, str]]:
    order = {
        Category.AI: ("assistant", "notes", "quiz", "roadmap"),
        Category.SOFTWARE: ("notes", "courses", "quiz", "assistant"),
        Category.HARDWARE: ("courses", "notes", "roadmap", "blog"),
        Category.AUTOMATION: ("roadmap", "notes", "timer", "assistant"),
        Category.FINANCE: ("courses", "notes", "blog", "roadmap"),
        Category.OTHER: ("notes", "courses", "blog", "assistant"),
    }.get(category, ("notes", "courses", "blog"))
    selected: list[tuple[str, str]] = []
    for key in order:
        for label, href in ATLAS_INTERNAL_LINKS:
            if key in href:
                selected.append((label, href))
                break
    return selected


def is_finance(category: Category) -> bool:
    return category is Category.FINANCE


async def write_article(
    client: BlogProviderClient,
    topic: TopicCandidate,
    keywords: KeywordPlan,
    research: ResearchBundle,
    *,
    min_words: int = 900,
    max_words: int = 2200,
    finance: bool | None = None,
) -> WriterResult:
    messages = writer_messages(
        topic,
        keywords,
        research,
        min_words=min_words,
        max_words=max_words,
        finance=is_finance(topic.category) if finance is None else finance,
        internal_links=internal_links_for(topic.category, topic.topic),
    )
    result = await client.complete_text(BlogStage.WRITER, messages)
    return _writer_result(result)


async def repair_article(
    client: BlogProviderClient,
    topic: TopicCandidate,
    keywords: KeywordPlan,
    research: ResearchBundle,
    draft: WriterOutput,
    report: GateReport,
    *,
    min_words: int = 900,
    max_words: int = 2200,
    finance: bool | None = None,
) -> WriterResult:
    messages = writer_repair_messages(
        topic,
        keywords,
        research,
        draft,
        report,
        min_words=min_words,
        max_words=max_words,
        finance=is_finance(topic.category) if finance is None else finance,
        internal_links=internal_links_for(topic.category, topic.topic),
    )
    result = await client.complete_text(BlogStage.WRITER, messages)
    return _writer_result(result)


def _writer_result(result: ProviderTextResult) -> WriterResult:
    return WriterResult(
        output=parse_writer_output(result.text),
        provider=result.provider,
        model=result.model,
        attempts=result.attempts,
    )


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style") and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def _allowed_attrs(tag: str) -> set[str]:
    if tag == "a":
        return {"href", "rel", "target", "title"}
    if tag == "img":
        return {"src", "alt", "width", "height", "loading"}
    if tag == "div":
        return {"class"}
    if tag == "th":
        return {"scope"}
    if tag == "svg":
        return {"viewbox", "viewBox", "role", "aria-label", "xmlns"}
    if tag in ("rect",):
        return {"x", "y", "width", "height", "rx", "ry", "fill", "opacity"}
    if tag in ("line",):
        return {"x1", "y1", "x2", "y2", "stroke", "stroke-width", "opacity"}
    if tag in ("polyline", "path"):
        return {"points", "d", "fill", "stroke", "stroke-width", "stroke-linejoin", "stroke-linecap"}
    if tag in ("text", "tspan"):
        return {
            "x",
            "y",
            "dx",
            "dy",
            "fill",
            "font-size",
            "font-family",
            "font-weight",
            "text-anchor",
            "dominant-baseline",
        }
    if tag == "circle":
        return {"cx", "cy", "r", "fill", "opacity"}
    return set()


class _HtmlValidator(HTMLParser):
    def __init__(self, *, allow_visuals: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.allow_visuals = allow_visuals
        self.allowed = VISUAL_TAGS if allow_visuals else WRITER_TAGS
        self.errors: list[str] = []
        self.stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        values = {name.lower(): (value or "") for name, value in attrs}
        if lower == "h1":
            self.errors.append("H1 is not allowed.")
        if lower not in self.allowed:
            self.errors.append(f"Tag <{lower}> is not allowed.")
        if any(name.startswith("on") or name in GLOBAL_FORBIDDEN_ATTRS for name in values):
            self.errors.append(f"Forbidden attribute on <{lower}>.")
        allowed_attrs = _allowed_attrs(lower)
        if lower == "svg":
            allowed_attrs = {name.lower() for name in allowed_attrs}
        for name in values:
            if name not in allowed_attrs:
                self.errors.append(f"Attribute {name} is not allowed on <{lower}>.")
        if lower == "div" and values.get("class") != "table-wrap":
            self.errors.append("Only div.table-wrap is allowed.")
        if lower == "a":
            href = values.get("href", "").strip()
            if not href:
                self.errors.append("Links must have a non-empty href.")
            elif href.lower().startswith("javascript:"):
                self.errors.append("javascript: links are forbidden.")
        if lower == "img":
            if not self.allow_visuals:
                self.errors.append("Images are not allowed in writer output.")
            else:
                if not values.get("src"):
                    self.errors.append("Images need a src.")
                if not values.get("alt"):
                    self.errors.append("Images need alt text.")
                for dimension in ("width", "height"):
                    if not values.get(dimension, "").isdigit():
                        self.errors.append(f"Images need numeric {dimension}.")
        if lower == "th" and values.get("scope") not in ("col", "row"):
            self.errors.append("Table headers need scope=col or row.")
        if lower not in VOID_TAGS:
            self.stack.append(lower)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        lower = tag.lower()
        if lower not in VOID_TAGS and self.stack and self.stack[-1] == lower:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in VOID_TAGS:
            return
        if lower in self.stack:
            while self.stack:
                opened = self.stack.pop()
                if opened == lower:
                    break
        else:
            self.errors.append(f"Unexpected closing tag </{lower}>.")

    def close(self) -> None:
        super().close()
        if self.stack:
            self.errors.append(f"Unclosed tags: {', '.join(self.stack)}.")


class _HtmlSanitizer(HTMLParser):
    def __init__(self, *, allow_visuals: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.allowed = VISUAL_TAGS if allow_visuals else WRITER_TAGS
        self.parts: list[str] = []
        self.stack: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower in ("script", "style"):
            self._skip_depth += 1
            return
        if self._skip_depth or lower not in self.allowed:
            return
        values = {name.lower(): (value or "") for name, value in attrs}
        if lower == "div" and values.get("class") != "table-wrap":
            return
        rendered = []
        allowed_attrs = _allowed_attrs(lower)
        if lower == "svg":
            allowed_attrs = {name.lower() for name in allowed_attrs}
        for name, value in values.items():
            if name.startswith("on") or name in GLOBAL_FORBIDDEN_ATTRS:
                continue
            if name not in allowed_attrs:
                continue
            if lower == "a" and name == "href" and value.strip().lower().startswith("javascript:"):
                continue
            output_name = "viewBox" if lower == "svg" and name == "viewbox" else name
            rendered.append(
                f'{output_name}="{html_lib.escape(value, quote=True)}"'
            )
        if lower == "a":
            href = values.get("href", "").strip()
            if href and not any(item.startswith("rel=") for item in rendered):
                rendered.append('rel="noopener noreferrer"')
            if href and not any(item.startswith("target=") for item in rendered):
                rendered.append('target="_blank"')
        self.parts.append(f"<{lower}{(' ' + ' '.join(rendered)) if rendered else ''}>")
        if lower not in VOID_TAGS:
            self.stack.append(lower)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        before = len(self.parts)
        self.handle_starttag(tag, attrs)
        lower = tag.lower()
        if (
            len(self.parts) > before
            and lower not in VOID_TAGS
            and self.stack
            and self.stack[-1] == lower
        ):
            self.stack.pop()
            self.parts[-1] = self.parts[-1][:-1] + "/>"

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in ("script", "style"):
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth or lower in VOID_TAGS:
            return
        if lower in self.stack:
            while self.stack:
                opened = self.stack.pop()
                if opened == lower:
                    self.parts.append(f"</{lower}>")
                    break

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(html_lib.escape(data, quote=False))
