from __future__ import annotations

import math
import re
from typing import Callable, Sequence
from urllib.parse import urlparse

from .research import BLACKLISTED_HOSTS, canonicalize_url
from .topics import find_duplicate
from .types import (
    Category,
    GateCheck,
    GateFailure,
    GateMetrics,
    GateReport,
    GateWarning,
    KeywordPlan,
    ResearchBundle,
    TopicCandidate,
    TopicHistoryEntry,
    VisualPlan,
    VisualProduction,
    VisualType,
    WriterOutput,
)
from .visuals import qa_visuals
from .writer import (
    extract_h2s,
    extract_links,
    first_words,
    is_internal_link,
    plain_text,
    validate_html,
    word_count,
)

BANNED_PHRASES = (
    "delve",
    "landscape",
    "realm",
    "tapestry",
    "unlock",
    "elevate",
    "seamless",
    "in today's fast-paced world",
    "it's important to note",
)
AI_STYLE_RE = re.compile(
    r"\b(as an ai|when i|in my experience|i tried|i tested|"
    r"i have personally|i learned firsthand)\b",
    re.IGNORECASE,
)
CLICKBAIT_RE = re.compile(
    r"\b(you won't believe|shocking|unbelievable|this one trick|"
    r"mind-blowing|insane|must-see|secret hack|guaranteed)\b",
    re.IGNORECASE,
)
FINANCE_BAD_RE = re.compile(
    r"\b(guaranteed returns?|risk[- ]free|get rich|you should invest|"
    r"buy this stock|double your money)\b",
    re.IGNORECASE,
)
NUMERIC_CLAIM_RE = re.compile(
    r"(?<![\w.])(?:[$€£]\s?\d|\d[\d,]*(?:\.\d+)?\s?%|"
    r"\d[\d,]*(?:\.\d+)?\s?(?:million|billion|thousand|hours?|minutes?|"
    r"days?|weeks?|months?|years?|GB|TB|MB|ms))\b",
    re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(
    r"\b(todo|lorem ipsum|example\.com|placeholder|insert here|"
    r"your keyword|\[insert)\b",
    re.IGNORECASE,
)
VISUAL_BLOCK_RE = re.compile(
    r"<figure\b[^>]*>.*?</figure>", re.IGNORECASE | re.DOTALL
)
TABLE_BLOCK_RE = re.compile(
    r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL
)
INTERNAL_LINK_TARGETS = 1
INTERNAL_LINK_MAX = 3


def run_gate(
    output: WriterOutput,
    keywords: KeywordPlan,
    research: ResearchBundle,
    visuals: VisualProduction | None,
    *,
    topic: TopicCandidate | None = None,
    history: Sequence[TopicHistoryEntry] = (),
    visual_plan: VisualPlan | None = None,
    min_score: int = 90,
    visual_word_threshold: int = 1200,
    finance: bool | None = None,
    citation_checker: Callable[[str], bool] | None = None,
    collision_checker: Callable[[str, str], bool] | None = None,
) -> GateReport:
    editorial = _editorial_content(output.content)
    text = plain_text(editorial)
    words = word_count(editorial)
    h2s = extract_h2s(output.content)
    editorial_h2s = extract_h2s(editorial)
    links = extract_links(output.content)
    external_links = [
        href for href in links if _is_external(href) and not is_internal_link(href)
    ]
    unique_sources = {
        canonicalize_url(href)
        for href in external_links
        if canonicalize_url(href)
    }
    citation_count = len(external_links)
    internal_links = [href for href in links if is_internal_link(href)]
    density = _keyword_density(text, keywords.primary, words)
    flesch = _flesch_score(text)
    required_visuals = 2 if words <= visual_word_threshold else 3
    visual_failures = _visual_failures(
        visuals, visual_plan, required_visuals
    )
    if finance is not None:
        is_finance_article = finance
    else:
        is_finance_article = bool(
            topic and topic.category is Category.FINANCE
        )

    checks: list[GateCheck] = []
    failures: list[GateFailure] = []
    warnings: list[GateWarning] = []

    def check(
        check_id: str,
        category: str,
        passed: bool,
        detail: str = "",
    ) -> None:
        checks.append(
            GateCheck(
                id=check_id,
                category=category,
                passed=passed,
                detail=detail,
            )
        )
        if not passed and detail:
            failures.append(GateFailure(id=check_id, detail=detail))

    check(
        "H01",
        "structure",
        900 <= words <= 2200,
        f"Word count is {words}; required 900-2,200.",
    )
    clickbait = CLICKBAIT_RE.search(output.title)
    check(
        "H02",
        "seo",
        30 <= len(output.title) <= 70 and not clickbait,
        f"Title is {len(output.title)} chars or contains clickbait.",
    )
    slug_valid = bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", output.slug))
    collision = bool(
        collision_checker and collision_checker(output.title, output.slug)
    )
    check(
        "H03",
        "seo",
        slug_valid and not collision,
        "Slug is invalid or already used.",
    )
    check(
        "H04",
        "seo",
        100 <= len(output.excerpt) <= 200,
        f"Excerpt is {len(output.excerpt)} chars; required 100-200.",
    )

    skipped_h3 = _has_skipped_h3(output.content)
    empty_sections = [
        heading.text
        for heading in editorial_h2s
        if not plain_text(
            editorial[heading.body_start : heading.body_end]
        )
    ]
    check(
        "H05",
        "structure",
        4 <= len(h2s) <= 8 and not empty_sections and not skipped_h3,
        "Need 4-8 H2s, non-empty sections, and no skipped heading levels.",
    )
    title_has = _contains(output.title, keywords.primary)
    intro_has = _contains(first_words(editorial, 100), keywords.primary)
    h2_hits = sum(
        1 for heading in h2s if _contains(heading.text, keywords.primary)
    )
    check(
        "H06",
        "seo",
        title_has and intro_has and h2_hits >= 2,
        "Primary keyword needs title, first 100 words, and 2 H2s.",
    )
    check(
        "H07",
        "seo",
        0.5 <= density <= 2.5,
        f"Keyword density is {density:.2f}%; required 0.5-2.5%.",
    )
    secondary_hits = sum(
        1 for keyword in keywords.secondary if _contains(text, keyword)
    )
    secondary_ratio = (
        secondary_hits / len(keywords.secondary) if keywords.secondary else 0
    )
    long_tail_hits = sum(
        1 for keyword in keywords.long_tail if _contains(text, keyword)
    )
    lsi_hits = sum(1 for keyword in keywords.lsi if _contains(text, keyword))
    check(
        "H08",
        "seo",
        secondary_ratio >= 0.6 and long_tail_hits >= 2 and lsi_hits >= 5,
        "Secondary coverage, long-tail count, or LSI coverage is too low.",
    )

    minimum_citations = math.ceil(words / 600)
    check(
        "H09",
        "research",
        len(unique_sources) >= 3 and citation_count >= minimum_citations,
        f"Need 3 unique sources and at least {minimum_citations} citations.",
    )
    malformed = [
        href
        for href in external_links
        if not _valid_external_url(href)
        or (citation_checker is not None and not citation_checker(href))
    ]
    check(
        "H10",
        "research",
        not malformed,
        f"Malformed or dead citation URL(s): {', '.join(malformed[:3])}",
    )
    numeric_blocks = _unlinked_numeric_blocks(output.content)
    visual_trace_errors = _visual_trace_errors(
        visual_plan, visuals, research
    )
    numeric_details = list(visual_trace_errors)
    if numeric_blocks:
        numeric_details.insert(0, "Numeric claims need inline citations.")
    check(
        "H11",
        "research",
        not numeric_blocks and not visual_trace_errors,
        "; ".join(numeric_details) or "Numeric claims need inline citations.",
    )
    answer = _direct_answer_paragraph(editorial)
    check(
        "H12",
        "structure",
        answer is not None,
        "Need a 40-70 word direct answer in the first 150 words.",
    )
    check(
        "H13",
        "readability",
        45 <= flesch <= 70,
        f"Flesch score is {flesch:.1f}; required 45-70.",
    )
    found_banned = _found_phrases(output.content, BANNED_PHRASES)
    check(
        "H14",
        "readability",
        not found_banned,
        f"Banned phrase(s): {', '.join(found_banned)}",
    )
    ai_style = bool(AI_STYLE_RE.search(text)) or "—" in output.content
    check(
        "H15",
        "readability",
        not ai_style,
        "Fake experience, em dash, or AI-style language found.",
    )
    html_errors = validate_html(output.content, allow_visuals=True)
    check(
        "H16",
        "compliance",
        not html_errors,
        "; ".join(html_errors[:3]) or "Unsafe HTML.",
    )
    check(
        "H17",
        "visuals",
        not visual_failures,
        "; ".join(visual_failures) or "Visual QA failed.",
    )
    finance_errors = _finance_errors(output.content, is_finance_article)
    check(
        "H18",
        "compliance",
        not finance_errors,
        "; ".join(finance_errors) or "Finance compliance failed.",
    )
    duplicate = (
        find_duplicate(topic, history)
        if topic is not None
        else None
    )
    check(
        "H19",
        "product",
        duplicate is None,
        duplicate or "Topic or keyword repeats inside the dedupe window.",
    )
    domains = {
        (urlparse(href).hostname or "").lower().removeprefix("www.")
        for href in external_links
    }
    blacklisted = [domain for domain in domains if _blacklisted(domain)]
    check(
        "H20",
        "research",
        len({domain for domain in domains if domain}) >= 2
        and not blacklisted,
        f"Sources need 2+ domains and no blacklisted domains: {blacklisted}",
    )
    section_errors = _section_length_errors(editorial)
    check(
        "H21",
        "structure",
        not section_errors,
        "; ".join(section_errors) or "A section is outside 80-500 words.",
    )
    placeholders = PLACEHOLDER_RE.findall(output.content)
    check(
        "H22",
        "compliance",
        not placeholders,
        "Placeholder text remains in the article.",
    )
    check(
        "H23",
        "compliance",
        not collision,
        "Title or slug collides with an existing post.",
    )

    if len(internal_links) < INTERNAL_LINK_TARGETS:
        warnings.append(
            GateWarning(
                id="S01",
                detail="Add at least one internal ATLAS link.",
            )
        )
    if len(internal_links) > INTERNAL_LINK_MAX:
        warnings.append(
            GateWarning(
                id="S02",
                detail="Use no more than three internal ATLAS links.",
            )
        )
    if visuals is None:
        warnings.append(
            GateWarning(id="S03", detail="Visual production was not available.")
        )

    metrics = GateMetrics(
        word_count=words,
        h2_count=len(h2s),
        citation_count=citation_count,
        unique_sources=len(unique_sources),
        flesch=round(flesch, 1),
        keyword_density=round(density, 2),
        internal_links=len(internal_links),
        visual_count=(1 + len(visuals.visuals)) if visuals else 0,
    )
    score = _score(checks, internal_links)
    passed = not failures and score >= min_score
    return GateReport(
        passed=passed,
        score=score,
        hard_failures=failures,
        warnings=warnings,
        metrics=metrics,
        checks=checks,
    )


def _score(checks: Sequence[GateCheck], internal_links: Sequence[str]) -> int:
    groups = {
        "seo": (25, {"H02", "H03", "H04", "H06", "H07", "H08"}),
        "structure": (20, {"H01", "H05", "H12", "H21"}),
        "research": (20, {"H09", "H10", "H11", "H20"}),
        "readability": (15, {"H13", "H14", "H15"}),
        "visuals": (10, {"H17"}),
        "product": (5, {"H19"}),
        "compliance": (5, {"H16", "H18", "H22", "H23"}),
    }
    score = 0.0
    for weight, ids in groups.values():
        selected = [check for check in checks if check.id in ids]
        passed = sum(1 for check in selected if check.passed)
        score += weight * (passed / len(selected) if selected else 1)
    if not INTERNAL_LINK_TARGETS <= len(internal_links) <= INTERNAL_LINK_MAX:
        score -= 2
    return max(0, min(100, round(score)))


def _visual_failures(
    production: VisualProduction | None,
    plan: VisualPlan | None,
    required_count: int,
) -> list[str]:
    if production is None:
        return ["Visual production is missing."]
    errors = []
    if len(production.visuals) != required_count:
        errors.append(
            f"Expected {required_count} content visuals, got {len(production.visuals)}."
        )
    if (production.cover_width, production.cover_height) != (1200, 630):
        errors.append("Cover must be 1200x630.")
    if not production.cover_image_url:
        errors.append("Cover URL is missing.")
    if not production.cover_alt.strip() or not production.cover_caption.strip():
        errors.append("Cover alt and caption are required.")
    if plan is not None:
        errors.extend(qa_visuals(production, plan))
    forbidden = {"faq", "sources", "frequently asked questions"}
    photo_counts: dict[str, int] = {}
    for visual in production.visuals:
        heading = _normalize(visual.after_heading)
        if heading in forbidden:
            errors.append("A visual was placed in FAQ or Sources.")
        if visual.type is VisualType.PHOTO:
            photo_counts[heading] = photo_counts.get(heading, 0) + 1
    if any(count > 1 for count in photo_counts.values()):
        errors.append("Only one photo is allowed per H2.")
    if plan is None and len(production.visuals) >= 2 and len(
        {visual.type for visual in production.visuals}
    ) < 2:
        errors.append("Content visual types should vary.")
    image_urls = [production.cover_image_url] + [
        image.url for image in production.content_images
    ]
    if len(image_urls) != len(set(image_urls)):
        errors.append("Duplicate media URLs detected.")
    for image in production.content_images:
        if not image.width or not image.height:
            errors.append("Content images need width and height.")
        if not image.alt.strip() or not image.caption.strip():
            errors.append("Content images need alt text and captions.")
    if production.visuals and not any(
        visual.type in (VisualType.CHART, VisualType.TABLE)
        for visual in production.visuals
    ) and not any(
        visual.type is VisualType.PHOTO for visual in production.visuals
    ):
        errors.append("No usable visual type was produced.")
    return errors


def _keyword_density(text: str, phrase: str, words: int) -> float:
    if words <= 0 or not phrase.strip():
        return 0.0
    occurrences = _count_phrase(text, phrase)
    return occurrences / words * 100


def _count_phrase(text: str, phrase: str) -> int:
    normalized_text = f" {_normalize(text)} "
    normalized_phrase = f" {_normalize(phrase)} "
    if not normalized_phrase.strip():
        return 0
    return normalized_text.count(normalized_phrase)


def _contains(text: str, phrase: str) -> bool:
    return _count_phrase(text, phrase) > 0


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _found_phrases(text: str, phrases: Sequence[str]) -> list[str]:
    lowered = (text or "").lower().replace("’", "'")
    return [phrase for phrase in phrases if phrase in lowered]


def _has_skipped_h3(content: str) -> bool:
    headings = re.findall(r"<h([23])\b", content or "", flags=re.IGNORECASE)
    seen_h2 = False
    for level in headings:
        if level == "2":
            seen_h2 = True
        elif not seen_h2:
            return True
    return False


def _direct_answer_paragraph(content: str) -> str | None:
    paragraphs = re.findall(
        r"<p\b[^>]*>(.*?)</p>", content or "", flags=re.IGNORECASE | re.DOTALL
    )
    consumed = 0
    for paragraph in paragraphs:
        text = plain_text(paragraph)
        count = len(text.split())
        if consumed <= 150 and 40 <= count <= 70 and text:
            return text
        consumed += count
        if consumed > 150:
            break
    return None


def _editorial_content(content: str) -> str:
    value = VISUAL_BLOCK_RE.sub(" ", content or "")
    return TABLE_BLOCK_RE.sub(" ", value)


def _visual_trace_errors(
    plan: VisualPlan | None,
    production: VisualProduction | None,
    research: ResearchBundle,
) -> list[str]:
    research_urls = {
        canonicalize_url(source.url) for source in research.sources
    }
    errors: list[str] = []
    if plan is not None:
        for item in plan.visuals:
            if item.type not in (VisualType.CHART, VisualType.TABLE):
                continue
            if item.data is None:
                errors.append("Chart or table data is missing.")
                continue
            if canonicalize_url(item.data.source_url) not in research_urls:
                errors.append("Chart or table data is not traceable to research.")
    if production is not None:
        for visual in production.visuals:
            if visual.type not in (VisualType.CHART, VisualType.TABLE):
                continue
            if (
                visual.source_url
                and canonicalize_url(visual.source_url) not in research_urls
            ):
                errors.append("Produced chart or table has an untraceable source.")
    return sorted(set(errors))


def _unlinked_numeric_blocks(content: str) -> list[str]:
    return [
        match.group(0)
        for match in re.finditer(
            r"<(p|li|blockquote)\b[^>]*>.*?</\1>",
            content or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        if NUMERIC_CLAIM_RE.search(match.group(0))
        and not re.search(r"<a\b[^>]*\bhref\s*=", match.group(0), re.IGNORECASE)
    ]


def _finance_errors(content: str, is_finance_article: bool) -> list[str]:
    if not is_finance_article:
        return []
    lowered = (content or "").lower()
    errors = []
    if "not financial advice" not in lowered and "educational" not in lowered:
        errors.append("Finance disclaimer is missing.")
    if "risk" not in lowered:
        errors.append("Finance risk disclosure is missing.")
    if FINANCE_BAD_RE.search(content or ""):
        errors.append("Finance article contains advice or a guarantee.")
    return errors


def _section_length_errors(content: str) -> list[str]:
    errors = []
    for heading in extract_h2s(content):
        if _normalize(heading.text) == "sources":
            continue
        count = word_count(content[heading.body_start : heading.body_end])
        if not 80 <= count <= 500:
            errors.append(f"{heading.text}: {count} words")
    return errors


def _valid_external_url(href: str) -> bool:
    parsed = urlparse(href)
    return (
        parsed.scheme in ("http", "https")
        and bool(parsed.netloc)
        and "." in parsed.netloc
        and " " not in href
    )


def _is_external(href: str) -> bool:
    return _valid_external_url(href)


def _blacklisted(domain: str) -> bool:
    return any(
        domain == blocked or domain.endswith(f".{blocked}")
        for blocked in BLACKLISTED_HOSTS
    )


def _flesch_score(text: str) -> float:
    words = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?", text or "")
    if not words:
        return 0.0
    sentences = max(1, len(re.findall(r"[.!?]+", text or "")))
    syllables = sum(_syllables(word) for word in words)
    return (
        206.835
        - 1.015 * (len(words) / sentences)
        - 84.6 * (syllables / len(words))
    )


def _syllables(word: str) -> int:
    value = re.sub(r"[^a-z]", "", word.lower())
    if not value:
        return 1
    groups = re.findall(r"[aeiouy]+", value)
    count = len(groups)
    if value.endswith("e") and count > 1:
        count -= 1
    return max(1, count)
