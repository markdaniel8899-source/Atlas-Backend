"""SEO/AEO/GEO validation for the dynamic AutoBlog pipeline.

The validator scores seven structural buckets for a maximum of 100 points.
Publishing requires 90 or more, so a draft can miss one small item but not an
entire section category.
"""

from __future__ import annotations

import re
import urllib.parse

from app.autoblog.models import KeywordPlan, MetaBundle, SeoValidation


PUBLISH_THRESHOLD = 90
TITLE_MIN, TITLE_MAX = 50, 60
DESC_MIN, DESC_MAX = 150, 160
MIN_WORDS = 1200
MIN_H2 = 4
MIN_BODY_H3 = 1
MIN_FAQ = 3
FAQ_MIN_WORDS, FAQ_MAX_WORDS = 40, 60

AUTHORITATIVE_DOMAINS: tuple[str, ...] = (
    "openai.com",
    "anthropic.com",
    "ai.google",
    "deepmind.com",
    "huggingface.co",
    "arxiv.org",
    "techcrunch.com",
    "mit.edu",
    "stanford.edu",
    "nvidia.com",
    "microsoft.com",
    "meta.com",
    "reuters.com",
    "theverge.com",
    "wired.com",
    "nature.com",
    "science.org",
    "github.com",
)

_TAG_RE = re.compile(r"<[^>]+>")
_HREF_RE = re.compile(r"""<a\b[^>]*\bhref=["']([^"']+)["']""", re.IGNORECASE)
_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_ALT_RE = re.compile(r"""\balt=["']([^"']*)["']""", re.IGNORECASE)
_H2_RE = re.compile(r"<h2\b[^>]*>.*?</h2>", re.IGNORECASE | re.DOTALL)
_H3_RE = re.compile(r"<h3\b[^>]*>(.*?)</h3>", re.IGNORECASE | re.DOTALL)
_FAQ_RE = re.compile(
    r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b).*?</h2>",
    re.IGNORECASE | re.DOTALL,
)


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", fragment)).strip()


def _words(fragment: str) -> list[str]:
    return _text(fragment).split()


def _keyword_in(fragment: str, keyword: str) -> bool:
    needle = str(keyword or "").strip().lower()
    if not needle:
        return False
    haystack = _text(fragment).lower()
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def _host(url: str) -> str:
    host = urllib.parse.urlsplit(str(url or "")).netloc.lower().split(":", 1)[0]
    return host[4:] if host.startswith("www.") else host


def _site_host(site_url: str) -> str:
    return _host(site_url or "https://atlas-frontend-pearl.vercel.app")


def _faq_entries(content_html: str) -> list[dict[str, str]]:
    match = _FAQ_RE.search(content_html)
    if not match:
        return []

    tail = content_html[match.end():]
    next_h2 = re.search(r"<h2\b", tail, re.IGNORECASE)
    block = tail[: next_h2.start()] if next_h2 else tail

    questions = list(_H3_RE.finditer(block))
    entries: list[dict[str, str]] = []
    for index, question in enumerate(questions):
        start = question.end()
        end = questions[index + 1].start() if index + 1 < len(questions) else len(block)
        answer = block[start:end]
        entries.append(
            {
                "question": _text(question.group(1)),
                "answer": _text(answer),
            }
        )
    return entries


def _body_h3_count(content_html: str) -> int:
    faq = _FAQ_RE.search(content_html)
    body = content_html[: faq.start()] if faq else content_html
    return len(re.findall(r"<h3\b", body, re.IGNORECASE))


def _count_links(content_html: str, site_url: str) -> tuple[int, int, int]:
    """Return (internal_count, authoritative_external_count, external_count)."""
    site_host = _site_host(site_url)
    internal = 0
    authoritative = 0
    external = 0
    seen: set[str] = set()

    for href in _HREF_RE.findall(content_html):
        href = href.strip()
        if not href or href.startswith("#") or href in seen:
            continue
        seen.add(href)

        if href.startswith("/") and not href.startswith("//"):
            internal += 1
            continue
        if not href.lower().startswith(("http://", "https://")):
            continue

        host = _host(href)
        if not host:
            continue
        if host == site_host or host.endswith(f".{site_host}"):
            internal += 1
        else:
            external += 1
            if any(
                host == domain or host.endswith(f".{domain}")
                for domain in AUTHORITATIVE_DOMAINS
            ):
                authoritative += 1

    return internal, authoritative, external


def _visual_score(content_html: str) -> tuple[int, int]:
    images = _IMG_RE.findall(content_html)
    tables = len(re.findall(r"<table\b", content_html, re.IGNORECASE))
    svgs = len(re.findall(r"<svg\b", content_html, re.IGNORECASE))
    visual_count = len(images) + tables + svgs
    if visual_count == 0:
        return 0, 0

    missing_alt = 0
    for image in images:
        alt_match = _ALT_RE.search(image)
        if not alt_match or not alt_match.group(1).strip():
            missing_alt += 1

    if missing_alt:
        return 5, visual_count
    return 10, visual_count


def validate_article(
    *,
    title: str,
    content_html: str,
    meta: MetaBundle,
    keywords: KeywordPlan,
    site_url: str = "",
) -> SeoValidation:
    """Score one assembled draft and return actionable rewrite instructions."""
    keyword = keywords.primary_keyword
    breakdown: dict[str, int] = {}
    issues: list[str] = []

    # 1. Meta title (8)
    title_len = len(meta.meta_title)
    title_has_keyword = _keyword_in(meta.meta_title, keyword)
    keyword_needle = str(keyword or "").strip().lower()
    title_starts_keyword = bool(keyword_needle) and meta.meta_title.lower().startswith(
        keyword_needle
    )
    if (
        TITLE_MIN <= title_len <= TITLE_MAX
        and title_has_keyword
        and title_starts_keyword
    ):
        breakdown["meta_title"] = 8
    elif title_has_keyword or TITLE_MIN <= title_len <= TITLE_MAX:
        breakdown["meta_title"] = 4
        issues.append(
            f"Meta title must be {TITLE_MIN}-{TITLE_MAX} characters and start "
            f"with '{keyword}' (currently {title_len} characters, "
            f"keyword={'yes' if title_has_keyword else 'no'})."
        )
    else:
        breakdown["meta_title"] = 0
        issues.append(
            f"Rewrite the meta title to {TITLE_MIN}-{TITLE_MAX} characters and start "
            f"it with '{keyword}'."
        )

    # 2. Meta description (7)
    desc_len = len(meta.meta_description)
    desc_has_keyword = _keyword_in(meta.meta_description, keyword)
    if DESC_MIN <= desc_len <= DESC_MAX and desc_has_keyword:
        breakdown["meta_description"] = 7
    elif desc_has_keyword or DESC_MIN <= desc_len <= DESC_MAX:
        breakdown["meta_description"] = 3
        issues.append(
            f"Meta description must be {DESC_MIN}-{DESC_MAX} characters and include "
            f"'{keyword}' (currently {desc_len} characters, "
            f"keyword={'yes' if desc_has_keyword else 'no'})."
        )
    else:
        breakdown["meta_description"] = 0
        issues.append(
            f"Rewrite the meta description to {DESC_MIN}-{DESC_MAX} characters with "
            f"'{keyword}' and a clear call to action."
        )

    # 3. Headings (15)
    h2_count = len(_H2_RE.findall(content_html))
    body_h3 = _body_h3_count(content_html)
    if h2_count >= 5 and body_h3 >= 2:
        breakdown["headings"] = 15
    elif h2_count >= MIN_H2 and body_h3 >= MIN_BODY_H3:
        breakdown["headings"] = 12
        issues.append("Add one more H2 section and at least one more H3 subheading.")
    elif h2_count >= MIN_H2:
        breakdown["headings"] = 8
        issues.append("Add at least two H3 subheadings inside body sections.")
    elif h2_count >= 3:
        breakdown["headings"] = 4
        issues.append(
            f"Use at least {MIN_H2} H2 sections (currently {h2_count}) and two H3 subheadings."
        )
    else:
        breakdown["headings"] = 0
        issues.append(
            f"Restructure the article into at least {MIN_H2} H2 sections with H3 subheadings "
            f"(currently {h2_count} H2s)."
        )

    # 4. Word count (15)
    total_words = len(_words(content_html))
    if total_words >= MIN_WORDS:
        breakdown["word_count"] = 15
    elif total_words >= 1000:
        breakdown["word_count"] = 12
        issues.append(
            f"Expand the article to at least {MIN_WORDS} words (currently {total_words})."
        )
    elif total_words >= 800:
        breakdown["word_count"] = 8
        issues.append(
            f"Expand the article to at least {MIN_WORDS} words (currently {total_words}); "
            "add concrete examples and explanations."
        )
    elif total_words >= 600:
        breakdown["word_count"] = 4
        issues.append(
            f"The article is too short at {total_words} words. Rewrite it to "
            f"{MIN_WORDS}-1600 words with deeper sections."
        )
    else:
        breakdown["word_count"] = 0
        issues.append(
            f"The article is far too short at {total_words} words. Write a complete "
            f"{MIN_WORDS}-1600 word article."
        )

    # 5-7. Keyword placement (15)
    keyword_points = 0
    if _keyword_in(title, keyword) or _keyword_in(meta.meta_title, keyword):
        keyword_points += 5
    else:
        issues.append(f"Include '{keyword}' in the article title or meta title.")

    first_100 = " ".join(_words(content_html)[:100])
    if _keyword_in(first_100, keyword):
        keyword_points += 5
    else:
        issues.append(f"Use '{keyword}' naturally in the first 100 words.")

    h2_texts = " ".join(_H2_RE.findall(content_html))
    if _keyword_in(h2_texts, keyword):
        keyword_points += 5
    else:
        issues.append(f"Include '{keyword}' in at least two H2 headings.")

    breakdown["keyword_placement"] = keyword_points

    # 8. FAQ count (8)
    faq_entries = _faq_entries(content_html)
    faq_count = len(faq_entries)
    if faq_count >= MIN_FAQ:
        breakdown["faq_count"] = 8
    elif faq_count >= 1:
        breakdown["faq_count"] = 4
        issues.append(
            f"Add FAQ questions so the article has at least {MIN_FAQ} "
            f"(currently {faq_count})."
        )
    else:
        breakdown["faq_count"] = 0
        issues.append("Add an <h2>FAQ</h2> section with 3-4 question and answer pairs.")

    # 9. FAQ answer length (7)
    answer_lengths = [len(entry["answer"].split()) for entry in faq_entries]
    if answer_lengths and all(FAQ_MIN_WORDS <= length <= FAQ_MAX_WORDS for length in answer_lengths):
        breakdown["faq_answers"] = 7
    elif answer_lengths and all(length >= 20 for length in answer_lengths):
        breakdown["faq_answers"] = 4
        issues.append(
            f"Each FAQ answer should be 40-60 words (current range "
            f"{min(answer_lengths)}-{max(answer_lengths)})."
        )
    else:
        breakdown["faq_answers"] = 0
        if answer_lengths:
            issues.append("Rewrite FAQ answers so each is 40-60 words and directly answers the question.")
        else:
            issues.append("Write 3-4 FAQ answers of 40-60 words each.")

    # 10. External links (8)
    internal_links, authoritative_links, external_links = _count_links(
        content_html, site_url
    )
    if authoritative_links >= 2:
        breakdown["external_links"] = 8
    elif external_links >= 2:
        breakdown["external_links"] = 6
        issues.append("Add a citation link to a recognized authoritative source.")
    elif external_links == 1:
        breakdown["external_links"] = 4
        issues.append("Add at least one more citation link to an external source.")
    else:
        breakdown["external_links"] = 0
        issues.append("Add at least two external links from the fact sheet.")

    # 11. Internal links (7)
    if internal_links >= 2:
        breakdown["internal_links"] = 7
    elif internal_links == 1:
        breakdown["internal_links"] = 3
        issues.append("Add at least one more internal link to an ATLAS page or article.")
    else:
        breakdown["internal_links"] = 0
        issues.append("Add two internal links, such as /blog and /app.")

    # 12. Visual (10)
    visual_points, visual_count = _visual_score(content_html)
    breakdown["visual"] = visual_points
    if visual_points == 0:
        issues.append("Add a chart, table or in-article image with descriptive alt text.")
    elif visual_points < 10:
        issues.append("Every in-article image needs a descriptive alt attribute.")

    total = sum(breakdown.values())
    validation = SeoValidation(
        total=total,
        passed=total >= PUBLISH_THRESHOLD,
        breakdown=breakdown,
        issues=issues,
    )

    print(
        f"[AutoBlog] SEO score: {total}/100 "
        f"({'PASS' if validation.passed else 'FAIL'}, threshold {PUBLISH_THRESHOLD}) "
        f"words={total_words} h2={h2_count} body_h3={body_h3} faq={faq_count} "
        f"internal_links={internal_links} external_links={authoritative_links} "
        f"visuals={visual_count}",
        flush=True,
    )
    return validation


__all__ = ["PUBLISH_THRESHOLD", "validate_article"]
