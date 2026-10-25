"""Deterministic meta-tag builder for the SEO/AEO/GEO pipeline.

The writer proposes copy; this module enforces the hard constraints search
engines and social previews actually care about:

* meta title 50-60 characters, starting with the primary keyword
* meta description 150-160 characters, containing the keyword and a CTA
* excerpt 140-160 characters, so the existing frontend stays consistent
"""

from __future__ import annotations

import re

from app.autoblog.models import DynamicArticleDraft, KeywordPlan, MetaBundle, TopicTrend


TITLE_MIN = 50
TITLE_MAX = 60
DESC_MIN = 150
DESC_MAX = 160
EXCERPT_MIN = 140
EXCERPT_MAX = 160

_ACRONYMS = {"ai", "api", "gpt", "llm", "seo", "geo", "aeo", "sql", "ux", "ui", "ml", "nlp"}
_CTA_TOKENS = (
    "learn more",
    "read more",
    "discover",
    "start ",
    "see how",
    "explore",
    "get the",
    "find out",
    "read the",
)
_DESC_PADS = (
    "See the full breakdown with examples and practical takeaways.",
    "Read the data-backed explanation and what it means for students.",
    "Get the key numbers, risks and next steps in one place.",
    "Compare the trade-offs before you change your study workflow.",
)
_EXCERPT_PADS = (
    "Here is what changed and how to use it.",
    "A practical guide for students and self-directed learners.",
    "The key numbers, trade-offs and next steps, explained.",
)


def _plain(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\u2014", ",").replace("\u2013", ",")
    text = re.sub(r"\s*,\s*,+", ", ", text)
    return text.strip(" \"'")


def _clean(value: str) -> str:
    return _plain(value).rstrip(" ,;:-")


def _title_keyword(keyword: str) -> str:
    words = []
    for word in _plain(keyword).split():
        words.append(word.upper() if word.lower() in _ACRONYMS else word.capitalize())
    return " ".join(words)


def _trim_to_max(text: str, limit: int) -> str:
    clean = _plain(text)
    if len(clean) <= limit:
        return clean
    cut = clean[:limit].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-")


def _trim_sentence(text: str, limit: int) -> str:
    clean = _plain(text)
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    for mark in (". ", "! ", "? ", "; "):
        index = cut.rfind(mark)
        if index >= limit // 2:
            return cut[: index + 1].rstrip()
    return _trim_to_max(clean, limit)


def _contains_keyword(text: str, keyword: str) -> bool:
    needle = _plain(keyword).lower()
    if not needle:
        return False
    haystack = _plain(text).lower()
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def _ensure_keyword(text: str, keyword: str) -> str:
    if not keyword:
        return text
    if _contains_keyword(text, keyword):
        return text
    return f"{_title_keyword(keyword)}: {text}"


def enforce_meta_title(raw: str, fallback_title: str, primary_keyword: str) -> str:
    keyword = _clean(primary_keyword)
    display = _title_keyword(keyword)
    base = _clean(raw) or _clean(fallback_title) or display

    if keyword and not base.lower().startswith(keyword.lower()):
        base = f"{display}: {base}"

    base = _trim_to_max(base, TITLE_MAX)

    if len(base) < TITLE_MIN:
        for suffix in (" | ATLAS Blog", " Guide", " in 2026", " for Students"):
            candidate = f"{base.rstrip(' ,;:-')}{suffix}"
            if len(candidate) <= TITLE_MAX:
                base = candidate
            if len(base) >= TITLE_MIN:
                break

    if len(base) < TITLE_MIN:
        tail = " Practical Guide for Students and Self-Directed Learners"
        base = _trim_to_max(f"{base.rstrip(' ,;:-')}{tail}", TITLE_MAX)

    if len(base) < TITLE_MIN:
        for suffix in (" Guide", " for Students", " | ATLAS Blog", " in 2026"):
            candidate = _trim_to_max(f"{base.rstrip(' ,;:-')}{suffix}", TITLE_MAX)
            if len(candidate) > len(base):
                base = candidate
            if len(base) >= TITLE_MIN:
                break

    if keyword and not base.lower().startswith(keyword.lower()):
        base = _trim_to_max(f"{display}: {base}", TITLE_MAX)

    return base.rstrip(" ,;:-")


def enforce_meta_description(
    raw: str,
    fallback_text: str,
    primary_keyword: str,
    excerpt: str = "",
) -> str:
    keyword = _clean(primary_keyword)
    base = _clean(raw) or _clean(excerpt) or _clean(fallback_text)
    if not base:
        base = f"A practical guide to {keyword or 'AI learning'} for students."

    base = _ensure_keyword(base, keyword)
    if len(base) > 120:
        base = _trim_sentence(base, 120)

    has_cta = any(token in base.lower() for token in _CTA_TOKENS)
    cta = "" if has_cta else " Learn more."
    room = DESC_MAX - len(cta)

    for pad in _DESC_PADS:
        candidate = f"{base.rstrip()} {pad}"
        if len(candidate) <= room:
            base = candidate
        if len(base) + len(cta) >= DESC_MIN:
            break

    if len(base) > room:
        base = _trim_sentence(base, room)

    pad_index = 0
    while len(base) + len(cta) < DESC_MIN and len(base) < room:
        pad = _DESC_PADS[pad_index % len(_DESC_PADS)]
        candidate = _trim_to_max(f"{base.rstrip()} {pad}", room)
        if candidate == base:
            break
        base = candidate
        pad_index += 1

    description = f"{base.rstrip()}{cta}".strip()
    if len(description) > DESC_MAX:
        description = description[:DESC_MAX].rstrip(" ,;:-")
    if not description.endswith((".", "!", "?")):
        if len(description) >= DESC_MAX:
            description = description[: DESC_MAX - 1].rstrip(" ,;:-") + "."
        else:
            description += "."
    return description


def enforce_excerpt(raw: str, description: str, primary_keyword: str) -> str:
    keyword = _clean(primary_keyword)
    base = _clean(raw) or _clean(description)
    if not base:
        base = f"A practical guide to {keyword or 'AI learning'} for students."

    base = _ensure_keyword(base, keyword)
    base = _trim_to_max(base, EXCERPT_MAX)

    if len(base) < EXCERPT_MIN:
        for pad in _EXCERPT_PADS:
            candidate = f"{base.rstrip()} {pad}"
            if len(candidate) <= EXCERPT_MAX:
                base = candidate
            if len(base) >= EXCERPT_MIN:
                break

    if len(base) < EXCERPT_MIN:
        tail = " Practical guidance for students and self-directed learners."
        base = _trim_to_max(f"{base.rstrip()} {tail}", EXCERPT_MAX)

    if not base.endswith((".", "!", "?")):
        if len(base) >= EXCERPT_MAX:
            base = base[: EXCERPT_MAX - 1].rstrip(" ,;:-") + "."
        else:
            base += "."
    return _trim_to_max(base, EXCERPT_MAX)


def build_meta(
    trend: TopicTrend,
    draft: DynamicArticleDraft,
    keywords: KeywordPlan,
) -> MetaBundle:
    """Step 4: turn the writer's proposals into publish-ready meta tags."""
    keyword = keywords.primary_keyword

    meta_title = enforce_meta_title(
        draft.meta_title_raw or draft.title,
        draft.title or trend.headline,
        keyword,
    )
    meta_description = enforce_meta_description(
        draft.meta_description_raw,
        draft.excerpt_raw or draft.intro_html or trend.angle,
        keyword,
        excerpt=draft.excerpt_raw,
    )
    excerpt = enforce_excerpt(draft.excerpt_raw, meta_description, keyword)

    twitter_title = meta_title if len(meta_title) <= 70 else _trim_to_max(meta_title, 70)
    twitter_description = meta_description if len(meta_description) <= 200 else _trim_to_max(meta_description, 200)

    bundle = MetaBundle(
        meta_title=meta_title,
        meta_description=meta_description,
        og_title=meta_title,
        og_description=meta_description,
        twitter_title=twitter_title,
        twitter_description=twitter_description,
        excerpt=excerpt,
    )

    print(
        f"[AutoBlog] Meta ready: title={len(bundle.meta_title)} chars, "
        f"description={len(bundle.meta_description)} chars, "
        f"excerpt={len(bundle.excerpt)} chars",
        flush=True,
    )
    return bundle


__all__ = [
    "build_meta",
    "enforce_excerpt",
    "enforce_meta_description",
    "enforce_meta_title",
]
