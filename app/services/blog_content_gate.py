"""Hard content quality gate for AutoBlog articles.

Replaces the lightweight SEO checker with comprehensive validation.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

from app.plugins.seo_checker import evaluate_seo_lightweight

# Banned words from the writer prompt (case-insensitive, phrase match)
BANNED_PHRASES = [
    "delve",
    "delve into",
    "moreover",
    "furthermore",
    "in conclusion",
    "additionally",
    "it's worth noting",
    "it is important to note",
    "in today's fast-paced world",
    "ever-evolving landscape",
    "landscape of",
    "tapestry",
    "testament",
    "testament to",
    "unlock",
    "unleash",
    "dive deep",
    "dive into",
    "game-changer",
    "revolutionize",
    "revolutionary",
    "seamless",
    "robust",
    "cutting-edge",
    "elevate",
    "harness the power",
    "in a world where",
    "boasts",
    "myriad",
    "plethora",
    "navigating",
    "embarking",
    "fostering",
    "realm",
]

_TAG_RE = re.compile(r"<[^>]+>")
_H2_RE = re.compile(r"(<h2[^>]*>.*?</h2>)", re.IGNORECASE | re.DOTALL)
_H3_RE = re.compile(r"<h3[^>]*>.*?</h3>", re.IGNORECASE | re.DOTALL)
_FAQ_H2_RE = re.compile(r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b)", re.IGNORECASE)


def _plain_text(html_text: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", html_text)).strip()


def _word_count(html_text: str) -> int:
    return len(_plain_text(html_text).split())


def _first_n_words(html_text: str, n: int = 100) -> str:
    words = _plain_text(html_text).split()
    return " ".join(words[:n]).lower()


def _extract_headings(html_text: str) -> list[tuple[str, str]]:
    """Extract (level, text) for all h2/h3 headings."""
    headings = []
    for match in _H2_RE.finditer(html_text):
        text = _TAG_RE.sub("", match.group(1)).strip()
        if text:
            headings.append(("h2", text))
    for match in _H3_RE.finditer(html_text):
        text = _TAG_RE.sub("", match.group(0)).strip()
        if text:
            headings.append(("h3", text))
    return headings


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, remove punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _has_heading_echo(html_text: str) -> list[str]:
    """Check if any heading is echoed as the first sentence of the following paragraph.

    Returns list of echoed heading texts.
    """
    parts = _H2_RE.split(html_text)
    echoes = []

    # parts[0] is before first h2, parts[1] is h2, parts[2] is content, parts[3] is h2, etc.
    for i in range(1, len(parts), 2):
        heading_tag = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""

        heading_text = _TAG_RE.sub("", heading_tag).strip()
        if not heading_text:
            continue

        # Find first paragraph in body
        p_match = re.search(r"<p[^>]*>(.*?)</p>", body, re.IGNORECASE | re.DOTALL)
        if p_match:
            first_para = _TAG_RE.sub("", p_match.group(1)).strip()
            if first_para:
                norm_heading = _normalize_text(heading_text)
                norm_para = _normalize_text(first_para)
                # Check if first paragraph starts with heading text (allowing some variation)
                if norm_para.startswith(norm_heading) or norm_heading in norm_para[:len(norm_heading) + 10]:
                    echoes.append(heading_text)

    return echoes


def _fix_heading_echo(html_text: str) -> tuple[str, list[str]]:
    """Remove heading echoes from the first paragraph after each h2/h3.

    Returns (fixed_html, list_of_fixed_headings).
    """
    parts = _H2_RE.split(html_text)
    fixed_parts = [parts[0]]
    fixed = []

    for i in range(1, len(parts), 2):
        heading_tag = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""

        heading_text = _TAG_RE.sub("", heading_tag).strip()
        if not heading_text:
            fixed_parts.append(heading_tag)
            fixed_parts.append(body)
            continue

        # Fix first paragraph if it echoes the heading
        def fix_para(match):
            para_content = match.group(1)
            norm_heading = _normalize_text(heading_text)
            norm_para = _normalize_text(para_content)
            if norm_para.startswith(norm_heading) or norm_heading in norm_para[:len(norm_heading) + 10]:
                # Remove the echoed prefix
                remaining = para_content[len(heading_text):].lstrip(" .,;:—–-")
                if not remaining.strip():
                    fixed.append(heading_text)
                    return ""  # Remove empty paragraph
                fixed.append(heading_text)
                return f"<p>{remaining}</p>"
            return match.group(0)

        fixed_body = re.sub(r"<p[^>]*>(.*?)</p>", fix_para, body, count=1, flags=re.IGNORECASE | re.DOTALL)

        fixed_parts.append(heading_tag)
        fixed_parts.append(fixed_body)

    return "".join(fixed_parts), fixed


def _check_banned_words(text: str) -> list[str]:
    """Check for banned words/phrases in text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for phrase in BANNED_PHRASES:
        if phrase.lower() in text_lower:
            found.append(phrase)
    return found


def _fix_dashes(text: str) -> str:
    """Replace em/en dashes with commas or hyphens."""
    # Replace em dash (—) and en dash (–) with comma+space
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    return text


def _keyword_word_boundary(keyword: str, text: str) -> bool:
    """Check if keyword appears as whole word(s) in text (case-insensitive)."""
    if not keyword or not text:
        return False
    kw = keyword.strip().lower()
    text_lower = text.lower()
    # Use word boundaries for each word in keyword
    words = kw.split()
    for word in words:
        if not re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text_lower):
            return False
    return True


def _count_faq_questions(html_text: str) -> int:
    """Count FAQ questions (h3 inside FAQ section)."""
    faq_match = _FAQ_H2_RE.search(html_text)
    if not faq_match:
        return 0
    faq_content = html_text[faq_match.end():]
    # Count h3 until next h2
    count = 0
    for match in _H3_RE.finditer(faq_content):
        # Check if there's an h2 before this h3
        next_h2 = _H2_RE.search(faq_content[match.start():])
        if next_h2 and next_h2.start() < match.start():
            break
        count += 1
    return count


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]
    fixed_html: str | None = None
    fixed_meta_title: str | None = None
    fixed_meta_description: str | None = None
    meta_only_failure: bool = False


def evaluate_article(
    article: dict[str, Any],
    meta_title: str,
    meta_description: str,
    research: Any,
    profile: Any,
    json_repaired: bool = False,
) -> GateResult:
    """Evaluate article against hard content gate.

    Returns GateResult with pass/fail and any fixes applied.
    """
    content_html = article.get("content_html", "")
    reasons = []

    # 1. Word count: 900-1200
    words = _word_count(content_html)
    if words < 900:
        reasons.append(f"Word count {words} < 900 (need 900-1200)")
    elif words > 1200:
        reasons.append(f"Word count {words} > 1200 (need 900-1200)")

    # 2. FAQ at the end: h2 FAQ + 3-4 h3 questions
    faq_count = _count_faq_questions(content_html)
    if faq_count < 3 or faq_count > 4:
        reasons.append(f"FAQ has {faq_count} questions (need 3-4)")
    # Ensure FAQ is at the end (no h2 after FAQ)
    faq_match = _FAQ_H2_RE.search(content_html)
    if faq_match:
        after_faq = content_html[faq_match.end():]
        if _H2_RE.search(after_faq):
            reasons.append("Content found after FAQ section")

    # 3. Headings unique + no echo
    headings = _extract_headings(content_html)
    seen = set()
    for _, text in headings:
        norm = _normalize_text(text)
        if norm in seen:
            reasons.append(f"Duplicate heading: {text}")
        seen.add(norm)

    echoes = _has_heading_echo(content_html)
    if echoes:
        # Auto-fix
        fixed_html, fixed_list = _fix_heading_echo(content_html)
        if fixed_list:
            # Re-evaluate with fixed HTML
            content_html = fixed_html
            # Update article for further checks
            article = {**article, "content_html": fixed_html}

    # 4. Banned words
    banned_found = _check_banned_words(content_html)
    if banned_found:
        reasons.append(f"Banned words/phrases found: {', '.join(banned_found)}")

    # 5. Em/en dashes (auto-fixed)
    if "—" in content_html or "–" in content_html:
        # This is a warning, auto-fix happens in writer
        reasons.append("Em/en dashes found (will be auto-fixed)")

    # 6. Truncated/repaired JSON
    if json_repaired:
        reasons.append("JSON was repaired from truncation — draft discarded")

    # 7. Primary keyword checks
    primary_keyword = research.keywords[0] if research.keywords else research.topic
    if primary_keyword:
        # In meta title
        if not _keyword_word_boundary(primary_keyword, meta_title):
            reasons.append(f"Primary keyword '{primary_keyword}' missing from meta title")
        # In first 100 words
        first_100 = _first_n_words(content_html, 100)
        if not _keyword_word_boundary(primary_keyword, first_100):
            reasons.append(f"Primary keyword '{primary_keyword}' missing from first 100 words")
        # In an h2
        h2_texts = [t for level, t in headings if level == "h2"]
        kw_in_h2 = any(_keyword_word_boundary(primary_keyword, t) for t in h2_texts)
        if not kw_in_h2:
            reasons.append(f"Primary keyword '{primary_keyword}' missing from any <h2>")

    # 8. Meta description 120-160 chars
    meta_len = len(meta_description)
    if meta_len < 120 or meta_len > 160:
        reasons.append(f"Meta description {meta_len} chars (need 120-160)")

    # 9. Meta title contains keyword
    if primary_keyword and not _keyword_word_boundary(primary_keyword, meta_title):
        reasons.append(f"Meta title missing primary keyword '{primary_keyword}'")

    # 10. Visual validation (delegated to caller)
    # We just check if profile requires visuals and data exists
    visual_failures = article.get("_visual_failures", [])
    reasons.extend(visual_failures)

    # Determine if only meta failures
    meta_failures = [r for r in reasons if "meta" in r.lower() or "Meta" in r]
    content_failures = [r for r in reasons if r not in meta_failures]

    passed = len(reasons) == 0

    return GateResult(
        passed=passed,
        reasons=reasons,
        fixed_html=content_html if content_html != article.get("content_html") else None,
        meta_only_failure=not content_failures and meta_failures,
    )


def regenerate_meta(
    research: Any,
    feedback: str,
) -> tuple[str, str]:
    """Regenerate meta title and description with feedback."""
    from app.services.auto_blog import _groq_with_gemini_fallback, _META_SYSTEM, get_settings

    sources = "\n".join(f"- {s}" for s in research.snippets[:3]) or "(none)"
    primary_keyword = research.keywords[0] if research.keywords else research.topic
    other_keywords = ", ".join(research.keywords[1:]) or "(none)"

    user_prompt = (
        f"Topic: {research.topic}\n"
        f"Primary keyword: {primary_keyword}\n"
        f"Other keywords: {other_keywords}\n"
        f"Research snippets:\n{sources}\n\n"
        f"Previous meta failed: {feedback}\n"
        "Regenerate meta title and description fixing the issues above."
    )

    # This would need to be async; caller should handle
    # Placeholder for now
    return _meta_chunk_sync(research)


def _meta_chunk_sync(research: Any) -> tuple[str, str]:
    """Sync fallback for meta generation."""
    from app.services.auto_blog import _meta_chunk
    import asyncio
    return asyncio.run(_meta_chunk(research))