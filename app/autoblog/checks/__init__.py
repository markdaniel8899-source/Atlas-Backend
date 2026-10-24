"""Hard content quality gate for AutoBlog v2 - SEO/AEO/GEO checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ─── Banned Words ──────────────────────────────────────────────────────────────

BANNED_WORDS = frozenset({
    "delve", "delve into", "moreover", "furthermore", "in conclusion", "additionally",
    "it's worth noting", "it is worth noting", "it is important to note",
    "in today's fast-paced world", "ever-evolving landscape", "landscape of",
    "tapestry", "testament", "testament to", "unlock", "unleash",
    "dive deep", "dive into", "game-changer", "revolutionize", "revolutionary",
    "seamless", "robust", "cutting-edge", "elevate", "harness the power",
    "in a world where", "boasts", "myriad", "plethora", "navigating",
    "embarking", "fostering", "realm",
})

# ─── Regex ─────────────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_H2_RE = re.compile(r"<h2[^>]*>.*?</h2>", re.IGNORECASE | re.DOTALL)
_H3_RE = re.compile(r"<h3[^>]*>.*?</h3>", re.IGNORECASE | re.DOTALL)
_FAQ_H2_RE = re.compile(r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b)", re.IGNORECASE)
_HEADING_RE = re.compile(r"<h[23][\s>]", re.IGNORECASE)


def _plain_text(html_text: str) -> str:
    """Strip HTML tags and collapse whitespace - words only, no markup."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_text)).strip()


def _word_count(html_text: str) -> int:
    return len(_plain_text(html_text).split())


def _first_n_words(html_text: str, n: int = 100) -> str:
    words = _plain_text(html_text).split()
    return " ".join(words[:n]).lower()


def _extract_headings(html_text: str) -> list[tuple[str, str]]:
    """Extract (level, text) for all h2/h3 headings."""
    headings = []
    for m in re.finditer(r"<h2[^>]*>(.*?)</h2>", html_text, re.IGNORECASE | re.DOTALL):
        text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if text:
            headings.append(("h2", text))
    for m in re.finditer(r"<h3[^>]*>(.*?)</h3>", html_text, re.IGNORECASE | re.DOTALL):
        text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if text:
            headings.append(("h3", text))
    return headings


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, remove punctuation."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keyword_boundary(keyword: str, text: str) -> bool:
    """Check if keyword appears as whole word(s) in text (case-insensitive)."""
    if not keyword or not text:
        return False
    kw = keyword.strip().lower()
    text_lower = text.lower()
    for w in kw.split():
        if not re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text_lower):
            return False
    return True


def _has_banned(text: str) -> list[str]:
    """Check for banned words/phrases (case-insensitive)."""
    text_lower = text.lower()
    return [w for w in BANNED_WORDS if w.lower() in text_lower]


def _fix_dashes(text: str) -> str:
    """Replace em/en dashes with comma+space."""
    return re.sub(r"\s*[—–]\s*", ", ", text)


def _has_heading_echo(html_text: str) -> list[str]:
    """Check if any h2/h3 heading is echoed as first sentence of following paragraph."""
    parts = re.split(r"(<h[23][^>]*>.*?</h[23]>)", html_text, flags=re.IGNORECASE | re.DOTALL)
    echoes = []
    for i in range(1, len(parts), 2):
        heading_text = re.sub(r"<[^>]+>", "", parts[i]).strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        p_match = re.search(r"<p[^>]*>(.*?)</p>", body, re.IGNORECASE | re.DOTALL)
        if p_match:
            first_para = re.sub(r"<[^>]+>", "", p_match.group(1)).strip()
            if first_para:
                norm_h = _normalize(heading_text)
                norm_p = _normalize(first_para)
                if norm_p.startswith(norm_h) or norm_h in norm_p[:len(norm_h) + 10]:
                    echoes.append(heading_text)
    return echoes


def _fix_heading_echo(html_text: str) -> tuple[str, list[str]]:
    """Remove heading echoes from first paragraph after each h2/h3."""
    parts = re.split(r"(<h[23][^>]*>.*?</h[23]>)", html_text, flags=re.IGNORECASE | re.DOTALL)
    fixed_parts = [parts[0]]
    fixed = []
    for i in range(1, len(parts), 2):
        heading_tag = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        heading_text = re.sub(r"<[^>]+>", "", heading_tag).strip()
        if not heading_text:
            fixed_parts.append(heading_tag)
            fixed_parts.append(body)
            continue

        def fix_para(match):
            para_content = match.group(1)
            norm_h = _normalize(heading_text)
            norm_p = _normalize(para_content)
            if norm_p.startswith(_normalize(heading_text)) or _normalize(heading_text) in _normalize(para_content)[:len(_normalize(heading_text)) + 10]:
                remaining = para_content[len(heading_text):].lstrip(" .,;:—–-")
                if not remaining.strip():
                    fixed.append(heading_text)
                    return ""
                fixed.append(heading_text)
                return f"<p>{remaining}</p>"
            return match.group(0)

        fixed_body = re.sub(r"<p[^>]*>(.*?)</p>", fix_para, body, count=1, flags=re.IGNORECASE | re.DOTALL)
        fixed_parts.append(heading_tag)
        fixed_parts.append(fixed_body)
    return "".join(fixed_parts), fixed


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]
    meta_only: bool = False
    fixed_html: str | None = None


def _count_faq(html_text: str) -> int:
    faq_match = re.search(r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b)", html_text, re.IGNORECASE)
    if not faq_match:
        return 0
    after_faq = html_text[faq_match.end():]
    return len(re.findall(r"<h3[^>]*>.*?</h3>", after_faq, re.IGNORECASE | re.DOTALL))


def _validate_visual(visual: dict[str, Any], research: Any) -> bool:
    """Validate structured visual is grounded in research."""
    kind = visual.get("kind")
    if kind == "table":
        columns = visual.get("columns", [])
        rows = visual.get("rows", [])
        if not columns or not rows or len(rows) < 2:
            return False
        # Check source_url exists
        if not visual.get("source_url"):
            return False
        # Check intro_sentence and takeaway exist
        if not visual.get("intro_sentence") or not visual.get("takeaway"):
            return False
    elif kind in ("bar", "line"):
        labels = visual.get("labels", [])
        values = visual.get("values", [])
        if not labels or not values or len(labels) != len(values) or len(labels) < 3:
            return False
        if not visual.get("unit"):
            return False
        if not visual.get("source_url"):
            return False
        if not visual.get("intro_sentence") or not visual.get("takeaway"):
            return False
    else:
        return False
    return True


def evaluate_article(
    article: dict[str, Any],
    meta_title: str,
    meta_description: str,
    research: Any,
    json_repaired: bool = False,
) -> GateResult:
    """
    Hard quality gate - ALL checks must pass.
    Returns GateResult with passed=True only if ALL checks pass.
    """
    content = article.get("content_html", "")
    pk = research.keywords[0] if research.keywords else research.topic
    reasons = []

    # 1. Word count 900-1200 (tags stripped)
    words = _word_count(content)
    if words < 900 or words > 1200:
        reasons.append(f"Word count {words} not in 900-1200")

    # 2. FAQ at end: last h2 is "FAQ" with 3-4 h3
    faq_count = _count_faq(content)
    if faq_count < 3 or faq_count > 4:
        reasons.append(f"FAQ has {faq_count} questions (need 3-4)")
    faq_match = re.search(r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b)", content, re.IGNORECASE)
    if faq_match:
        after_faq = content[faq_match.end():]
        if re.search(r"<h2[^>]*>", after_faq):
            reasons.append("Content found after FAQ section")

    # 3. Headings unique + no heading echo
    headings = _extract_headings(content)
    seen = set()
    for _, text in headings:
        norm = _normalize(text)
        if norm in seen:
            reasons.append(f"Duplicate heading: {text}")
        seen.add(norm)

    echoes = _has_heading_echo(content)
    if echoes:
        fixed, fixed_list = _fix_heading_echo(content)
        return GateResult(
            passed=False,
            reasons=reasons + [f"Heading echo: {', '.join(echoes)}"],
            fixed_html=fixed,
        )

    # 4. Banned words
    banned = _has_banned(content)
    if banned:
        reasons.append(f"Banned words: {', '.join(banned)}")

    # 4b. Em/en dashes
    if "—" in content or "–" in content:
        reasons.append("Em/en dashes found (auto-fix failed)")

    # 5. Visual reference check
    visual = article.get("visual")
    if visual:
        kind = visual.get("kind")
        if kind == "table" and not ("table below" in content.lower() or "table above" in content.lower() or "table shows" in content.lower() or "table compares" in content.lower()):
            reasons.append("Table referenced but not introduced in text (need 'table below' etc.)")
        if kind in ("bar", "line") and not ("chart below" in content.lower() or "chart above" in content.lower() or "chart shows" in content.lower()):
            reasons.append("Chart referenced but not introduced in text (need 'chart below' etc.)")

    # 6. Primary keyword checks
    if pk:
        if not _keyword_boundary(pk, article.get("title", "")):
            reasons.append(f"Primary keyword '{pk}' missing from title")
        if not _keyword_boundary(pk, _first_n_words(content, 100)):
            reasons.append(f"Primary keyword '{pk}' missing from first 100 words")
        h2_texts = [t for l, t in _extract_headings(content) if l == "h2"]
        if not any(_keyword_boundary(pk, t) for t in h2_texts):
            reasons.append(f"Primary keyword '{pk}' missing from any <h2>")

    # 7. Excerpt length 120-160
    meta_len = len(meta_description)
    if meta_len < 120 or meta_len > 160:
        reasons.append(f"Meta description {meta_len} chars (need 120-160)")

    # 8. Meta title has keyword
    if pk and not _keyword_boundary(pk, article.get("title", "")):
        reasons.append(f"Title missing primary keyword '{pk}'")

    # 9. JSON truncated/repaired
    if json_repaired:
        reasons.append("JSON was repaired from truncation")

    # 10. Visual validation
    if visual:
        if not _validate_visual(visual, research):
            reasons.append("Visual validation failed (ungrounded numbers, missing source/intro/takeaway, or no variation)")

    passed = len(reasons) == 0
    meta_fail = any("meta" in r.lower() or "title" in r.lower() or "description" in r.lower() for r in reasons)
    content_fail = [r for r in reasons if not ("meta" in r.lower() or "title" in r.lower() or "description" in r.lower())]

    return GateResult(
        passed=passed,
        reasons=reasons,
        meta_only=meta_fail and not content_fail,
    )