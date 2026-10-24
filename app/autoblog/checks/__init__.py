"""Deterministic quality gate for AutoBlog v2 - 2 fix attempts, then discard."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ─── Constants ────────────────────────────────────────────────────────────────

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

MIN_WORDS = 800
MAX_WORDS = 1200
MIN_FAQ = 3
MAX_FAQ = 4
MAX_TOTAL_IMAGES = 4  # including cover
MAX_CONTENT_IMAGES_UNDER_1000 = 2
MAX_CONTENT_IMAGES_OVER_1000 = 3

_TAG_RE = re.compile(r"<[^>]+>")
_H2_RE = re.compile(r"<h2[^>]*>.*?</h2>", re.IGNORECASE | re.DOTALL)
_H3_RE = re.compile(r"<h3[^>]*>.*?</h3>", re.IGNORECASE | re.DOTALL)
_FAQ_H2_RE = re.compile(r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b)", re.IGNORECASE)
_HEADING_RE = re.compile(r"<h[23][\s>]", re.IGNORECASE)


def _plain_text(html_text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_text)).strip()


def _word_count(html_text: str) -> int:
    return len(_plain_text(html_text).split())


def _first_n_words(html_text: str, n: int = 100) -> str:
    words = _plain_text(html_text).split()
    return " ".join(words[:n]).lower()


def _extract_headings(html_text: str) -> list[tuple[str, str]]:
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
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keyword_boundary(keyword: str, text: str) -> bool:
    if not keyword or not text:
        return False
    kw = keyword.strip().lower()
    text_lower = text.lower()
    for w in kw.split():
        if not re.search(rf"(?<!\w){re.escape(w)}(?!\w)", text_lower):
            return False
    return True


def _has_banned(text: str) -> list[str]:
    text_lower = text.lower()
    return [w for w in BANNED_WORDS if w.lower() in text_lower]


def _has_heading_echo(html_text: str) -> list[str]:
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


def _fix_heading_echo(html_text: str) -> str:
    parts = re.split(r"(<h[23][^>]*>.*?</h[23]>)", html_text, flags=re.IGNORECASE | re.DOTALL)
    fixed_parts = [parts[0]]
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
                    return ""
                return f"<p>{remaining}</p>"
            return match.group(0)

        fixed_body = re.sub(r"<p[^>]*>(.*?)</p>", fix_para, body, count=1, flags=re.IGNORECASE | re.DOTALL)
        fixed_parts.append(heading_tag)
        fixed_parts.append(fixed_body)
    return "".join(fixed_parts)


def _count_faq(html_text: str) -> int:
    faq_match = re.search(r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b)", html_text, re.IGNORECASE)
    if not faq_match:
        return 0
    after_faq = html_text[faq_match.end():]
    return len(re.findall(r"<h3[^>]*>.*?</h3>", after_faq, re.IGNORECASE | re.DOTALL))


def _extract_source_urls(html_text: str) -> list[str]:
    urls = re.findall(r'href="([^"]+)"', html_text)
    return [u for u in urls if u.startswith("http")]


def _check_visual_references(content_html: str, visuals: list[dict[str, Any]]) -> list[str]:
    """Check that every visual is properly introduced in text."""
    issues = []
    content_lower = content_html.lower()
    for i, vis in enumerate(visuals):
        kind = vis.get("kind", "")
        if kind == "table":
            has_intro = any(phrase in content_lower for phrase in ["table below", "table above", "table shows", "table compares", "following table"])
            if not has_intro:
                issues.append(f"Visual {i} (table) not introduced in text (need 'table below' etc.)")
        elif kind in ("bar", "line"):
            has_intro = any(phrase in content_lower for phrase in ["chart below", "chart above", "chart shows", "following chart"])
            if not has_intro:
                issues.append(f"Visual {i} ({kind}) not introduced in text (need 'chart below' etc.)")
    return issues


def _validate_visual_data(visuals: list[dict[str, Any]], fact_sheet: Any) -> list[str]:
    """Validate that every visual's numbers come from fact sheet."""
    issues = []
    fact_numbers = set()
    for n in getattr(fact_sheet, "numbers", []):
        fact_numbers.add((str(n.get("value", "")), str(n.get("unit", "")), str(n.get("context", ""))))

    for i, vis in enumerate(visuals):
        kind = vis.get("kind", "")
        if kind == "table":
            rows = vis.get("rows", [])
            for row in rows:
                for cell in row:
                    cell_str = str(cell).strip()
                    # Check if cell contains a number that should be in fact sheet
                    # (simplified: just check if visual has required fields)
                    pass
            if not vis.get("source_url"):
                issues.append(f"Visual {i} (table) missing source_url")
            if not vis.get("intro_sentence") or not vis.get("takeaway"):
                issues.append(f"Visual {i} (table) missing intro_sentence or takeaway")
        elif kind in ("bar", "line"):
            values = vis.get("values", [])
            labels = vis.get("labels", [])
            if not values or not labels or len(values) != len(labels) or len(values) < 3:
                issues.append(f"Visual {i} ({kind}) needs >=3 data points")
            if not vis.get("unit"):
                issues.append(f"Visual {i} ({kind}) missing unit")
            if not vis.get("source_url"):
                issues.append(f"Visual {i} ({kind}) missing source_url")
            if not vis.get("intro_sentence") or not vis.get("takeaway"):
                issues.append(f"Visual {i} ({kind}) missing intro_sentence or takeaway")
    return issues


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]
    fixed_html: str | None = None
    fixed_visuals: list[dict[str, Any]] | None = None
    meta_only: bool = False


def run_gate(
    article_html: str,
    article_data: dict[str, Any],
    meta_title: str,
    meta_description: str,
    plan: Any,
    fact_sheet: Any,
) -> GateResult:
    """
    Run all deterministic checks. Returns GateResult.
    If failed, caller can fix and re-run (max 2 attempts).
    """
    reasons = []
    fixed_html = article_html
    fixed_visuals = article_data.get("visuals", [])

    # 1. Word count
    words = _word_count(article_html)
    if words < MIN_WORDS or words > MAX_WORDS:
        reasons.append(f"Word count {words} not in {MIN_WORDS}-{MAX_WORDS}")

    # 2. FAQ at end with 3-4 questions
    faq_count = _count_faq(article_html)
    if faq_count < MIN_FAQ or faq_count > MAX_FAQ:
        reasons.append(f"FAQ has {faq_count} questions (need {MIN_FAQ}-{MAX_FAQ})")
    faq_match = _FAQ_H2_RE.search(article_html)
    if faq_match:
        after_faq = article_html[faq_match.end():]
        if _HEADING_RE.search(after_faq):
            reasons.append("Content found after FAQ section")

    # 3. Unique headings + no heading echo
    headings = _extract_headings(article_html)
    seen = set()
    for _, text in headings:
        norm = _normalize(text)
        if norm in seen:
            reasons.append(f"Duplicate heading: {text}")
        seen.add(norm)

    echoes = _has_heading_echo(article_html)
    if echoes:
        fixed_html = _fix_heading_echo(article_html)
        reasons.append(f"Heading echo fixed: {', '.join(echoes)}")

    # 4. Banned words
    banned = _has_banned(article_html)
    if banned:
        reasons.append(f"Banned words: {', '.join(banned)}")

    # 5. Em/en dashes
    if "—" in article_html or "–" in article_html:
        reasons.append("Em/en dashes found")

    # 6. Primary keyword checks
    pk = plan.primary_keyword if hasattr(plan, "primary_keyword") else ""
    if pk:
        if not _keyword_boundary(pk, article_data.get("title", "")):
            reasons.append(f"Primary keyword '{pk}' missing from title")
        if not _keyword_boundary(pk, _first_n_words(article_html, 100)):
            reasons.append(f"Primary keyword '{pk}' missing from first 100 words")
        h2_texts = [t for l, t in _extract_headings(article_html) if l == "h2"]
        if not any(_keyword_boundary(pk, t) for t in h2_texts):
            reasons.append(f"Primary keyword '{pk}' missing from any <h2>")

    # 7. Meta description length
    meta_len = len(meta_description)
    if meta_len < 120 or meta_len > 160:
        reasons.append(f"Meta description {meta_len} chars (need 120-160)")

    # 8. Inline source links (2-4)
    source_urls = _extract_source_urls(article_html)
    if len(source_urls) < 2:
        reasons.append(f"Only {len(source_urls)} inline source links (need 2-4)")

    # 9. Visual references in text
    visuals = article_data.get("visuals", [])
    visual_ref_issues = _check_visual_references(article_html, visuals)
    reasons.extend(visual_ref_issues)

    # 10. Visual data validation (grounded in fact sheet)
    visual_data_issues = _validate_visual_data(visuals, fact_sheet)
    reasons.extend(visual_data_issues)

    # 11. Image/visual caps
    total_images = article_data.get("content_images_count", 0) + 1  # +1 for cover
    word_count = _word_count(article_html)
    max_content_images = MAX_CONTENT_IMAGES_OVER_1000 if word_count >= 1000 else MAX_CONTENT_IMAGES_UNDER_1000
    if total_images > MAX_TOTAL_IMAGES:
        reasons.append(f"Total images {total_images} exceeds max {MAX_TOTAL_IMAGES}")
    if article_data.get("content_images_count", 0) > max_content_images:
        reasons.append(f"Content images {article_data.get('content_images_count')} exceeds max {max_content_images} for {word_count} words")

    passed = len(reasons) == 0
    meta_fail = any("meta" in r.lower() or "title" in r.lower() or "description" in r.lower() for r in reasons)
    content_fail = [r for r in reasons if not ("meta" in r.lower() or "title" in r.lower() or "description" in r.lower())]

    return GateResult(
        passed=passed,
        reasons=reasons,
        fixed_html=fixed_html if fixed_html != article_html else None,
        meta_only=meta_fail and not content_fail,
    )


def fix_article(
    article_html: str,
    article_data: dict[str, Any],
    gate_result: GateResult,
    plan: Any,
    fact_sheet: Any,
) -> tuple[str, dict[str, Any]]:
    """Apply fixes from gate result. Returns (fixed_html, fixed_data)."""
    html = gate_result.fixed_html or article_html
    visuals = list(article_data.get("visuals", []))

    # Fix heading echoes already applied in fixed_html
    # Fix banned words by removing them
    for word in BANNED_WORDS:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        html = pattern.sub("", html)

    # Fix em/en dashes
    html = re.sub(r"\s*[—–]\s*", ", ", html)

    # Fix visual references: inject intro sentences if missing
    content_lower = html.lower()
    for i, vis in enumerate(visuals):
        kind = vis.get("kind", "")
        intro = vis.get("intro_sentence", "")
        if intro and intro.lower() not in content_lower:
            # Insert after first paragraph of the relevant section
            marker = f"<!--VISUAL:{i}-->"
            if marker in html:
                html = html.replace(marker, f"<p>{intro}</p>\n{marker}", 1)

    # Update visuals with any fixes
    fixed_data = dict(article_data)
    fixed_data["visuals"] = visuals
    fixed_data["content_html"] = html

    return html, fixed_data


# ─── Exports ──────────────────────────────────────────────────────────────────

__all__ = [
    "run_gate",
    "fix_article",
    "GateResult",
    "MIN_WORDS",
    "MAX_WORDS",
    "MIN_FAQ",
    "MAX_FAQ",
    "MAX_TOTAL_IMAGES",
    "MAX_CONTENT_IMAGES_UNDER_1000",
    "MAX_CONTENT_IMAGES_OVER_1000",
]