"""Single-call dynamic article writer for the SEO/AEO/GEO pipeline.

Nemotron writes the complete article as strict JSON (with Gemini as the
automatic fallback). The function then deterministically assembles clean HTML,
normalises FAQ answers to 40-60 words and guarantees the structure the SEO
validator expects.
"""

from __future__ import annotations

import html as html_lib
import re
from typing import Any

from app.autoblog.internal_links import ensure_internal_links
from app.autoblog.llm.clients import (
    BlogPipelineError,
    extract_json,
    gemini_final_pass,
    nemotron_with_gemini_fallback,
)
from app.autoblog.models import (
    DynamicArticleDraft,
    InternalLink,
    KeywordPlan,
    TopicTrend,
)
from app.autoblog.research import FactSheet


TARGET_MIN_WORDS = 1250
TARGET_MAX_WORDS = 1700
MIN_FAQ_QUESTIONS = 3
MAX_FAQ_QUESTIONS = 5
FAQ_MIN_WORDS = 40
FAQ_MAX_WORDS = 60


_SYSTEM = """You are the lead writer for ATLAS, an AI learning OS for students.

Write one complete, publish-ready article as strict JSON. Do not wrap the JSON itself in markdown fences. Mermaid fenced blocks inside section HTML are allowed and required for data topics.

Required schema:
{
  "title": "...",
  "meta_title": "...",
  "meta_description": "...",
  "excerpt": "...",
  "tags": ["...", "..."],
  "intro": "<p>...</p>",
  "sections": [
    {"heading": "...", "level": 2, "html": "<p>...</p><ul>...</ul>"},
    {"heading": "...", "level": 3, "html": "<p>...</p>"}
  ],
  "conclusion": "<h2>...</h2><p>...</p>",
  "faq": [
    {"question": "...", "answer": "..."}
  ],
  "chart": {
    "title": "...",
    "labels": ["...", "..."],
    "values": [1, 2],
    "unit": "...",
    "source_url": "...",
    "alt": "..."
  }
}

VISUALS REQUIREMENT (MANDATORY FOR DATA/COMPARISON TOPICS):
If the topic involves financial data, benchmarks, comparisons, or statistics, you MUST generate the visuals directly in the content:
1. DATA TABLE: Create a clean, well-formatted HTML or Markdown table with realistic, researched metrics.
2. MERMAID CHART: Immediately after the table, include a Mermaid.js code block wrapped in ```mermaid and ```.
DYNAMIC RULE: Extract the EXACT categories and values from the HTML table you just generated. If your table has 2 items, the chart must have 2. If it has 5, the chart must have 5. Do not use fixed examples.
Format:
```mermaid
xychart-beta
    title "Your Dynamic Chart Title"
    x-axis ["Item 1", "Item 2", "Item 3", ...]
    y-axis "Metric Name" 0 --> MaxValue
    bar [val1, val2, val3, ...]
```
Do not use placeholder text. Use the actual facts and numbers from the research phase.
For this renderer, prefer a real HTML `<table>` because table markup is preserved reliably.

RULES:
- title: 45-65 characters, specific, no clickbait, no colon-stuffed keyword string
- meta_title: 50-60 characters and starts with the primary keyword
- meta_description: 150-160 characters, contains the primary keyword and one clear call to action
- excerpt: 140-160 characters, plain text
- intro: 2 short paragraphs, no heading. Answer the reader's core question in the first 2 sentences (AEO)
- sections: 5-7 body sections. At least 5 use level 2 and at least 2 use level 3
- section html: 120-260 words of prose each, short paragraphs (1-3 sentences), bullet lists where useful. No <h1> and no <img> tags. HTML tables and Mermaid blocks are allowed for data topics.
- conclusion: starts with <h2> and summarises the practical takeaway in 80-140 words
- faq: exactly 4 questions. Each answer MUST be 40-60 words, direct, plain text (no HTML). Each FAQ answer must end with a UNIQUE, specific insight related to that exact question. DO NOT repeat phrases like "The practical test is whether..." or "Treat the tool as a tutor..." across different answers. STRICT RULE: Do NOT use generic filler sentences in FAQ answers. You are BANNED from using phrases like: "Students should verify claims", "The next step is to try", "A useful check is to compare", or "The answer depends on the task". Each answer must end with a direct, specific fact related to that exact question.
- chart: only when the fact sheet contains at least 3 real numbers. Labels must name the metrics, values must be numbers, source_url must come from the fact sheet
- Use the primary keyword in the first 100 words, in at least 2 section headings, and once in the conclusion
- HUMAN FLOW: vary sentence length and rhythm. Mix short, punchy sentences (5-12 words) with longer explanatory ones (18-30 words). Do not start three consecutive sentences with the same subject, and avoid starting every paragraph with the subject. Use natural transitions such as "Still," "In practice," "That said," "For students," and "The catch is."
- Include 2-4 external links to source URLs. Use exactly this HTML shape: <a href="https://example.com/page" target="_blank" rel="noopener">descriptive anchor text</a>. Every citation must be a real anchor with a URL from the fact sheet. When citing, never write the bare word "source" or a raw URL as plain text.
- Include 2 internal links from the supplied list using relative hrefs
- For data/comparison topics, place the HTML table and the Mermaid block inside the most relevant section HTML, with the Mermaid block immediately after the table.
- Insert the exact marker <!--DATA_VISUAL--> after a paragraph in the section where the chart belongs only when you are not including both an HTML table and a Mermaid chart; the fallback visual builder uses this marker.
- Escape newlines inside JSON string values so the returned JSON stays valid.
- Ground every claim, number and quote in the fact sheet. Never invent data, studies or quotes
- BANNED words: delve, moreover, furthermore, additionally, in conclusion, it's worth noting, it is important to note, in today's fast-paced world, ever-evolving landscape, landscape of, tapestry, testament, unlock, unleash, dive deep, dive into, game-changer, revolutionize, revolutionary, seamless, robust, cutting-edge, elevate, harness the power, in a world where, boasts, myriad, plethora, navigating, embarking, fostering, realm
- NEVER use em dashes or en dashes. Use commas, colons, or parentheses
- Write for curious students, not for search engines
- Return ONLY the JSON object
"""


_REPAIR_SYSTEM = """You repair malformed JSON from another language model.

Return ONLY a valid JSON object with the same content and these keys:
title, meta_title, meta_description, excerpt, tags, intro, sections, conclusion, faq, chart.
Preserve every HTML table and Mermaid block exactly. Do not add commentary. Do not wrap the JSON in markdown fences."""

_BANNED_FAQ_PHRASES: tuple[str, ...] = (
    "students should verify claims",
    "the next step is to try",
    "a useful check is to compare",
    "the answer depends on the task",
)

def _strip_banned_phrases(text: str) -> str:
    """Remove any banned filler phrases from FAQ answers."""
    for phrase in _BANNED_FAQ_PHRASES:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^\s*[.,;:]\s*", "", text)
    text = re.sub(r"\s*[.,;:]\s*$", "", text)
    return text.strip()


_FILLER_SENTENCES: tuple[str, ...] = (
    "The evidence points to a clear conclusion you can act on.",
    "The data supports a specific outcome you can measure.",
    "The result is a concrete takeaway you can apply immediately.",
    "The benchmark gives you a definitive reference point.",
)

_FAQ_CLOSERS: tuple[str, ...] = (
    "Test it on one real study task before changing your routine.",
    "Check the original source before trusting the claim.",
    "Judge the feedback quality, not the polish of the demo.",
    "Measure the result against how you studied before.",
    "Keep a short log of what actually changed.",
    "Students should compare the new workflow with their old routine and note what changed.",
    "A small trial with a clear before-and-after comparison is safer than a broad overhaul.",
)


# ─── Text helpers ─────────────────────────────────────────────────────────────


def _clean_text(value: Any, limit: int = 300) -> str:
    clean = re.sub(r"\s+", " ", str(value or "")).strip()
    clean = clean.replace("\u2014", ",").replace("\u2013", ",")
    clean = re.sub(r"\s*,\s*,+", ", ", clean)
    return clean[:limit].strip(" ,.;:-")


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", str(value or ""))


def word_count(html_text: str) -> int:
    return len(_strip_tags(html_text).split())


_MERMAID_FENCE_RE = re.compile(
    r"```mermaid[ \t]*\r?\n?(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_MERMAID_CODE_RE = re.compile(
    r"""<pre\b[^>]*>\s*<code\b[^>]*\bclass\s*=\s*["'][^"']*language-mermaid[^"']*["'][^>]*>(.*?)</code>\s*</pre>""",
    re.IGNORECASE | re.DOTALL,
)
_MARKDOWN_TABLE_RE = re.compile(
    r"(?m)(?:^[ \t]*\|.*\|[ \t]*\r?\n)"
    r"(?:^[ \t]*\|[ \t:|-]+\|[ \t]*\r?\n)"
    r"(?:^[ \t]*\|.*\|[ \t]*(?:\r?\n|$))+",
)


def _mermaid_block(code: str) -> str:
    decoded = html_lib.unescape(str(code or "")).strip()
    if not decoded:
        return ""
    return f'<pre class="mermaid">{html_lib.escape(decoded)}</pre>'


def _normalize_mermaid_blocks(fragment: str) -> str:
    """Convert fenced Mermaid source into a preserveable HTML block."""
    clean = _MERMAID_FENCE_RE.sub(
        lambda match: _mermaid_block(match.group(1)),
        str(fragment or ""),
    )
    return _MERMAID_CODE_RE.sub(
        lambda match: _mermaid_block(match.group(1)),
        clean,
    )


def _markdown_table_block(source: str) -> str:
    lines = [line.strip() for line in source.strip().splitlines()]
    if len(lines) < 3:
        return source

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    header = cells(lines[0])
    rows = [cells(line) for line in lines[2:]]
    if not header or any(len(row) != len(header) for row in rows):
        return source

    head_html = "".join(f"<th>{html_lib.escape(cell)}</th>" for cell in header)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{html_lib.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>"


def _normalize_markdown_tables(fragment: str) -> str:
    return _MARKDOWN_TABLE_RE.sub(
        lambda match: _markdown_table_block(match.group(0)),
        str(fragment or ""),
    )


def _sanitize_fragment(fragment: str) -> str:
    """Strip dangerous or duplicate-document tags from an LLM HTML fragment."""
    clean = _normalize_mermaid_blocks(str(fragment or "").strip())
    clean = _normalize_markdown_tables(clean)
    clean = re.sub(
        r"<(script|style|iframe|form|input)\b[^>]*>.*?</\1>",
        "",
        clean,
        flags=re.IGNORECASE | re.DOTALL,
    )
    clean = re.sub(
        r"<(script|style|iframe|form|input)\b[^>]*/?>",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"<h1\b[^>]*>.*?</h1>", "", clean, flags=re.IGNORECASE | re.DOTALL)
    clean = re.sub(r"\s+on\w+\s*=\s*\"[^\"]*\"", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+on\w+\s*=\s*'[^']*'", "", clean, flags=re.IGNORECASE)
    return clean.strip()


def _strip_leading_heading(fragment: str) -> str:
    return re.sub(
        r"^\s*<h[1-6][^>]*>.*?</h[1-6]>",
        "",
        fragment,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def _first_heading(fragment: str) -> str:
    match = re.search(r"<h[2-6][^>]*>(.*?)</h[2-6]>", fragment, re.IGNORECASE | re.DOTALL)
    return _clean_text(_strip_tags(match.group(1)), 120) if match else ""


def _keyword_in(text: str, keyword: str) -> bool:
    needle = _clean_text(keyword).lower()
    if not needle:
        return False
    haystack = _strip_tags(text).lower()
    pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
    return re.search(pattern, haystack) is not None


def _display_keyword(keyword: str) -> str:
    acronyms = {"ai", "api", "gpt", "llm", "seo", "geo", "aeo", "sql", "ux", "ui", "ml", "nlp"}
    words = []
    for word in _clean_text(keyword).split():
        words.append(word.upper() if word.lower() in acronyms else word.capitalize())
    return " ".join(words) or "AI learning"


def _format_facts(fact_sheet: FactSheet) -> str:
    key_points = "\n".join(f"- {p}" for p in fact_sheet.raw_snippets[:8]) or "(none)"
    summary = " ".join(fact_sheet.claims[:4]) or "(no summary)"
    numbers = "\n".join(
        f"- {item.get('value')} {item.get('unit')}: {item.get('context', '')[:120]}"
        for item in fact_sheet.numbers[:6]
    ) or "(none)"
    sources = "\n".join(f"- {url}" for url in fact_sheet.source_urls[:6]) or "(none)"
    return (
        f"Key Data Points:\n{key_points}\n\n"
        f"Summary:\n{summary}\n\n"
        f"Key Numbers:\n{numbers}\n\n"
        f"Sources:\n{sources}"
    )


def _format_links(links: list[InternalLink]) -> str:
    return "\n".join(f"- {link.title} -> {link.url}" for link in links) or "(none)"


_EXTERNAL_HREF_RE = re.compile(
    r"""<a\b[^>]*\bhref=["'](https?://[^"']+)["']""",
    re.IGNORECASE,
)


def _external_host(url: str) -> str:
    host = re.sub(r"^https?://", "", url.strip(), flags=re.IGNORECASE)
    return host.split("/", 1)[0].removeprefix("www.")


def ensure_external_links(
    content_html: str,
    fact_sheet: FactSheet,
    minimum: int = 2,
) -> str:
    """Guarantee proper external anchors when the model forgets citations."""
    existing = {
        url.rstrip("/")
        for url in _EXTERNAL_HREF_RE.findall(content_html)
    }
    needed = max(minimum - len(existing), 0)
    if not needed:
        return content_html

    candidates = [
        url
        for url in fact_sheet.source_urls
        if url.startswith("http") and url.rstrip("/") not in existing
    ]
    if not candidates:
        return content_html

    anchors = []
    for url in candidates[:needed]:
        label = html_lib.escape(_external_host(url) or "Original source")
        safe_url = html_lib.escape(url, quote=True)
        anchors.append(
            f'<a href="{safe_url}" target="_blank" rel="noopener">{label}</a>'
        )

    block = f"<p><strong>References:</strong> {', '.join(anchors)}.</p>"
    faq_match = re.search(
        r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b).*?</h2>",
        content_html,
        re.IGNORECASE | re.DOTALL,
    )
    if faq_match:
        return (
            content_html[: faq_match.start()]
            + block
            + "\n"
            + content_html[faq_match.start():]
        )
    return content_html + "\n" + block


def _build_prompt(
    trend: TopicTrend,
    keywords: KeywordPlan,
    fact_sheet: FactSheet,
    links: list[InternalLink],
    issues: list[str] | None,
    attempt: int,
) -> str:
    secondary = ", ".join(keywords.secondary_keywords) or "(none)"
    long_tail = ", ".join(keywords.long_tail_keywords[:5]) or "(none)"
    lsi = ", ".join(keywords.lsi_keywords[:6]) or "(none)"
    has_numbers = len(fact_sheet.numbers) >= 3

    issue_block = ""
    if issues:
        issue_lines = "\n".join(f"- {issue}" for issue in issues[:12])
        issue_block = (
            "\nPREVIOUS DRAFT FAILED SEO VALIDATION. Rewrite the whole article and fix every issue:\n"
            f"{issue_lines}\n"
        )

    return f"""Write attempt {attempt}.

TOPIC
Headline: {trend.headline}
Angle: {trend.angle}
Category: {trend.category}
Why now: {trend.why_now}

KEYWORDS
Primary: {keywords.primary_keyword}
Secondary: {secondary}
Long-tail: {long_tail}
LSI terms: {lsi}

INTERNAL LINKS TO USE (relative URLs):
{_format_links(links)}

FACT SHEET (the only facts you may use):
{_format_facts(fact_sheet)}

CHART DECISION: {"include a chart object grounded in the numbers above" if has_numbers else "return chart as null because there are fewer than 3 reliable numbers"}
TARGET LENGTH: {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS} words of body content.

QUALITY REMINDERS:
- Each FAQ answer must end with a UNIQUE, specific insight related to that exact question. DO NOT repeat phrases like "The practical test is whether..." or "Treat the tool as a tutor..." across different answers. STRICT RULE: Do NOT use generic filler sentences in FAQ answers. You are BANNED from using phrases like: "Students should verify claims", "The next step is to try", "A useful check is to compare", or "The answer depends on the task". Each answer must end with a direct, specific fact related to that exact question.
- Every external citation must be a full HTML anchor with a real URL from the fact sheet, never the bare word "source".
- Vary sentence length and paragraph openings. Mix short and long sentences; do not start every paragraph with the subject.
{issue_block}
Write the full article JSON now."""


def _parse_payload(text: str) -> dict[str, Any]:
    parsed = extract_json(text)
    data = parsed.data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        raise BlogPipelineError("Writer returned invalid JSON.")
    return data


# ─── Section assembly ─────────────────────────────────────────────────────────


def _build_sections(raw_sections: Any) -> list[dict[str, str]]:
    if not isinstance(raw_sections, list):
        return []

    sections: list[dict[str, str]] = []
    for index, item in enumerate(raw_sections):
        if not isinstance(item, dict):
            continue
        source_html = str(item.get("html") or item.get("content") or "").strip()
        source_html = _sanitize_fragment(source_html)
        heading = _clean_text(
            item.get("heading") or item.get("title") or _first_heading(source_html),
            140,
        )
        body = _strip_leading_heading(source_html)

        level_raw = str(item.get("level") or "").strip().lower()
        level = 3 if level_raw in {"3", "h3", "subheading"} else 2

        if not heading and not body:
            continue
        if not heading:
            heading = f"Key insight {index + 1}"

        tag = "h3" if level == 3 else "h2"
        heading_html = f"<{tag}>{html_lib.escape(heading)}</{tag}>"
        sections.append(
            {
                "heading": heading,
                "level": str(level),
                "html": f"{heading_html}\n{body}".strip(),
            }
        )

    if sections and all(section["level"] == "3" for section in sections):
        sections[0]["level"] = "2"
        sections[0]["html"] = re.sub(
            r"^<h3>", "<h2>", sections[0]["html"], count=1
        ).replace("</h3>", "</h2>", 1)
    return sections


def _normalize_answer(
    answer: str,
    *,
    used_fillers: set[str] | None = None,
) -> str:
    clean = re.sub(r"\s+", " ", _strip_tags(answer)).strip()
    if not clean:
        clean = "This is a fast-moving area, so the practical answer starts with the source."

    words = clean.split()
    if len(words) > FAQ_MAX_WORDS:
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        kept: list[str] = []
        total = 0
        for sentence in sentences:
            size = len(sentence.split())
            if total + size > FAQ_MAX_WORDS:
                break
            kept.append(sentence)
            total += size
        clean = " ".join(kept).strip() if kept else " ".join(words[:FAQ_MAX_WORDS]).strip()
        if not clean.endswith((".", "!", "?")):
            clean += "."

    while len(clean.split()) < FAQ_MIN_WORDS:
        added = False
        for filler in _FILLER_SENTENCES:
            if used_fillers is not None and filler in used_fillers:
                continue
            candidate = f"{filler} {clean}".strip()
            if len(candidate.split()) <= FAQ_MAX_WORDS:
                clean = candidate
                if used_fillers is not None:
                    used_fillers.add(filler)
                added = True
                break
        if not added:
            break

    if not clean.endswith((".", "!", "?")):
        clean += "."
    clean = _strip_banned_phrases(clean)
    return clean


def _split_sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.strip())
        if part.strip()
    ]


def _ending_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _dedupe_faq_endings(faq: list[dict[str, str]]) -> list[dict[str, str]]:
    """Replace repeated FAQ closing sentences with unique conclusions."""
    used: set[str] = set()

    for item in faq:
        answer = str(item.get("answer") or "").strip()
        sentences = _split_sentences(answer)
        if not sentences:
            continue

        key = _ending_key(sentences[-1])
        if key not in used:
            used.add(key)
            continue

        base = " ".join(sentences[:-1]).strip()
        replacement = ""
        for closer in _FAQ_CLOSERS:
            closer_key = _ending_key(closer)
            if closer_key in used:
                continue

            candidate = f"{base} {closer}".strip()
            if len(candidate.split()) > FAQ_MAX_WORDS:
                while len(candidate.split()) > FAQ_MAX_WORDS and base:
                    base_sentences = _split_sentences(base)
                    if len(base_sentences) <= 1:
                        break
                    base = " ".join(base_sentences[:-1]).strip()
                    candidate = f"{base} {closer}".strip()

            if len(candidate.split()) > FAQ_MAX_WORDS:
                available = FAQ_MAX_WORDS - len(closer.split())
                if available > 0:
                    base_words = base.split()
                    candidate = (
                        f"{' '.join(base_words[:available]).rstrip('.,;:')}. "
                        f"{closer}"
                    )

            if not base:
                available = FAQ_MAX_WORDS - len(closer.split())
                if available > 0:
                    words = answer.split()
                    candidate = (
                        f"{' '.join(words[:available]).rstrip('.,;:')}. "
                        f"{closer}"
                    )

            if FAQ_MIN_WORDS <= len(candidate.split()) <= FAQ_MAX_WORDS:
                replacement = candidate
                used.add(closer_key)
                break

        if replacement:
            item["answer"] = replacement
        else:
            used.add(key)

    return faq


def _fallback_faq(
    trend: TopicTrend,
    keywords: KeywordPlan,
    fact_sheet: FactSheet,
    *,
    used_fillers: set[str] | None = None,
) -> list[dict[str, str]]:
    primary = keywords.primary_keyword
    display = _display_keyword(primary)
    claims = fact_sheet.claims or [trend.why_now, trend.angle]
    answers = [
        f"{display} refers to {claims[0]}",
        f"It matters because {claims[min(1, len(claims) - 1)]}",
        f"The safest approach is to use {primary} for feedback and practice, then verify every claim against the original source before trusting it.",
        f"Watch for measurable changes in study time, recall and confidence, and compare them with how you learned before using {primary}.",
    ]
    questions = [
        f"What is {primary}?",
        f"Why does {primary} matter for students?",
        f"How can students use {primary} without cutting corners?",
        f"What should students watch next with {primary}?",
    ]
    return [
        {
            "question": question,
            "answer": _normalize_answer(answer, used_fillers=used_fillers),
        }
        for question, answer in zip(questions, answers)
    ]


def _build_faq(
    raw_faq: Any,
    trend: TopicTrend,
    keywords: KeywordPlan,
    fact_sheet: FactSheet,
) -> list[dict[str, str]]:
    faq: list[dict[str, str]] = []
    used_fillers: set[str] = set()
    if isinstance(raw_faq, list):
        for item in raw_faq:
            if not isinstance(item, dict):
                continue
            question = _clean_text(item.get("question") or item.get("q"), 200)
            answer = str(item.get("answer") or item.get("a") or "").strip()
            if question and answer:
                faq.append(
                    {
                        "question": question,
                        "answer": _normalize_answer(
                            answer,
                            used_fillers=used_fillers,
                        ),
                    }
                )

    if len(faq) < MIN_FAQ_QUESTIONS:
        existing = {item["question"].lower() for item in faq}
        for fallback in _fallback_faq(
            trend,
            keywords,
            fact_sheet,
            used_fillers=used_fillers,
        ):
            if fallback["question"].lower() not in existing:
                faq.append(fallback)
                existing.add(fallback["question"].lower())
            if len(faq) >= MIN_FAQ_QUESTIONS:
                break

    return _dedupe_faq_endings(faq[:MAX_FAQ_QUESTIONS])


def _parse_chart(raw_chart: Any, fact_sheet: FactSheet) -> dict[str, Any] | None:
    if not isinstance(raw_chart, dict):
        return None
    labels = raw_chart.get("labels")
    values = raw_chart.get("values")
    if not isinstance(labels, list) or not isinstance(values, list):
        return None
    if len(labels) != len(values) or not 3 <= len(labels) <= 8:
        return None

    clean_labels: list[str] = []
    clean_values: list[float] = []
    for label, value in zip(labels, values):
        try:
            number = float(str(value).replace(",", "").replace("%", "").strip())
        except (TypeError, ValueError):
            return None
        text = _clean_text(label, 60)
        if not text:
            return None
        clean_labels.append(text)
        clean_values.append(number)

    source_url = _clean_text(raw_chart.get("source_url"), 500)
    if not source_url.startswith("http"):
        source_url = fact_sheet.source_urls[0] if fact_sheet.source_urls else ""

    return {
        "title": _clean_text(raw_chart.get("title") or "Key numbers", 120),
        "labels": clean_labels,
        "values": clean_values,
        "unit": _clean_text(raw_chart.get("unit"), 30),
        "source_url": source_url,
        "alt": _clean_text(raw_chart.get("alt"), 200),
    }


def _has_native_data_visuals(content_html: str) -> bool:
    has_table = bool(re.search(r"<table\b", content_html, re.IGNORECASE))
    has_mermaid = bool(
        re.search(
            r"""<[a-z][^>]*\bclass\s*=\s*["'][^"']*\bmermaid\b""",
            content_html,
            re.IGNORECASE,
        )
        or re.search(r"```mermaid\b", content_html, re.IGNORECASE)
    )
    return has_table and has_mermaid


def _ensure_data_marker(content_html: str, chart: dict[str, Any] | None) -> str:
    if _has_native_data_visuals(content_html):
        return content_html.replace("<!--DATA_VISUAL-->", "")
    if not chart or "<!--DATA_VISUAL-->" in content_html:
        return content_html
    faq_match = re.search(
        r"<h2[^>]*>\s*(?:FAQ\b|Frequently Asked Questions\b).*?</h2>",
        content_html,
        re.IGNORECASE | re.DOTALL,
    )
    if faq_match:
        return content_html[: faq_match.start()] + "<!--DATA_VISUAL-->\n" + content_html[faq_match.start():]
    first_h2 = re.search(r"</h2>", content_html, re.IGNORECASE)
    if first_h2:
        return content_html[: first_h2.end()] + "\n<!--DATA_VISUAL-->\n" + content_html[first_h2.end():]
    return content_html + "\n<!--DATA_VISUAL-->"


def _sanitize_tags(raw_tags: Any, keywords: KeywordPlan) -> list[str]:
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            clean = _clean_text(tag, 30).strip("#").lower()
            if clean and clean not in tags:
                tags.append(clean)
    for keyword in keywords.secondary_keywords[:3]:
        clean = _clean_text(keyword, 30).lower()
        if clean and clean not in tags:
            tags.append(clean)
    if not tags:
        tags = ["ai", "learning", "students"]
    return tags[:6]


# ─── Main entry point ─────────────────────────────────────────────────────────


async def write_dynamic_article(
    trend: TopicTrend,
    keywords: KeywordPlan,
    fact_sheet: FactSheet,
    links: list[InternalLink],
    *,
    issues: list[str] | None = None,
    attempt: int = 1,
) -> DynamicArticleDraft:
    """Generate one full draft, optionally rewriting to fix SEO issues."""
    prompt = _build_prompt(trend, keywords, fact_sheet, links, issues, attempt)

    result = await nemotron_with_gemini_fallback(
        _SYSTEM,
        prompt,
        temperature=0.55 if attempt == 1 else 0.35,
        max_tokens=8000,
    )

    try:
        data = _parse_payload(result.text)
    except Exception:
        print("[AutoBlog] Writer JSON invalid - requesting a Gemini repair pass.", flush=True)
        repair = await gemini_final_pass(
            _REPAIR_SYSTEM,
            result.text[:20000],
            temperature=0.0,
            max_tokens=8000,
        )
        data = _parse_payload(repair.text)

    title = _clean_text(data.get("title") or trend.headline, 120)
    if not title:
        title = trend.headline

    intro = _sanitize_fragment(str(data.get("intro") or data.get("introduction") or ""))
    intro = _strip_leading_heading(intro)
    if not intro:
        intro = f"<p>{html_lib.escape(trend.angle)}</p>"

    if not _keyword_in(intro, keywords.primary_keyword):
        display = html_lib.escape(_display_keyword(keywords.primary_keyword))
        intro = (
            f"<p>{display} is changing how students learn and work with AI.</p>\n{intro}"
        )

    sections = _build_sections(data.get("sections"))
    if not sections:
        raise BlogPipelineError("Writer returned no usable article sections.")

    conclusion = _sanitize_fragment(str(data.get("conclusion") or ""))
    if not conclusion:
        conclusion = (
            f"<h2>Final takeaway</h2>"
            f"<p>{html_lib.escape(trend.angle)}. The practical next step is to test one "
            f"small workflow change, measure the result, and keep what works.</p>"
        )
    elif not re.search(r"<h2\b", conclusion, re.IGNORECASE):
        conclusion = f"<h2>Final takeaway</h2>\n{conclusion}"

    faq = _build_faq(data.get("faq"), trend, keywords, fact_sheet)
    faq_html = "<h2>FAQ</h2>\n" + "\n".join(
        f"<h3>{html_lib.escape(item['question'])}</h3>"
        f"<p>{html_lib.escape(item['answer'])}</p>"
        for item in faq
    )

    chart = _parse_chart(data.get("chart"), fact_sheet)

    content_html = "\n".join(
        [intro, *(section["html"] for section in sections), conclusion, faq_html]
    )
    content_html = _ensure_data_marker(content_html, chart)
    content_html = ensure_external_links(content_html, fact_sheet, minimum=2)
    content_html = ensure_internal_links(content_html, links, minimum=2)

    draft = DynamicArticleDraft(
        title=title,
        intro_html=intro,
        sections=sections,
        conclusion_html=conclusion,
        faq=faq,
        tags=_sanitize_tags(data.get("tags"), keywords),
        content_html=content_html,
        word_count=word_count(content_html),
        meta_title_raw=_clean_text(data.get("meta_title"), 200),
        meta_description_raw=_clean_text(data.get("meta_description"), 300),
        excerpt_raw=_clean_text(data.get("excerpt"), 300),
        chart_spec=chart,
    )

    print(
        f"[AutoBlog] Draft ready: {draft.word_count} words, "
        f"{len(sections)} sections, {len(faq)} FAQs, chart={'yes' if chart else 'no'}",
        flush=True,
    )
    return draft


__all__ = ["write_dynamic_article", "word_count"]
