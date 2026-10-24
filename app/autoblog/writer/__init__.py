"""Writer module for AutoBlog v2 - Article and Meta generation."""

from __future__ import annotations

import asyncio
import random
import re
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.llm.clients import (
    groq_with_gemini_fallback,
    extract_json,
    BlogPipelineError,
)
from app.autoblog.planner import ARTICLE_PROFILES


# ─── System Prompts ───────────────────────────────────────────────────────────


_WRITER_SYSTEM = """You are an expert tech journalist and industry analyst writing for ATLAS, a learning OS for students. Your articles are published on our public blog and must be citable by both Google and AI assistants.

STRICT SEO, AEO, & GEO REQUIREMENTS (MUST FOLLOW):

1. DIRECT ANSWER FIRST (AEO):
   - Answer the core user query directly within the first 2 sentences of the introduction.
   - Never open with fluff such as "In today's fast-paced world" or a long wind-up.

2. STRUCTURE & READABILITY:
   - Use proper HTML tags: <h2>, <h3>, <p>, <ul>, <li>, <strong>, and <table> only when the topic calls for it.
   - Keep paragraphs short (1-3 sentences).
   - Use bullet points for features, steps, or comparisons. At least one bullet list per article.
   - Write 3-5 body sections, each opened by an engaging <h2>; use <h3> where it helps.

3. DYNAMIC FORMATTING & VISUALS (Context-Aware):
   - Analyze the nature of the topic and choose the format that informs best. Never force a table into an article where it does not fit.
   - Theoretical, conceptual, or opinion topics: use narrative paragraphs and bullets; do NOT use a <table>.
   - Research, data, or statistics-heavy topics: MUST include one HTML <table> with <thead>, <tbody>, <tr>, <th> and <td>, plus a one-sentence introduction before it.
   - Comparison or "versus" topics: MUST include one comparison <table> (features, pros and cons, benchmarks) plus a short bullet summary.
   - Visuals: the publishing pipeline places 1-3 highly relevant images automatically, including a conceptual image mid-article. Do NOT output <img> tags, placeholders, or image URLs; just keep the middle sections visual-friendly.
   - Every <h2> and <h3> must be unique and descriptive. Never repeat the same heading text, and never restate a heading's wording as plain text directly beneath it.

4. KEYWORD DENSITY (SEO):
   - Naturally include the exact primary keyword 5-7 times. Never stuff.
   - Include it in the first 100 words, in at least one <h2> heading, and in the closing section.
   - Use semantic/LSI keywords related to the topic naturally.

5. AUTHORITY & CITATIONS (GEO):
   - Cite primary sources naturally in the text (for example, "According to the official documentation..." or "As announced by...").
   - Link 2-3 of the supplied source URLs inline with <a href="..." rel="noopener">.
   - Ground every claim in the research provided. If unsure about a statistic, phrase it generally. Never invent studies or numbers.
   - Maintain an objective, expert, professional tone. E-E-A-T compliant.

6. MANDATORY FAQ SECTION (AEO):
   - At the very end of the article, add an <h2>FAQ</h2> section.
   - Include exactly 3-4 specific, long-tail questions related to the topic, each followed by a concise, direct answer of 40-60 words.
   - Format: <h3>Question?</h3><p>Direct, clear answer.</p>

7. LENGTH:
   - Write between 900 and 1200 words for comprehensive coverage.

VOICE & OUTPUT RULES:
- Write like a sharp, authoritative tech journalist: contractions, direct "you", specific examples. A human wrote this.
- BANNED words and phrases (using any of them fails the brief): delve, delve into, moreover, furthermore, in conclusion, additionally, it's worth noting, it is important to note, in today's fast-paced world, ever-evolving landscape, landscape of, tapestry, testament, testament to, unlock, unleash, dive deep, dive into, game-changer, revolutionize, revolutionary, seamless, robust, cutting-edge, elevate, harness the power, in a world where, boasts, myriad, plethora, navigating, embarking, fostering, realm.
- Never use em dashes (—) or en dashes (–) anywhere in the title, excerpt or body. Use commas, colons or parentheses instead.
- The content_html value is a plain HTML fragment: never <html>, <head> or <body>. The article must end with the FAQ section.

Return ONLY strict JSON (no markdown fences, no commentary before or after):
{"title": "...", "excerpt": "...", "content_html": "...", "tags": ["...", "..."], "visuals": [...]}

- title: max 110 chars, specific and engaging, contains the primary keyword, no clickbait lies.
- excerpt: 130-155 chars of plain text containing the primary keyword (used as the meta description).
- content_html: the full article body as an HTML fragment, ending with the FAQ section.
- tags: 3-5 lowercase topic tags.
- visuals: array of structured visuals. Each visual is an object:
  * For tables: {"kind": "table", "title": "...", "caption": "...", "columns": ["...", "..."], "rows": [["...", "..."], ...], "source_url": "...", "intro_sentence": "...", "takeaway": "..."}
  * For charts: {"kind": "bar"|"line", "title": "...", "caption": "...", "unit": "...", "labels": ["...", "..."], "values": [number, ...], "source_url": "...", "intro_sentence": "...", "takeaway": "..."}
Rules for visuals: use ONLY numbers/facts present in the supplied research. 3-8 data points (charts), 2-6 rows (tables). Every visual needs intro_sentence (a sentence that points to it and says why it matters, e.g. "The chart below shows how X grew from A to B.") and takeaway (one or two sentences interpreting what the reader should conclude, not repeating the numbers). If the research has no usable data, return "visuals": [].
The content_html must contain the marker <!--VISUAL:0--> (and <!--VISUAL:1--> if two) at the exact spot where each visual belongs, directly after a paragraph that sets it up. Do not write <img>, <svg> or tables for visuals yourself.

Tables that are simple parts of the prose may still be written as <table>; structured visuals is for the data ones."""


_META_SYSTEM = """You are an SEO meta-copy expert. From the topic and keywords provided, write ONE catchy meta title and ONE meta description for the blog post.

Return ONLY strict JSON (no fences, no commentary):
{"meta_title": "...", "meta_description": "..."}

- meta_title: catchy and specific, max 110 chars, MUST contain the exact primary keyword supplied by the user, may include a number or power word.
- meta_description: 130-155 chars (never below 120 or above 160), MUST contain the primary keyword, entices a click. Plain text only.
- No em dashes (—) or en dashes (–); use commas or colons instead."""


_TOPIC_SYSTEM = """You are an SEO trend analyst for ATLAS, a learning OS for students.
From the research snippets, pick ONE specific, currently trending, searchable blog topic (not a generic pillar term).

Return ONLY strict JSON (no fences, no commentary):
{"topic": "..."}

- topic: a concrete angle a reader would click, max 120 chars.
- Prefer a specific model, company, research paper, or measurable news event over generic advice.
- Never repeat an angle from the recently published posts supplied in the prompt."""


_KEYWORD_SYSTEM = """You are an SEO content strategist for ATLAS, a learning OS for students.
Extract the top 3 to 5 SEO keywords for the given blog topic.

Return ONLY strict JSON (no fences, no commentary):
{"keywords": ["...", "...", "..."], "article_type": "..."}

- keywords: 3 to 5, each 1-3 words, lowercase, no dupes, primary keyword first.
- article_type: one of: news, explainer, how_to, opinion, comparison, benchmark, financial, growth, research.
  Choose the type that best fits the topic. Prefer specific types (comparison, benchmark, financial, growth, research) when the topic clearly fits.
  If unsure, use "explainer"."""


# ─── Article Writing ──────────────────────────────────────────────────────────


async def write_article(
    research_topic: str,
    primary_keyword: str,
    secondary_keywords: str,
    sources: str,
    seo_feedback: str | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the article via Groq (Gemini fallback on 429/5xx)."""
    user_prompt = (
        f"Trending topic: {research_topic}\n"
        f"Primary keyword: {primary_keyword}\n"
        f"Secondary keywords (weave in naturally): {secondary_keywords}\n"
        f"Today's research (ground the article in these, link 2-3 inline):\n"
        f"{sources}\n\n"
        "Write the article now. Open by answering the reader's core question "
        "directly in the first 2 sentences, with no wind-up. Include the exact "
        "primary keyword 5-7 times naturally, plus once in the first 100 words, "
        "once in an <h2>, and once in the closing section. Use short paragraphs "
        "(1-3 sentences), at least one <ul><li> list, and 3-5 body sections. "
        "End with a mandatory <h2>FAQ</h2> section containing exactly 3-4 "
        "<h3>questions</h3>, each answered in 40-60 words. Target 800-1200 "
        "words. Banned words fail the brief. Write in simple, human-like "
        "language. 100% unique, no plagiarism."
    )
    if profile:
        required = profile.get("required_visuals", ())
        if required:
            visual_hint = ", ".join(required)
            user_prompt += (
                f"\n\nThe article profile requires these structured visuals: {visual_hint}. "
                "If the research contains relevant data, include them in the 'visuals' array. "
                "Place <!--VISUAL:0--> marker in content_html after the paragraph that sets up each visual."
            )
        max_visuals = profile.get("max_visuals", 1)
        if max_visuals:
            user_prompt += f" Maximum {max_visuals} structured visual(s)."

    if seo_feedback:
        user_prompt += (
            "\n\nIMPORTANT: a lightweight SEO checker rejected your previous "
            f"draft. Fix ALL of these issues in this rewrite:\n- {seo_feedback}\n"
            "Keep the same topic, keywords and research; make the fixes "
            "minimally and keep the article between 800 and 1200 words. Keep "
            "the direct 2-sentence opening, 5-7 uses of the primary keyword, "
            "and the final <h2>FAQ</h2> section with 3-4 Q&A pairs."
        )

    text = await groq_with_gemini_fallback(
        _WRITER_SYSTEM, user_prompt, temperature=0.8, max_tokens=4096
    )

    result = extract_json(text)
    data = result.data
    if not isinstance(data, dict):
        raise BlogPipelineError("Article JSON was not an object.")

    content_html = str(data.get("content_html") or data.get("content") or "").strip()
    if not content_html:
        raise BlogPipelineError("Article had no content_html.")

    title = str(data.get("title") or "").strip()[:300] or research_topic[:300]
    plain = re.sub(r"<[^>]+>", " ", content_html)
    plain = re.sub(r"\s+", " ", plain).strip()
    excerpt = str(data.get("excerpt") or "").strip() or plain[:170].rstrip() + "..."

    tags_raw = data.get("tags")
    tags = (
        [str(t).strip().lower() for t in tags_raw if str(t).strip()]
        if isinstance(tags_raw, list)
        else []
    )
    seen: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.append(tag)
    tags = seen[:5]

    visuals_raw = data.get("visuals")
    visuals = (
        visuals_raw
        if isinstance(visuals_raw, list)
        else []
    )

    words = len(plain.split())
    if words < 500:
        print(f"[AutoBlog] WARNING - article is only {words} words (target ~1000)")
    else:
        print(f"[AutoBlog] article written - {words} words")

    return {
        "title": title,
        "excerpt": excerpt,
        "content_html": content_html,
        "tags": tags,
        "visuals": visuals,
        "_json_meta": {"repaired": result.repaired},
    }


# ─── Meta Generation ──────────────────────────────────────────────────────────


async def generate_meta(
    topic: str,
    primary_keyword: str,
    other_keywords: str,
    sources: str,
) -> tuple[str, str]:
    """Generate meta title and description via Groq (Gemini fallback)."""
    try:
        text = await groq_with_gemini_fallback(
            _META_SYSTEM,
            f"Topic: {topic}\n"
            f"Primary keyword: {primary_keyword}\n"
            f"Other keywords: {other_keywords}\n"
            f"Research snippets:\n{sources}",
            temperature=0.7,
            max_tokens=1024,
        )
        result = extract_json(text)
        data = result.data
        if isinstance(data, dict):
            title = str(data.get("meta_title") or "").strip()
            description = str(data.get("meta_description") or "").strip()
            if title and description:
                return title[:300], description[:300]
        print("[AutoBlog] meta chunk returned incomplete JSON - using fallbacks.")
    except Exception as exc:
        print(f"[AutoBlog] meta generation failed (Groq/Gemini): {type(exc).__name__}: {exc}")

    fallback_desc = sources[:170].rstrip() + "..." if sources else topic
    fallback_title = topic
    if primary_keyword and primary_keyword.lower() not in fallback_title.lower():
        fallback_title = f"{fallback_title}: {primary_keyword}"
    return fallback_title[:300], fallback_desc


# ─── Keyword Extraction (Nemotron) ────────────────────────────────────────────


async def extract_keywords_nim(
    topic: str, snippets: list[str]
) -> tuple[list[str], str]:
    """Extract keywords and article type using Nemotron Ultra."""
    blob = "\n".join(
        f"- {s['title']} ({s['url']}): {s['content'][:500]}"
        for s in snippets
    ) or "(no research snippets)"
    try:
        text = await nim_chat(
            _KEYWORD_SYSTEM,
            f"Topic: {topic}\nResearch snippets:\n{blob}",
            temperature=0.3,
            max_tokens=2048,
        )
        result = extract_json(text, prefer=("keywords", "article_type"))
        data = result.data
        if isinstance(data, dict):
            raw_keywords = data.get("keywords")
            keywords = [
                str(k) for k in raw_keywords
            ] if isinstance(raw_keywords, list) else []
            article_type = str(data.get("article_type") or "explainer").strip().lower()
            from app.autoblog.planner import ARTICLE_TYPES, upgrade_article_type
            if article_type not in ARTICLE_TYPES:
                article_type = "explainer"
            article_type = upgrade_article_type(topic, article_type)
            return keywords, article_type
    except Exception as exc:
        print(f"[AutoBlog] keyword extraction failed (Nemotron): {type(exc).__name__}: {exc}")

    # Heuristic fallback
    from app.autoblog.sources import _pad_keywords
    return _pad_keywords([], topic), "explainer"