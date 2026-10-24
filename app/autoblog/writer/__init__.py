"""Writer for AutoBlog v2 - Section-by-section article generation."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.llm.clients import (
    nemotron_with_gemini_fallback,
    gemini_final_pass,
    extract_json,
    BlogPipelineError,
)
from app.autoblog.planner import ArticlePlan, SectionPlan, VisualPlan, ImagePlan
from app.autoblog.research import FactSheet


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class WrittenSection:
    heading: str
    html: str
    word_count: int
    visual_marker: str | None = None  # e.g. "<!--VISUAL:0-->"


@dataclass(slots=True)
class ArticleDraft:
    title: str
    excerpt: str
    sections: list[WrittenSection]
    faq_html: str
    tags: list[str]
    visuals: list[dict[str, Any]]  # rendered visual data for visual_plan
    word_count: int


# ─── System Prompts ───────────────────────────────────────────────────────────


_SECTION_WRITER_SYSTEM = """You are an expert tech journalist writing ONE section of an article for ATLAS (learning OS for students).

CONTEXT:
- Topic: {topic}
- Angle: {angle}
- Tone: {tone} (analytical/conversational/authoritative/investigative)
- Article Type: {article_type}
- Target Audience: Students learning tech/AI
- Primary Keyword: {primary_keyword} (must appear naturally)
- Secondary Keywords: {secondary_keywords}
- Word Budget for THIS section: {word_budget}
- Previous Sections Summary: {prev_summary}

FACT SHEET (you may ONLY use these facts, must link 2-4 source URLs inline):
{claims}
{numbers}
{entities}
Source URLs: {source_urls}

VISUAL FOR THIS SECTION: {visual_info}

SECTION TO WRITE:
Heading: {heading}
Purpose: {purpose}

RULES:
- Write ONLY this section's HTML content (no <h1>, no full article wrapper)
- Start with the heading as <h2> or <h3> as appropriate
- Word count MUST be close to word_budget (±20%)
- Include the primary keyword naturally at least once if not used yet
- Use short paragraphs (1-3 sentences), bullet points where helpful
- If visual_info provided, include the marker <!--VISUAL:n--> right after the paragraph that sets it up
- Link 1-2 source URLs inline with <a href="..." rel="noopener">anchor text</a>
- Ground every claim in the fact sheet. Never invent numbers, studies, or quotes.
- Tone: {tone}. BANNED words: delve, delve into, moreover, furthermore, in conclusion, additionally, it's worth noting, it is important to note, in today's fast-paced world, ever-evolving landscape, landscape of, tapestry, testament, testament to, unlock, unleash, dive deep, dive into, game-changer, revolutionize, revolutionary, seamless, robust, cutting-edge, elevate, harness the power, in a world where, boasts, myriad, plethora, navigating, embarking, fostering, realm.
- NO em dashes (—) or en dashes (–). Use commas, colons, or parentheses.
- Return ONLY the HTML fragment for this section."""


_INTRO_WRITER_SYSTEM = """You are an expert tech journalist writing the INTRODUCTION of an article for ATLAS.

CONTEXT:
- Topic: {topic}
- Angle: {angle}
- Tone: {tone}
- Article Type: {article_type}
- Primary Keyword: {primary_keyword} (MUST appear in first 100 words)
- Word Budget: {word_budget}
- Target Audience: Students learning tech/AI

FACT SHEET (you may ONLY use these facts):
{claims}
{numbers}
{entities}
Source URLs: {source_urls}

RULES:
- Write the introduction HTML (no heading, just paragraphs)
- Answer the reader's core question DIRECTLY in the first 2 sentences (AEO)
- Include primary keyword in first 100 words
- Word count close to word_budget (±20%)
- Link 1-2 source URLs inline
- NO banned words, NO em/en dashes
- Return ONLY the intro HTML fragment (paragraphs only)."""


_CONCLUSION_WRITER_SYSTEM = """You are an expert tech journalist writing the CONCLUSION of an article for ATLAS.

CONTEXT:
- Topic: {topic}
- Angle: {angle}
- Tone: {tone}
- Primary Keyword: {primary_keyword} (MUST appear once in closing)
- Word Budget: {word_budget}
- Article Summary: {article_summary}

RULES:
- Write conclusion HTML (start with <h2>Conclusion</h2> or similar)
- Summarize key takeaway in 1-2 paragraphs
- Include primary keyword once naturally
- Word count close to word_budget (±20%)
- End with a forward-looking or actionable statement
- NO banned words, NO em/en dashes
- Return ONLY the conclusion HTML fragment."""


_FAQ_WRITER_SYSTEM = """You are an expert tech journalist writing the FAQ section for ATLAS.

CONTEXT:
- Topic: {topic}
- Primary Keyword: {primary_keyword}
- Article Type: {article_type}
- Key Facts: {claims}

RULES:
- Write FAQ HTML starting with <h2>FAQ</h2>
- Exactly 3-4 questions, each as <h3>Question?</h3><p>Answer.</p>
- Questions must be specific, long-tail (what students would actually search)
- Answers: 40-60 words each, direct and clear
- Include primary keyword in at least one answer
- NO banned words, NO em/en dashes
- Return ONLY the FAQ HTML fragment."""


_META_FINAL_SYSTEM = """You are an SEO meta-copy expert. Write the final meta title, meta description, excerpt, and 3-5 tags.

CONTEXT:
- Topic: {topic}
- Article Title: {title}
- Primary Keyword: {primary_keyword}
- Secondary Keywords: {secondary_keywords}
- Article Summary: {summary}
- Word Count: {word_count}

Return ONLY strict JSON:
{{
  "meta_title": "...",       // max 110 chars, contains primary keyword, catchy
  "meta_description": "...", // 130-155 chars, contains primary keyword
  "excerpt": "...",          // 130-155 chars plain text, contains primary keyword
  "tags": ["...", "..."]     // 3-5 lowercase tags
}}"""


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _word_count(html_text: str) -> int:
    return len(re.sub(r"<[^>]+>", " ", html_text).split())


def _format_fact_sheet(fact_sheet: FactSheet) -> dict[str, str]:
    claims_text = "\n".join(f"- {c}" for c in fact_sheet.claims[:15]) or "(none)"
    numbers_text = "\n".join(
        f"- {n['value']} {n['unit']}: {n['context']} (source: {n['source_url']})"
        for n in fact_sheet.numbers[:20]
    ) or "(none)"
    entities_text = "\n".join(
        f"- {e['name']} ({e['type']}): {e['context']} (source: {e['source_url']})"
        for e in fact_sheet.entities[:15]
    ) or "(none)"
    sources_text = "\n".join(f"- {u}" for u in fact_sheet.source_urls[:10]) or "(none)"
    return {
        "claims": claims_text,
        "numbers": numbers_text,
        "entities": entities_text,
        "source_urls": sources_text,
    }


def _build_section_prompt(
    plan: ArticlePlan,
    fact_sheet: FactSheet,
    section: SectionPlan,
    prev_summary: str,
    visual_info: str,
) -> str:
    fs = _format_fact_sheet(fact_sheet)
    secondary = ", ".join(plan.secondary_keywords) if plan.secondary_keywords else "(none)"
    return _SECTION_WRITER_SYSTEM.format(
        topic=plan.topic,
        angle=plan.angle,
        tone=plan.tone,
        article_type=plan.article_type,
        primary_keyword=plan.primary_keyword,
        secondary_keywords=secondary,
        word_budget=section.word_budget,
        prev_summary=prev_summary or "(this is the first body section)",
        claims=fs["claims"],
        numbers=fs["numbers"],
        entities=fs["entities"],
        source_urls=fs["source_urls"],
        visual_info=visual_info or "(no visual for this section)",
        heading=section.heading,
        purpose=section.purpose,
    )


# ─── Section Writers ──────────────────────────────────────────────────────────


async def write_intro(
    plan: ArticlePlan,
    fact_sheet: FactSheet,
    intro_budget: int,
) -> str:
    """Write the introduction (no heading, direct answer first)."""
    fs = _format_fact_sheet(fact_sheet)
    secondary = ", ".join(plan.secondary_keywords) if plan.secondary_keywords else "(none)"
    prompt = _INTRO_WRITER_SYSTEM.format(
        topic=plan.topic,
        angle=plan.angle,
        tone=plan.tone,
        article_type=plan.article_type,
        primary_keyword=plan.primary_keyword,
        secondary_keywords=secondary,
        word_budget=intro_budget,
        claims=fs["claims"],
        numbers=fs["numbers"],
        entities=fs["entities"],
        source_urls=fs["source_urls"],
    )
    result = await nemotron_with_gemini_fallback(prompt, "", temperature=0.5, max_tokens=1024)
    html = result.text.strip()
    # Ensure no heading in intro
    html = re.sub(r"^<h[1-6][^>]*>.*?</h[1-6]>", "", html, flags=re.IGNORECASE | re.DOTALL)
    return html


async def write_section(
    plan: ArticlePlan,
    fact_sheet: FactSheet,
    section: SectionPlan,
    prev_summary: str,
    visual_plan: list[VisualPlan],
    visual_index: int,
) -> WrittenSection:
    """Write one body section."""
    # Find matching visual
    visual_info = "(no visual for this section)"
    visual_marker = None
    if section.needs_visual and visual_index < len(visual_plan):
        vis = visual_plan[visual_index]
        visual_info = f"Kind: {vis.kind}, Title: {vis.title}, Purpose: {vis.takeaway}"
        visual_marker = f"<!--VISUAL:{visual_index}-->"

    prompt = _build_section_prompt(plan, fact_sheet, section, prev_summary, visual_info)
    result = await nemotron_with_gemini_fallback(prompt, "", temperature=0.6, max_tokens=2048)
    html = result.text.strip()

    # Ensure heading is present
    if not re.search(r"<h[23][^>]*>", html):
        html = f"<h2>{section.heading}</h2>\n{html}"

    # Insert visual marker if not present
    if visual_marker and visual_marker not in html:
        # Insert after first paragraph
        if "</p>" in html:
            html = html.replace("</p>", f"</p>\n{visual_marker}", 1)
        else:
            html = f"{html}\n{visual_marker}"

    words = _word_count(html)
    # Check word count - if way off, rewrite once
    if words < section.word_budget * 0.6 or words > section.word_budget * 1.5:
        print(f"[AutoBlog] Section '{section.heading}' {words} words (budget {section.word_budget}) - rewriting...", flush=True)
        rewrite_prompt = prompt + f"\n\nPREVIOUS ATTEMPT WAS {words} WORDS (budget {section.word_budget}). REWRITE to hit the budget."
        result = await nemotron_with_gemini_fallback(rewrite_prompt, "", temperature=0.5, max_tokens=2048)
        html = result.text.strip()
        words = _word_count(html)

    return WrittenSection(
        heading=section.heading,
        html=html,
        word_count=words,
        visual_marker=visual_marker,
    )


async def write_conclusion(
    plan: ArticlePlan,
    fact_sheet: FactSheet,
    article_summary: str,
    conclusion_budget: int,
) -> str:
    """Write the conclusion."""
    fs = _format_fact_sheet(fact_sheet)
    prompt = _CONCLUSION_WRITER_SYSTEM.format(
        topic=plan.topic,
        angle=plan.angle,
        tone=plan.tone,
        primary_keyword=plan.primary_keyword,
        word_budget=conclusion_budget,
        article_summary=article_summary,
        claims=fs["claims"],
        numbers=fs["numbers"],
        entities=fs["entities"],
    )
    result = await nemotron_with_gemini_fallback(prompt, "", temperature=0.5, max_tokens=1024)
    return result.text.strip()


async def write_faq(
    plan: ArticlePlan,
    fact_sheet: FactSheet,
) -> str:
    """Write the FAQ section."""
    fs = _format_fact_sheet(fact_sheet)
    prompt = _FAQ_WRITER_SYSTEM.format(
        topic=plan.topic,
        primary_keyword=plan.primary_keyword,
        article_type=plan.article_type,
        claims=fs["claims"],
    )
    result = await nemotron_with_gemini_fallback(prompt, "", temperature=0.4, max_tokens=1024)
    html = result.text.strip()

    # Validate FAQ structure
    faq_count = len(re.findall(r"<h3[^>]*>.*?</h3>", html, re.IGNORECASE | re.DOTALL))
    if faq_count < 3 or faq_count > 4:
        # Fallback: rewrite with strict instruction
        retry_prompt = prompt + f"\n\nPREVIOUS ATTEMPT HAD {faq_count} QUESTIONS. YOU MUST WRITE EXACTLY 3-4 <h3>QUESTION?</h3><p>ANSWER</p> PAIRS."
        result = await gemini_final_pass(retry_prompt, "", temperature=0.3, max_tokens=1024)
        html = result.text.strip()

    return html


async def finalize_meta(
    plan: ArticlePlan,
    article_title: str,
    article_summary: str,
    word_count: int,
) -> dict[str, Any]:
    """Final pass for meta title, description, excerpt, tags (Gemini)."""
    secondary = ", ".join(plan.secondary_keywords) if plan.secondary_keywords else "(none)"
    prompt = _META_FINAL_SYSTEM.format(
        topic=plan.topic,
        title=article_title,
        primary_keyword=plan.primary_keyword,
        secondary_keywords=secondary,
        summary=article_summary[:500],
        word_count=word_count,
    )
    result = await gemini_final_pass(prompt, "", temperature=0.3, max_tokens=1024)
    parsed = extract_json(result.text)
    data = parsed.data
    if not isinstance(data, dict):
        raise BlogPipelineError("Meta final pass returned invalid JSON.")

    meta_title = str(data.get("meta_title", "")).strip()[:110] or article_title[:110]
    meta_description = str(data.get("meta_description", "")).strip()
    excerpt = str(data.get("excerpt", "")).strip()
    tags = [str(t).strip().lower() for t in data.get("tags", []) if str(t).strip()][:5]

    # Ensure primary keyword in meta fields
    if plan.primary_keyword.lower() not in meta_title.lower():
        meta_title = f"{meta_title}: {plan.primary_keyword}"[:110]
    if plan.primary_keyword.lower() not in meta_description.lower():
        meta_description = f"{meta_description} {plan.primary_keyword}"[:160]
    if plan.primary_keyword.lower() not in excerpt.lower():
        excerpt = f"{excerpt} {plan.primary_keyword}"[:160]

    return {
        "meta_title": meta_title,
        "meta_description": meta_description,
        "excerpt": excerpt,
        "tags": tags,
    }


# ─── Main Assembly ────────────────────────────────────────────────────────────


async def write_article(
    plan: ArticlePlan,
    fact_sheet: FactSheet,
) -> ArticleDraft:
    """Write full article section by section, then finalize meta."""
    # Budget allocation: intro ~150, sections per plan, conclusion ~100, FAQ ~200
    intro_budget = 150
    conclusion_budget = 100

    print(f"[AutoBlog] Writing intro (budget {intro_budget})...", flush=True)
    intro_html = await write_intro(plan, fact_sheet, intro_budget)

    print(f"[AutoBlog] Writing {len(plan.outline)} body sections...", flush=True)
    sections: list[WrittenSection] = []
    prev_summaries: list[str] = []
    visual_index = 0

    for i, section in enumerate(plan.outline):
        prev_summary = " ".join(prev_summaries[-2:]) if prev_summaries else ""
        written = await write_section(
            plan, fact_sheet, section, prev_summary, plan.visual_plan, visual_index
        )
        sections.append(written)
        prev_summaries.append(f"{section.heading}: {written.html[:200]}")
        if written.visual_marker:
            visual_index += 1
        print(f"[AutoBlog] Section {i+1}/{len(plan.outline)}: '{section.heading}' - {written.word_count} words", flush=True)

    print(f"[AutoBlog] Writing conclusion (budget {conclusion_budget})...", flush=True)
    article_summary = intro_html + " " + " ".join(s.html for s in sections)
    conclusion_html = await write_conclusion(plan, fact_sheet, article_summary, conclusion_budget)

    print("[AutoBlog] Writing FAQ...", flush=True)
    faq_html = await write_faq(plan, fact_sheet)

    # Assemble full content
    all_sections_html = "\n".join(s.html for s in sections)
    content_html = f"{intro_html}\n{all_sections_html}\n{conclusion_html}\n{faq_html}"

    # Final meta pass
    print("[AutoBlog] Final meta pass...", flush=True)
    meta = await finalize_meta(plan, plan.topic, article_summary, _word_count(content_html))

    # Collect visuals data for rendering
    visuals_data = []
    for vis in plan.visual_plan:
        visuals_data.append({
            "kind": vis.kind,
            "title": vis.title,
            "caption": vis.caption,
            "columns": vis.columns,
            "labels": vis.labels,
            "unit": vis.unit,
            "source_url": vis.source_url,
            "intro_sentence": vis.intro_sentence,
            "takeaway": vis.takeaway,
        })

    return ArticleDraft(
        title=meta["meta_title"],
        excerpt=meta["excerpt"],
        sections=sections,
        faq_html=faq_html,
        tags=meta["tags"],
        visuals=visuals_data,
        word_count=_word_count(content_html),
    )


# ─── Exports ──────────────────────────────────────────────────────────────────

__all__ = [
    "write_article",
    "ArticleDraft",
    "WrittenSection",
]