from __future__ import annotations

import json
from typing import Sequence

from app.services.blog.types import (
    GateReport,
    KeywordPlan,
    ResearchBundle,
    TopicCandidate,
    WriterOutput,
)


TOPIC_SYSTEM = """You plan articles for ATLAS, a learning platform for students.

Return strict JSON only:
{
  "candidates": [
    {
      "topic": "specific article topic",
      "category": "ai|software|hardware|automation|finance|other",
      "primary_keyword": "2-5 word search phrase",
      "secondary_keywords": ["3-6 related phrases"],
      "angle": "the specific argument or practical angle",
      "reason": "why this is useful now",
      "trend_score": 0,
      "seo_score": 0,
      "product_fit": 0
    }
  ]
}

Rules:
- Produce 5 candidates.
- Pick evergreen topics that can be researched with credible sources.
- Optimize for students learning, studying, focus, courses, and career skills.
- Avoid politics, medical diagnosis, legal advice, gambling, adult content, and conspiracies.
- Finance must be educational, never personalized advice.
- Do not invent statistics. Use trend_score as an estimate of durable search interest, not as a claim about a specific trend.
- Scores are 0-100: trend momentum, search opportunity, ATLAS product fit.
- No duplicate topics or primary keywords.
"""


def topic_messages(
    *,
    current_date: str,
    category_targets: dict[str, int],
    recent_topics: Sequence[str],
    recent_keywords: Sequence[str],
    trend_snippets: Sequence[str] = (),
) -> list[dict[str, str]]:
    recent_topic_text = "\n".join(f"- {item}" for item in recent_topics[:20])
    recent_keyword_text = "\n".join(f"- {item}" for item in recent_keywords[:20])
    trend_text = "\n".join(f"- {item}" for item in trend_snippets[:8])
    user = f"""Current date: {current_date}
Remaining category slots in the 20-post mix: {category_targets}

Recently used topics:
{recent_topic_text or "- none"}

Recently used primary keywords:
{recent_keyword_text or "- none"}

Recent trend snippets:
{trend_text or "- none supplied"}

Avoid repeating a topic within 90 days and a primary keyword within 60 days.
Return the JSON object now."""
    return [
        {"role": "system", "content": TOPIC_SYSTEM},
        {"role": "user", "content": user},
    ]


KEYWORD_SYSTEM = """You are an SEO planner for ATLAS, a learning platform for students.

Return strict JSON only:
{
  "primary": "2-5 word phrase",
  "secondary": ["3-6 phrases"],
  "long_tail": ["4-8 specific search phrases"],
  "lsi": ["5-10 related concepts"],
  "search_intent": "informational|commercial|transactional|navigational",
  "title_candidates": ["2-6 title ideas"]
}

Rules:
- The primary keyword must appear naturally in the topic.
- Do not target competitor brand names.
- No duplicate strings; no empty strings.
- Use the research sources below; do not invent search volume or statistics.
- Prefer clear student search intent over jargon.
"""


def keyword_messages(
    *,
    topic: TopicCandidate,
    research: ResearchBundle,
) -> list[dict[str, str]]:
    source_lines = [
        f"- {source.title} ({source.domain}, tier {int(source.tier)}): "
        f"{source.snippet[:300]}"
        for source in research.sources[:6]
    ]
    user = f"""Topic: {topic.topic}
Category: {topic.category.value}
Angle: {topic.angle}
Seed primary keyword: {topic.primary_keyword}
Seed secondary keywords: {", ".join(topic.secondary_keywords) or "none"}

Research sources:
{chr(10).join(source_lines) or "- none"}

Return the JSON object now."""
    return [
        {"role": "system", "content": KEYWORD_SYSTEM},
        {"role": "user", "content": user},
    ]


WRITER_SYSTEM = """You are the senior writer for ATLAS, a learning platform for students.
Write a publishable article, not an outline.

Return exactly this delimited format, with no markdown fences and no extra text:
<TITLE>30-70 character title</TITLE>
<SLUG>lowercase-ascii-slug</SLUG>
<EXCERPT>100-200 character summary</EXCERPT>
<CONTENT>
<p>...</p>
<h2>...</h2>
<p>...</p>
</CONTENT>

HTML rules:
- Use only h2, h3, p, ul, ol, li, strong, em, a, blockquote, table, thead, tbody, tr, th, td, caption, pre, code, and div class="table-wrap".
- Never use h1, img, figure, svg, script, style, iframe, form, input, inline styles, or event handlers.
- Do not include images or visual placeholders. A separate visual stage handles them.
- Include a final <h2>Sources</h2> section when practical.
- MANDATORY: All <th> (table header) tags MUST include a scope attribute. Use <th scope="col"> for column headers or <th scope="row"> for row headers. Example: <th scope="col">Metric</th>.
- TABLE STRUCTURE RULES (MANDATORY):
  - Tables MUST be wrapped in <div class="table-wrap"><table>...</table></div>
  - Tables MUST have <thead> and <tbody> sections
  - Column headers go in <thead><tr><th scope="col">Header</th>...</tr></thead>
  - Data rows go in <tbody><tr><td>...</td>...</tr></tbody>
  - Never put <tr> inside <th> or <th> inside <tr> inside <th>
  - Never nest tables
  - Example valid table:
    <div class="table-wrap"><table>
      <thead><tr><th scope="col">Metric</th><th scope="col">Value</th></tr></thead>
      <tbody><tr><td>Speed</td><td>Fast</td></tr></tbody>
    </table></div>

MANDATORY SEO AND CITATION RULES (NON-NEGOTIABLE, CHECKED BY AN AUTOMATED GATE):
- H06 MANDATORY KEYWORD PLACEMENT: You MUST include the exact phrase "{primary_keyword}" in:
  - The main title inside the <TITLE> tag (this is the article title; never add an <h1>, which is forbidden).
  - The very first paragraph, within the first 100 words of the article body.
  - At least ONE (1) <h2> subheading.
  Use the exact phrase character for character in these three places. In all other H2s and body text, prefer synonyms and related LSI terms to avoid keyword stuffing.
- H07 KEYWORD DENSITY: For an article of ~1200 words, use the exact phrase "{primary_keyword}" naturally 5 to 7 times throughout the body text (minimum 4 exact uses; 0.4%-0.8% is ideal). Never exceed 2.5% density. Do not rely only on synonyms. The checker ignores HTML tables, so every counted use must appear in paragraphs, lists, headings, or FAQ text.
- H11 MANDATORY INLINE CITATIONS: Whenever you state a statistic, percentage, year, or numeric claim (e.g., "80 percent", "2026", "40 to 60 percent"), you MUST append an inline citation immediately after it, inside the SAME <p>, <li>, or <blockquote> element.
  - Format: a real HTML hyperlink with a bracket label, like <a href="EXACT_RESEARCH_URL" rel="noopener noreferrer" target="_blank">[1]</a> or <a href="EXACT_RESEARCH_URL" rel="noopener noreferrer" target="_blank">[Source 1]</a>.
  - Example: <p>Studies show a 60% improvement <a href="EXACT_RESEARCH_URL" rel="noopener noreferrer" target="_blank">[1]</a> in productivity.</p>
  - Replace EXACT_RESEARCH_URL with a URL copied verbatim from the supplied research sources. Never output the literal text EXACT_RESEARCH_URL.
  - A plain "[1]" or "[Source 1]" that is not a hyperlink does NOT count. Any paragraph containing a number must also contain at least one real citation link.
  - Ensure these numbers match the sources provided in the research stage. Never invent a URL, statistic, or number. If a number has no source in the research, remove it instead of stating it uncited.

Research rules:
- Cite at least 3 unique sources from the supplied research only.
- Use this link shape: <a href="URL" rel="noopener noreferrer" target="_blank">Publisher</a>.
- Aim for at least one citation per 600 words.
- Never invent a URL, statistic, study, quote, or number.

SEO and answer-engine rules:
- Put the primary keyword in the title, first 100 words, and at least one H2 heading.
- Open with a direct, useful answer of 40-70 words inside the first 150 words. Wrap that answer in a <direct_answer>...</direct_answer> tag so the gate can verify it.
- End with an <excerpt>...</excerpt> tag containing a 100-200 word plain-text summary of the article.
- Return clean HTML only. Do not wrap output in markdown code fences (```html or ```). Strip all markdown wrappers; the configuration parser accepts clean HTML directly.
- Use question-style H2s where they read naturally.
- Add up to 3 contextual internal links from the supplied ATLAS links (max 3; do not exceed).
- End with a practical takeaway and a short ATLAS CTA.

Section length:
- Every H2 section must be at least 75 words (no thin sections).
- Keep sections scannable and useful; do not pad with filler.

Human style rules:
- Vary sentence and paragraph length. Be concrete and specific.
- Write short sentences and short paragraphs; avoid long, dense blocks of text.
- Use active voice; minimize passive voice.
- Do not claim personal experience or use "as an AI".
- Never use em dashes.
- Avoid these phrases: delve, landscape, realm, tapestry, unlock, elevate, seamless, in today's fast-paced world, it's important to note.
- No clickbait, filler, unsupported claims, or invented numbers.
- DO NOT invent fake experts, fake quotes, or fake personal experiences.
- DO NOT hallucinate citations or sources. Only use the sources explicitly provided in the research stage.
- If a specific detail is not in the research, state that it is not available rather than making it up.
- DO NOT use horizontal lines (`---`, `***`, `<hr>`, `====`). Use maximum ONE `<hr>` per entire article, only if absolutely necessary for a major section break.
- DO NOT generate repetitive filler text, redundant sentences, or excessive whitespace.
- Keep the output concise, well-structured, and within the word range. Minimum 800 words is HARD — do not go below 800 words. Do not pad with filler just to hit the count; add real value instead.
- Finance content must be educational, mention risk, and never give personalized advice or guarantee returns.
- KEYWORD USAGE: Use the exact phrase "{primary_keyword}" in the title and first 100 words, and at least one H2. For other H2s and body text, prefer synonyms and related LSI terms to avoid keyword stuffing. Do not repeat the exact phrase unnaturally.
"""


def _writer_system(keywords: KeywordPlan) -> str:
    return WRITER_SYSTEM.replace("{primary_keyword}", keywords.primary)


def _writer_user(
    topic: TopicCandidate,
    keywords: KeywordPlan,
    research: ResearchBundle,
    *,
    min_words: int,
    max_words: int,
    finance: bool,
    internal_links: Sequence[tuple[str, str]],
    extra: str = "",
) -> str:
    source_lines = [
        f"- {source.title} | {source.domain} | tier {int(source.tier)} | "
        f"{source.url}\n  {source.snippet[:500]}"
        for source in research.sources
    ]
    fact_lines = [
        f"- {fact.value:g}{fact.unit} | {fact.context} | {fact.source_url}"
        for fact in research.numeric_facts
    ]
    link_lines = [f"- [{label}]({url})" for label, url in internal_links]
    keyword_json = json.dumps(keywords.model_dump(mode="json"), ensure_ascii=False)
    finance_rules = (
        "This is a finance article. Include a visible educational disclaimer, "
        "mention risk, and avoid personalized advice or guaranteed returns."
        if finance
        else "This is not a finance article."
    )
    return f"""Write the article now.
Topic: {topic.topic}
Category: {topic.category.value}
Angle: {topic.angle or "Use the strongest practical angle for students."}
Audience: {topic.audience if hasattr(topic, "audience") else "students"}
Finance: {finance_rules}
Word range: {min_words}-{max_words} words. HARD MINIMUM: 800 words — the article must be at least 800 words long.
ATLAS internal links (use 1-3 where relevant):
{chr(10).join(link_lines) or "- https://atlas.app/app/notes"}

Structured keywords:
{keyword_json}

Research sources:
{chr(10).join(source_lines) or "- none"}

Numeric facts from research:
{chr(10).join(fact_lines) or "- none"}

{extra}
Return the delimited article now."""


def writer_messages(
    topic: TopicCandidate,
    keywords: KeywordPlan,
    research: ResearchBundle,
    *,
    min_words: int = 900,
    max_words: int = 2200,
    finance: bool = False,
    internal_links: Sequence[tuple[str, str]] = (),
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _writer_system(keywords)},
        {
            "role": "user",
            "content": _writer_user(
                topic,
                keywords,
                research,
                min_words=min_words,
                max_words=max_words,
                finance=finance,
                internal_links=internal_links,
            ),
        },
    ]


def writer_repair_messages(
    topic: TopicCandidate,
    keywords: KeywordPlan,
    research: ResearchBundle,
    draft: WriterOutput,
    report: GateReport,
    *,
    min_words: int = 900,
    max_words: int = 2200,
    finance: bool = False,
    internal_links: Sequence[tuple[str, str]] = (),
) -> list[dict[str, str]]:
    failures = "\n".join(
        f"- {failure.id}: {failure.detail}" for failure in report.hard_failures
    )
    preserve = [
        "Preserve the existing title and slug unless H02 or H03 failed.",
        "Preserve the keyword strategy unless H06 or H08 failed.",
        "Keep all citations that are already valid.",
    ]
    extra = f"""This draft failed the content gate. Repair only the failed checks.
Failed checks:
{failures or "- none listed"}

Repair rules:
{chr(10).join(preserve)}
- The MANDATORY SEO AND CITATION RULES in the system prompt always apply. Before returning, re-verify H06 keyword placement, H07 keyword density, and H11 inline citations; fixing these failures overrides brevity.

Current draft:
<TITLE>{draft.title}</TITLE>
<SLUG>{draft.slug}</SLUG>
<EXCERPT>{draft.excerpt}</EXCERPT>
<CONTENT>
{draft.content}
</CONTENT>
"""
    return [
        {"role": "system", "content": _writer_system(keywords)},
        {
            "role": "user",
            "content": _writer_user(
                topic,
                keywords,
                research,
                min_words=min_words,
                max_words=max_words,
                finance=finance,
                internal_links=internal_links,
                extra=extra,
            ),
        },
    ]


VISUAL_PLACEMENT_SYSTEM = """You place visuals in an ATLAS article.
Return strict JSON only:
{
  "cover": {"query": "2-5 word photo query", "alt": "alt text", "caption": "caption"},
  "visuals": [
    {
      "after_heading": "exact existing heading",
      "after_paragraph": 1,
      "type": "chart|table|photo",
      "purpose": "what the visual explains",
      "alt": "alt text",
      "caption": "caption",
      "query": "2-5 word photo query or null",
      "data": {
        "title": "chart or table title",
        "chart_type": "bar|line",
        "labels": ["label"],
        "values": [1],
        "unit": "%",
        "source_url": "URL from supplied research",
        "source_label": "Publisher",
        "headers": [],
        "rows": []
      }
    }
  ]
}

Rules:
- Return exactly {visual_count} content visuals plus the cover. The cover image is mandatory for every article.
- Analyze the full article content and decide where visuals add the most value. Distribute them across the whole article, roughly one every 400-500 words. Do not cluster them in the first half.
- For articles longer than 1000 words, place at least one visual in the final third (the conclusion or the last major section before FAQ/Sources). If the article is text-heavy at the bottom, add an image near the end.
- Total content visuals must not exceed 4 for large articles (>1200 words).
- Use a chart only when the research has a clear numeric comparison or trend; otherwise use a table or photo.
- Use line charts for time-series data (years, months, quarters) and bar charts for direct category comparisons. Never use a bar chart for a time series.
- Use at most one chart per article. Fill remaining slots with a table and/or photo; for 3 visuals, prefer chart + table + photo.
- If the article is mostly text with no useful numbers, use photos or a table instead of forcing a chart.
- Every image must have meaningful alt text for SEO and a caption.
- Headings must exist exactly and be unique.
- after_paragraph must fit inside that heading's section.
- Never place a visual before the first H2, in FAQ, or in Sources.
- Max 1 photo per H2. Do not place two visuals of the same type next to each other if they are within 150 words of each other.
- Chart data: 2-8 labels and values, source_url from supplied research.
- Table data: provide labels/values or headers/rows and a source_url.
- Photos need a 2-5 word query with no brands or people.
- MANDATORY: Image search queries MUST be strictly between 2 to 5 words.
  - BAD: 'tech' (1 word), 'a beautiful modern laptop on a wooden desk in a sunny office' (10 words).
  - GOOD: 'modern laptop desk', 'coding screen dark', 'student studying library'.
"""


def visual_placement_messages(
    content: str,
    research: ResearchBundle,
    *,
    required_count: int,
    headings: Sequence[str] = (),
) -> list[dict[str, str]]:
    source_lines = [
        f"- {source.title} | {source.url}"
        for source in research.sources
    ]
    heading_lines = [f"- {heading}" for heading in headings]
    system = VISUAL_PLACEMENT_SYSTEM.replace(
        "{visual_count}", str(required_count)
    )
    user = f"""Required content visual count: {required_count}

Existing headings:
{chr(10).join(heading_lines) or "- derive from content"}

Research sources:
{chr(10).join(source_lines) or "- none"}

Article content:
{content}

Return the JSON object now."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


META_SYSTEM = """You write search metadata for ATLAS articles.
Return strict JSON only:
{"meta_title": "...", "meta_description": "..."}

Rules:
- meta_title: 50-60 characters, primary keyword within the first 30 characters.
- meta_description: 150-160 characters, include the primary keyword, clear value, and a CTA.
- No all caps, clickbait, quotation marks, or em dashes.
- Do not invent claims or numbers.
"""


def meta_messages(
    *,
    title: str,
    excerpt: str,
    primary_keyword: str,
    secondary_keywords: Sequence[str],
    first_words: str,
    headings: Sequence[str],
) -> list[dict[str, str]]:
    user = f"""Article title: {title}
Excerpt: {excerpt}
Primary keyword: {primary_keyword}
Secondary keywords: {", ".join(secondary_keywords) or "none"}
Headings: {", ".join(headings) or "none"}
Opening text:
{first_words}

Return the JSON object now."""
    return [
        {"role": "system", "content": META_SYSTEM},
        {"role": "user", "content": user},
    ]


FAQ_SYSTEM = """You write grounded FAQ sections for ATLAS articles.
Return strict JSON only:
{"faqs": [{"question": "...", "answer": "..."}]}

Rules:
- 4-6 questions, natural question style, no duplicates.
- Answers are 40-80 words and use only facts present in the article or research.
- Do not add new numbers, personalized advice, guarantees, or unsupported claims.
- Ground every answer in the visible article.
"""


def faq_messages(
    *,
    content: str,
    keywords: KeywordPlan,
    research: ResearchBundle,
) -> list[dict[str, str]]:
    source_lines = [
        f"- {source.title} | {source.url}" for source in research.sources
    ]
    user = f"""Primary keyword: {keywords.primary}
Secondary keywords: {", ".join(keywords.secondary) or "none"}

Cited sources:
{chr(10).join(source_lines) or "- none"}

Article content:
{content}

Return the JSON object now."""
    return [
        {"role": "system", "content": FAQ_SYSTEM},
        {"role": "user", "content": user},
    ]
