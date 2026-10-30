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
- Do not invent statistics or claim a trend without a supplied snippet.
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
- Use only h2, h3, p, ul, ol, li, strong, em, a, blockquote, table, thead, tbody, tr, th, td, caption, and div class="table-wrap".
- Never use h1, img, figure, svg, script, style, iframe, form, input, inline styles, or event handlers.
- Do not include images or visual placeholders. A separate visual stage handles them.
- Do not include an FAQ section. A separate FAQ stage adds it.
- Include a final <h2>Sources</h2> section when practical.
- MANDATORY: All <th> (table header) tags MUST include a scope attribute. Use <th scope="col"> for column headers or <th scope="row"> for row headers. Example: <th scope="col">Metric</th>.

Research rules:
- Cite at least 3 unique sources from the supplied research only.
- Use this link shape: <a href="URL" rel="noopener noreferrer" target="_blank">Publisher</a>.
- Aim for at least one citation per 600 words.
- Never invent a URL, statistic, study, quote, or number.

SEO and answer-engine rules:
- Put the primary keyword in the title, first 100 words, and at least 2 H2 headings.
- Open with a direct, useful answer of 40-70 words inside the first 150 words.
- Use 4-8 H2 sections with scannable paragraphs, lists, or a table.
- Use question-style H2s where they read naturally.
- Add 1-3 contextual internal links from the supplied ATLAS links.
- End with a practical takeaway and a short ATLAS CTA.

Human style rules:
- Vary sentence and paragraph length. Be concrete and specific.
- Do not claim personal experience or use "as an AI".
- Never use em dashes.
- Avoid these phrases: delve, landscape, realm, tapestry, unlock, elevate, seamless, in today's fast-paced world, it's important to note.
- No clickbait, filler, unsupported claims, or invented numbers.
- Finance content must be educational, mention risk, and never give personalized advice or guarantee returns.
"""


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
Word range: {min_words}-{max_words} words.
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
        {"role": "system", "content": WRITER_SYSTEM},
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

Current draft:
<TITLE>{draft.title}</TITLE>
<SLUG>{draft.slug}</SLUG>
<EXCERPT>{draft.excerpt}</EXCERPT>
<CONTENT>
{draft.content}
</CONTENT>
"""
    return [
        {"role": "system", "content": WRITER_SYSTEM},
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
- Return exactly {visual_count} content visuals plus the cover.
- Prefer a chart or table when the research contains useful numbers.
- Headings must exist exactly and be unique.
- after_paragraph must fit inside that heading's section.
- Never place a visual before the first H2, in FAQ, or in Sources.
- Max 1 photo per H2. Do not place two visuals of the same type next to each other.
- Every visual needs alt text and a caption.
- Chart data: 2-8 labels and values, source_url from supplied research.
- Table data: provide labels/values or headers/rows and a source_url.
- Photos need a 2-5 word query with no brands or people.
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
