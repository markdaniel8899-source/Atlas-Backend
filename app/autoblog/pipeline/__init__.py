"""Pipeline orchestrator for the dynamic SEO/AEO/GEO AutoBlog.

Seven steps:
  1. Groq trend discovery + Tavily research + fact extraction
  2. Keyword map (primary, secondary, long-tail, LSI)
  3. Nemotron/Gemini full-article generation
  4. Meta title/description enforcement
  5. Visual strategy (data table + chart, or cover + content photos)
  6. SEO validation, with targeted rewrites until the score reaches 85
  7. Supabase publish + Vercel ISR revalidation
"""

from __future__ import annotations

import asyncio
import sys
import threading
import traceback
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.internal_links import gather_internal_links
from app.autoblog.keywords import generate_keyword_plan
from app.autoblog.meta import build_meta
from app.autoblog.models import TopicTrend
from app.autoblog.publisher.seo import publish_seo_post
from app.autoblog.research import FactSheet
from app.autoblog.seo import validate_article
from app.autoblog.visuals.strategy import assemble_content, build_visual_strategy
from app.autoblog.writer.dynamic import write_dynamic_article, word_count


# Global lock - only one pipeline run at a time
_RUN_LOCK = threading.Lock()

MAX_VALIDATION_ATTEMPTS = 3


def is_run_active() -> bool:
    return _RUN_LOCK.locked()


async def run_pipeline() -> dict[str, Any] | None:
    """Run the dynamic pipeline once and return the published post."""
    print("[AutoBlog] TASK STARTED EXECUTION!", flush=True)

    try:
        if not _RUN_LOCK.acquire(blocking=False):
            print("[AutoBlog] a run is already in progress - skipped.", flush=True)
            return None

        try:
            settings = get_autoblog_settings()
            print(f"[AutoBlog] dynamic pipeline start {_utc_iso()}", flush=True)

            # ─── Step 1: Hardcoded 3-way model benchmark ───
            print(
                "[AutoBlog] Step 1/7: Using hardcoded 3-way model benchmark topic...",
                flush=True,
            )

            # TODO: REMOVE THIS HARDCODED TEST TOPIC
            topic = {
                "headline": "GPT-4o vs Claude 3.5 Sonnet vs Gemini 1.5 Pro: The 2026 Developer Benchmark",
                "category": "benchmark_comparison",
                "why_now": "New independent benchmarks reveal significant shifts in coding accuracy and pricing.",
                "key_data_points": [
                    "GPT-4o: 88% HumanEval, $5/1M tokens, 128k context",
                    "Claude 3.5 Sonnet: 92% HumanEval, $3/1M tokens, 200k context",
                    "Gemini 1.5 Pro: 85% HumanEval, $1.25/1M tokens, 1M+ context",
                ],
            }

            openai_source = "https://openai.com/api/pricing/"
            anthropic_source = "https://www.anthropic.com/pricing"
            google_source = "https://ai.google.dev/gemini-api/docs/pricing"
            humaneval_source = "https://github.com/openai/human-eval"

            trend = TopicTrend(
                headline=topic["headline"],
                angle=(
                    "Compare GPT-4o, Claude 3.5 Sonnet, and Gemini 1.5 Pro on "
                    "coding accuracy, token pricing, and context window to "
                    "identify the best developer value in 2026."
                ),
                category=topic["category"],
                why_now=topic["why_now"],
                search_intent="commercial",
            )
            fact_sheet = FactSheet(
                claims=[
                    "GPT-4o scored 88% on HumanEval, costs $5 per 1M tokens, "
                    "and supports a 128k context window.",
                    "Claude 3.5 Sonnet scored 92% on HumanEval, costs $3 per "
                    "1M tokens, and supports a 200k context window.",
                    "Gemini 1.5 Pro scored 85% on HumanEval, costs $1.25 per "
                    "1M tokens, and supports a 1M+ context window.",
                ],
                numbers=[
                    {
                        "value": "88",
                        "unit": "%",
                        "context": "GPT-4o HumanEval score",
                        "source_url": humaneval_source,
                    },
                    {
                        "value": "5",
                        "unit": "USD per 1M tokens",
                        "context": "GPT-4o API price",
                        "source_url": openai_source,
                    },
                    {
                        "value": "128000",
                        "unit": "tokens",
                        "context": "GPT-4o context window",
                        "source_url": openai_source,
                    },
                    {
                        "value": "92",
                        "unit": "%",
                        "context": "Claude 3.5 Sonnet HumanEval score",
                        "source_url": humaneval_source,
                    },
                    {
                        "value": "3",
                        "unit": "USD per 1M tokens",
                        "context": "Claude 3.5 Sonnet API price",
                        "source_url": anthropic_source,
                    },
                    {
                        "value": "200000",
                        "unit": "tokens",
                        "context": "Claude 3.5 Sonnet context window",
                        "source_url": anthropic_source,
                    },
                    {
                        "value": "85",
                        "unit": "%",
                        "context": "Gemini 1.5 Pro HumanEval score",
                        "source_url": humaneval_source,
                    },
                    {
                        "value": "1.25",
                        "unit": "USD per 1M tokens",
                        "context": "Gemini 1.5 Pro API price",
                        "source_url": google_source,
                    },
                    {
                        "value": "1000000",
                        "unit": "tokens",
                        "context": "Gemini 1.5 Pro context window",
                        "source_url": google_source,
                    },
                ],
                entities=[
                    {
                        "name": "GPT-4o",
                        "type": "AI model",
                        "context": "OpenAI model in the comparison",
                        "source_url": openai_source,
                    },
                    {
                        "name": "Claude 3.5 Sonnet",
                        "type": "AI model",
                        "context": "Anthropic model in the comparison",
                        "source_url": anthropic_source,
                    },
                    {
                        "name": "Gemini 1.5 Pro",
                        "type": "AI model",
                        "context": "Google model in the comparison",
                        "source_url": google_source,
                    },
                ],
                source_urls=[
                    openai_source,
                    anthropic_source,
                    google_source,
                    humaneval_source,
                ],
                raw_snippets=topic["key_data_points"],
            )

            print(
                f"[AutoBlog] Step 1 complete: {trend.headline!r}, "
                f"{len(fact_sheet.claims)} claims, {len(fact_sheet.numbers)} numbers",
                flush=True,
            )

            # ─── Step 2: Keywords ───
            print("[AutoBlog] Step 2/7: Building keyword map...", flush=True)
            keywords = await generate_keyword_plan(trend, fact_sheet)
            print(
                f"[AutoBlog] Step 2 complete: primary={keywords.primary_keyword!r}",
                flush=True,
            )

            # ─── Step 3: Full article ───
            print("[AutoBlog] Step 3/7: Writing the article...", flush=True)
            internal_links = await gather_internal_links(4)
            draft = await write_dynamic_article(
                trend,
                keywords,
                fact_sheet,
                internal_links,
                attempt=1,
            )

            # ─── Step 4: Meta ───
            print("[AutoBlog] Step 4/7: Enforcing meta tags...", flush=True)
            meta = build_meta(trend, draft, keywords)

            # ─── Step 5: Visuals ───
            print("[AutoBlog] Step 5/7: Preparing visuals...", flush=True)
            strategy = await build_visual_strategy(
                trend,
                keywords,
                fact_sheet,
                draft,
            )
            content_html, content_images = assemble_content(
                draft,
                strategy,
                internal_links,
                keyword=keywords.primary_keyword,
            )
            draft.content_html = content_html
            draft.word_count = word_count(content_html)
            print(
                f"[AutoBlog] Step 5 complete: mode={strategy.mode}, "
                f"{len(strategy.data_visuals)} data visual(s), "
                f"{len(content_images)} content image(s)",
                flush=True,
            )

            # ─── Step 6: Validation loop ───
            print("[AutoBlog] Step 6/7: SEO validation...", flush=True)
            validation = validate_article(
                title=draft.title,
                content_html=draft.content_html,
                meta=meta,
                keywords=keywords,
                site_url=settings.site_url,
            )

            attempt = 1
            while not validation.passed and attempt < MAX_VALIDATION_ATTEMPTS:
                attempt += 1
                print(
                    f"[AutoBlog] Attempt {attempt}/{MAX_VALIDATION_ATTEMPTS}: "
                    f"rewriting to fix {len(validation.issues)} issue(s).",
                    flush=True,
                )
                draft = await write_dynamic_article(
                    trend,
                    keywords,
                    fact_sheet,
                    internal_links,
                    issues=validation.issues,
                    attempt=attempt,
                )
                meta = build_meta(trend, draft, keywords)
                content_html, content_images = assemble_content(
                    draft,
                    strategy,
                    internal_links,
                    keyword=keywords.primary_keyword,
                )
                draft.content_html = content_html
                draft.word_count = word_count(content_html)
                validation = validate_article(
                    title=draft.title,
                    content_html=draft.content_html,
                    meta=meta,
                    keywords=keywords,
                    site_url=settings.site_url,
                )

            if not validation.passed:
                print(
                    f"[AutoBlog] SEO score stuck at {validation.total}/100 after "
                    f"{MAX_VALIDATION_ATTEMPTS} attempts - discarding post. "
                    f"Breakdown: {validation.breakdown}",
                    flush=True,
                )
                print(f"[AutoBlog] Remaining issues: {validation.issues}", flush=True)
                return None

            print(
                f"[AutoBlog] Step 6 complete: score {validation.total}/100 PASS.",
                flush=True,
            )

            # ─── Step 7: Publish + revalidate ───
            print("[AutoBlog] Step 7/7: Publishing to Supabase + Vercel...", flush=True)
            post = await publish_seo_post(
                article=draft,
                meta=meta,
                keywords=keywords,
                cover_image_url=strategy.cover_url,
                content_images=content_images,
                seo_score=validation.total,
            )

            print("[AutoBlog] TASK COMPLETED SUCCESSFULLY!", flush=True)
            return {
                "post": post,
                "topic": trend.headline,
                "keywords": keywords.all_keywords(),
                "seo_score": validation.total,
                "seo_breakdown": validation.breakdown,
            }
        finally:
            _RUN_LOCK.release()

    except Exception as exc:
        print(
            f"[AutoBlog] CRITICAL TASK ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )
        print(_redact_secrets(traceback.format_exc()), flush=True)
        sys.stdout.flush()
        return None


def run_scheduled_job() -> None:
    """APScheduler entry point."""
    try:
        result = asyncio.run(run_pipeline())
        if result:
            print(f"[AutoBlog] scheduled run OK -> /blog/{result['post'].get('slug')}")
        else:
            print("[AutoBlog] scheduled run exited gracefully")
    except Exception as exc:
        print(f"[AutoBlog] scheduled run failed: {type(exc).__name__}: {exc}")


def _utc_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact_secrets(text: str) -> str:
    import re

    return re.sub(
        r"(?i)(api[_-]?key|secret|token|password)[\s:=]+[^\s]+",
        r"\1=***",
        text,
    )


__all__ = [
    "run_pipeline",
    "run_scheduled_job",
    "is_run_active",
]
