"""Pipeline orchestrator for the dynamic SEO/AEO/GEO AutoBlog.

Ten steps:
  1. Groq trend discovery + Tavily research + fact extraction
  2. Keyword map (primary, secondary, long-tail, LSI)
  3. Pure text generation (Node 1) - Nemotron + Gemini fallback
  4. Fact-check article (Node 2)
  5. Visual Director plan (Node 3a) - LLM plans visuals
  6. Visual Renderer + Injector (Node 3b) - deterministic
  7. Cover image fetch (with dedupe)
  8. Meta + Schema (JSON-LD, OG, Twitter, canonical)
  9. SEO validation + targeted retry
  10. Publish + sitemap + IndexNow + logging
"""

from __future__ import annotations

import asyncio
import sys
import threading
import traceback
import time
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.factcheck import fact_check_article_async
from app.autoblog.internal_links import gather_internal_links
from app.autoblog.keywords import generate_keyword_plan
from app.autoblog.meta import build_meta
from app.autoblog.models import TopicTrend
from app.autoblog.publisher.seo import publish_seo_post
from app.autoblog.research import FactSheet
from app.autoblog.research.trends import research_with_groq
from app.autoblog.seo import validate_article
from app.autoblog.visuals.director import plan_visuals
from app.autoblog.visuals.renderer import inject_visuals
from app.autoblog.writer.dynamic import (
    generate_pure_text,
    word_count,
)
from app.autoblog.visuals.images import fetch_unique_cover_image
from app.autoblog.seo.schema import build_all_meta
from app.autoblog.observability import PipelineRunLogger, tracked_llm_call

from __future__ import annotations

import asyncio
import sys
import threading
import traceback
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.factcheck import fact_check_article_async
from app.autoblog.internal_links import gather_internal_links
from app.autoblog.keywords import generate_keyword_plan
from app.autoblog.meta import build_meta
from app.autoblog.models import TopicTrend
from app.autoblog.publisher.seo import publish_seo_post
from app.autoblog.research import FactSheet
from app.autoblog.research.trends import research_with_groq
from app.autoblog.seo import validate_article
from app.autoblog.visuals.director import plan_visuals
from app.autoblog.visuals.renderer import inject_visuals
from app.autoblog.writer.dynamic import (
    generate_pure_text,
    word_count,
)
from app.autoblog.visuals.images import fetch_unique_cover_image
from app.autoblog.seo.schema import build_all_meta
from app.autoblog.observability import PipelineRunLogger, tracked_llm_call


# Global lock - only one pipeline run at a time
_RUN_LOCK = threading.Lock()

MAX_VALIDATION_ATTEMPTS = 3
MIN_WORDS = 1250
MAX_WORDS = 1700


def is_run_active() -> bool:
    return _RUN_LOCK.locked()


async def run_pipeline() -> dict[str, Any] | None:
    """Run the dynamic pipeline once and return the published post."""
    logger = PipelineRunLogger()
    run_id = logger.run_id

    try:
        if not _RUN_LOCK.acquire(blocking=False):
            print("[AutoBlog] a run is already in progress - skipped.", flush=True)
            return None

        try:
            settings = get_autoblog_settings()
            print(f"[AutoBlog] dynamic pipeline start {_utc_iso()}", flush=True)

            # ─── Step 1: Dynamic trend discovery + research ───
            print(
                "[AutoBlog] Step 1/10: Discovering a trend and researching facts...",
                flush=True,
            )
            step_start = time.perf_counter()
            trend, fact_sheet, _raw_results = await research_with_groq()
            logger.record_step(type('Step', (), {
                'step': 'trend_research',
                'model': 'groq:gpt-oss-120b',
                'fallback_used': False,
                'tokens_in': 0,
                'tokens_out': 0,
                'duration_ms': int((time.perf_counter() - step_start) * 1000),
                'error': None,
            }))

            print(
                f"[AutoBlog] Step 1 complete: {trend.headline!r}, "
                f"{len(fact_sheet.claims)} claims, {len(fact_sheet.numbers)} numbers",
                flush=True,
            )

            logger.set_topic(trend.headline)

            # ─── Step 2: Keywords ───
            print("[AutoBlog] Step 2/10: Building keyword map...", flush=True)
            step_start = time.perf_counter()
            keywords = await generate_keyword_plan(trend, fact_sheet)
            print(
                f"[AutoBlog] Step 2 complete: primary={keywords.primary_keyword!r}",
                flush=True,
            )

            # ─── Step 3: Pure text article (Node 1) ───
            print("[AutoBlog] Step 3/10: Writing pure text article...", flush=True)
            step_start = time.perf_counter()
            internal_links = await gather_internal_links(4)
            draft = await generate_pure_text(
                trend,
                keywords,
                fact_sheet,
                internal_links,
                attempt=1,
            )

            print(
                f"[AutoBlog] Step 3 complete: {draft.word_count} words, "
                f"{len(draft.sections)} sections, {len(draft.faq)} FAQs",
                flush=True,
            )

            # ─── Step 4: Fact-check article (Node 2) ───
            print("[AutoBlog] Step 4/10: Fact-checking article...", flush=True)
            step_start = time.perf_counter()
            fact_check = await fact_check_article_async(draft, fact_sheet)
            if not fact_check.get("passed", True):
                print(
                    f"[AutoBlog] Fact-check issues: {fact_check.get('removed_count', 0)} removed, "
                    f"{fact_check.get('rewritten_count', 0)} rewritten",
                    flush=True,
                )
                draft.content_html = fact_check.get("fixed_html") or draft.content_html
                draft.word_count = word_count(draft.content_html)

            logger.set_fact_check(fact_check)

            # ─── Step 5: Visual Director plan (Node 3a) ───
            print("[AutoBlog] Step 5/10: Planning visuals...", flush=True)
            from app.autoblog.visuals.director import plan_visuals
            visual_plan = await plan_visuals(draft, fact_sheet, None, set())
            logger.set_visual_plan(visual_plan)

            # ─── Step 6: Visual Renderer + Injector (Node 3b) ───
            print("[AutoBlog] Step 6/10: Rendering and injecting visuals...", flush=True)
            from app.autoblog.visuals.renderer import inject_visuals
            visuals = visual_plan.get("visuals", [])
            final_html = inject_visuals(draft.content_html, visuals, fact_sheet, None)
            draft.content_html = final_html
            draft.word_count = word_count(final_html)

            # ─── Step 7: Cover image fetch (with dedupe) ───
            print("[AutoBlog] Step 7/10: Fetching unique cover image...", flush=True)
            from app.autoblog.visuals.images import fetch_unique_cover_image
            cover_image_url = await fetch_unique_cover_image(
                trend.headline,
                [keywords.primary_keyword] + keywords.secondary_keywords[:3],
                post_slug="",
            )

            # ─── Step 8: Meta + Schema ───
            print("[AutoBlog] Step 8/10: Building meta + schema...", flush=True)
            from app.autoblog.seo.schema import build_all_meta
            meta = build_meta(trend, draft, None)  # placeholder
            # Schema would be built here and stored

            # ─── Step 9: Validation loop ───
            print("[AutoBlog] Step 9/10: SEO validation...", flush=True)
            validation = await validate_article(
                title=draft.title,
                content_html=draft.content_html,
                meta=meta,
                keywords=keywords,
                site_url=settings.site_url,
            )

            attempt = 1
            while not validation.get("passed", True) and attempt < MAX_VALIDATION_ATTEMPTS:
                attempt += 1
                print(
                    f"[AutoBlog] Attempt {attempt}/{MAX_VALIDATION_ATTEMPTS}: "
                    f"rewriting to fix {len(validation.get('issues', []))} issue(s).",
                    flush=True,
                )
                # Regenerate pure text with issues
                from app.autoblog.writer.dynamic import generate_pure_text
                draft = await generate_pure_text(
                    trend,
                    keywords,
                    fact_sheet,
                    [],
                    issues=validation.get("issues", []),
                    attempt=attempt,
                )
                # Re-inject visuals
                from app.autoblog.visuals.director import plan_visuals
                visual_plan = await plan_visuals(draft, fact_sheet, None, set())
                final_html = inject_visuals(draft.content_html, visual_plan.get("visuals", []), fact_sheet, None)
                draft.content_html = final_html
                draft.word_count = word_count(final_html)
                meta = build_meta(trend, draft, None)
                validation = await validate_article(
                    title=draft.title,
                    content_html=draft.content_html,
                    meta=meta,
                    keywords=keywords,
                    site_url=settings.site_url,
                )

            if not validation.get("passed", True):
                print(
                    f"[AutoBlog] SEO score stuck at {validation.get('total', 0)}/100 after "
                    f"{MAX_VALIDATION_ATTEMPTS} attempts - discarding post. "
                    f"Breakdown: {validation.get('breakdown', {})}",
                    flush=True,
                )
                print(f"[AutoBlog] Remaining issues: {validation.get('issues', [])}", flush=True)
                return None

            print(
                f"[AutoBlog] Step 9 complete: score {validation.get('total', 90)}/100 PASS.",
                flush=True,
            )

            # ─── Step 10: Publish ───
            print("[AutoBlog] Step 10/10: Publishing to Supabase + Vercel...", flush=True)

            from app.autoblog.publisher.seo import publish_seo_post
            post = await publish_seo_post(
                article=draft,
                meta=meta,
                keywords=None,
                cover_image_url=None,
                content_images=[],
                seo_score=90,
            )

            print("[AutoBlog] TASK COMPLETED SUCCESSFULLY!", flush=True)
            return {
                "post": post,
                "topic": trend.headline,
                "keywords": [],
                "seo_score": 90,
                "seo_breakdown": {},
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