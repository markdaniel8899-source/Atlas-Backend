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
from app.autoblog.research.trends import research_with_groq
from app.autoblog.seo import validate_article
from app.autoblog.visuals.intelligent_injector import (
    analyze_content_needs,
    generate_dynamic_visual,
    inject_visuals_intelligently,
)
from app.autoblog.writer.dynamic import (
    generate_pure_text,
    write_dynamic_article,
    word_count,
)
from app.autoblog.images import fetch_cover_image


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

            # ─── Step 1: Dynamic trend discovery + research ───
            print(
                "[AutoBlog] Step 1/7: Discovering a trend and researching facts...",
                flush=True,
            )
            trend, fact_sheet, _raw_results = await research_with_groq()

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

            # ─── Step 3: Pure text article (Node 1) ───
            print("[AutoBlog] Step 3/7: Writing pure text article...", flush=True)
            internal_links = await gather_internal_links(4)
            draft = await generate_pure_text(
                trend,
                keywords,
                fact_sheet,
                internal_links,
                attempt=1,
            )

            # ─── Step 4: Visual injection (Intelligent Visual Injector) ───
            print("[AutoBlog] Step 4/7: Injecting visuals intelligently...", flush=True)

            # Step 4a: Analyze content for visual needs
            visual_needs = await analyze_content_needs(
                draft.content_html, fact_sheet.raw_snippets, fact_sheet, keywords
            )

            # Step 4b: Generate visuals for each need
            generated_visuals = []
            for req in visual_needs:
                visual = await generate_dynamic_visual(req, fact_sheet, keywords)
                if visual.get("html"):
                    generated_visuals.append(visual)

            # Step 4c: Inject visuals intelligently
            final_html = inject_visuals_intelligently(draft.content_html, generated_visuals, fact_sheet, keywords)
            draft.content_html = final_html
            draft.word_count = word_count(final_html)

            print(
                f"[AutoBlog] Step 4 complete: injected {len(generated_visuals)} visual(s)",
                flush=True,
            )

            # ─── Step 4b: Cover image fetch ───
            print("[AutoBlog] Step 4b/7: Fetching cover image...", flush=True)
            _, cover_image_url = await fetch_cover_image(
                draft.title,
                [keywords.primary_keyword, *keywords.secondary_keywords[:3]],
            )

            # ─── Step 5: Meta ───
            print("[AutoBlog] Step 5/7: Enforcing meta tags...", flush=True)
            meta = build_meta(trend, draft, keywords)

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
                # Regenerate pure text with issues, then re-inject visuals intelligently
                draft = await generate_pure_text(
                    trend,
                    keywords,
                    fact_sheet,
                    internal_links,
                    issues=validation.issues,
                    attempt=attempt,
                )
                # Re-analyze and re-inject visuals
                visual_needs = await analyze_content_needs(
                    draft.content_html, fact_sheet.raw_snippets, fact_sheet, keywords
                )
                generated_visuals = []
                for req in visual_needs:
                    visual = await generate_dynamic_visual(req, fact_sheet, keywords)
                    if visual.get("html"):
                        generated_visuals.append(visual)
                final_html = inject_visuals_intelligently(draft.content_html, generated_visuals, fact_sheet, keywords)
                draft.content_html = final_html
                draft.word_count = word_count(final_html)
                meta = build_meta(trend, draft, keywords)
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
                cover_image_url=cover_image_url,
                content_images=[],
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


def _inject_content_images(
    html_text: str,
    content_images: list[dict[str, str]],
    keyword: str,
) -> str:
    """Inject content images after the 1st and 3rd H2 headings.
    Falls back to inserting after first paragraph if not enough H2s."""
    if not content_images:
        return html_text

    from app.autoblog.images.dynamic import inject_content_images, ImageResult

    image_results = [
        ImageResult(url=img["url"], caption=img["caption"], provider_id="")
        for img in content_images
    ]
    return inject_content_images(html_text, image_results, keyword=keyword, positions=(0, 2))


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
