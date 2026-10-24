"""Pipeline orchestrator for AutoBlog v2 - New flow with editor-planner."""

from __future__ import annotations

import asyncio
import sys
import threading
import traceback
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.research import run_research, FactSheet
from app.autoblog.planner import create_article_plan, ArticlePlan
from app.autoblog.writer import write_article, ArticleDraft
from app.autoblog.checks import run_gate, fix_article, GateResult, MAX_TOTAL_IMAGES
from app.autoblog.images import fetch_cover_image, inject_images_into_content, ImageResult
from app.autoblog.visuals import replace_visual_markers
from app.autoblog.publisher import publish_post
from app.autoblog.llm.clients import BlogPipelineError


# Global lock - only one pipeline run at a time
_RUN_LOCK = threading.Lock()

# Timing
RESEARCH_PAUSE_SECONDS = 2
PLAN_PAUSE_SECONDS = 1
WRITE_PAUSE_SECONDS = 3


def is_run_active() -> bool:
    return _RUN_LOCK.locked()


async def run_pipeline() -> dict[str, Any] | None:
    """New v2 pipeline: Research -> Plan -> Write -> Gate/Fix -> Images -> Publish."""
    print("[AutoBlog] TASK STARTED EXECUTION!", flush=True)
    try:
        if not _RUN_LOCK.acquire(blocking=False):
            print("[AutoBlog] a run is already in progress - skipped.", flush=True)
            return None
        try:
            print(f"[AutoBlog] v2 pipeline start {_utc_iso()}", flush=True)

            # ─── Step 1: Research (topic discovery + fact sheet) ───
            print("[AutoBlog] Step 1: Research (queries + fact extraction)...", flush=True)
            try:
                settings = get_autoblog_settings()
                # run_research handles topic discovery internally using rotating queries
                fact_sheet, raw_results = await run_research("trending AI topic", "exploring the topic", "explainer")
                print(f"[AutoBlog] Step 1 complete: {len(fact_sheet.claims)} claims, {len(fact_sheet.numbers)} numbers", flush=True)
            except Exception as exc:
                tb = traceback.format_exc()
                tb = _redact_secrets(tb)
                print(f"[AutoBlog] Step 1 failed: {type(exc).__name__}: {exc}. Traceback: {tb}", flush=True)
                return None
            await asyncio.sleep(RESEARCH_PAUSE_SECONDS)

            # ─── Step 2: Editor-Planner ───
            print("[AutoBlog] Step 2: Editor-planner creating article plan...", flush=True)
            try:
                plan = await create_article_plan(fact_sheet)
                print(f"[AutoBlog] Step 2 complete: {plan.topic} ({plan.article_type}, {plan.target_words} words, {len(plan.outline)} sections, {len(plan.visual_plan)} visuals, {len(plan.image_plan)} images)", flush=True)
            except Exception as exc:
                tb = traceback.format_exc()
                tb = _redact_secrets(tb)
                print(f"[AutoBlog] Step 2 failed: {type(exc).__name__}: {exc}. Traceback: {tb}", flush=True)
                return None
            await asyncio.sleep(PLAN_PAUSE_SECONDS)

            # ─── Step 3: Section-by-section Writing ───
            print("[AutoBlog] Step 3: Writing article section by section...", flush=True)
            try:
                draft = await write_article(plan, fact_sheet)
                print(f"[AutoBlog] Step 3 complete: {draft.word_count} words, title={draft.title[:60]}", flush=True)
            except Exception as exc:
                tb = traceback.format_exc()
                tb = _redact_secrets(tb)
                print(f"[AutoBlog] Step 3 failed: {type(exc).__name__}: {exc}. Traceback: {tb}", flush=True)
                return None
            await asyncio.sleep(WRITE_PAUSE_SECONDS)

            # ─── Step 4: Gate + Fix Loop (max 2 attempts) ───
            print("[AutoBlog] Step 4: Quality gate...", flush=True)
            article_data = {
                "title": draft.title,
                "excerpt": draft.excerpt,
                "content_html": replace_visual_markers(
                    "\n".join(s.html for s in draft.sections) + "\n" + draft.faq_html,
                    draft.visuals
                ),
                "visuals": draft.visuals,
                "tags": draft.tags,
            }

            fix_attempt = 0
            max_fixes = 2
            current_html = article_data["content_html"]
            current_visuals = draft.visuals

            while fix_attempt <= max_fixes:
                gate = run_gate(
                    current_html,
                    article_data,
                    draft.title,
                    draft.excerpt,
                    plan,
                    fact_sheet,
                )
                if gate.passed:
                    print("[AutoBlog] Quality gate PASSED.", flush=True)
                    break

                if fix_attempt >= max_fixes:
                    print(f"[AutoBlog] Gate FAILED after {max_fixes} fixes: {gate.reasons}. Discarding post.", flush=True)
                    return None

                print(f"[AutoBlog] Gate failed (attempt {fix_attempt + 1}/{max_fixes}): {gate.reasons}. Fixing...", flush=True)
                current_html, article_data = fix_article(current_html, article_data, gate, plan, fact_sheet)
                fix_attempt += 1

            article_data["content_html"] = current_html

            # ─── Step 5: Images (cover + content per image_plan) ───
            print("[AutoBlog] Step 5: Fetching images...", flush=True)
            try:
                used_provider_ids: set[str] = set()
                cover_result, cover_source = await fetch_cover_image(
                    draft.title, [plan.primary_keyword] + plan.secondary_keywords, used_provider_ids
                )
                used_provider_ids.add(cover_result.provider_id)

                content_html, content_images, content_count = await inject_images_into_content(
                    current_html,
                    plan,
                    fact_sheet,
                    keyword=plan.primary_keyword,
                    used_provider_ids=used_provider_ids,
                )

                # Validate total image count
                total_images = 1 + content_count  # cover + content
                if total_images > MAX_TOTAL_IMAGES:
                    print(f"[AutoBlog] Total images {total_images} exceeds {MAX_TOTAL_IMAGES}, trimming content images", flush=True)
                    content_images = content_images[:MAX_TOTAL_IMAGES - 1]
                    content_count = len(content_images)

                print(f"[AutoBlog] Step 5 complete: cover + {content_count} content images", flush=True)
            except Exception as exc:
                tb = traceback.format_exc()
                tb = _redact_secrets(tb)
                print(f"[AutoBlog] Step 5 failed: {type(exc).__name__}: {exc}. Traceback: {tb}", flush=True)
                return None

            # ─── Step 6: Publish ───
            print("[AutoBlog] Step 6: Publishing to Supabase...", flush=True)
            try:
                content_images_list = [{"url": img.url, "caption": img.caption} for img in content_images]
                post = await publish_post(
                    title=draft.title,
                    content_html=content_html,
                    excerpt=draft.excerpt,
                    cover_image_url=cover_result.url,
                    content_images=content_images_list,
                    tags=draft.tags,
                    keywords=[plan.primary_keyword] + plan.secondary_keywords,
                )
                print(f"[AutoBlog] Step 6 complete: post id={post.get('id')} slug={post.get('slug')}", flush=True)
            except Exception as exc:
                tb = traceback.format_exc()
                tb = _redact_secrets(tb)
                print(f"[AutoBlog] Step 6 failed: {type(exc).__name__}: {exc}. Traceback: {tb}", flush=True)
                return None

            print("[AutoBlog] TASK COMPLETED SUCCESSFULLY!", flush=True)
            return {
                "post": post,
                "topic": plan.topic,
                "keywords": [plan.primary_keyword] + plan.secondary_keywords,
            }
        finally:
            _RUN_LOCK.release()
    except Exception as e:
        print(f"[AutoBlog] CRITICAL TASK ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
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
        r'(?i)(api[_-]?key|secret|token|password)[\s:=]+[^\s]+',
        r'\1=***',
        text,
    )


# ─── Exports ──────────────────────────────────────────────────────────────────

__all__ = [
    "run_pipeline",
    "run_scheduled_job",
    "is_run_active",
]