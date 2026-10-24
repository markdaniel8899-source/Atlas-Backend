"""Pipeline orchestrator for AutoBlog v2 - Staggered 6-chunk pipeline."""

from __future__ import annotations

import asyncio
import sys
import threading
import traceback
from typing import Any

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.research import research_and_select_topic, extract_keywords, analyze_data_and_profile
from app.autoblog.writer import write_article, generate_meta, extract_keywords_nim
from app.autoblog.checks import evaluate_article, GateResult
from app.autoblog.images import fetch_cover_image, inject_images_into_content
from app.autoblog.publisher import publish_post
from app.autoblog.planner import build_profile
from app.autoblog.llm.clients import BlogPipelineError


# Global lock - only one pipeline run at a time (route + scheduler share this)
_RUN_LOCK = threading.Lock()

# Timing constants (from settings in practice)
RESEARCH_PAUSE_SECONDS = 2
WRITE_PAUSE_SECONDS = 3


def is_run_active() -> bool:
    """True while a pipeline run holds the lock (route 409 pre-check)."""
    return _RUN_LOCK.locked()


async def run_pipeline() -> dict[str, Any] | None:
    """Staggered 6-chunk pipeline with deliberate API pauses (never raises).

    Designed to run inside a FastAPI BackgroundTask:
      - the FIRST line prints "TASK STARTED EXECUTION!" so we can prove the
        task actually entered the coroutine,
      - the ENTIRE body is wrapped in one try/except that prints
        "CRITICAL TASK ERROR" + a traceback, so nothing can die silently,
      - every step logs a [AutoBlog] Step N line (flushed immediately)
        before and after the call,
      - sleeps use `asyncio.sleep` only (this is an `async def` running on
        the event loop - `time.sleep` would block every request).

    Returns the saved row plus research metadata, or None when any chunk
    failed - the run aborts gracefully without ever crashing the server.
    """
    print("[AutoBlog] TASK STARTED EXECUTION!", flush=True)
    try:
        if not _RUN_LOCK.acquire(blocking=False):
            print(
                "[AutoBlog] a run is already in progress - staggered run skipped.",
                flush=True,
            )
            return None
        try:
            print(
                f"[AutoBlog] staggered pipeline start "
                f"{_utc_iso()}",
                flush=True,
            )

            # Step 1/6 - Research + Topic Selection
            results: list[dict[str, str]] = []
            print(
                "[AutoBlog] Step 1: Starting research (recent-post dedupe + "
                "rotating Tavily news + NVIDIA Nemotron topic)...",
                flush=True,
            )
            try:
                from app.autoblog.sources import _fetch_recent_posts
                settings = get_autoblog_settings()
                recent_posts = await asyncio.to_thread(
                    _fetch_recent_posts, settings.recent_posts_limit
                )
                print(
                    f"[AutoBlog] Step 1/6: Loaded {len(recent_posts)} recent post(s) "
                    f"for dedupe.",
                    flush=True,
                )
                research, results = await research_and_select_topic()
                print(
                    f"[AutoBlog] Step 1/6: Research complete "
                    f"({len(results)} results, topic={research.topic!r}). "
                    f"Waiting {RESEARCH_PAUSE_SECONDS}s...",
                    flush=True,
                )
            except Exception as exc:
                from app.autoblog.sources import FALLBACK_TOPICS, _day_index
                topic = FALLBACK_TOPICS[_day_index() % len(FALLBACK_TOPICS)]
                from app.autoblog.research import Research
                research = Research(topic=topic, keywords=[], snippets=[])
                print(
                    f"[AutoBlog] Step 1/6: Research failed "
                    f"({type(exc).__name__}: {exc}); fallback topic. "
                    f"Waiting {RESEARCH_PAUSE_SECONDS}s...",
                    flush=True,
                )
            await asyncio.sleep(RESEARCH_PAUSE_SECONDS)

            # Step 2/6 - Keywords + Article Type (Nemotron Ultra)
            print(
                "[AutoBlog] Step 2: Extracting keywords + article type "
                "(NVIDIA Nemotron Ultra)...",
                flush=True,
            )
            try:
                keywords, article_type = await extract_keywords_nim(
                    research.topic, results
                )
                research.keywords = keywords
                research.article_type = article_type
                print(
                    f"[AutoBlog] Step 2/6: Keywords complete "
                    f"({', '.join(research.keywords)}, type={research.article_type}). "
                    f"Waiting {RESEARCH_PAUSE_SECONDS}s...",
                    flush=True,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                tb = _redact_secrets(tb)
                print(
                    f"[AutoBlog] Step 2/6: Keywords failed "
                    f"({type(exc).__name__}: {exc}). Traceback: {tb}. Aborting run.",
                    flush=True,
                )
                return None
            await asyncio.sleep(RESEARCH_PAUSE_SECONDS)

            # Step 2b/6 - Data Availability & Article Profile
            print("[AutoBlog] Step 2b: Analyzing data availability...", flush=True)
            try:
                data, profile = await analyze_data_and_profile(
                    research.topic, research.article_type, research.snippets
                )
                if profile.get("downgraded"):
                    print(f"[AutoBlog] Profile downgraded: {profile.get('downgrade_reason')}")
                else:
                    print(
                        f"[AutoBlog] Profile: {profile['article_type']}, "
                        f"chart_data={data.has_chart_data}, table_data={data.has_table_data}"
                    )
                print(
                    f"[AutoBlog] Step 2b/6: Data analysis complete. "
                    f"Waiting {RESEARCH_PAUSE_SECONDS}s...",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[AutoBlog] Step 2b/6: Data analysis failed "
                    f"({type(exc).__name__}: {exc}). Using default profile.",
                    flush=True,
                )
                from app.autoblog.planner import ARTICLE_PROFILES
                profile = ARTICLE_PROFILES.get(research.article_type, ARTICLE_PROFILES["explainer"])
            await asyncio.sleep(RESEARCH_PAUSE_SECONDS)

            # Step 3/6 - Meta title + description (Groq -> Gemini on 429/5xx)
            print(
                "[AutoBlog] Step 3: Generating meta title and description "
                "(Groq -> Gemini fallback)...",
                flush=True,
            )
            try:
                primary_keyword = research.keywords[0] if research.keywords else research.topic
                other_keywords = ", ".join(research.keywords[1:]) or "(none)"
                sources = "\n".join(research.snippets[:3]) or "(none)"
                meta_title, meta_description = await generate_meta(
                    research.topic, primary_keyword, other_keywords, sources
                )
                print(
                    f"[AutoBlog] Step 3/6: Meta data complete. "
                    f"Waiting {WRITE_PAUSE_SECONDS}s...",
                    flush=True,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                tb = _redact_secrets(tb)
                print(
                    f"[AutoBlog] Step 3/6: Meta failed "
                    f"({type(exc).__name__}: {exc}). Traceback: {tb}. Aborting run.",
                    flush=True,
                )
                return None
            await asyncio.sleep(WRITE_PAUSE_SECONDS)

            # Step 4/6 - Article writing (Groq -> Gemini on 429/5xx)
            print(
                "[AutoBlog] Step 4: Writing article "
                "(Groq -> Gemini fallback)...",
                flush=True,
            )
            try:
                article = await write_article(
                    research.topic,
                    primary_keyword,
                    other_keywords,
                    "\n".join(research.snippets) or "(none)",
                    profile=profile,
                )
                print(
                    f"[AutoBlog] Step 4/6: Article written. "
                    f"Waiting {WRITE_PAUSE_SECONDS}s...",
                    flush=True,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                tb = _redact_secrets(tb)
                print(
                    f"[AutoBlog] Step 4/6: Writing failed "
                    f"({type(exc).__name__}: {exc}). Traceback: {tb}. Aborting run.",
                    flush=True,
                )
                return None
            await asyncio.sleep(WRITE_PAUSE_SECONDS)

            # Content Gate - hard checks, all must pass
            rewrite_count = 0
            max_rewrites = 1
            json_repaired = article.get("_json_meta", {}).get("repaired", False)

            while True:
                gate = evaluate_article(
                    {
                        "title": article["title"],
                        "excerpt": article["excerpt"],
                        "content_html": article["content_html"],
                        "visual": article.get("visuals", [None])[0] if article.get("visuals") else None,
                        "tags": article["tags"],
                    },
                    meta_title,
                    article["excerpt"],
                    research,
                    json_repaired,
                )
                if gate.passed:
                    print("[AutoBlog] Content gate passed.", flush=True)
                    break
                if gate.meta_only and rewrite_count == 0:
                    print("[AutoBlog] Meta-only failure, regenerating meta...", flush=True)
                    meta_title, meta_description = await generate_meta(
                        research.topic, primary_keyword, other_keywords,
                        "\n".join(research.snippets[:3]) or "(none)"
                    )
                    continue
                if rewrite_count >= 1:
                    print(f"[AutoBlog] Content gate FAILED after rewrite: {gate.reasons}", flush=True)
                    return None
                print(
                    f"[AutoBlog] Content gate failed: {gate.reasons} - "
                    f"rewrite {rewrite_count + 1}/{max_rewrites}",
                    flush=True,
                )
                rewrite_count += 1
                try:
                    article = await write_article(
                        research.topic,
                        primary_keyword,
                        other_keywords,
                        "\n".join(research.snippets) or "(none)",
                        seo_feedback="; ".join(gate.reasons),
                        profile=profile,
                    )
                except Exception as exc:
                    print(
                        f"[AutoBlog] Rewrite failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    return None

            # Step 5/6 - Images (cover first, then 1-3 unique in-article images)
            print(
                "[AutoBlog] Step 5: Fetching images (Pixabay, fallback Unsplash)...",
                flush=True,
            )
            try:
                cover, cover_source = await fetch_cover_image(
                    meta_title, research.keywords
                )
                content_html, content_images = await inject_images_into_content(
                    article["content_html"],
                    research.keywords,
                    exclude_urls={cover, cover_source},
                    profile=profile,
                )
                print(
                    f"[AutoBlog] Step 5/6: Images complete "
                    f"({len(content_images)} in-article + cover).",
                    flush=True,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                tb = _redact_secrets(tb)
                print(
                    f"[AutoBlog] Step 5/6: Images failed "
                    f"({type(exc).__name__}: {exc}). Traceback: {tb}. Aborting run.",
                    flush=True,
                )
                return None

            # Step 6/6 - Upload (Supabase)
            print("[AutoBlog] Step 6: Saving post to Supabase...", flush=True)
            try:
                post = await publish_post(
                    title=meta_title,
                    content_html=content_html,
                    excerpt=meta_description,
                    cover_image_url=cover,
                    content_images=content_images,
                    tags=article["tags"],
                    keywords=research.keywords,
                )
                print(
                    f"[AutoBlog] Step 6/6: Saved post "
                    f"id={post.get('id')} slug={post.get('slug')}",
                    flush=True,
                )
                print("[AutoBlog] TASK COMPLETED SUCCESSFULLY!", flush=True)
                return {
                    "post": post,
                    "topic": research.topic,
                    "keywords": research.keywords,
                }
            except Exception as exc:
                tb = traceback.format_exc()
                tb = _redact_secrets(tb)
                print(
                    f"[AutoBlog] Step 6/6: Upload failed "
                    f"({type(exc).__name__}: {exc}). Traceback: {tb}. Aborting run.",
                    flush=True,
                )
                return None
        finally:
            _RUN_LOCK.release()
    except Exception as e:
        print(f"[AutoBlog] CRITICAL TASK ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return None


def run_scheduled_job() -> None:
    """APScheduler entry point (runs on a worker thread - owns its own loop)."""
    try:
        result = asyncio.run(run_pipeline())
        if result:
            print(f"[AutoBlog] scheduled run OK -> /blog/{result['post'].get('slug')}")
        else:
            print("[AutoBlog] scheduled run exited gracefully (see error above)")
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