"""Routes for AutoBlog v2 - POST /api/blog/generate with admin auth."""

from __future__ import annotations

import asyncio
import secrets
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.pipeline import run_pipeline, is_run_active
from app.autoblog.scheduler import get_scheduler_status
from app.deps import fetch_profile_role, verify_bearer_token

router = APIRouter(prefix="/api/blog", tags=["blog-admin"])


async def require_blog_admin(
    authorization: str | None = Header(default=None),
    x_admin_secret: str | None = Header(default=None),
) -> None:
    """Gate privileged endpoints: 403 unless the caller is an admin.

    Two accepted proofs:
      * X-Admin-Secret header matching the server's ADMIN_SECRET
        (for cron jobs / scripts), compared in constant time; or
      * a valid Supabase session JWT whose profile row has role = 'admin'.

    Every failure - missing header, bad secret, invalid token, non-admin
    user, unverifiable role - returns 403 Forbidden.
    """
    settings = get_autoblog_settings()
    provided = (x_admin_secret or "").strip()
    expected = settings.admin_secret
    if expected and provided and secrets.compare_digest(provided, expected):
        return

    try:
        user_id = verify_bearer_token(authorization)
    except HTTPException:
        raise HTTPException(status_code=403, detail="Admin access required.")

    role = await asyncio.to_thread(fetch_profile_role, user_id)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")


@router.post("/generate", status_code=202)
async def generate_blog_post(
    background_tasks: BackgroundTasks,
    request: Request,
    _admin: None = Depends(require_blog_admin),
) -> dict[str, Any]:
    """Kick off the AutoBlog v2 pipeline in the background (returns 202 immediately).

    The work runs as a FastAPI BackgroundTask:
    - Tavily research + Nemotron Ultra topic/keywords
    - Groq meta + article (automatic Gemini fallback on Groq 429/5xx)
    - Pixabay photos (cover + 1-3 in-article, one by one)
    - Supabase save

    The pauses keep every provider under its rate limit, so the run takes a few minutes.
    """
    settings = get_autoblog_settings()

    if not settings.blog_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Auto Blog is not configured: set GROQ_BLOG_API_KEY (dedicated "
                "blog writer) and SUPABASE_SERVICE_ROLE_KEY on the server. "
                "GOOGLE_API_KEY enables the Gemini fallback; NVIDIA_API_KEY + "
                "TAVILY_API_KEY power the research steps."
            ),
        )

    if is_run_active():
        raise HTTPException(
            status_code=409,
            detail="A blog generation run is already in progress.",
        )

    background_tasks.add_task(run_pipeline)
    print(
        "[AutoBlog Background] POST /api/blog/generate -> task queued "
        "(run_pipeline). Next log line MUST be "
        "'[AutoBlog] TASK STARTED EXECUTION!'.",
        flush=True,
    )
    return {
        "status": "success",
        "message": (
            "Blog generation started in background. It will take a few "
            "minutes to avoid rate limits."
        ),
    }


@router.get("/status")
async def blog_status(
    _admin: None = Depends(require_blog_admin),
) -> dict[str, Any]:
    """Get AutoBlog scheduler and configuration status."""
    settings = get_autoblog_settings()
    scheduler_status = get_scheduler_status()

    return {
        "configured": settings.blog_ready,
        "research_ready": settings.research_ready,
        "images_ready": settings.images_ready,
        "gemini_fallback_ready": settings.gemini_fallback_ready,
        "draft_mode": settings.autoblog_draft,
        "scheduler": scheduler_status,
    }