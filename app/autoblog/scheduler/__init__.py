"""Scheduler for AutoBlog v2 - APScheduler integration."""

from __future__ import annotations

from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

from app.autoblog.config.settings import get_autoblog_settings
from app.autoblog.pipeline import run_scheduled_job, is_run_active


def setup_scheduler(app: FastAPI) -> None:
    """Configure and start the AutoBlog scheduler on app startup."""
    settings = get_autoblog_settings()

    # Store scheduler in app state for shutdown
    app.state.autoblog_scheduler = None

    if not settings.blog_ready:
        print(
            "[AutoBlog] scheduler disabled - set GROQ_BLOG_API_KEY and "
            "SUPABASE_SERVICE_ROLE_KEY (manual POST /api/blog/generate "
            "will also be unavailable)"
        )
        return

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_scheduled_job,
        trigger=CronTrigger(
            hour=settings.autoblog_schedule_hour,
            minute=settings.autoblog_schedule_minute,
            timezone="UTC",
        ),
        id="autoblog-daily",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,  # boot within 1h of schedule -> still fires
    )
    scheduler.start()
    app.state.autoblog_scheduler = scheduler
    print(
        f"[AutoBlog] scheduled - every day at "
        f"{settings.autoblog_schedule_hour:02d}:{settings.autoblog_schedule_minute:02d} UTC"
    )


def shutdown_scheduler(app: FastAPI) -> None:
    """Shutdown the scheduler on app shutdown."""
    scheduler = getattr(app.state, "autoblog_scheduler", None)
    if scheduler is not None:
        scheduler.shutdown(wait=False)
        print("[AutoBlog] scheduler shutdown complete")


def get_scheduler_status() -> dict[str, Any]:
    """Get scheduler status for health/debug endpoints."""
    settings = get_autoblog_settings()
    return {
        "enabled": settings.blog_ready,
        "schedule": f"{settings.autoblog_schedule_hour:02d}:{settings.autoblog_schedule_minute:02d} UTC",
        "max_posts_per_day": settings.autoblog_max_posts_per_day,
        "catchup_enabled": settings.autoblog_catchup_enabled,
        "run_active": is_run_active(),
    }