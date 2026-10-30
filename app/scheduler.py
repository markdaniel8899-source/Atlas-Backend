from __future__ import annotations

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import get_settings
from app.services.blog.pipeline import (
    PipelineBusy,
    PipelineNotConfigured,
    execute_pipeline,
    start_pipeline,
)
from app.services.blog.publish import BlogRepository, BlogStorageError

logger = logging.getLogger(__name__)

JOB_ID = "atlas-daily-blog"


async def run_daily_blog() -> None:
    """Generate at most the configured number of articles per UTC day."""
    settings = get_settings()
    if not settings.blog_pipeline_ready:
        logger.warning("Scheduled blog run skipped: pipeline is not configured.")
        return

    repo = BlogRepository()
    midnight = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    try:
        completed_today = await repo.count_runs_since(midnight)
    except BlogStorageError as exc:
        logger.warning("Scheduled blog run skipped: %s", exc)
        return
    if completed_today >= settings.blog_daily_article_limit:
        logger.info(
            "Scheduled blog run skipped: daily limit of %s already reached.",
            settings.blog_daily_article_limit,
        )
        return

    try:
        run_id = await start_pipeline("cron", settings=settings)
    except PipelineBusy:
        logger.info("Scheduled blog run skipped: another run is active.")
        return
    except (PipelineNotConfigured, BlogStorageError, ValueError) as exc:
        logger.warning("Scheduled blog run could not start: %s", exc)
        return

    outcome = await execute_pipeline(
        run_id, "cron", settings=settings, repository=repo
    )
    logger.info(
        "Scheduled blog run %s finished with status %s",
        run_id,
        outcome.status,
    )


def start_scheduler() -> AsyncIOScheduler | None:
    settings = get_settings()
    if not settings.blog_cron_enabled:
        logger.info("Blog cron is disabled.")
        return None

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        run_daily_blog,
        CronTrigger(
            hour=settings.blog_cron_hour_utc,
            minute=settings.blog_cron_minute_utc,
            timezone="UTC",
        ),
        id=JOB_ID,
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.start()
    logger.info(
        "Blog cron scheduled daily at %02d:%02d UTC.",
        settings.blog_cron_hour_utc,
        settings.blog_cron_minute_utc,
    )
    return scheduler


def stop_scheduler(scheduler: AsyncIOScheduler | None) -> None:
    if scheduler is not None and scheduler.running:
        scheduler.shutdown(wait=False)
