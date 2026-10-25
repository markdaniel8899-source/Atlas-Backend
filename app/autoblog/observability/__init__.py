"""Observability module for pipeline runs and LLM call logging."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.autoblog.config.settings import get_autoblog_settings


@dataclass
class LLMStep:
    """Record of a single LLM call."""
    step: str
    model: str
    fallback_used: bool
    tokens_in: int
    tokens_out: int
    duration_ms: int
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PipelineStep:
    """Record of a pipeline step."""
    step: str
    model: Optional[str] = None
    fallback_used: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PipelineRun:
    """Complete pipeline run record."""
    id: str
    started_at: str
    finished_at: Optional[str] = None
    topic: str = ""
    slug: str = ""
    status: str = "running"  # running, published, draft, failed
    seo_score: int = 0
    attempts: int = 0
    fact_check_json: Optional[str] = None
    visual_plan_json: Optional[str] = None
    steps_json: str = "[]"
    error: Optional[str] = None


class PipelineRunLogger:
    """Logger for pipeline runs with automatic step tracking."""

    def __init__(self):
        self.run_id = str(uuid.uuid4())[:8]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.topic = ""
        self.slug = ""
        self.status = "running"
        self.seo_score = 0
        self.attempts = 0
        self.fact_check_json = None
        self.visual_plan_json = None
        self.steps: list[PipelineStep] = []
        self.fact_check_data: Optional[dict] = None
        self.visual_plan_data: Optional[dict] = None
        self.error: Optional[str] = None

    def set_topic(self, topic: str, slug: str = "") -> None:
        self.topic = topic
        self.slug = slug

    def set_fact_check(self, fact_check: dict) -> None:
        self.fact_check_data = fact_check
        self.fact_check_json = __import__('json').dumps(fact_check, ensure_ascii=False)

    def set_visual_plan(self, visual_plan: dict) -> None:
        self.visual_plan_data = visual_plan
        self.visual_plan_json = __import__('json').dumps(visual_plan, ensure_ascii=False)

    def record_step(self, step: PipelineStep) -> None:
        self.steps.append(step)

    def set_final_status(
        self,
        status: str,
        seo_score: int = 0,
        attempts: int = 1,
        error: Optional[str] = None,
    ) -> None:
        self.status = status
        self.seo_score = seo_score
        self.attempts = attempts
        self.error = error

    def to_dict(self) -> dict:
        return {
            "id": self.run_id,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "topic": self.topic,
            "slug": self.slug,
            "status": self.status,
            "seo_score": self.seo_score,
            "attempts": self.attempts,
            "fact_check_json": self.fact_check_json,
            "visual_plan_json": self.visual_plan_json,
            "steps_json": __import__('json').dumps([asdict(s) for s in self.steps], ensure_ascii=False),
            "error": self.error,
        }

    async def save(self) -> None:
        """Save to Supabase pipeline_runs table."""
        # In production, insert into Supabase
        # For now, just log
        import logging
        logger = __import__('logging').getLogger(__name__)
        logger.info(f"Pipeline run {self.run_id}: {self.status} - {self.topic}")


@asynccontextmanager
async def llm_call(
    logger: PipelineRunLogger,
    step_name: str,
    model: str,
    call_fn: Callable,
    *args,
    **kwargs,
):
    """
    Context manager to wrap LLM calls with automatic logging.

    Usage:
        async with llm_call(logger, "trend_discovery", "groq:gpt-oss-120b", groq_research, system, prompt):
            result = await call_fn()
    """
    start = time.perf_counter()
    fallback_used = False
    error = None
    tokens_in = 0
    tokens_out = 0

    try:
        result = await call_fn()
        return result
    except Exception as exc:
        error = str(exc)
        fallback_used = True
        raise
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        step = PipelineStep(
            step=step_name,
            fallback_used=fallback_used,
            duration_ms=duration_ms,
            error=error,
        )
        # Note: tokens_in/out would need to be extracted from result
        # This is a simplified version
        # In production, extract from result object


async def tracked_llm_call(
    logger: PipelineRunLogger,
    step_name: str,
    model: str,
    call_fn: Callable,
    *args,
    **kwargs,
) -> Any:
    """
    Track an LLM call with automatic logging.

    Usage:
        result = await tracked_llm_call(logger, "trend_discovery", "groq:gpt-oss-120b", groq_research, system, prompt)
    """
    start = time.perf_counter()
    fallback_used = False
    error = None
    tokens_in = 0
    tokens_out = 0

    try:
        result = await call_fn()
        return result
    except Exception as exc:
        error = str(exc)
        fallback_used = True
        raise
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        step = PipelineStep(
            step=step_name,
            model=model,
            fallback_used=fallback_used,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            error=error,
        )
        # In production, save to logger
        # logger.record_step(step)


def create_run_logger() -> PipelineRunLogger:
    """Create a new pipeline run logger."""
    return PipelineRunLogger()


async def save_pipeline_run(logger: PipelineRunLogger) -> None:
    """Save pipeline run to database."""
    await logger.save()


__all__ = [
    "PipelineRunLogger",
    "PipelineStep",
    "PipelineRun",
    "llm_call",
    "tracked_llm_call",
    "create_run_logger",
    "save_pipeline_run",
]