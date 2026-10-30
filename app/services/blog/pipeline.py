from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence

from app.config import Settings, get_settings

from .faq import generate_faqs, insert_faq_section
from .gate import run_gate
from .keywords import generate_keywords
from .meta import generate_meta
from .providers import BlogProviderClient
from .publish import BlogRepository
from .research import ResearchService
from .topics import (
    category_counts,
    generate_topic_candidates,
    seed_candidates,
    select_topic,
)
from .types import (
    FAQBundle,
    GateReport,
    KeywordPlan,
    MetaResult,
    ResearchBundle,
    TopicCandidate,
    TopicHistoryEntry,
    VisualPlan,
    VisualProduction,
    WriterOutput,
)
from .vercel import DeployError, trigger_deploy
from .visuals import PixabayClient, VisualService, plan_visuals
from .writer import repair_article, word_count, write_article

logger = logging.getLogger(__name__)

VALID_TRIGGERS = {"cron", "admin", "manual", "retry"}


class PipelineError(RuntimeError):
    pass


class PipelineNotConfigured(PipelineError):
    pass


class PipelineBusy(PipelineError):
    pass


class PipelineStageError(PipelineError):
    pass


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    run_id: str
    status: str
    post_id: str | None = None
    score: int | None = None
    word_count: int | None = None
    error: str | None = None


def _short(value: Any, limit: int = 1800) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _attempt_payload(attempts: Sequence[Any]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for attempt in attempts:
        payload.append(
            {
                "provider": str(getattr(attempt, "provider", "") or ""),
                "model": str(getattr(attempt, "model", "") or ""),
                "attempt": int(getattr(attempt, "attempt", 0) or 0),
                "status": str(getattr(attempt, "status", "") or ""),
                "duration_ms": int(getattr(attempt, "duration_ms", 0) or 0),
                "error": getattr(attempt, "error", None),
            }
        )
    return payload


def _visual_payload(
    production: VisualProduction,
) -> dict[str, Any]:
    return {
        "cover_image_url": production.cover_image_url,
        "cover_alt": production.cover_alt,
        "cover_caption": production.cover_caption,
        "cover_width": production.cover_width,
        "cover_height": production.cover_height,
        "content_images": [
            image.model_dump(mode="json")
            for image in production.content_images
        ],
        "visuals": [
            visual.model_dump(mode="json") for visual in production.visuals
        ],
    }


class PipelineRunner:
    def __init__(
        self,
        run_id: str,
        trigger: str = "admin",
        *,
        settings: Settings | Any | None = None,
        repository: BlogRepository | None = None,
        client: BlogProviderClient | None = None,
    ) -> None:
        self.run_id = run_id
        self.trigger = trigger
        self.settings = settings or get_settings()
        self.repo = repository or BlogRepository()
        self.client = client or BlogProviderClient(settings=self.settings)
        self.provider_summary: dict[str, Any] = {}
        self.history: list[TopicHistoryEntry] = []
        self.topic: TopicCandidate | None = None
        self.topic_id: str | None = None
        self.research: ResearchBundle | None = None
        self.keywords: KeywordPlan | None = None
        self.output: WriterOutput | None = None
        self.post_id: str | None = None
        self.plan: VisualPlan | None = None
        self.production: VisualProduction | None = None
        self.report: GateReport | None = None
        self.meta: MetaResult | None = None
        self.faqs: FAQBundle | None = None
        self.current_stage = "preflight"

    async def run(self) -> PipelineOutcome:
        try:
            async with self._stage("preflight") as meta:
                self._require_ready()
                meta["ready"] = True

            topic, topic_id = await self._topic_stage()
            research = await self._research_stage(topic)
            keywords = await self._keyword_stage(topic, research)
            output = await self._writer_stage(topic, keywords, research)
            plan, production = await self._visual_stage(
                topic, output, research
            )
            output, report = await self._gate_stage(
                topic, keywords, research, output, plan, production
            )
            meta = await self._meta_stage(topic, keywords, output)
            bundle, output = await self._faq_stage(
                topic, keywords, research, output
            )
            final_words = word_count(output.content)
            await self._publish_stage(
                topic,
                keywords,
                research,
                output,
                report,
                meta,
                bundle,
                final_words,
            )
            await self._deploy_stage()
            await self.repo.update_run(
                self.run_id, current_stage="completed"
            )
            await self.repo.finish_run(
                self.run_id,
                "completed",
                score=report.score,
                word_count=final_words,
                provider_summary=self.provider_summary,
            )
            return PipelineOutcome(
                run_id=self.run_id,
                status="completed",
                post_id=self.post_id,
                score=report.score,
                word_count=final_words,
            )
        except Exception as exc:  # noqa: BLE001 - failures are persisted, not raised
            logger.exception("Blog pipeline run %s failed", self.run_id)
            error = _short(exc)
            score = self.report.score if self.report else None
            words = (
                self.report.metrics.word_count if self.report else None
            )
            try:
                await self.repo.update_run(
                    self.run_id,
                    current_stage=self.current_stage,
                    error=error,
                )
                await self.repo.finish_run(
                    self.run_id,
                    "failed",
                    error=error,
                    score=score,
                    word_count=words,
                    provider_summary=self.provider_summary,
                )
            except Exception:  # noqa: BLE001 - preserve the original failure
                logger.exception(
                    "Could not persist failure for run %s", self.run_id
                )
            return PipelineOutcome(
                run_id=self.run_id,
                status="failed",
                post_id=self.post_id,
                score=score,
                word_count=words,
                error=error,
            )

    def _require_ready(self) -> None:
        if not self.settings.blog_enabled:
            raise PipelineNotConfigured("Blog generation is disabled.")
        if not self.settings.blog_pipeline_ready:
            raise PipelineNotConfigured(
                "Blog pipeline configuration is incomplete."
            )

    @asynccontextmanager
    async def _stage(
        self, name: str
    ) -> AsyncIterator[dict[str, Any]]:
        self.current_stage = name
        await self.repo.update_run(self.run_id, current_stage=name)
        started = time.perf_counter()
        step_id = await self._safe_start_step(name)
        meta: dict[str, Any] = {}
        try:
            yield meta
        except Exception as exc:  # noqa: BLE001 - stage boundary reporting
            await self._safe_finish_step(
                step_id,
                name,
                started,
                status="failed",
                detail=meta,
                error=_short(exc),
            )
            raise
        else:
            await self._safe_finish_step(
                step_id,
                name,
                started,
                status="completed",
                provider=meta.get("provider"),
                model=meta.get("model"),
                detail=meta,
            )

    async def _safe_start_step(self, name: str) -> int | None:
        try:
            return await self.repo.start_step(self.run_id, name)
        except Exception:  # noqa: BLE001 - step logging is best-effort
            logger.exception("Could not start step %s", name)
            return None

    async def _safe_finish_step(
        self,
        step_id: int | None,
        name: str,
        started: float,
        *,
        status: str,
        provider: str | None = None,
        model: str | None = None,
        detail: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        try:
            await self.repo.finish_step(
                step_id,
                status=status,
                duration_ms=int((time.perf_counter() - started) * 1000),
                provider=provider,
                model=model,
                detail=detail,
                error=error,
            )
        except Exception:  # noqa: BLE001 - step logging is best-effort
            logger.exception("Could not finish step %s", name)

    async def _record_attempts(
        self, stage: str, attempts: Sequence[Any]
    ) -> None:
        if not attempts:
            return
        self.provider_summary[stage] = _attempt_payload(attempts)
        try:
            await self.repo.record_provider_usage(attempts)
        except Exception:  # noqa: BLE001 - usage persistence is best-effort
            logger.exception("Could not record provider usage for %s", stage)

    async def _topic_stage(self) -> tuple[TopicCandidate, str]:
        async with self._stage("topic") as meta:
            self.history = await self.repo.topic_history()
            candidates = list(seed_candidates())
            try:
                result = await generate_topic_candidates(
                    self.client,
                    recent_topics=[
                        entry.topic for entry in self.history[:30]
                    ],
                    recent_keywords=[
                        entry.primary_keyword
                        for entry in self.history[:30]
                        if entry.primary_keyword
                    ],
                    counts=category_counts(self.history),
                )
                candidates = [*result.data, *candidates]
                meta.update(
                    provider=result.provider,
                    model=result.model,
                    fallback=False,
                )
                await self._record_attempts("topic", result.attempts)
            except Exception as exc:  # noqa: BLE001 - seeds are the fallback
                logger.warning("Topic provider failed: %s", exc)
                meta.update(fallback=True, reason=_short(exc, 400))
                self.provider_summary["topic"] = {
                    "fallback": True,
                    "reason": _short(exc, 400),
                }

            selected = select_topic(candidates, self.history)
            if selected is None:
                raise PipelineStageError(
                    "No eligible topic remained after dedupe and caps."
                )
            topic_id = await self.repo.save_topic(selected)
            await self.repo.update_run(self.run_id, topic_id=topic_id)
            self.topic = selected
            self.topic_id = topic_id
            meta.update(
                topic=selected.topic,
                category=selected.category.value,
                topic_id=topic_id,
                score=selected.total_score,
            )
            return selected, topic_id

    async def _research_stage(
        self, topic: TopicCandidate
    ) -> ResearchBundle:
        async with self._stage("research") as meta:
            research = await ResearchService().research(
                topic.topic, topic.primary_keyword
            )
            self.research = research
            meta.update(
                provider="tavily",
                sources=len(research.sources),
                facts=len(research.numeric_facts),
            )
            return research

    async def _keyword_stage(
        self, topic: TopicCandidate, research: ResearchBundle
    ) -> KeywordPlan:
        async with self._stage("keywords") as meta:
            result = await generate_keywords(
                self.client, topic, research
            )
            self.keywords = result.data
            meta.update(provider=result.provider, model=result.model)
            await self._record_attempts("keywords", result.attempts)
            return result.data

    async def _writer_stage(
        self,
        topic: TopicCandidate,
        keywords: KeywordPlan,
        research: ResearchBundle,
    ) -> WriterOutput:
        async with self._stage("writer") as meta:
            result = await write_article(
                self.client,
                topic,
                keywords,
                research,
                min_words=self.settings.blog_min_words,
                max_words=self.settings.blog_max_words,
            )
            output = await self._unique_slug(result.output)
            post_id = await self.repo.insert_draft(
                output,
                run_id=self.run_id,
                topic_id=self.topic_id or "",
                topic=topic,
                keywords=keywords,
            )
            self.output = output
            self.post_id = post_id
            await self.repo.update_run(self.run_id, post_id=post_id)
            await self.repo.insert_sources(post_id, research)
            meta.update(
                provider=result.provider,
                model=result.model,
                post_id=post_id,
                word_count=word_count(output.content),
            )
            await self._record_attempts("writer", result.attempts)
            return output

    async def _unique_slug(
        self,
        output: WriterOutput,
        *,
        exclude_post_id: str | None = None,
    ) -> WriterOutput:
        if not await self.repo.slug_exists(
            output.slug, exclude_post_id=exclude_post_id
        ):
            return output
        base = output.slug[:70].rstrip("-")
        for index in range(2, 50):
            candidate = f"{base}-{index}"
            if not await self.repo.slug_exists(
                candidate, exclude_post_id=exclude_post_id
            ):
                return output.model_copy(update={"slug": candidate})
        raise PipelineStageError("Could not produce a unique blog slug.")

    async def _produce_visuals(
        self,
        topic: TopicCandidate,
        output: WriterOutput,
        research: ResearchBundle,
    ) -> tuple[VisualPlan, VisualProduction]:
        content_words = word_count(output.content)
        required = (
            2
            if content_words <= self.settings.blog_visual_word_threshold
            else 3
        )
        service = VisualService(
            pixabay=PixabayClient(self.settings.pixabay_api_key)
        )
        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                plan = await plan_visuals(
                    self.client,
                    output.content,
                    research,
                    required_count=required,
                )
                production = await service.produce(
                    plan,
                    output.content,
                    research,
                    slug=output.slug,
                    title=output.title,
                    category=topic.category,
                )
                return plan, production
            except Exception as exc:  # noqa: BLE001 - one visual retry
                last_error = exc
                if attempt == 1:
                    await asyncio.sleep(1.0)
        raise PipelineStageError(
            f"Visual production failed: {_short(last_error)}"
        )

    async def _visual_stage(
        self,
        topic: TopicCandidate,
        output: WriterOutput,
        research: ResearchBundle,
    ) -> tuple[VisualPlan, VisualProduction]:
        async with self._stage("visuals") as meta:
            plan, production = await self._produce_visuals(
                topic, output, research
            )
            self.plan = plan
            self.production = production
            if not self.post_id:
                raise PipelineStageError("Draft post is missing.")
            await self.repo.record_media(self.post_id, production)
            await self.repo.update_post(
                self.post_id,
                content=production.content,
                cover_image_url=production.cover_image_url,
                content_images=[
                    image.model_dump(mode="json")
                    for image in production.content_images
                ],
                visuals_json=_visual_payload(production),
            )
            meta.update(
                provider=self.client.last_provider or "local",
                model=self.client.last_model,
                media="pixabay"
                if production.content_images
                else "generated",
                visuals=len(production.visuals),
            )
            return plan, production

    def _run_gate(
        self,
        topic: TopicCandidate,
        keywords: KeywordPlan,
        research: ResearchBundle,
        output: WriterOutput,
        production: VisualProduction,
        plan: VisualPlan,
    ) -> GateReport:
        def collision_checker(*_: Any, **__: Any) -> bool:
            return self.repo.title_or_slug_exists_sync(
                output.title,
                output.slug,
                exclude_post_id=self.post_id,
            )

        return run_gate(
            output,
            keywords,
            research,
            production,
            topic=topic,
            history=self.history,
            visual_plan=plan,
            min_score=max(90, self.settings.blog_min_score),
            visual_word_threshold=self.settings.blog_visual_word_threshold,
            collision_checker=collision_checker,
        )

    async def _gate_stage(
        self,
        topic: TopicCandidate,
        keywords: KeywordPlan,
        research: ResearchBundle,
        output: WriterOutput,
        plan: VisualPlan,
        production: VisualProduction,
    ) -> tuple[WriterOutput, GateReport]:
        async with self._stage("gate") as meta:
            report = await asyncio.to_thread(
                self._run_gate,
                topic, keywords, research, output, production, plan
            )
            repairs = 0
            max_repairs = max(0, int(self.settings.blog_max_repairs))
            while not report.passed and repairs < max_repairs:
                repairs += 1
                try:
                    async with self._stage("repair") as repair_meta:
                        repaired = await repair_article(
                            self.client,
                            topic,
                            keywords,
                            research,
                            output,
                            report,
                            min_words=self.settings.blog_min_words,
                            max_words=self.settings.blog_max_words,
                        )
                        repaired_output = await self._unique_slug(
                            repaired.output,
                            exclude_post_id=self.post_id,
                        )
                        plan, production = await self._produce_visuals(
                            topic, repaired_output, research
                        )
                        report = await asyncio.to_thread(
                            self._run_gate,
                            topic,
                            keywords,
                            research,
                            repaired_output,
                            production,
                            plan,
                        )
                        repair_meta.update(
                            provider=repaired.provider,
                            model=repaired.model,
                            attempt=repairs,
                            score=report.score,
                        )
                        await self._record_attempts(
                            "repair", repaired.attempts
                        )
                    output = repaired_output
                    self.plan = plan
                    self.production = production
                    if self.post_id:
                        await self.repo.record_media(
                            self.post_id, production
                        )
                        await self.repo.update_post(
                            self.post_id,
                            title=output.title,
                            slug=output.slug,
                            excerpt=output.excerpt,
                            content=production.content,
                            cover_image_url=production.cover_image_url,
                            content_images=[
                                image.model_dump(mode="json")
                                for image in production.content_images
                            ],
                            visuals_json=_visual_payload(production),
                        )
                except Exception as exc:  # noqa: BLE001 - keep the draft
                    logger.warning("Gate repair failed: %s", exc)
                    meta["repair_error"] = _short(exc, 600)
                    break

            self.output = output
            self.report = report
            if self.post_id:
                await self.repo.update_post(
                    self.post_id,
                    content=production.content,
                    cover_image_url=production.cover_image_url,
                    content_images=[
                        image.model_dump(mode="json")
                        for image in production.content_images
                    ],
                    visuals_json=_visual_payload(production),
                    gate_report=report.model_dump(mode="json"),
                    seo_score=report.score,
                    word_count=report.metrics.word_count,
                )
            meta.update(
                score=report.score,
                passed=report.passed,
                repairs=repairs,
                failures=len(report.hard_failures),
            )
            if not report.passed:
                detail = "; ".join(
                    f"{item.id}: {item.detail}"
                    for item in report.hard_failures[:6]
                )
                raise PipelineStageError(
                    f"Content gate failed: {_short(detail, 900)}"
                )
            return output, report

    async def _meta_stage(
        self,
        topic: TopicCandidate,
        keywords: KeywordPlan,
        output: WriterOutput,
    ) -> MetaResult:
        async with self._stage("meta") as meta:
            result = await generate_meta(
                self.client, topic, keywords, output
            )
            self.meta = result
            meta.update(
                provider=(
                    None if result.fallback else self.client.last_provider
                ),
                model=None if result.fallback else self.client.last_model,
                fallback=result.fallback,
                title_length=len(result.meta_title),
                description_length=len(result.meta_description),
            )
            return result

    async def _faq_stage(
        self,
        topic: TopicCandidate,
        keywords: KeywordPlan,
        research: ResearchBundle,
        output: WriterOutput,
    ) -> tuple[FAQBundle, WriterOutput]:
        async with self._stage("faq") as meta:
            bundle = await generate_faqs(
                self.client,
                content=output.content,
                keywords=keywords,
                research=research,
            )
            final_content = insert_faq_section(
                output.content, bundle.faqs
            )
            final_output = output.model_copy(
                update={"content": final_content}
            )
            self.faqs = bundle
            self.output = final_output
            meta.update(
                provider=self.client.last_provider,
                model=self.client.last_model,
                count=len(bundle.faqs),
            )
            return bundle, final_output

    async def _publish_stage(
        self,
        topic: TopicCandidate,
        keywords: KeywordPlan,
        research: ResearchBundle,
        output: WriterOutput,
        report: GateReport,
        meta: MetaResult,
        bundle: FAQBundle,
        final_words: int,
    ) -> None:
        async with self._stage("publish") as stage_meta:
            if not self.post_id:
                raise PipelineStageError("Draft post is missing.")
            if not self.production:
                raise PipelineStageError("Visual production is missing.")
            if not report.passed:
                raise PipelineStageError("Gate did not pass.")
            flat_keywords = list(
                dict.fromkeys(
                    [
                        keywords.primary,
                        *keywords.secondary,
                        *keywords.long_tail,
                        *keywords.lsi,
                    ]
                )
            )
            tags = list(
                dict.fromkeys(
                    [topic.category.value, *keywords.secondary[:3], "atlas"]
                )
            )
            await self.repo.update_post(
                self.post_id,
                title=output.title,
                slug=output.slug,
                excerpt=output.excerpt,
                content=output.content,
                cover_image_url=self.production.cover_image_url,
                content_images=[
                    image.model_dump(mode="json")
                    for image in self.production.content_images
                ],
                tags=tags,
                keywords=flat_keywords,
                keywords_json=keywords.model_dump(mode="json"),
                meta_title=meta.meta_title,
                meta_description=meta.meta_description,
                word_count=final_words,
                seo_score=report.score,
                gate_report=report.model_dump(mode="json"),
                faq_json=bundle.model_dump(mode="json", by_alias=True),
                visuals_json=_visual_payload(self.production),
                research_json=research.model_dump(mode="json"),
            )
            published = await self.repo.publish(self.post_id)
            if not published:
                raise PipelineStageError(
                    "Publish RPC rejected the draft; requirements not met."
                )
            if self.topic_id:
                try:
                    await self.repo.mark_topic_used(
                        self.topic_id, self.post_id
                    )
                except Exception:  # noqa: BLE001 - post is already published
                    logger.exception(
                        "Could not mark topic used for run %s", self.run_id
                    )
                    stage_meta["topic_update_warning"] = True
            stage_meta.update(
                post_id=self.post_id,
                published=True,
                score=report.score,
            )

    async def _deploy_stage(self) -> None:
        async with self._stage("deploy") as meta:
            try:
                result = await trigger_deploy(self.settings)
                meta.update(result.as_dict())
                self.provider_summary["deploy"] = result.as_dict()
                if not result.triggered:
                    logger.warning(
                        "Deploy hook failed for run %s: %s",
                        self.run_id,
                        result.error,
                    )
            except DeployError as exc:
                meta.update(triggered=False, error=_short(exc))
                self.provider_summary["deploy"] = {
                    "triggered": False,
                    "error": _short(exc),
                }
            except Exception as exc:  # noqa: BLE001 - deploy is non-fatal
                logger.exception("Deploy hook failed for run %s", self.run_id)
                meta.update(triggered=False, error=_short(exc))
                self.provider_summary["deploy"] = {
                    "triggered": False,
                    "error": _short(exc),
                }


async def start_pipeline(
    trigger: str = "admin",
    *,
    settings: Settings | Any | None = None,
    repository: BlogRepository | None = None,
) -> str:
    current = settings or get_settings()
    if not current.blog_enabled:
        raise PipelineNotConfigured("Blog generation is disabled.")
    if not current.blog_pipeline_ready:
        raise PipelineNotConfigured(
            "Blog pipeline configuration is incomplete."
        )
    normalized = (trigger or "admin").strip().lower()
    if normalized not in VALID_TRIGGERS:
        raise ValueError(f"Unsupported trigger: {trigger}")
    repo = repository or BlogRepository()
    run_id = await repo.acquire_run(normalized)
    if not run_id:
        raise PipelineBusy("A blog pipeline run is already active.")
    return run_id


async def execute_pipeline(
    run_id: str,
    trigger: str = "admin",
    *,
    settings: Settings | Any | None = None,
    repository: BlogRepository | None = None,
) -> PipelineOutcome:
    runner = PipelineRunner(
        run_id,
        trigger,
        settings=settings,
        repository=repository,
    )
    return await runner.run()
