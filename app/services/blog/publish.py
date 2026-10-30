from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from supabase import Client, create_client

from app.cache import clear_all
from app.config import get_settings

from .types import (
    KeywordPlan,
    ResearchBundle,
    TopicCandidate,
    TopicHistoryEntry,
    VisualProduction,
    WriterOutput,
)

logger = logging.getLogger(__name__)


class BlogStorageError(RuntimeError):
    pass


class PublishRejected(RuntimeError):
    pass


_client: Client | None = None


def _get_client() -> Client:
    global _client
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise BlogStorageError(
            "Supabase write access is not configured: set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY."
        )
    if _client is None:
        _client = create_client(
            settings.supabase_url, settings.supabase_service_role_key
        )
    return _client


def _rows(result: Any) -> list[dict[str, Any]]:
    data = getattr(result, "data", None)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _first_id(result: Any) -> str | None:
    rows = _rows(result)
    if rows:
        value = rows[0].get("id")
        return str(value) if value else None
    return None


def _scalar_id(data: Any) -> str | None:
    if isinstance(data, str):
        return data or None
    if isinstance(data, list):
        return _scalar_id(data[0]) if data else None
    if isinstance(data, dict):
        for key in ("blog_acquire_run", "id", "result"):
            if key in data:
                return _scalar_id(data[key])
        for value in data.values():
            if isinstance(value, str):
                return value or None
    return None


def _as_bool(data: Any) -> bool:
    if isinstance(data, bool):
        return data
    if isinstance(data, list):
        return _as_bool(data[0]) if data else False
    if isinstance(data, dict):
        for key in ("blog_publish_post", "blog_unpublish_post", "result"):
            if key in data:
                return _as_bool(data[key])
    return bool(data)


def _iso(value: datetime | None = None) -> str:
    return (value or datetime.now(timezone.utc)).isoformat()


def _attr(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


class BlogRepository:
    async def acquire_run(self, trigger: str) -> str | None:
        def run() -> str | None:
            result = (
                _get_client()
                .rpc("blog_acquire_run", {"p_trigger": trigger})
                .execute()
            )
            return _scalar_id(getattr(result, "data", None))

        return await asyncio.to_thread(run)

    async def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return

        def run() -> None:
            _get_client().table("blog_pipeline_runs").update(fields).eq(
                "id", run_id
            ).execute()

        await asyncio.to_thread(run)

    async def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        score: int | None = None,
        word_count: int | None = None,
        provider_summary: dict[str, Any] | None = None,
    ) -> None:
        fields: dict[str, Any] = {
            "status": status,
            "finished_at": _iso(),
        }
        if error is not None:
            fields["error"] = error[:4000]
        if score is not None:
            fields["score"] = max(0, min(100, int(score)))
        if word_count is not None:
            fields["word_count"] = max(0, int(word_count))
        if provider_summary is not None:
            fields["provider_summary"] = provider_summary
        await self.update_run(run_id, **fields)

    async def start_step(
        self,
        run_id: str,
        stage: str,
        *,
        attempt: int = 1,
        provider: str | None = None,
        model: str | None = None,
    ) -> int | None:
        payload = {
            "run_id": run_id,
            "stage": stage,
            "status": "started",
            "attempt": max(1, attempt),
            "provider": provider,
            "model": model,
        }

        def run() -> int | None:
            result = (
                _get_client()
                .table("blog_pipeline_steps")
                .insert(payload)
                .execute()
            )
            rows = _rows(result)
            if not rows:
                return None
            value = rows[0].get("id")
            return int(value) if value is not None else None

        return await asyncio.to_thread(run)

    async def finish_step(
        self,
        step_id: int | None,
        *,
        status: str,
        duration_ms: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        detail: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if step_id is None:
            return
        payload: dict[str, Any] = {
            "status": status,
            "provider": provider,
            "model": model,
            "detail": detail or {},
        }
        if duration_ms is not None:
            payload["duration_ms"] = max(0, int(duration_ms))
        if error is not None:
            payload["error"] = error[:4000]

        def run() -> None:
            _get_client().table("blog_pipeline_steps").update(payload).eq(
                "id", step_id
            ).execute()

        await asyncio.to_thread(run)

    async def active_run(self) -> dict[str, Any] | None:
        def run() -> dict[str, Any] | None:
            result = (
                _get_client()
                .table("blog_pipeline_runs")
                .select("*")
                .eq("status", "running")
                .order("started_at", desc=True)
                .limit(1)
                .execute()
            )
            rows = _rows(result)
            return rows[0] if rows else None

        return await asyncio.to_thread(run)

    async def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        def run() -> list[dict[str, Any]]:
            result = (
                _get_client()
                .table("blog_pipeline_runs")
                .select("*")
                .order("started_at", desc=True)
                .limit(max(1, min(100, limit)))
                .execute()
            )
            return _rows(result)

        return await asyncio.to_thread(run)

    async def count_runs_since(
        self, since: datetime, *, statuses: Sequence[str] = ("completed",)
    ) -> int:
        def run() -> int:
            query = (
                _get_client()
                .table("blog_pipeline_runs")
                .select("id", count="exact", head=True)
                .gte("started_at", since.isoformat())
            )
            if statuses:
                query = query.in_("status", list(statuses))
            result = query.execute()
            return int(getattr(result, "count", 0) or 0)

        return await asyncio.to_thread(run)

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        def run() -> dict[str, Any] | None:
            result = (
                _get_client()
                .table("blog_pipeline_runs")
                .select("*")
                .eq("id", run_id)
                .limit(1)
                .execute()
            )
            rows = _rows(result)
            return rows[0] if rows else None

        return await asyncio.to_thread(run)

    async def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        def run() -> list[dict[str, Any]]:
            result = (
                _get_client()
                .table("blog_pipeline_steps")
                .select("*")
                .eq("run_id", run_id)
                .order("created_at")
                .execute()
            )
            return _rows(result)

        return await asyncio.to_thread(run)

    async def topic_history(self, limit: int = 100) -> list[TopicHistoryEntry]:
        def run() -> list[TopicHistoryEntry]:
            result = (
                _get_client()
                .table("blog_topics")
                .select("topic, primary_keyword, category, used_at")
                .eq("status", "used")
                .order("used_at", desc=True)
                .limit(max(1, min(200, limit)))
                .execute()
            )
            history: list[TopicHistoryEntry] = []
            for row in _rows(result):
                if not row.get("used_at"):
                    continue
                try:
                    history.append(TopicHistoryEntry.model_validate(row))
                except Exception:
                    continue
            return history

        return await asyncio.to_thread(run)

    async def save_topic(self, topic: TopicCandidate) -> str:
        from .topics import topic_slug

        payload = {
            "topic": topic.topic,
            "slug": topic_slug(topic.topic),
            "category": topic.category.value,
            "primary_keyword": topic.primary_keyword,
            "secondary_keywords": list(topic.secondary_keywords),
            "source": topic.source.value,
            "trend_score": topic.trend_score,
            "seo_score": topic.seo_score,
            "product_fit": topic.product_fit,
            "diversity_score": getattr(topic, "diversity_score", None),
            "total_score": getattr(topic, "total_score", None),
            "score_breakdown": getattr(topic, "score_breakdown", {}),
            "status": "selected",
        }

        def run() -> str:
            result = (
                _get_client()
                .table("blog_topics")
                .upsert(payload, on_conflict="slug")
                .execute()
            )
            topic_id = _first_id(result)
            if topic_id:
                return topic_id
            lookup = (
                _get_client()
                .table("blog_topics")
                .select("id")
                .eq("slug", payload["slug"])
                .limit(1)
                .execute()
            )
            found = _first_id(lookup)
            if not found:
                raise BlogStorageError("Could not persist the selected topic.")
            return found

        return await asyncio.to_thread(run)

    async def mark_topic_used(self, topic_id: str, post_id: str) -> None:
        def run() -> None:
            _get_client().table("blog_topics").update(
                {
                    "status": "used",
                    "used_post_id": post_id,
                    "used_at": _iso(),
                }
            ).eq("id", topic_id).execute()

        await asyncio.to_thread(run)

    async def insert_draft(
        self,
        output: WriterOutput,
        *,
        run_id: str,
        topic_id: str,
        topic: TopicCandidate,
        keywords: KeywordPlan,
    ) -> str:
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
        payload = {
            "title": output.title,
            "slug": output.slug,
            "content": output.content,
            "excerpt": output.excerpt,
            "tags": tags,
            "keywords": flat_keywords,
            "keywords_json": keywords.model_dump(mode="json"),
            "published_at": None,
            "topic_id": topic_id,
            "pipeline_run_id": run_id,
            "word_count": len(output.content.split()),
        }

        def run() -> str:
            result = (
                _get_client()
                .table("blog_posts")
                .insert(payload)
                .execute()
            )
            post_id = _first_id(result)
            if not post_id:
                raise BlogStorageError("Could not insert the blog draft.")
            return post_id

        return await asyncio.to_thread(run)

    async def update_post(self, post_id: str, **fields: Any) -> None:
        if not fields:
            return

        def run() -> None:
            _get_client().table("blog_posts").update(fields).eq(
                "id", post_id
            ).execute()

        await asyncio.to_thread(run)

    async def get_post(self, post_id: str) -> dict[str, Any] | None:
        def run() -> dict[str, Any] | None:
            result = (
                _get_client()
                .table("blog_posts")
                .select("*")
                .eq("id", post_id)
                .limit(1)
                .execute()
            )
            rows = _rows(result)
            return rows[0] if rows else None

        return await asyncio.to_thread(run)

    def _exists_sync(
        self,
        column: str,
        value: str,
        *,
        exclude_post_id: str | None = None,
    ) -> bool:
        query = (
            _get_client()
            .table("blog_posts")
            .select("id")
            .eq(column, value)
            .limit(1)
        )
        if exclude_post_id:
            query = query.neq("id", exclude_post_id)
        return bool(_rows(query.execute()))

    def title_or_slug_exists_sync(
        self,
        title: str,
        slug: str,
        *,
        exclude_post_id: str | None = None,
    ) -> bool:
        return self._exists_sync(
            "slug", slug, exclude_post_id=exclude_post_id
        ) or self._exists_sync(
            "title", title, exclude_post_id=exclude_post_id
        )

    async def slug_exists(
        self,
        slug: str,
        *,
        exclude_post_id: str | None = None,
    ) -> bool:
        return await asyncio.to_thread(
            self._exists_sync, "slug", slug, exclude_post_id=exclude_post_id
        )

    async def insert_sources(
        self, post_id: str, research: ResearchBundle
    ) -> None:
        payloads: list[dict[str, Any]] = []
        for index, source in enumerate(research.sources):
            payloads.append(
                {
                    "post_id": post_id,
                    "url": source.url,
                    "domain": source.domain,
                    "title": source.title,
                    "accessed_at": research.accessed_at.isoformat(),
                    "used_for": "citation",
                    "is_primary": index == 0,
                }
            )
        if not payloads:
            return

        def run() -> None:
            table = _get_client().table("blog_post_sources")
            for payload in payloads:
                try:
                    table.insert(payload).execute()
                except Exception as exc:  # noqa: BLE001 - duplicate URLs are expected
                    logger.warning("Source insert skipped: %s", exc)

        await asyncio.to_thread(run)

    async def record_media(
        self, post_id: str, production: VisualProduction
    ) -> None:
        produced = getattr(production, "visuals", ()) or ()
        by_url = {
            str(_attr(item, "url")): item
            for item in produced
            if _attr(item, "url")
        }
        rows: list[dict[str, Any]] = [
            {
                "provider": "generated",
                "external_id": None,
                "url": production.cover_image_url,
                "kind": "cover",
            }
        ]
        for image in getattr(production, "content_images", ()) or ():
            url = str(_attr(image, "url", "") or "")
            if not url:
                continue
            visual = by_url.get(url)
            provider = str(
                _attr(visual, "provider", "generated") or "generated"
            )
            if provider not in {"pixabay", "branded", "generated"}:
                provider = "generated"
            rows.append(
                {
                    "provider": provider,
                    "external_id": _attr(visual, "external_id"),
                    "url": url,
                    "kind": "content",
                }
            )

        seen: set[str] = set()
        payloads: list[dict[str, Any]] = []
        for row in rows:
            url = str(row.get("url") or "")
            if not url:
                continue
            url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
            if url_hash in seen:
                continue
            seen.add(url_hash)
            payloads.append(
                {
                    **row,
                    "url_hash": url_hash,
                    "post_id": post_id,
                }
            )
        if not payloads:
            return

        def run() -> None:
            table = _get_client().table("blog_media_usage")
            for payload in payloads:
                try:
                    table.insert(payload).execute()
                except Exception as exc:  # noqa: BLE001 - URL/media reuse is expected
                    logger.warning("Media usage insert skipped: %s", exc)

        await asyncio.to_thread(run)

    async def record_provider_usage(self, attempts: Sequence[Any]) -> None:
        buckets: dict[tuple[str, str], dict[str, int]] = {}
        for attempt in attempts:
            provider = str(_attr(attempt, "provider", "") or "")
            model = str(_attr(attempt, "model", "") or "")
            status = str(_attr(attempt, "status", "") or "")
            if not provider or not model or status == "skipped":
                continue
            bucket = buckets.setdefault(
                (provider, model),
                {"requests": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0},
            )
            bucket["requests"] += 1
            bucket["input_tokens"] += max(
                0, int(_attr(attempt, "input_tokens", 0) or 0)
            )
            bucket["output_tokens"] += max(
                0, int(_attr(attempt, "output_tokens", 0) or 0)
            )
            if status == "failed":
                bucket["errors"] += 1
        if not buckets:
            return

        usage_date = datetime.now(timezone.utc).date().isoformat()

        def run() -> None:
            client = _get_client()
            table = client.table("blog_provider_usage")
            for (provider, model), bucket in buckets.items():
                existing = (
                    table.select("requests, input_tokens, output_tokens, errors")
                    .eq("provider", provider)
                    .eq("model", model)
                    .eq("usage_date", usage_date)
                    .limit(1)
                    .execute()
                )
                rows = _rows(existing)
                values = {
                    "requests": bucket["requests"],
                    "input_tokens": bucket["input_tokens"],
                    "output_tokens": bucket["output_tokens"],
                    "errors": bucket["errors"],
                }
                if rows:
                    previous = rows[0]
                    for key in values:
                        merged = int(previous.get(key, 0) or 0) + values[key]
                        values[key] = merged
                    table.update(values).eq("provider", provider).eq(
                        "model", model
                    ).eq("usage_date", usage_date).execute()
                else:
                    table.insert(
                        {
                            "provider": provider,
                            "model": model,
                            "usage_date": usage_date,
                            **values,
                        }
                    ).execute()

        await asyncio.to_thread(run)

    async def publish(self, post_id: str) -> bool:
        def run() -> bool:
            result = (
                _get_client()
                .rpc("blog_publish_post", {"p_post_id": post_id})
                .execute()
            )
            published = _as_bool(getattr(result, "data", None))
            if published:
                clear_all()
            return published

        return await asyncio.to_thread(run)

    async def unpublish(self, post_id: str) -> bool:
        def run() -> bool:
            result = (
                _get_client()
                .rpc("blog_unpublish_post", {"p_post_id": post_id})
                .execute()
            )
            unpublished = _as_bool(getattr(result, "data", None))
            if unpublished:
                clear_all()
            return unpublished

        return await asyncio.to_thread(run)
