from __future__ import annotations

from typing import Any, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
)

from app.config import get_settings
from app.deps import require_blog_admin
from app.services.blog.pipeline import (
    PipelineBusy,
    PipelineNotConfigured,
    execute_pipeline,
    start_pipeline,
)
from app.services.blog.publish import BlogRepository, BlogStorageError
from app.services.blog.vercel import DeployError, trigger_deploy

router = APIRouter(
    prefix="/api/blog",
    tags=["blog-admin"],
    dependencies=[Depends(require_blog_admin)],
)


def _repo() -> BlogRepository:
    return BlogRepository()


async def _deploy_safely() -> dict[str, Any]:
    try:
        result = await trigger_deploy(get_settings())
        return result.as_dict()
    except DeployError as exc:
        return {"triggered": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - publish already succeeded
        return {"triggered": False, "error": f"{type(exc).__name__}: {exc}"}


@router.post("/pipeline/run", status_code=202)
async def trigger_pipeline(
    background_tasks: BackgroundTasks,
    trigger: Literal["cron", "admin", "manual", "retry"] = Query(
        default="admin"
    ),
) -> dict[str, Any]:
    settings = get_settings()
    if not settings.blog_pipeline_ready:
        raise HTTPException(
            status_code=503,
            detail="Blog pipeline configuration is incomplete.",
        )
    try:
        run_id = await start_pipeline(trigger)
    except PipelineBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PipelineNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BlogStorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    background_tasks.add_task(execute_pipeline, run_id, trigger)
    return {"run_id": run_id, "status": "running", "trigger": trigger}


@router.get("/pipeline/status")
async def pipeline_status() -> dict[str, Any]:
    try:
        run = await _repo().active_run()
    except BlogStorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"run": run}


@router.get("/pipeline/runs")
async def pipeline_runs(
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    try:
        items = await _repo().list_runs(limit)
    except BlogStorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"items": items, "count": len(items)}


@router.get("/pipeline/runs/{run_id}")
async def pipeline_run_detail(run_id: str) -> dict[str, Any]:
    repo = _repo()
    try:
        run = await repo.get_run(run_id)
        if run is None:
            raise HTTPException(
                status_code=404, detail="Pipeline run not found."
            )
        steps = await repo.get_steps(run_id)
        post = None
        if run.get("post_id"):
            post = await repo.get_post(str(run["post_id"]))
    except BlogStorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "run": run,
        "steps": steps,
        "gate_report": (post or {}).get("gate_report"),
        "post": post,
    }


@router.delete("/pipeline/runs/{run_id}")
async def delete_pipeline_run(run_id: str) -> dict[str, Any]:
    repo = _repo()
    try:
        deleted = await repo.delete_run(run_id)
    except BlogStorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Pipeline run not found.")
    return {"deleted": True, "run_id": run_id}


@router.get("/admin/posts/{post_id}/preview")
async def preview_post(post_id: str) -> dict[str, Any]:
    try:
        post = await _repo().get_post(post_id)
    except BlogStorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if post is None:
        raise HTTPException(status_code=404, detail="Blog post not found.")
    return post


@router.post("/admin/posts/{post_id}/publish")
async def publish_post(post_id: str) -> dict[str, Any]:
    repo = _repo()
    try:
        if await repo.get_post(post_id) is None:
            raise HTTPException(
                status_code=404, detail="Blog post not found."
            )
        published = await repo.publish(post_id)
    except BlogStorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not published:
        raise HTTPException(
            status_code=422,
            detail="Post does not meet the publish requirements.",
        )
    return {
        "published": True,
        "post_id": post_id,
        "deploy": await _deploy_safely(),
    }


@router.post("/admin/posts/{post_id}/unpublish")
async def unpublish_post(post_id: str) -> dict[str, Any]:
    repo = _repo()
    try:
        if await repo.get_post(post_id) is None:
            raise HTTPException(
                status_code=404, detail="Blog post not found."
            )
        unpublished = await repo.unpublish(post_id)
    except BlogStorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not unpublished:
        raise HTTPException(
            status_code=422,
            detail="Post is not currently published.",
        )
    return {
        "published": False,
        "post_id": post_id,
        "deploy": await _deploy_safely(),
    }


@router.post("/admin/deploy")
async def rebuild_site() -> dict[str, Any]:
    settings = get_settings()
    if not settings.vercel_deploy_hook_url:
        raise HTTPException(
            status_code=503,
            detail="VERCEL_DEPLOY_HOOK_URL is not configured.",
        )
    try:
        result = await trigger_deploy(settings)
    except DeployError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not result.triggered:
        raise HTTPException(
            status_code=502,
            detail=result.error or "Deploy hook failed.",
        )
    return result.as_dict()
