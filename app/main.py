from __future__ import annotations

from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import ai_service
from app.config import get_settings
from app.routers import chat, quizzes, roadmaps
from app.services import auto_blog

REQUIRED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://atlas-frontend-pearl.vercel.app",
]


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Auto Blog Writer: one article every day at 09:00 UTC.
        # NOTE: BackgroundScheduler runs per-process — deploy with a single
        # uvicorn worker (or accept one job per worker; the pipeline lock
        # only protects within a process).
        app.state.blog_scheduler = None
        if settings.blog_ready:
            scheduler = BackgroundScheduler(timezone="UTC")
            scheduler.add_job(
                auto_blog.run_scheduled_job,
                trigger=CronTrigger(hour=9, minute=0, timezone="UTC"),
                id="auto-blog-daily",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=3600,  # boot within 1h of 9:00 → still fires
            )
            scheduler.start()
            app.state.blog_scheduler = scheduler
            print("[AutoBlog] scheduled — every day at 09:00 UTC")
        else:
            print(
                "[AutoBlog] scheduler disabled — set GOOGLE_API_KEY and "
                "SUPABASE_SERVICE_ROLE_KEY (manual POST /api/blog/generate "
                "will also be unavailable)"
            )
        yield
        if app.state.blog_scheduler is not None:
            app.state.blog_scheduler.shutdown(wait=False)

    app = FastAPI(
        title="ATLAS AI Service",
        description="Model-routed learning assistant backed by the Groq API.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(
            dict.fromkeys([*REQUIRED_ORIGINS, *settings.allowed_origins])
        ),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router)
    app.include_router(quizzes.router)
    app.include_router(roadmaps.router)

    @app.exception_handler(ai_service.AIServiceError)
    async def handle_ai_error(
        request: Request, exc: ai_service.AIServiceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc), "error": "ai_service_error"},
        )

    @app.exception_handler(auto_blog.BlogPipelineError)
    async def handle_blog_error(
        request: Request, exc: auto_blog.BlogPipelineError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc), "error": "blog_pipeline_error"},
        )

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "ai_configured": settings.ai_ready,
            "blog_configured": settings.blog_ready,
            "routes": {
                task.value: ai_service.pick_model(task)
                for task in ai_service.Task
            },
        }

    @app.post("/api/blog/generate", tags=["blog"], status_code=201)
    async def generate_blog_post() -> dict[str, object]:
        """Run the Auto Blog pipeline now: Tavily research → Gemini article →
        Pollinations cover URL → Supabase save."""
        if not settings.blog_ready:
            raise auto_blog.BlogPipelineError(
                "Auto Blog is not configured: set GOOGLE_API_KEY and "
                "SUPABASE_SERVICE_ROLE_KEY on the server.",
                status_code=503,
            )
        result = await auto_blog.run_pipeline()
        if result is None:
            # Stage errors are already logged as [AutoBlog] … — graceful exit.
            raise auto_blog.BlogPipelineError(
                "Blog generation failed — see server logs for the stage error.",
                status_code=502,
            )
        return {"status": "created", **result}

    return app


app = create_app()
