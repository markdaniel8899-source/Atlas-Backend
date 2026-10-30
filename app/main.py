from __future__ import annotations

import sys
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import ai_service
from app.config import get_settings
from app.routers import blog, blog_admin, chat, gamification, quizzes, roadmaps
from app.scheduler import start_scheduler, stop_scheduler

# Background-task logs go to stdout (the nohup log). A non-tty stdout is
# block-buffered, so those prints never appeared next to uvicorn's 202 line.
# Line-buffer so every print flushes immediately; UTF-8 + errors="replace"
# so emoji/arrow characters can never raise UnicodeEncodeError in a print.
try:
    sys.stdout.reconfigure(
        line_buffering=True, encoding="utf-8", errors="replace"
    )
except Exception:  # noqa: BLE001 - stdout swapped by a test harness → skip
    pass

REQUIRED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://atlas-frontend-pearl.vercel.app",
]


class ErrorCatchMiddleware:
    """Last line of defence for unhandled exceptions, INSIDE CORS.

    Starlette's ServerErrorMiddleware sits *outside* CORSMiddleware, so a
    bare 500 it emits carries no Access-Control-* headers — the browser
    then reports the request as a CORS failure and hides the real error.
    This middleware is registered BEFORE CORSMiddleware (add_middleware
    prepends, so the CORS layer added afterwards ends up outermost), which
    means any response it sends still travels through CORSMiddleware's
    send-wrapper and keeps its CORS headers. The full traceback is printed
    to stdout so it lands in the nohup server log.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:  # noqa: BLE001 - convert to a clean, CORS-safe 500
            if response_started:
                raise
            print(
                "[500] Unhandled exception:\n" + traceback.format_exc(),
                flush=True,
            )
            await JSONResponse(
                status_code=500,
                content={
                    "detail": (
                        "Internal server error. "
                        "Check the backend log for the full traceback."
                    )
                },
            )(scope, receive, send)


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler = start_scheduler()
        app.state.blog_scheduler = scheduler
        try:
            yield
        finally:
            stop_scheduler(scheduler)

    app = FastAPI(
        title="ATLAS AI Service",
        description="Model-routed learning assistant backed by the Groq API.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Added FIRST: add_middleware prepends, so CORSMiddleware below ends up
    # outermost and ErrorCatchMiddleware sits directly inside it.
    app.add_middleware(ErrorCatchMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(
            dict.fromkeys([*REQUIRED_ORIGINS, *settings.allowed_origins])
        ),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(blog.router)
    app.include_router(blog_admin.router)
    app.include_router(chat.router)
    app.include_router(quizzes.router)
    app.include_router(roadmaps.router)
    app.include_router(gamification.router)

    @app.exception_handler(ai_service.AIServiceError)
    async def handle_ai_error(
        request: Request, exc: ai_service.AIServiceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc), "error": "ai_service_error"},
        )

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "ai_configured": settings.ai_ready,
            "blog_configured": settings.blog_ready,
            "blog_pipeline_configured": settings.blog_pipeline_ready,
            "blog_writer_configured": settings.blog_writer_ready,
            "r2_configured": settings.r2_ready,
            "deploy_hook_configured": settings.deploy_ready,
            "blog_cron_enabled": settings.blog_cron_enabled,
            "routes": {
                task.value: ai_service.pick_model(task)
                for task in ai_service.Task
            },
        }

    return app


app = create_app()
