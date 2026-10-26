from __future__ import annotations

import asyncio
import secrets
import sys
import traceback
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app import ai_service
from app.config import get_settings
from app.deps import fetch_profile_role, verify_bearer_token
from app.routers import blog, chat, gamification, quizzes, roadmaps

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
    settings = get_settings()
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
        yield

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
            "routes": {
                task.value: ai_service.pick_model(task)
                for task in ai_service.Task
            },
        }

    return app


app = create_app()
