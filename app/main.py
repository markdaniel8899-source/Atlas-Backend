from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import ai_service
from app.config import get_settings
from app.routers import chat, quizzes, roadmaps

REQUIRED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://atlas-frontend-pearl.vercel.app",
]


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ATLAS AI Service",
        description="Model-routed learning assistant backed by the Groq API.",
        version="1.0.0",
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

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "ai_configured": settings.ai_ready,
            "routes": {
                task.value: ai_service.pick_model(task)
                for task in ai_service.Task
            },
        }

    return app


app = create_app()
