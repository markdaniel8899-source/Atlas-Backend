from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str
    groq_base_url: str
    groq_blog_api_key: str
    groq_blog_model: str
    allowed_origins: tuple[str, ...]
    request_timeout: float
    max_retries: int
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    google_api_key: str
    blog_model: str
    tavily_api_key: str

    @property
    def ai_ready(self) -> bool:
        return len(self.groq_api_key) > 0

    @property
    def supabase_ready(self) -> bool:
        return bool(self.supabase_url) and bool(self.supabase_anon_key)

    @property
    def blog_ready(self) -> bool:
        """Auto Blog needs the DEDICATED Groq blog key to write and the
        service-role key to save. GOOGLE_API_KEY is optional — it only
        powers the Gemini fallback when Groq hits 429/5xx."""
        return bool(self.groq_blog_api_key) and bool(
            self.supabase_service_role_key
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:5173"
        ).split(",")
        if origin.strip()
    )
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_base_url=os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        ).strip(),
        # Dedicated blog writer key — deliberately NOT GROQ_API_KEY, which is
        # reserved for Chat/Quiz so blog traffic can never starve the app.
        groq_blog_api_key=os.getenv("GROQ_BLOG_API_KEY", "").strip(),
        # NOTE: llama-3.1-70b-versatile was decommissioned by Groq (Jan 2025);
        # openai/gpt-oss-120b is its live replacement.
        groq_blog_model=os.getenv("GROQ_BLOG_MODEL", "openai/gpt-oss-120b").strip()
        or "openai/gpt-oss-120b",
        allowed_origins=origins,
        request_timeout=float(os.getenv("AI_TIMEOUT_SECONDS", "180")),
        max_retries=int(os.getenv("AI_MAX_RETRIES", "1")),
        supabase_url=os.getenv("SUPABASE_URL", "").strip().rstrip("/"),
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", "").strip(),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip(),
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        blog_model=os.getenv("GOOGLE_BLOG_MODEL", "gemini-3.8-flash").strip()
        or "gemini-3.8-flash",
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
    )
